"""
train_xlnet.py
Replication of Kwako & Ormerod (BEA 2024):
  - XLNet-base-cased fine-tuned per prompt on PERSUADE 2.0
  - Regression head (MSE loss)
  - AdamW + linear LR schedule, early stopping on QWK
  - Evaluates QWK, SMD per demographic group

Paths default to repo-relative locations and can be overridden via env vars:
    DATA_BASE     defaults to <repo_root>/../DATA
    RESULTS_DIR   defaults to <repo_root>/results/xlnet/<RUN_VERSION>
    RUN_VERSION   defaults to "baseline_replication"
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    XLNetTokenizer,
    XLNetModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "xlnet-base-cased"
MAX_LEN      = int(os.environ.get("MAX_LEN", 2048))
LR           = float(os.environ.get("LR", 5e-6))    # Kwako & Ormerod (BEA 2024)
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 8)) # Kwako & Ormerod (BEA 2024)
EPOCHS       = int(os.environ.get("EPOCHS", 20))    # Kwako & Ormerod (BEA 2024)
DEV_SPLIT    = 0.10
RANDOM_SEED  = 42
SCORE_COL    = "holistic_essay_score"
TEXT_COL     = "full_text"
PROMPT_COL   = "prompt_name"
DEMO_COLS    = ["gender", "race_ethnicity", "economically_disadvantaged",
                "student_disability_status", "ell_status"]

RUN_VERSION  = os.environ.get("RUN_VERSION", "baseline_replication")

# Paths default to repo-relative locations; override via env vars on RunPod etc.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR  = os.environ.get(
    "RESULTS_DIR",
    os.path.join(REPO_ROOT, "results", "xlnet", RUN_VERSION)
)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{RUN_VERSION}] device={device}  results={RESULTS_DIR}")

# ── Metrics ────────────────────────────────────────────────────────────────
def quadratic_weighted_kappa(y_true, y_pred, min_score=1, max_score=6):
    y_true = np.clip(np.round(y_true).astype(int), min_score, max_score)
    y_pred = np.clip(np.round(y_pred).astype(int), min_score, max_score)
    n = max_score - min_score + 1
    O = np.zeros((n, n))
    for t, p in zip(y_true, y_pred):
        O[t - min_score][p - min_score] += 1
    hist_true = np.bincount(y_true - min_score, minlength=n)
    hist_pred = np.bincount(y_pred - min_score, minlength=n)
    E = np.outer(hist_true, hist_pred).astype(float)
    E /= E.sum()
    O /= O.sum()
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    return 1 - (W * O).sum() / (W * E).sum()

def smd(group_a, group_b):
    pooled = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
    return (group_a.mean() - group_b.mean()) / pooled if pooled > 0 else 0.0

def to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def compute_bias_metrics(df, pred_col, score_col):
    results = {}
    pairs = {
        "gender":                    ("f", "m"),
        "ell_status":                ("no", "yes"),
        "economically_disadvantaged":("not economically disadvantaged",
                                      "economically disadvantaged"),
        "student_disability_status": ("not identified as having disability",
                                      "identified as having disability"),
    }
    for col, (a, b) in pairs.items():
        if col not in df.columns:
            continue
        col_vals = df[col].astype(str).str.lower().str.strip()
        ga_pred = df.loc[col_vals == a, pred_col].values
        gb_pred = df.loc[col_vals == b, pred_col].values
        ga_true = df.loc[col_vals == a, score_col].values
        gb_true = df.loc[col_vals == b, score_col].values
        if len(ga_pred) < 5 or len(gb_pred) < 5:
            continue
        results[col] = {
            "human_smd":  round(smd(ga_true, gb_true), 4),
            "model_smd":  round(smd(ga_pred, gb_pred), 4),
            "n_a": len(ga_pred),
            "n_b": len(gb_pred),
        }
    return results

# ── Model ──────────────────────────────────────────────────────────────────
class XLNetRegressor(torch.nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.xlnet   = XLNetModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.regressor = torch.nn.Linear(self.xlnet.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, -1, :]
        cls = self.dropout(cls)
        return self.regressor(cls).squeeze(-1)

# ── Dataset ────────────────────────────────────────────────────────────────
class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, texts, scores, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get(
                "token_type_ids", torch.zeros_like(self.encodings["input_ids"])
            )[idx],
            "labels": self.scores[idx],
        }

# ── Training per prompt ────────────────────────────────────────────────────
def train_prompt(prompt_name, train_df, test_df, tokenizer):
    tr, dev = train_test_split(train_df, test_size=DEV_SPLIT, random_state=RANDOM_SEED)
    tr  = tr.reset_index(drop=True)
    dev = dev.reset_index(drop=True)

    print(f"\n[{prompt_name}] train={len(tr)} dev={len(dev)} test={len(test_df)}")

    train_ds = EssayDataset(tr[TEXT_COL].tolist(),  tr[SCORE_COL].tolist(),  tokenizer)
    dev_ds   = EssayDataset(dev[TEXT_COL].tolist(), dev[SCORE_COL].tolist(), tokenizer)
    test_ds  = EssayDataset(test_df[TEXT_COL].tolist(), test_df[SCORE_COL].tolist(), tokenizer)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = XLNetRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn = torch.nn.MSELoss()

    best_qwk   = -1
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            loss = loss_fn(preds, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        dev_preds, dev_true = [], []
        with torch.no_grad():
            for batch in dev_dl:
                preds = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device),
                )
                dev_preds.extend(preds.cpu().numpy())
                dev_true.extend(batch["labels"].numpy())

        dev_qwk = quadratic_weighted_kappa(np.array(dev_true), np.array(dev_preds))
        print(f"  ep{epoch+1:02d} loss={train_loss/len(train_dl):.4f} dev_qwk={dev_qwk:+.4f}")

        if dev_qwk > best_qwk:
            best_qwk   = dev_qwk
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_dl:
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch["labels"].numpy())

    test_preds = np.array(test_preds)
    test_true  = np.array(test_true)

    qwk = quadratic_weighted_kappa(test_true, test_preds)
    exact_acc = np.mean(np.round(test_preds).clip(1, 6) == test_true)

    test_df = test_df.copy()
    test_df["model_score"] = test_preds
    bias = compute_bias_metrics(test_df, "model_score", SCORE_COL)

    result = {
        "prompt":    prompt_name,
        "n_test":    len(test_true),
        "qwk":       round(qwk, 4),
        "exact_acc": round(exact_acc, 4),
        "bias":      bias,
    }

    print(f"  test_qwk={qwk:+.4f} test_acc={exact_acc:.3f}")

    save_path = os.path.join(RESULTS_DIR, f"xlnet_{prompt_name.replace(' ', '_')}.pt")
    torch.save(best_state, save_path)

    return to_python(result)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Default DATA_BASE: <repo_root>/../DATA (data sits alongside the repo)
    BASE = os.environ.get(
        "DATA_BASE",
        os.path.join(REPO_ROOT, "..", "DATA")
    )

    persuade_train = pd.read_csv(
        os.path.join(BASE, "PERSUADE/train/persuade_corpus_2.0_train.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    persuade_test = pd.read_csv(
        os.path.join(BASE, "PERSUADE/test/persuade_corpus_2.0_test.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    # Alternative: load from HuggingFace once dataset is public
    # persuade_train = pd.read_csv(
    #     "hf://datasets/nlpscu/Analyzing-Demographic-Biases/PERSUADE/persuade_corpus_2.0_train.csv",
    #     low_memory=False
    # ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    for df in [persuade_train, persuade_test]:
        for c in DEMO_COLS:
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower().str.strip().replace("nan", pd.NA)

    persuade_train = persuade_train[
        persuade_train[TEXT_COL].notna() & persuade_train[SCORE_COL].notna()
    ].reset_index(drop=True)
    persuade_test = persuade_test[
        persuade_test[TEXT_COL].notna() & persuade_test[SCORE_COL].notna()
    ].reset_index(drop=True)

    prompts = sorted(persuade_train[PROMPT_COL].dropna().unique())
    print(f"PERSUADE loaded: train={len(persuade_train):,} test={len(persuade_test):,} prompts={len(prompts)}")

    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)

    all_results = []

    for prompt in prompts:
        tr = persuade_train[persuade_train[PROMPT_COL] == prompt].reset_index(drop=True)
        te = persuade_test[persuade_test[PROMPT_COL] == prompt].reset_index(drop=True)

        if len(tr) < 20 or len(te) < 5:
            print(f"[{prompt}] skipped (train={len(tr)}, test={len(te)})")
            continue

        result = train_prompt(prompt, tr, te, tokenizer)
        all_results.append(result)

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    qwks = [r['qwk'] for r in all_results]
    print(f"\nmacro_qwk={np.mean(qwks):.4f}  saved={results_path}")

if __name__ == "__main__":
    main()