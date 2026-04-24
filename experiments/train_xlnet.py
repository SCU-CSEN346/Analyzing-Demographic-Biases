"""
train_xlnet.py
Replication of Kwako & Ormerod (BEA 2024):
  - XLNet-base-cased fine-tuned per prompt on PERSUADE 2.0
  - Regression head (MSE loss)
  - AdamW + linear LR schedule, early stopping on QWK
  - Evaluates QWK, SMD per demographic group
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
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import wandb
import warnings
warnings.filterwarnings("ignore")

WANDB_PROJECT = "xlnet-aes-replication"

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "xlnet-base-cased"
MAX_LEN      = 512
LR           = 5e-6          # exact replication: Kwako & Ormerod (BEA 2024)
BATCH_SIZE   = 8             # exact replication: Kwako & Ormerod (BEA 2024)
EPOCHS       = 20            # exact replication: Kwako & Ormerod (BEA 2024)
DEV_SPLIT    = 0.10
RANDOM_SEED  = 42
SCORE_COL    = "holistic_essay_score"
TEXT_COL     = "full_text"
PROMPT_COL   = "prompt_name"
DEMO_COLS    = ["gender", "race_ethnicity", "economically_disadvantaged",
                "student_disability_status", "ell_status"]

RUN_VERSION  = "v3_exact_replication"  # lr=5e-6, batch=8, epochs=20
RESULTS_DIR  = os.path.expanduser(f"~/CSEN364/PROJECT/Analyzing-Demographic-Biases/results/xlnet_{RUN_VERSION}")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
        # Use last token's hidden state (XLNet CLS equivalent)
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
    print(f"\n{'='*60}\nPrompt: {prompt_name}\n{'='*60}")

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{RUN_VERSION}/{prompt_name}",
        group=RUN_VERSION,
        config={
            "model":      MODEL_NAME,
            "prompt":     prompt_name,
            "max_len":    MAX_LEN,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "epochs":     EPOCHS,
            "dev_split":  DEV_SPLIT,
            "run_version": RUN_VERSION,
        },
        reinit=True,
    )

    # Split train → train/dev
    tr, dev = train_test_split(train_df, test_size=DEV_SPLIT, random_state=RANDOM_SEED)
    tr  = tr.reset_index(drop=True)
    dev = dev.reset_index(drop=True)

    print(f"  Train: {len(tr)}  Dev: {len(dev)}  Test: {len(test_df)}")

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
        # Train
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

        # Dev
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
        print(f"  Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_dl):.4f} | Dev QWK: {dev_qwk:.4f}")

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss / len(train_dl),
            "dev_qwk":    dev_qwk,
        })

        if dev_qwk > best_qwk:
            best_qwk   = dev_qwk
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test with best model
    print(f"  Best Dev QWK: {best_qwk:.4f} — evaluating on test set...")
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
    smd_overall = smd(test_preds, test_true)  # model vs human overall
    exact_acc = np.mean(np.round(test_preds).clip(1, 6) == test_true)

    # Bias metrics
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

    print(f"  Test QWK: {qwk:.4f} | Exact Acc: {exact_acc:.4f}")
    for attr, vals in bias.items():
        print(f"    {attr}: human_smd={vals['human_smd']:+.3f} → model_smd={vals['model_smd']:+.3f}")

    # Log test metrics to wandb
    test_log = {"test_qwk": qwk, "test_exact_acc": exact_acc, "best_dev_qwk": best_qwk}
    for attr, vals in bias.items():
        test_log[f"bias/{attr}/human_smd"] = vals["human_smd"]
        test_log[f"bias/{attr}/model_smd"] = vals["model_smd"]
        test_log[f"bias/{attr}/amplification"] = round(abs(vals["model_smd"]) - abs(vals["human_smd"]), 4)
    wandb.log(test_log)
    wandb.finish()

    # Save model
    save_path = os.path.join(RESULTS_DIR, f"xlnet_{prompt_name.replace(' ', '_')}.pt")
    torch.save(best_state, save_path)

    return to_python(result)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # ── Data Loading ──────────────────────────────────────────────────────
    # LOCAL (temporary): switch back to HuggingFace paths once repo is public
    BASE = os.path.expanduser("~/CSEN364/PROJECT/DATA")
    print("Loading PERSUADE from local files...")
    persuade_train = pd.read_csv(
        os.path.join(BASE, "PERSUADE/train/persuade_corpus_2.0_train.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    persuade_test = pd.read_csv(
        os.path.join(BASE, "PERSUADE/test/persuade_corpus_2.0_test.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    # ── HuggingFace loading (uncomment once repo is public) ───────────────
    # print("Loading PERSUADE from HuggingFace...")
    # persuade_train = pd.read_csv(
    #     "hf://datasets/nlpscu/Analyzing-Demographic-Biases/PERSUADE/persuade_corpus_2.0_train.csv",
    #     low_memory=False
    # ).drop_duplicates(subset="essay_id").reset_index(drop=True)
    # persuade_test = pd.read_csv(
    #     "hf://datasets/nlpscu/Analyzing-Demographic-Biases/PERSUADE/persuade_corpus_2.0_test.csv",
    #     low_memory=False
    # ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    # Clean
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

    print(f"Train: {len(persuade_train):,}  Test: {len(persuade_test):,}")
    print(f"Prompts: {sorted(persuade_train[PROMPT_COL].unique())}")

    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)

    all_results = []
    prompts = sorted(persuade_train[PROMPT_COL].dropna().unique())

    for prompt in prompts:
        tr = persuade_train[persuade_train[PROMPT_COL] == prompt].reset_index(drop=True)
        te = persuade_test[persuade_test[PROMPT_COL] == prompt].reset_index(drop=True)

        if len(tr) < 20 or len(te) < 5:
            print(f"Skipping '{prompt}' — too few samples (train={len(tr)}, test={len(te)})")
            continue

        result = train_prompt(prompt, tr, te, tokenizer)
        all_results.append(result)

    # Save all results
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Prompt':<45} {'QWK':>6} {'Exact':>6}")
    print("-" * 60)
    qwks = []
    for r in all_results:
        print(f"{r['prompt']:<45} {r['qwk']:>6.4f} {r['exact_acc']:>6.4f}")
        qwks.append(r['qwk'])
    print("-" * 60)
    print(f"{'Macro Average':<45} {np.mean(qwks):>6.4f}")
    print(f"\nResults saved to: {results_path}")

    # Log macro summary run
    summary_run = wandb.init(project=WANDB_PROJECT, name=f"{RUN_VERSION}/summary", group=RUN_VERSION, reinit=True)
    wandb.log({
        "macro_avg_qwk": float(np.mean(qwks)),
        **{f"prompt_qwk/{r['prompt']}": r["qwk"] for r in all_results},
    })
    wandb.finish()

if __name__ == "__main__":
    main()