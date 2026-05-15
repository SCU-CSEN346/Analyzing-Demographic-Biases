"""
train_xlnet_grl.py
Adversarial demographic debiasing via Gradient Reversal Layer (GRL).
Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation",
repurposed here for fairness following the framing in Kwako & Ormerod (BEA 2024)
§4.3 "Further Research".

Architecture:
              ┌──→ regressor  → score        (MSE loss, L_s)
              │
  text → XLNet ─→ hidden h ──┤
              │
              └──→ GRL(λ) → demographic_classifier → demo_label   (CE loss, L_d)

Forward: GRL is identity.
Backward: dL/dh ← -λ · dL_d/dh   (gradient flips when passing into encoder)

The encoder is pulled by L_s to encode score-relevant features, and pushed by
the reversed L_d to remove demographic-predictive features. Adversarial.

Lambda schedule (Ganin & Lempitsky):
    λ(p) = LAMBDA_MAX · (2 / (1 + exp(-GAMMA · p)) - 1)
where p ∈ [0,1] is fractional training progress, measured in OPTIMIZER STEPS
(not micro-batches — important when grad_accum > 1).

Runs per prompt, per dataset. Saves .pt + results.json identical in shape to
train_xlnet.py so downstream tools (compute_weighted_smd.py, probe_hidden_states.py)
work without modification.

Usage:
    DATASET=PERSUADE python train_xlnet_grl.py
    DATASET=ASAP     python train_xlnet_grl.py

Key env overrides:
    DATASET            PERSUADE | ASAP   (required, no default — pick deliberately)
    DEMO_ATTR          gender (default), ell_status, race_white_vs_black, ...
    LAMBDA_MAX         0.5 (default)
    BATCH_SIZE         8 on 80GB, 4 on 48/40GB
    GRAD_ACCUM         1 on 80GB, 2 on 48/40GB → effective BS stays 8
    MAX_LEN            2048
    RUN_VERSION        grl_gender_l05  (BUMP for every run)
"""

import os
import json
import math
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Function
from transformers import XLNetTokenizer, XLNetModel
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────
DATASET      = os.environ.get("DATASET", "").upper()
if DATASET not in ("PERSUADE", "ASAP"):
    raise ValueError("Set DATASET=PERSUADE or DATASET=ASAP")

MODEL_NAME   = "xlnet-base-cased"
MAX_LEN      = int(os.environ.get("MAX_LEN", 2048))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 8))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", 1))
LR           = float(os.environ.get("LR", 5e-6))
EPOCHS       = int(os.environ.get("EPOCHS", 20))
DROPOUT      = 0.1
WARMUP_FRAC  = 0.1
RANDOM_SEED  = 42

DEMO_ATTR    = os.environ.get("DEMO_ATTR", "gender")
LAMBDA_MAX   = float(os.environ.get("LAMBDA_MAX", 0.5))
GAMMA        = float(os.environ.get("GAMMA", 10.0))   # ramp steepness
DEMO_HEAD_HIDDEN = 128                                 # small MLP, not raw linear, helps stability

# bf16 autocast for forward+loss. Free 1.5-2x speedup on Blackwell (PRO 6000) /
# Hopper (H100). bf16 has fp32's exponent range so no loss scaling needed.
USE_BF16 = os.environ.get("USE_BF16", "1") == "1"

RUN_VERSION  = os.environ.get(
    "RUN_VERSION",
    f"grl_{DEMO_ATTR}_l{int(LAMBDA_MAX * 100):02d}_{DATASET.lower()}"
)
# Paths are resolved from environment variables; suitable defaults are
# provided for both the RunPod /workspace layout and a local repo layout.
# Override BASE and RESULTS_DIR at launch time as needed.
BASE         = os.environ.get(
    "BASE",
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "..", "DATA"
    )
)
RESULTS_DIR  = os.environ.get(
    "RESULTS_DIR",
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "results", "xlnet", RUN_VERSION
    )
)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{RUN_VERSION}] device={device} dataset={DATASET} demo={DEMO_ATTR} "
      f"λmax={LAMBDA_MAX} γ={GAMMA} bs={BATCH_SIZE}x{GRAD_ACCUM} bf16={USE_BF16}")


# ── Dataset-specific column wiring ─────────────────────────────────────────
if DATASET == "PERSUADE":
    DATA_TRAIN = os.path.join(BASE, "PERSUADE/train/persuade_corpus_2.0_train.csv")
    DATA_TEST  = os.path.join(BASE, "PERSUADE/test/persuade_corpus_2.0_test.csv")
    TEXT_COL   = "full_text"
    SCORE_COL  = "holistic_essay_score"
    PROMPT_COL = "prompt_name"
    ESSAY_ID   = "essay_id"
else:  # ASAP
    DATA_TRAIN = os.path.join(BASE, "ASAP/train/ASAP_2_Final_github_train.csv")
    DATA_TEST  = os.path.join(BASE, "ASAP/test/ASAP_2_Final_github_test.csv")
    TEXT_COL   = "full_text"
    SCORE_COL  = "score"
    PROMPT_COL = "prompt_name"
    ESSAY_ID   = "essay_id"


# ── Demographic label mapping ──────────────────────────────────────────────
# Each entry: (column, value_to_class_mapping). None means skip the row.
DEMO_TASKS = {
    "gender": {
        "col": "gender",
        "map": {"f": 0, "m": 1},
        "n_classes": 2,
    },
    "ell_status": {
        "col": "ell_status",
        "map": {"no": 0, "yes": 1},
        "n_classes": 2,
    },
    "econ_disadvantaged": {
        "col": "economically_disadvantaged",
        "map": {
            "not economically disadvantaged": 0,
            "economically disadvantaged":     1,
        },
        "n_classes": 2,
    },
    "disability": {
        "col": "student_disability_status",
        "map": {
            "not identified as having disability": 0,
            "identified as having disability":     1,
        },
        "n_classes": 2,
    },
    "race_white_vs_black": {
        "col": "race_ethnicity",
        "map": {"white": 0, "black/african american": 1},
        "n_classes": 2,
    },
}
if DEMO_ATTR not in DEMO_TASKS:
    raise ValueError(f"Unknown DEMO_ATTR={DEMO_ATTR}. Choose from {list(DEMO_TASKS)}")
DEMO_CFG = DEMO_TASKS[DEMO_ATTR]


# ── Gradient Reversal Layer ────────────────────────────────────────────────
class GradientReversalFn(Function):
    """Forward: identity. Backward: multiply incoming gradient by -lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)


# ── Model ──────────────────────────────────────────────────────────────────
class XLNetGRL(nn.Module):
    """
    XLNet encoder + scoring head + adversarial demographic head with GRL.
    Encoder is shared. The GRL sits between encoder and demographic head.
    """
    def __init__(self, model_name=MODEL_NAME, dropout=DROPOUT,
                 demo_n_classes=2, demo_hidden=DEMO_HEAD_HIDDEN):
        super().__init__()
        self.xlnet     = XLNetModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.xlnet.config.hidden_size, 1)

        # Small MLP for the adversary. Pure linear is also fine but a tiny
        # MLP is a more competent adversary, which makes the encoder work
        # harder to hide demographic info. Standard choice in DANN.
        self.demo_head = nn.Sequential(
            nn.Linear(self.xlnet.config.hidden_size, demo_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(demo_hidden, demo_n_classes),
        )

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        out = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.last_hidden_state[:, -1, :]  # (B, 768) — same as train_xlnet.py

    def forward(self, input_ids, attention_mask, token_type_ids=None, lambda_=0.0):
        h = self.encode(input_ids, attention_mask, token_type_ids)
        # Scoring path
        score_pred = self.regressor(self.dropout(h)).squeeze(-1)
        # Adversarial path — GRL flips gradients flowing back to encoder
        h_rev = grad_reverse(h, lambda_)
        demo_logits = self.demo_head(h_rev)
        return score_pred, demo_logits, h


# ── Dataset ────────────────────────────────────────────────────────────────
class EssayDataset(torch.utils.data.Dataset):
    """
    Yields (input_ids, attention_mask, token_type_ids, score, demo_label, demo_mask).
    demo_mask is 1 if the demographic label is present, else 0 — lets us compute
    the adversarial loss only on labeled rows without dropping the row entirely
    (so the scoring head still trains on the full data).
    """
    def __init__(self, texts, scores, demo_labels, demo_masks, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.demo_labels = torch.tensor(demo_labels, dtype=torch.long)
        self.demo_masks = torch.tensor(demo_masks, dtype=torch.float32)

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, idx):
        return (
            self.encodings["input_ids"][idx],
            self.encodings["attention_mask"][idx],
            self.encodings.get(
                "token_type_ids",
                torch.zeros_like(self.encodings["input_ids"])
            )[idx],
            self.scores[idx],
            self.demo_labels[idx],
            self.demo_masks[idx],
        )


# ── Helpers ────────────────────────────────────────────────────────────────
def to_python(obj):
    """Recursive numpy→python type converter for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def pooled_smd(a, b):
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def round_qwk(y_true, y_pred_continuous):
    """QWK after rounding regression outputs to nearest integer score."""
    y_pred = np.clip(np.round(y_pred_continuous), 1, 6).astype(int)
    return cohen_kappa_score(y_true.astype(int), y_pred, weights="quadratic")


def prepare_demo_labels(df, demo_cfg):
    """
    Returns (demo_label_array, demo_mask_array) aligned with df rows.
    demo_label=0 for rows where the label is missing — these are masked out
    of the adversarial loss via demo_mask=0.
    """
    col = demo_cfg["col"]
    mapping = demo_cfg["map"]
    if col not in df.columns:
        raise KeyError(f"Column {col} not in dataframe for demo attribute {DEMO_ATTR}")
    vals = df[col].astype(str).str.lower().str.strip()
    labels = np.zeros(len(df), dtype=np.int64)
    mask = np.zeros(len(df), dtype=np.float32)
    for v, cls in mapping.items():
        m = (vals == v).values
        labels[m] = cls
        mask[m] = 1.0
    return labels, mask


def lambda_for_step(step, total_steps):
    """DANN lambda schedule. Smooth ramp from 0 to LAMBDA_MAX."""
    p = step / max(total_steps, 1)
    return LAMBDA_MAX * (2.0 / (1.0 + math.exp(-GAMMA * p)) - 1.0)


# ── Train one prompt ───────────────────────────────────────────────────────
def train_one_prompt(prompt, df_train_p, df_test_p, tokenizer):
    print(f"\n[{prompt}] train={len(df_train_p)} test={len(df_test_p)}")

    # 10% dev split, stratified on score where possible
    try:
        df_tr, df_dev = train_test_split(
            df_train_p, test_size=0.1, random_state=RANDOM_SEED,
            stratify=df_train_p[SCORE_COL]
        )
    except ValueError:
        df_tr, df_dev = train_test_split(
            df_train_p, test_size=0.1, random_state=RANDOM_SEED
        )

    # Build labels for adversarial head
    y_demo_tr, m_demo_tr = prepare_demo_labels(df_tr, DEMO_CFG)
    y_demo_dev, m_demo_dev = prepare_demo_labels(df_dev, DEMO_CFG)
    y_demo_te, m_demo_te = prepare_demo_labels(df_test_p, DEMO_CFG)

    if m_demo_tr.mean() < 1.0:
        print(f"  demo_coverage train={m_demo_tr.mean():.1%} "
              f"dev={m_demo_dev.mean():.1%} test={m_demo_te.mean():.1%}")
    if m_demo_tr.sum() < 20:
        print(f"  warn: only {int(m_demo_tr.sum())} demo labels in train")

    # Datasets
    train_ds = EssayDataset(df_tr[TEXT_COL].tolist(), df_tr[SCORE_COL].values,
                             y_demo_tr, m_demo_tr, tokenizer)
    dev_ds   = EssayDataset(df_dev[TEXT_COL].tolist(), df_dev[SCORE_COL].values,
                             y_demo_dev, m_demo_dev, tokenizer)
    test_ds  = EssayDataset(df_test_p[TEXT_COL].tolist(), df_test_p[SCORE_COL].values,
                             y_demo_te, m_demo_te, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model + optim
    model = XLNetGRL(demo_n_classes=DEMO_CFG["n_classes"]).to(device)

    # Compute class weights for demographic loss (balance imbalanced groups)
    labels_present = y_demo_tr[m_demo_tr.astype(bool)]
    if len(labels_present) > 0:
        counts = np.bincount(labels_present, minlength=DEMO_CFG["n_classes"]).astype(float)
        counts = np.where(counts == 0, 1.0, counts)  # avoid div by 0
        class_weights = len(labels_present) / (DEMO_CFG["n_classes"] * counts)
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        cw = None

    score_loss_fn = nn.MSELoss()
    demo_loss_fn  = nn.CrossEntropyLoss(weight=cw, reduction="none")
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_optim_steps = (len(train_dl) // GRAD_ACCUM) * EPOCHS

    best_dev_qwk = -float("inf")
    best_state   = None
    history      = []
    optim_step   = 0

    for epoch in range(EPOCHS):
        model.train()
        running_score_loss = 0.0
        running_demo_loss  = 0.0
        running_demo_count = 0.0
        n_seen = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dl):
            input_ids, attn, tok_type, scores, demo_y, demo_m = [t.to(device) for t in batch]

            # Use lambda at the *current* optimizer step
            lam = lambda_for_step(optim_step, total_optim_steps)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                score_pred, demo_logits, _ = model(input_ids, attn, tok_type, lambda_=lam)

                l_score = score_loss_fn(score_pred, scores)

                # Per-sample demo loss, masked by availability, then mean over labeled
                per_sample = demo_loss_fn(demo_logits, demo_y)
                if demo_m.sum() > 0:
                    l_demo = (per_sample * demo_m).sum() / demo_m.sum()
                else:
                    l_demo = torch.tensor(0.0, device=device)

                # Combined loss. GRL handles the sign on the encoder gradient;
                # we just add the demo loss positively.
                loss = (l_score + l_demo) / GRAD_ACCUM

            loss.backward()

            running_score_loss += l_score.item() * scores.size(0)
            running_demo_loss  += l_demo.item() * float(demo_m.sum())
            running_demo_count += float(demo_m.sum())
            n_seen += scores.size(0)

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optim_step += 1

        # Flush any leftover gradient accumulation
        if (batch_idx + 1) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            optim_step += 1

        # ── Dev eval ──
        model.eval()
        dev_preds, dev_true = [], []
        dev_demo_correct, dev_demo_total = 0, 0
        with torch.no_grad():
            for batch in dev_dl:
                input_ids, attn, tok_type, scores, demo_y, demo_m = [t.to(device) for t in batch]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                    score_pred, demo_logits, _ = model(input_ids, attn, tok_type, lambda_=0.0)
                dev_preds.append(score_pred.float().cpu().numpy())
                dev_true.append(scores.cpu().numpy())
                if demo_m.sum() > 0:
                    labeled = demo_m.bool()
                    preds = demo_logits.argmax(dim=1)
                    dev_demo_correct += (preds[labeled] == demo_y[labeled]).sum().item()
                    dev_demo_total   += int(labeled.sum().item())
        dev_preds = np.concatenate(dev_preds)
        dev_true  = np.concatenate(dev_true)
        dev_qwk = round_qwk(dev_true, dev_preds)
        dev_demo_acc = (dev_demo_correct / dev_demo_total) if dev_demo_total else float("nan")

        avg_score_loss = running_score_loss / max(n_seen, 1)
        avg_demo_loss  = running_demo_loss / max(running_demo_count, 1)
        cur_lambda = lambda_for_step(optim_step - 1, total_optim_steps)

        print(f"  ep{epoch+1:02d} λ={cur_lambda:.3f} "
              f"l_s={avg_score_loss:.3f} l_d={avg_demo_loss:.3f} "
              f"dev_qwk={dev_qwk:+.4f} dev_demo_acc={dev_demo_acc:.3f}")

        history.append({
            "epoch": epoch + 1,
            "lambda": cur_lambda,
            "train_score_loss": avg_score_loss,
            "train_demo_loss": avg_demo_loss,
            "dev_qwk": dev_qwk,
            "dev_demo_acc": dev_demo_acc,
        })

        if dev_qwk > best_dev_qwk:
            best_dev_qwk = dev_qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_true, test_demo_y, test_demo_m = [], [], [], []
    with torch.no_grad():
        for batch in test_dl:
            input_ids, attn, tok_type, scores, demo_y, demo_m = [t.to(device) for t in batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                score_pred, _, _ = model(input_ids, attn, tok_type, lambda_=0.0)
            test_preds.append(score_pred.float().cpu().numpy())
            test_true.append(scores.cpu().numpy())
            test_demo_y.append(demo_y.cpu().numpy())
            test_demo_m.append(demo_m.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_true  = np.concatenate(test_true)
    test_demo_y = np.concatenate(test_demo_y)
    test_demo_m = np.concatenate(test_demo_m).astype(bool)

    test_qwk = round_qwk(test_true, test_preds)
    test_acc = float((np.clip(np.round(test_preds), 1, 6).astype(int) ==
                      test_true.astype(int)).mean())

    # SMD on the model predictions, for the trained-against attribute
    bias = {}
    for cls in range(DEMO_CFG["n_classes"]):
        mask = test_demo_m & (test_demo_y == cls)
        if mask.sum() >= 5:
            bias[f"class_{cls}_mean_pred"] = float(test_preds[mask].mean())
            bias[f"class_{cls}_mean_true"] = float(test_true[mask].mean())
            bias[f"class_{cls}_n"]         = int(mask.sum())
    if DEMO_CFG["n_classes"] == 2:
        a = test_preds[test_demo_m & (test_demo_y == 0)]
        b = test_preds[test_demo_m & (test_demo_y == 1)]
        ha = test_true[test_demo_m & (test_demo_y == 0)]
        hb = test_true[test_demo_m & (test_demo_y == 1)]
        if len(a) >= 5 and len(b) >= 5:
            bias["model_smd"] = pooled_smd(a, b)
            bias["human_smd"] = pooled_smd(ha, hb)
            bias["amplification"] = abs(bias["model_smd"]) - abs(bias["human_smd"])

    # Save .pt — full model state including demo head, since we may want to
    # ablate or probe it later. Downstream probe scripts only touch
    # xlnet.* and regressor.* keys, so the extra demo_head.* is harmless.
    pt_path = os.path.join(
        RESULTS_DIR, f"xlnet_{prompt.replace(' ', '_')}.pt"
    )
    torch.save(model.state_dict(), pt_path)
    print(f"  test_qwk={test_qwk:+.4f} test_acc={test_acc:.3f} "
          f"smd={bias.get('model_smd', float('nan')):+.4f}")

    return {
        "prompt": prompt,
        "n_train": int(len(df_tr)),
        "n_dev":   int(len(df_dev)),
        "n_test":  int(len(df_test_p)),
        "best_dev_qwk": float(best_dev_qwk),
        "test_qwk": float(test_qwk),
        "test_acc": float(test_acc),
        "bias": {DEMO_ATTR: bias},
        "history": history,
        "pt_path": pt_path,
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    df_train = pd.read_csv(DATA_TRAIN, low_memory=False)
    df_test  = pd.read_csv(DATA_TEST,  low_memory=False)

    if DATASET == "PERSUADE":
        df_train = df_train.drop_duplicates(subset=ESSAY_ID).reset_index(drop=True)
        df_test  = df_test.drop_duplicates(subset=ESSAY_ID).reset_index(drop=True)

    # Normalize string demographic columns
    string_demo_cols = ["gender", "ell_status", "economically_disadvantaged",
                        "student_disability_status", "race_ethnicity"]
    for df in [df_train, df_test]:
        for c in string_demo_cols:
            if c in df.columns:
                df[c] = (df[c].astype(str).str.lower().str.strip()
                              .replace("nan", pd.NA))

    df_train = df_train[df_train[TEXT_COL].notna() & df_train[SCORE_COL].notna()].reset_index(drop=True)
    df_test  = df_test[df_test[TEXT_COL].notna() & df_test[SCORE_COL].notna()].reset_index(drop=True)

    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)
    prompts = sorted(df_train[PROMPT_COL].dropna().unique())
    print(f"{DATASET} loaded: train={len(df_train):,} test={len(df_test):,} prompts={len(prompts)}")

    all_results = []
    for prompt in prompts:
        tr_p = df_train[df_train[PROMPT_COL] == prompt].reset_index(drop=True)
        te_p = df_test[df_test[PROMPT_COL] == prompt].reset_index(drop=True)
        if len(tr_p) < 50 or len(te_p) < 10:
            print(f"[{prompt}] skipped (train={len(tr_p)}, test={len(te_p)})")
            continue
        result = train_one_prompt(prompt, tr_p, te_p, tokenizer)
        all_results.append(result)

        # Save partial results after each prompt — protects long runs from crashes
        partial = {
            "run_version": RUN_VERSION,
            "dataset": DATASET,
            "demo_attr": DEMO_ATTR,
            "lambda_max": LAMBDA_MAX,
            "completed_prompts": len(all_results),
            "results": all_results,
        }
        partial_path = os.path.join(RESULTS_DIR, "results_partial.json")
        with open(partial_path, "w") as f:
            json.dump(to_python(partial), f, indent=2)

    # Final aggregated results
    qwks = [r["test_qwk"] for r in all_results]
    smds = [r["bias"][DEMO_ATTR].get("model_smd") for r in all_results
            if "model_smd" in r["bias"][DEMO_ATTR]]
    amps = [r["bias"][DEMO_ATTR].get("amplification") for r in all_results
            if "amplification" in r["bias"][DEMO_ATTR]]

    final = {
        "run_version": RUN_VERSION,
        "dataset": DATASET,
        "demo_attr": DEMO_ATTR,
        "lambda_max": LAMBDA_MAX,
        "gamma": GAMMA,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "macro_qwk": float(np.mean(qwks)) if qwks else None,
        "mean_model_smd": float(np.mean(smds)) if smds else None,
        "mean_amplification": float(np.mean(amps)) if amps else None,
        "results": all_results,
    }
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(to_python(final), f, indent=2)

    print(f"\ndone: macro_qwk={final['macro_qwk']:+.4f} "
          f"mean_smd={final['mean_model_smd']:+.4f} "
          f"mean_amp={final['mean_amplification']:+.4f} saved={out_path}")


if __name__ == "__main__":
    main()