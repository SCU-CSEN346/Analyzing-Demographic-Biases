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

# wandb is optional — set USE_WANDB=0 to disable. If the package isn't
# installed, fall back silently so the run is never blocked on it.
USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"
if USE_WANDB:
    try:
        import wandb
    except ImportError:
        print("wandb not installed — continuing without logging "
              "(pip install wandb to enable, or set USE_WANDB=0 to silence)")
        USE_WANDB = False


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
# Early-stop patience: bail after this many epochs without dev_qwk improvement.
# Baseline ASAP showed best dev_qwk usually at ep5–10; PATIENCE=4 catches that
# while letting a slow-improving run continue. Set high (e.g. 20) to disable.
PATIENCE     = int(os.environ.get("PATIENCE", 4))
DROPOUT      = 0.1
WARMUP_FRAC  = 0.1
RANDOM_SEED  = 42

# Joint multi-head training: list of demographic attributes the adversary
# pressures simultaneously. Five binary heads, matching Tom's RoBERTa setup
# (race kept binary Black-vs-White per K&O's headline result).
DEFAULT_JOINT_ATTRS = ["gender", "ell_status", "econ_disadvantaged",
                       "disability", "race_white_vs_black"]
JOINT_ATTRS = os.environ.get(
    "JOINT_ATTRS",
    ",".join(DEFAULT_JOINT_ATTRS),
).split(",")
JOINT_ATTRS = [a.strip() for a in JOINT_ATTRS if a.strip()]

LAMBDA_MAX   = float(os.environ.get("LAMBDA_MAX", 0.5))
GAMMA        = float(os.environ.get("GAMMA", 10.0))   # ramp steepness
DEMO_HEAD_HIDDEN = 128                                 # small MLP, not raw linear, helps stability

# bf16 autocast for forward+loss. Free 1.5-2x speedup on Blackwell (PRO 6000) /
# Hopper (H100). bf16 has fp32's exponent range so no loss scaling needed.
USE_BF16 = os.environ.get("USE_BF16", "1") == "1"

RUN_VERSION  = os.environ.get(
    "RUN_VERSION",
    f"grl_joint_l{int(LAMBDA_MAX * 100):02d}_{DATASET.lower()}"
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
print(f"[{RUN_VERSION}] device={device} dataset={DATASET} attrs={JOINT_ATTRS} "
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
unknown = [a for a in JOINT_ATTRS if a not in DEMO_TASKS]
if unknown:
    raise ValueError(f"Unknown JOINT_ATTRS entries {unknown}. "
                     f"Choose from {list(DEMO_TASKS)}")
# Ordered list of (name, cfg) actually being trained against
JOINT_CFG = [(a, DEMO_TASKS[a]) for a in JOINT_ATTRS]


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
    XLNet encoder + scoring head + multiple adversarial demographic heads
    with shared GRL. Encoder is shared. Each head pressures the encoder to
    drop signal for its own attribute; gradients sum at the encoder via GRL.
    """
    def __init__(self, joint_cfg, model_name=MODEL_NAME, dropout=DROPOUT,
                 demo_hidden=DEMO_HEAD_HIDDEN):
        super().__init__()
        self.xlnet     = XLNetModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.xlnet.config.hidden_size, 1)

        # One MLP adversary per attribute. Pure linear would also work; the
        # tiny MLP is a more competent adversary, which makes the encoder
        # work harder to hide demographic info. Standard DANN choice.
        # ModuleDict keys cannot contain '.' but our attribute names are clean.
        self.demo_heads = nn.ModuleDict()
        for attr, cfg in joint_cfg:
            self.demo_heads[attr] = nn.Sequential(
                nn.Linear(self.xlnet.config.hidden_size, demo_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(demo_hidden, cfg["n_classes"]),
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
        # Adversarial path — GRL flips gradients flowing back to encoder for
        # every head simultaneously (one h_rev shared across heads).
        h_rev = grad_reverse(h, lambda_)
        demo_logits = {attr: head(h_rev) for attr, head in self.demo_heads.items()}
        return score_pred, demo_logits, h


# ── Dataset ────────────────────────────────────────────────────────────────
class EssayDataset(torch.utils.data.Dataset):
    """
    Yields a dict per essay:
      {input_ids, attention_mask, token_type_ids, score, demo_y: {attr: int},
       demo_m: {attr: float}}.
    demo_m[attr]=1 iff the label for that attribute is present, else 0 — lets
    us compute each adversarial loss only on its labeled rows, while the
    scoring head trains on the full data.
    """
    def __init__(self, texts, scores, demo_labels_dict, demo_masks_dict,
                 tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.scores = torch.tensor(scores, dtype=torch.float32)
        # Per-attribute tensors keyed by attribute name
        self.demo_labels = {a: torch.tensor(v, dtype=torch.long)
                            for a, v in demo_labels_dict.items()}
        self.demo_masks = {a: torch.tensor(v, dtype=torch.float32)
                           for a, v in demo_masks_dict.items()}

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get(
                "token_type_ids",
                torch.zeros_like(self.encodings["input_ids"])
            )[idx],
            "score":  self.scores[idx],
            "demo_y": {a: self.demo_labels[a][idx] for a in self.demo_labels},
            "demo_m": {a: self.demo_masks[a][idx]  for a in self.demo_masks},
        }


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


def round_qwk(y_true, y_pred_continuous, score_low=None, score_high=None):
    """
    QWK after rounding regression outputs to nearest integer score, clipped
    to the actual score range. If score_low/high are None, derive from y_true
    (so per-prompt ranges that differ — e.g. ASAP set 1 is 2-12, set 7 is
    0-30 — work without code changes).
    """
    if score_low is None:
        score_low = int(np.floor(y_true.min()))
    if score_high is None:
        score_high = int(np.ceil(y_true.max()))
    y_pred = np.clip(np.round(y_pred_continuous), score_low, score_high).astype(int)
    return cohen_kappa_score(y_true.astype(int), y_pred, weights="quadratic")


def prepare_demo_labels(df, attr_name, demo_cfg):
    """
    Returns (demo_label_array, demo_mask_array) aligned with df rows for one
    attribute. demo_label=0 where the label is missing — masked out of the
    adversarial loss via demo_mask=0.
    """
    col = demo_cfg["col"]
    mapping = demo_cfg["map"]
    if col not in df.columns:
        raise KeyError(f"Column {col} not in dataframe for demo attribute {attr_name}")
    vals = df[col].astype(str).str.lower().str.strip()
    labels = np.zeros(len(df), dtype=np.int64)
    mask = np.zeros(len(df), dtype=np.float32)
    for v, cls in mapping.items():
        m = (vals == v).values
        labels[m] = cls
        mask[m] = 1.0
    return labels, mask


def prepare_joint_labels(df, joint_cfg):
    """Returns ({attr: labels}, {attr: mask}) over all joint attributes."""
    labels_dict, masks_dict = {}, {}
    for attr, cfg in joint_cfg:
        y, m = prepare_demo_labels(df, attr, cfg)
        labels_dict[attr] = y
        masks_dict[attr]  = m
    return labels_dict, masks_dict


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

    # Build labels for every adversarial head (dict keyed by attribute)
    y_tr_d, m_tr_d  = prepare_joint_labels(df_tr,     JOINT_CFG)
    y_dev_d, m_dev_d = prepare_joint_labels(df_dev,    JOINT_CFG)
    y_te_d, m_te_d  = prepare_joint_labels(df_test_p, JOINT_CFG)

    # Coverage logging — show coverage per attribute (a single attribute can
    # have very different missingness from the others, e.g. race).
    for attr, _ in JOINT_CFG:
        cov_tr  = float(m_tr_d[attr].mean())
        cov_dev = float(m_dev_d[attr].mean())
        cov_te  = float(m_te_d[attr].mean())
        print(f"  {attr:24s} demo_coverage train={cov_tr:.1%} "
              f"dev={cov_dev:.1%} test={cov_te:.1%}")
        if m_tr_d[attr].sum() < 20:
            print(f"  warn[{attr}]: only {int(m_tr_d[attr].sum())} labels in train")

    # Datasets
    train_ds = EssayDataset(df_tr[TEXT_COL].tolist(), df_tr[SCORE_COL].values,
                             y_tr_d, m_tr_d, tokenizer)
    dev_ds   = EssayDataset(df_dev[TEXT_COL].tolist(), df_dev[SCORE_COL].values,
                             y_dev_d, m_dev_d, tokenizer)
    test_ds  = EssayDataset(df_test_p[TEXT_COL].tolist(), df_test_p[SCORE_COL].values,
                             y_te_d, m_te_d, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model + optim
    model = XLNetGRL(JOINT_CFG).to(device)

    # Per-attribute class weights for demographic loss (balance imbalanced
    # groups). Each head gets its own CE function with its own weights.
    demo_loss_fns = {}
    cw_per_attr = {}
    for attr, cfg in JOINT_CFG:
        labels_present = y_tr_d[attr][m_tr_d[attr].astype(bool)]
        if len(labels_present) > 0:
            counts = np.bincount(labels_present, minlength=cfg["n_classes"]).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            class_weights = len(labels_present) / (cfg["n_classes"] * counts)
            cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            cw = None
        cw_per_attr[attr] = cw
        demo_loss_fns[attr] = nn.CrossEntropyLoss(weight=cw, reduction="none")

    score_loss_fn = nn.MSELoss()
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_optim_steps = (len(train_dl) // GRAD_ACCUM) * EPOCHS

    best_dev_qwk = -float("inf")
    best_state   = None
    epochs_since_best = 0
    history      = []
    optim_step   = 0

    for epoch in range(EPOCHS):
        model.train()
        running_score_loss = 0.0
        running_demo_loss = {a: 0.0 for a, _ in JOINT_CFG}
        running_demo_count = {a: 0.0 for a, _ in JOINT_CFG}
        n_seen = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            tok_type  = batch["token_type_ids"].to(device)
            scores    = batch["score"].to(device)
            demo_y    = {a: batch["demo_y"][a].to(device) for a, _ in JOINT_CFG}
            demo_m    = {a: batch["demo_m"][a].to(device) for a, _ in JOINT_CFG}

            # Use lambda at the *current* optimizer step
            lam = lambda_for_step(optim_step, total_optim_steps)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                score_pred, demo_logits, _ = model(input_ids, attn, tok_type, lambda_=lam)

                l_score = score_loss_fn(score_pred, scores)

                # Per-head demo loss, masked by per-attribute availability.
                # Each head's CE function has its own class weights (computed
                # once at setup); reduction="none" returns per-sample loss.
                # Mean aggregation across heads keeps total adversarial
                # pressure on the encoder at ~one head's worth, so λ_max
                # remains comparable to single-attribute runs.
                per_attr_losses = []
                for attr, _ in JOINT_CFG:
                    y_a = demo_y[attr]
                    m_a = demo_m[attr]
                    per_sample = demo_loss_fns[attr](demo_logits[attr], y_a)
                    if m_a.sum() > 0:
                        cw_a = cw_per_attr[attr]
                        if cw_a is not None:
                            sample_w = cw_a[y_a] * m_a
                        else:
                            sample_w = m_a
                        denom = sample_w.sum().clamp_min(1e-8)
                        l_attr = (per_sample * m_a).sum() / denom
                    else:
                        l_attr = torch.tensor(0.0, device=device)
                    per_attr_losses.append(l_attr)
                    running_demo_loss[attr]  += l_attr.item() * float(m_a.sum())
                    running_demo_count[attr] += float(m_a.sum())

                # MEAN aggregation across heads (not sum), so total
                # adversarial pressure stays at one head's worth
                l_demo_total = torch.stack(per_attr_losses).mean()

                # Combined loss. GRL handles the sign on the encoder gradient;
                # we just add the demo loss positively.
                loss = (l_score + l_demo_total) / GRAD_ACCUM

            loss.backward()

            running_score_loss += l_score.item() * scores.size(0)
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
        # Per-attribute dev demographic accuracy: how well can each adversary
        # still predict its target from h on held-out essays.
        dev_demo_correct = {a: 0 for a, _ in JOINT_CFG}
        dev_demo_total   = {a: 0 for a, _ in JOINT_CFG}
        with torch.no_grad():
            for batch in dev_dl:
                input_ids = batch["input_ids"].to(device)
                attn      = batch["attention_mask"].to(device)
                tok_type  = batch["token_type_ids"].to(device)
                scores    = batch["score"].to(device)
                demo_y_b  = {a: batch["demo_y"][a].to(device) for a, _ in JOINT_CFG}
                demo_m_b  = {a: batch["demo_m"][a].to(device) for a, _ in JOINT_CFG}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                    score_pred, demo_logits, _ = model(input_ids, attn, tok_type, lambda_=0.0)
                dev_preds.append(score_pred.float().cpu().numpy())
                dev_true.append(scores.cpu().numpy())
                for attr, _ in JOINT_CFG:
                    m_a = demo_m_b[attr]
                    if m_a.sum() > 0:
                        labeled = m_a.bool()
                        preds_a = demo_logits[attr].argmax(dim=1)
                        dev_demo_correct[attr] += (preds_a[labeled] == demo_y_b[attr][labeled]).sum().item()
                        dev_demo_total[attr]   += int(labeled.sum().item())
        dev_preds = np.concatenate(dev_preds)
        dev_true  = np.concatenate(dev_true)
        dev_qwk = round_qwk(dev_true, dev_preds)
        dev_demo_accs = {
            attr: (dev_demo_correct[attr] / dev_demo_total[attr])
                  if dev_demo_total[attr] else float("nan")
            for attr, _ in JOINT_CFG
        }

        avg_score_loss = running_score_loss / max(n_seen, 1)
        avg_demo_loss_per_attr = {
            attr: running_demo_loss[attr] / max(running_demo_count[attr], 1)
            for attr, _ in JOINT_CFG
        }
        avg_demo_loss = float(np.mean(list(avg_demo_loss_per_attr.values())))
        cur_lambda = lambda_for_step(optim_step - 1, total_optim_steps)

        # Compact one-line per-epoch summary; per-head accuracy stays readable
        acc_str = " ".join(
            f"{attr[:6]}={dev_demo_accs[attr]:.2f}"
            for attr, _ in JOINT_CFG
        )
        print(f"  ep{epoch+1:02d} λ={cur_lambda:.3f} "
              f"l_s={avg_score_loss:.3f} l_d={avg_demo_loss:.3f} "
              f"dev_qwk={dev_qwk:+.4f} | {acc_str}")

        history.append({
            "epoch": epoch + 1,
            "lambda": cur_lambda,
            "train_score_loss": avg_score_loss,
            "train_demo_loss_mean": avg_demo_loss,
            "train_demo_loss_per_attr": avg_demo_loss_per_attr,
            "dev_qwk": dev_qwk,
            "dev_demo_acc_per_attr": dev_demo_accs,
        })

        # wandb logging — one panel per attribute for dev demo accuracy,
        # plus shared scoring + lambda metrics
        if USE_WANDB:
            log = {
                f"{prompt}/lambda": cur_lambda,
                f"{prompt}/train_score_loss": avg_score_loss,
                f"{prompt}/train_demo_loss_mean": avg_demo_loss,
                f"{prompt}/dev_qwk": dev_qwk,
                f"{prompt}/epoch": epoch + 1,
            }
            for attr, _ in JOINT_CFG:
                log[f"{prompt}/dev_demo_acc/{attr}"] = dev_demo_accs[attr]
                log[f"{prompt}/train_demo_loss/{attr}"] = avg_demo_loss_per_attr[attr]
            wandb.log(log)

        if dev_qwk > best_dev_qwk:
            best_dev_qwk = dev_qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"  early stop @ ep{epoch+1:02d}: no dev_qwk improvement "
                      f"for {PATIENCE} epochs (best={best_dev_qwk:+.4f})")
                break

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_true = [], []
    test_demo_y_all = {a: [] for a, _ in JOINT_CFG}
    test_demo_m_all = {a: [] for a, _ in JOINT_CFG}
    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            tok_type  = batch["token_type_ids"].to(device)
            scores    = batch["score"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                score_pred, _, _ = model(input_ids, attn, tok_type, lambda_=0.0)
            test_preds.append(score_pred.float().cpu().numpy())
            test_true.append(scores.cpu().numpy())
            for attr, _ in JOINT_CFG:
                test_demo_y_all[attr].append(batch["demo_y"][attr].cpu().numpy())
                test_demo_m_all[attr].append(batch["demo_m"][attr].cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_true  = np.concatenate(test_true)
    test_demo_y_all = {a: np.concatenate(v) for a, v in test_demo_y_all.items()}
    test_demo_m_all = {a: np.concatenate(v).astype(bool)
                       for a, v in test_demo_m_all.items()}

    # Per-prompt score range derived from true scores; same as round_qwk uses.
    score_low  = int(np.floor(test_true.min()))
    score_high = int(np.ceil(test_true.max()))

    test_qwk = round_qwk(test_true, test_preds, score_low, score_high)
    test_acc = float((np.clip(np.round(test_preds), score_low, score_high).astype(int) ==
                      test_true.astype(int)).mean())

    # Per-attribute SMD on the model predictions — one bias dict per attribute
    bias = {}
    for attr, cfg in JOINT_CFG:
        ty = test_demo_y_all[attr]
        tm = test_demo_m_all[attr]
        attr_bias = {}
        for cls in range(cfg["n_classes"]):
            mask = tm & (ty == cls)
            if mask.sum() >= 5:
                attr_bias[f"class_{cls}_mean_pred"] = float(test_preds[mask].mean())
                attr_bias[f"class_{cls}_mean_true"] = float(test_true[mask].mean())
                attr_bias[f"class_{cls}_n"]         = int(mask.sum())
        if cfg["n_classes"] == 2:
            a = test_preds[tm & (ty == 0)]
            b = test_preds[tm & (ty == 1)]
            ha = test_true[tm & (ty == 0)]
            hb = test_true[tm & (ty == 1)]
            if len(a) >= 5 and len(b) >= 5:
                attr_bias["model_smd"] = pooled_smd(a, b)
                attr_bias["human_smd"] = pooled_smd(ha, hb)
                attr_bias["amplification"] = abs(attr_bias["model_smd"]) - abs(attr_bias["human_smd"])
        bias[attr] = attr_bias

    # Save .pt — full model state including all adversarial heads. Downstream
    # probe / projection scripts use strict=False and only load xlnet.* +
    # regressor.* keys, so the extra demo_heads.* are harmless.
    pt_path = os.path.join(
        RESULTS_DIR, f"xlnet_{prompt.replace(' ', '_')}.pt"
    )
    torch.save(model.state_dict(), pt_path)

    # Concise summary line: per-attribute SMD
    smd_str = " ".join(
        f"{attr[:6]}={bias[attr].get('model_smd', float('nan')):+.3f}"
        for attr, _ in JOINT_CFG
    )
    print(f"  test_qwk={test_qwk:+.4f} test_acc={test_acc:.3f} | {smd_str}")

    if USE_WANDB:
        log = {
            f"{prompt}/test_qwk": test_qwk,
            f"{prompt}/test_acc": test_acc,
        }
        for attr, _ in JOINT_CFG:
            for k, v in bias[attr].items():
                if isinstance(v, (int, float)):
                    log[f"{prompt}/test_bias/{attr}/{k}"] = v
        wandb.log(log)

    return {
        "prompt": prompt,
        "n_train": int(len(df_tr)),
        "n_dev":   int(len(df_dev)),
        "n_test":  int(len(df_test_p)),
        "best_dev_qwk": float(best_dev_qwk),
        "test_qwk": float(test_qwk),
        "test_acc": float(test_acc),
        "bias": bias,           # now: {attr: {model_smd, human_smd, amp, ...}}
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

    # wandb init (gracefully degrades if unavailable)
    global USE_WANDB
    if USE_WANDB:
        try:
            wandb.init(
                project="xlnet-aes-grl",
                name=RUN_VERSION,
                group=RUN_VERSION,
                config={
                    "dataset": DATASET, "model": MODEL_NAME, "max_len": MAX_LEN,
                    "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                    "epochs": EPOCHS, "lr": LR,
                    "joint_attrs": JOINT_ATTRS,
                    "lambda_max": LAMBDA_MAX, "gamma": GAMMA,
                    "loss_aggregation": "mean",
                },
            )
        except Exception as e:
            print(f"wandb init failed ({e}) — continuing without logging")
            USE_WANDB = False

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
            "joint_attrs": JOINT_ATTRS,
            "lambda_max": LAMBDA_MAX,
            "completed_prompts": len(all_results),
            "results": all_results,
        }
        partial_path = os.path.join(RESULTS_DIR, "results_partial.json")
        with open(partial_path, "w") as f:
            json.dump(to_python(partial), f, indent=2)

    # Final aggregated results — per-attribute means
    qwks = [r["test_qwk"] for r in all_results]
    mean_smd_per_attr = {}
    mean_amp_per_attr = {}
    for attr in JOINT_ATTRS:
        smds = [r["bias"][attr].get("model_smd") for r in all_results
                if attr in r["bias"] and "model_smd" in r["bias"][attr]]
        amps = [r["bias"][attr].get("amplification") for r in all_results
                if attr in r["bias"] and "amplification" in r["bias"][attr]]
        mean_smd_per_attr[attr] = float(np.mean(smds)) if smds else None
        mean_amp_per_attr[attr] = float(np.mean(amps)) if amps else None

    final = {
        "run_version": RUN_VERSION,
        "dataset": DATASET,
        "joint_attrs": JOINT_ATTRS,
        "lambda_max": LAMBDA_MAX,
        "gamma": GAMMA,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "macro_qwk": float(np.mean(qwks)) if qwks else None,
        "mean_model_smd_per_attr": mean_smd_per_attr,
        "mean_amplification_per_attr": mean_amp_per_attr,
        "results": all_results,
    }
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(to_python(final), f, indent=2)

    smd_summary = " ".join(
        f"{attr}={mean_smd_per_attr[attr]:+.3f}"
        if mean_smd_per_attr[attr] is not None else f"{attr}=n/a"
        for attr in JOINT_ATTRS
    )
    print(f"\ndone: macro_qwk={final['macro_qwk']:+.4f} | {smd_summary}\n"
          f"saved={out_path}")

    if USE_WANDB:
        log = {"summary/macro_qwk": final["macro_qwk"],
               "summary/n_prompts": len(all_results)}
        for attr in JOINT_ATTRS:
            if mean_smd_per_attr[attr] is not None:
                log[f"summary/mean_model_smd/{attr}"] = mean_smd_per_attr[attr]
            if mean_amp_per_attr[attr] is not None:
                log[f"summary/mean_amplification/{attr}"] = mean_amp_per_attr[attr]
        wandb.log(log)
        wandb.finish()


if __name__ == "__main__":
    main()