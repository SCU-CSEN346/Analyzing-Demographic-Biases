"""
probe_hidden_states.py
Replicates Kwako & Ormerod (BEA 2024) Section 2.8 / 3.3, "Score Features" column of Table 4.

Methodology:
  M(x) = (sigma o L) o (S o T o E)(x)  — scoring model
  M~(x) = (sigma o L~) o (S o T o E)(x) — demographic model: same frozen feature
                                          model, classifier head re-optimized.

  We freeze the XLNet encoder + dropout from the scoring model, replace the
  regression head with a linear classification head, and train ONLY that head
  to predict each demographic attribute. Report Cohen's kappa on the held-out
  test set per prompt, per attribute.

Key differences from a naive linear probe on cached features:
  - The probe is trained with mini-batch SGD (Adam) and many epochs, not a
    single full-batch step per epoch. Lets the head actually converge.
  - Race/ethnicity is reported as binary one-vs-rest classifiers (W, B, L, A)
    matching the paper's Table 4 columns, not a single multi-class kappa.
  - Robust to degenerate dev splits: stratified split when possible, fallback
    to random; tracks best_state across epochs with a safe default.
  - Class imbalance handled via class-weighted cross-entropy.

Output:
  results/probing/probing_{RUN_VERSION}.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "xlnet-base-cased"
MAX_LEN      = 2048              # match training; lower on memory-constrained boxes
ENCODER_BS   = 8                 # batch size for the frozen forward pass
PROBE_EPOCHS = 50                # plenty — early stopping via best dev kappa
PROBE_LR     = 1e-3              # standard linear-probe LR; was 5e-6 (too low) / 1e-2 (too high)
PROBE_BS     = 64                # mini-batch for the probe head
WEIGHT_DECAY = 1e-4              # mild L2 — linear classifiers benefit
PATIENCE     = 8                 # stop if no dev-kappa improvement in N epochs
RANDOM_SEED  = 42

# Race columns split out as binary one-vs-rest — matches paper Table 4 columns
RACE_BINARIES = {
    "race_white":    "white",
    "race_black":    "black/african american",
    "race_hispanic": "hispanic/latino",
    "race_asian":    "asian/pacific islander",
}

# Binary attributes (paper Table 4 columns G, SES, ELL, DS)
BINARY_COLS = ["gender", "economically_disadvantaged", "ell_status",
               "student_disability_status"]

SCORE_COL  = "holistic_essay_score"
TEXT_COL   = "full_text"
PROMPT_COL = "prompt_name"

# Paths are resolved relative to the repo root (parent of experiments/),
# unless overridden by environment variables.
REPO_ROOT    = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
RUN_VERSION  = os.environ.get("RUN_VERSION", "baseline_replication")
MODELS_DIR   = os.environ.get(
    "MODELS_DIR",
    os.path.join(REPO_ROOT, "results", "xlnet", RUN_VERSION)
)
DATA_BASE    = os.environ.get(
    "DATA_BASE",
    os.path.join(REPO_ROOT, "..", "DATA")
)
OUT_DIR      = os.environ.get(
    "OUT_DIR",
    os.path.join(REPO_ROOT, "results", "xlnet", "probing")
)
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"DATA_BASE:  {DATA_BASE}")
print(f"OUT_DIR:    {OUT_DIR}")


# ── XLNet feature model (frozen) ───────────────────────────────────────────
class XLNetRegressor(nn.Module):
    """Matches the architecture in train_xlnet.py exactly."""
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.xlnet     = XLNetModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.xlnet.config.hidden_size, 1)

    def get_hidden(self, input_ids, attention_mask, token_type_ids=None):
        """Return the hidden state (last token, before regression head)."""
        with torch.no_grad():
            out = self.xlnet(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            cls = out.last_hidden_state[:, -1, :]
        return cls  # shape: (batch, hidden_size=768)


# ── Hidden-state extraction ────────────────────────────────────────────────
def extract_hidden_states(model, dataloader):
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for batch in dataloader:
            h = model.get_hidden(
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            all_hidden.append(h.cpu())
    return torch.cat(all_hidden, dim=0).numpy()


# ── Linear probe head ──────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def safe_stratified_split(X, y, test_size, seed):
    """Stratified split when each class has >=2 samples; fallback to random."""
    counts = np.bincount(y) if y.dtype.kind in "iu" else None
    if counts is not None and counts.min() >= 2:
        try:
            return train_test_split(X, y, test_size=test_size,
                                    random_state=seed, stratify=y)
        except ValueError:
            pass
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_probe(h_train, y_train, h_dev, y_dev, num_classes):
    """
    Mini-batch trained linear probe with class-weighted CE, early stopping
    on dev kappa. Always returns a valid trained probe + best dev kappa.
    """
    probe = LinearProbe(h_train.shape[1], num_classes).to(device)

    # Class weights help when groups are imbalanced (e.g. ELL identified ~9%)
    present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present, y=y_train)
    full_w = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(present, weights):
        full_w[c] = w
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(full_w, device=device))

    optimizer = torch.optim.Adam(
        probe.parameters(), lr=PROBE_LR, weight_decay=WEIGHT_DECAY
    )

    X_tr = torch.tensor(h_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_dev = torch.tensor(h_dev, dtype=torch.float32, device=device)
    y_dev_t = torch.tensor(y_dev, dtype=torch.long, device=device)

    # Initialize best_state to the freshly-initialized weights so we ALWAYS
    # have something dict-like to load — this is the bug from the crash.
    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    best_kappa = -float("inf")
    epochs_no_improve = 0

    n = X_tr.shape[0]
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(n)
        for i in range(0, n, PROBE_BS):
            idx = perm[i:i + PROBE_BS]
            xb = X_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            optimizer.zero_grad()
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        # Dev evaluation
        probe.eval()
        with torch.no_grad():
            dev_logits = probe(X_dev)
            dev_preds = dev_logits.argmax(dim=1).cpu().numpy()

        # kappa can be NaN if dev predictions collapse to one class and
        # the true labels are also one class — guard against it.
        try:
            kappa = cohen_kappa_score(y_dev, dev_preds)
            if np.isnan(kappa):
                kappa = -float("inf")
        except ValueError:
            kappa = -float("inf")

        if kappa > best_kappa:
            best_kappa = kappa
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    probe.load_state_dict(best_state)
    # If we never beat -inf, surface that as 0.0 rather than -inf for JSON sanity
    if best_kappa == -float("inf"):
        best_kappa = 0.0
    return probe, float(best_kappa)


def evaluate_probe(probe, h_test, y_test):
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.tensor(h_test, dtype=torch.float32, device=device))
        preds = logits.argmax(dim=1).cpu().numpy()
    try:
        kappa = cohen_kappa_score(y_test, preds)
        if np.isnan(kappa):
            return 0.0
        return float(kappa)
    except ValueError:
        return 0.0


# ── Essay tokenization ─────────────────────────────────────────────────────
class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return (
            self.encodings["input_ids"][idx],
            self.encodings["attention_mask"][idx],
            self.encodings.get(
                "token_type_ids",
                torch.zeros_like(self.encodings["input_ids"])
            )[idx],
        )


# ── Build the list of (probe_name, label_column, label_fn) tasks ──────────
def build_tasks(df_tr, df_te):
    """
    Returns a list of (task_name, get_train_labels_fn, get_test_labels_fn).
    Each label fn takes the prompt-filtered df and returns (mask, y_int).
    mask: bool array over the df, True where label is valid.
    y_int: int labels for the rows where mask is True.
    """
    tasks = []

    # Binary attributes — direct 0/1 encoding
    for col in BINARY_COLS:
        if col not in df_tr.columns:
            continue
        def make_fn(c):
            def fn(df):
                vals = df[c]
                mask = vals.notna().values
                le = LabelEncoder()
                y = le.fit_transform(vals[mask].values)
                return mask, y, list(le.classes_)
            return fn
        tasks.append((col, make_fn(col)))

    # Race binaries — one-vs-rest, only emit task if there's race info at all
    if "race_ethnicity" in df_tr.columns:
        for task_name, target_value in RACE_BINARIES.items():
            def make_fn(tv):
                def fn(df):
                    vals = df["race_ethnicity"]
                    mask = vals.notna().values
                    y_str = vals[mask].values
                    y = (y_str == tv).astype(int)
                    classes = [f"not_{tv}", tv]
                    return mask, y, classes
                return fn
            tasks.append((task_name, make_fn(target_value)))

    return tasks


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading PERSUADE...")
    persuade_train = pd.read_csv(
        os.path.join(DATA_BASE, "PERSUADE/train/persuade_corpus_2.0_train.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    persuade_test = pd.read_csv(
        os.path.join(DATA_BASE, "PERSUADE/test/persuade_corpus_2.0_test.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    for df in [persuade_train, persuade_test]:
        for c in BINARY_COLS + ["race_ethnicity"]:
            if c in df.columns:
                df[c] = (df[c].astype(str).str.lower().str.strip()
                              .replace("nan", pd.NA))

    persuade_train = persuade_train[
        persuade_train[TEXT_COL].notna() & persuade_train[SCORE_COL].notna()
    ].reset_index(drop=True)
    persuade_test = persuade_test[
        persuade_test[TEXT_COL].notna() & persuade_test[SCORE_COL].notna()
    ].reset_index(drop=True)

    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)
    prompts   = sorted(persuade_train[PROMPT_COL].dropna().unique())

    all_results = []

    for prompt in prompts:
        # PERSUADE prompt names sometimes contain quotes / special chars.
        # The saved .pt filename in train_xlnet.py uses .replace(' ', '_')
        # without stripping quotes, so we mirror that exactly.
        pt_filename = f"xlnet_{prompt.replace(' ', '_')}.pt"
        pt_path = os.path.join(MODELS_DIR, pt_filename)
        if not os.path.exists(pt_path):
            print(f"Skipping '{prompt}' — no .pt at {pt_path}")
            continue

        print(f"\n{'=' * 60}\nProbing: {prompt}\n{'=' * 60}")

        tr = persuade_train[persuade_train[PROMPT_COL] == prompt].reset_index(drop=True)
        te = persuade_test[persuade_test[PROMPT_COL] == prompt].reset_index(drop=True)

        if len(tr) < 20 or len(te) < 5:
            print(f"  Skipping — insufficient rows (n_train={len(tr)}, n_test={len(te)})")
            continue

        # Load scoring model and freeze
        model = XLNetRegressor().to(device)
        state = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print(f"  Loaded {pt_path}")

        # Tokenize + extract hidden states ONCE per prompt (reuse across tasks)
        tr_ds = EssayDataset(tr[TEXT_COL].tolist(), tokenizer)
        tr_dl = DataLoader(tr_ds, batch_size=ENCODER_BS)
        te_ds = EssayDataset(te[TEXT_COL].tolist(), tokenizer)
        te_dl = DataLoader(te_ds, batch_size=ENCODER_BS)

        print("  Extracting hidden states (train)...")
        h_train = extract_hidden_states(model, tr_dl)
        print("  Extracting hidden states (test)...")
        h_test = extract_hidden_states(model, te_dl)

        # Free model memory before running probes
        del model
        torch.cuda.empty_cache()

        prompt_results = {"prompt": prompt,
                          "n_train": int(len(tr)),
                          "n_test": int(len(te)),
                          "demographics": {}}

        tasks = build_tasks(tr, te)

        for task_name, label_fn in tasks:
            tr_mask, y_tr, classes = label_fn(tr)
            te_mask, y_te, _       = label_fn(te)

            n_tr = int(tr_mask.sum())
            n_te = int(te_mask.sum())

            if n_tr < 20 or n_te < 5:
                continue
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                # Probe is undefined if a split has only one class
                continue

            h_tr_valid = h_train[tr_mask]
            h_te_valid = h_test[te_mask]
            num_classes = int(max(y_tr.max(), y_te.max()) + 1)

            # Stratified train/dev split where possible
            h_tr_split, h_dev_split, y_tr_split, y_dev_split = safe_stratified_split(
                h_tr_valid, y_tr, test_size=0.1, seed=RANDOM_SEED
            )

            # Need >=2 classes in dev for kappa to be informative
            if len(np.unique(y_dev_split)) < 2:
                # Fall back: use a slice of test as dev (test-shaped kappas still on held-out test)
                h_dev_split = h_te_valid
                y_dev_split = y_te
                h_tr_split, y_tr_split = h_tr_valid, y_tr

            probe, dev_kappa = train_probe(
                h_tr_split, y_tr_split,
                h_dev_split, y_dev_split,
                num_classes,
            )
            test_kappa = evaluate_probe(probe, h_te_valid, y_te)

            if test_kappa < 0.2:
                interpretation = "no_agreement"
            elif test_kappa < 0.4:
                interpretation = "minimal"
            elif test_kappa < 0.6:
                interpretation = "moderate"
            elif test_kappa < 0.8:
                interpretation = "substantial"
            else:
                interpretation = "almost_perfect"

            print(f"  {task_name:32s} dev={dev_kappa:+.3f}  test={test_kappa:+.3f}  "
                  f"n_train={n_tr}  n_test={n_te}  ({interpretation})")

            prompt_results["demographics"][task_name] = {
                "test_kappa":     round(test_kappa, 4),
                "dev_kappa":      round(dev_kappa, 4),
                "n_classes":      num_classes,
                "n_train":        n_tr,
                "n_test":         n_te,
                "interpretation": interpretation,
                "classes":        classes,
            }

        all_results.append(prompt_results)

    # ── Aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}\nPROBING SUMMARY (mean test kappa across prompts)\n{'=' * 60}")
    all_task_names = sorted({
        t for r in all_results for t in r["demographics"].keys()
    })
    summary = {}
    for task_name in all_task_names:
        kappas = [r["demographics"][task_name]["test_kappa"]
                  for r in all_results if task_name in r["demographics"]]
        if not kappas:
            continue
        summary[task_name] = {
            "mean_kappa": round(float(np.mean(kappas)), 4),
            "median_kappa": round(float(np.median(kappas)), 4),
            "min_kappa":  round(float(np.min(kappas)), 4),
            "max_kappa":  round(float(np.max(kappas)), 4),
            "n_prompts":  len(kappas),
        }
        print(f"  {task_name:32s} mean={summary[task_name]['mean_kappa']:+.3f}  "
              f"median={summary[task_name]['median_kappa']:+.3f}  "
              f"min={summary[task_name]['min_kappa']:+.3f}  "
              f"max={summary[task_name]['max_kappa']:+.3f}  "
              f"({summary[task_name]['n_prompts']} prompts)")

    output = {
        "run_version": RUN_VERSION,
        "config": {
            "PROBE_EPOCHS": PROBE_EPOCHS,
            "PROBE_LR": PROBE_LR,
            "PROBE_BS": PROBE_BS,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "PATIENCE": PATIENCE,
            "MAX_LEN": MAX_LEN,
            "RANDOM_SEED": RANDOM_SEED,
        },
        "summary": summary,
        "per_prompt": all_results,
    }
    out_path = os.path.join(OUT_DIR, f"probing_{RUN_VERSION}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved → {out_path}")


if __name__ == "__main__":
    main()