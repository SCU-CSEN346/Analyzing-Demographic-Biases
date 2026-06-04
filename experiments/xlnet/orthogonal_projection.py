"""
orthogonal_projection.py
Post-hoc bias mitigation by orthogonal projection of HIDDEN STATES.

Implements the framing of paper §6.2 / Kwako & Ormerod (2024) §4.3:
  - Freeze a trained scoring model.
  - Estimate a demographic direction w_d from a linear probe on the frozen
    encoder's hidden states (the SAME extraction used by probe_hidden_states.py).
  - Build P = I - (w_d w_d^T) / ||w_d||^2, the orthogonal projector onto the
    hyperplane normal to w_d.
  - Re-score with the frozen regression head on projected hidden states:
        y_hat = regressor( P h )
  - Optionally iterate (INLP, Ravfogel et al. 2020): re-probe the projected
    representations, get a new direction, project again, until a fresh probe
    on the projected space no longer beats chance.

This is the HIDDEN-STATE projection described in §6.2, distinct from projecting
the regression WEIGHTS. Choose this to match §6.2 as written.

Runs per prompt, per attribute, on PERSUADE test essays. Writes:
  - test_predictions_proj_<attr>.csv : per-essay projected predictions
    (same columns as dump_predictions.py, so compute_weighted_smd.py --predictions
    consumes it directly for the K&O regression z)
  - projection_results.json : per-prompt QWK before/after + probe kappa
    before/after, for the fairness-accuracy summary

Works on baseline AND GRL .pt (strict=False ignores GRL demo_head.* keys).

Usage:
    # single attribute
    DEMO_ATTR=ell_status python orthogonal_projection.py \
        --models  ../../results/xlnet/baseline_replication \
        --outdir  ../../results/xlnet/projection_ell

    # all four attributes in one run (writes one CSV + json per attribute)
    python orthogonal_projection.py \
        --models ../../results/xlnet/baseline_replication \
        --outdir ../../results/xlnet/projection_baseline \
        --attrs gender ell_status economically_disadvantaged student_disability_status

Env overrides:
    DATA_BASE   defaults to <repo_root>/../DATA
    MAX_LEN     2048
    INLP_ITERS  1 (single projection). Set >1 for iterative null-space projection.
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

# ── Paths / config ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_BASE  = os.environ.get("DATA_BASE", os.path.join(REPO_ROOT, "..", "DATA"))

MODEL_NAME = "xlnet-base-cased"
MAX_LEN    = int(os.environ.get("MAX_LEN", 2048))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
INLP_ITERS = int(os.environ.get("INLP_ITERS", 1))
RANDOM_SEED = 42

TEXT_COL   = "full_text"
SCORE_COL  = "holistic_essay_score"
PROMPT_COL = "prompt_name"
ESSAY_ID   = "essay_id"
ALL_DEMO_COLS = ["gender", "race_ethnicity", "ell_status",
                 "economically_disadvantaged", "student_disability_status"]

# Binary pairs (focal/reference) used for direction estimation + SMD downstream.
DEMO_PAIRS = {
    "gender": ("f", "m"),
    "ell_status": ("no", "yes"),
    "economically_disadvantaged": (
        "not economically disadvantaged", "economically disadvantaged"),
    "student_disability_status": (
        "not identified as having disability", "identified as having disability"),
}

# Race is handled as one-vs-rest (matches probe_hidden_states.py build_tasks,
# which emits race_black / race_white / ... tasks). The key here MUST match the
# probe's task name so probe_projected_states.py can find it via __attr__.
# Maps projection-attr-name -> (race_ethnicity column, target value treated as 1).
RACE_ONE_VS_REST = {
    "race_black": "black/african american",
    "race_white": "white",
    "race_hispanic": "hispanic/latino",
    "race_asian": "asian/pacific islander",
}

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Frozen scoring model (score path only; matches probe + dump) ────────────
class XLNetRegressor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.xlnet     = XLNetModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.xlnet.config.hidden_size, 1)

    @torch.no_grad()
    def get_hidden(self, input_ids, attention_mask, token_type_ids=None):
        out = self.xlnet(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        return out.last_hidden_state[:, -1, :]   # (B, 768), same as training/probe

    @torch.no_grad()
    def score_from_hidden(self, h):
        # Re-score from a (possibly projected) hidden state. Eval mode => dropout
        # is identity, so this matches inference exactly.
        return self.regressor(h).squeeze(-1)


def load_score_model(pt_path):
    model = XLNetRegressor().to(device)
    state = torch.load(pt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("demo_head")]
    if real_missing:
        print(f"    warn: missing score-path keys {real_missing[:4]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.enc = tokenizer(texts, max_length=max_len, padding="max_length",
                             truncation=True, return_tensors="pt")

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, idx):
        return (self.enc["input_ids"][idx],
                self.enc["attention_mask"][idx],
                self.enc.get("token_type_ids",
                             torch.zeros_like(self.enc["input_ids"]))[idx])


@torch.no_grad()
def extract_hidden(model, texts, tokenizer):
    ds = EssayDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE)
    hs = []
    for input_ids, attn, tok in dl:
        h = model.get_hidden(input_ids.to(device), attn.to(device), tok.to(device))
        hs.append(h.cpu())
    return torch.cat(hs, dim=0).numpy()   # (N, 768)


# ── Projection math ─────────────────────────────────────────────────────────
def direction_from_probe(h, y):
    """
    Fit a logistic-regression probe h -> y and return the unit normal of its
    decision boundary (the demographic direction w_d) plus the probe's kappa.
    """
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(h, y)
    w = clf.coef_.reshape(-1).astype(np.float64)   # (768,)
    nrm = np.linalg.norm(w)
    if nrm == 0:
        return None, 0.0
    w = w / nrm
    kappa = cohen_kappa_score(y, clf.predict(h))
    return w, float(kappa)


def projector(w):
    """P = I - w w^T (w already unit-norm)."""
    return np.eye(w.shape[0]) - np.outer(w, w)


def inlp_project(h_train, y_train, h_test, n_iters, extra=None):
    """
    Iterative null-space projection. Returns (P_total, h_train_proj, h_test_proj,
    kappa_before, kappa_after, n_directions_removed [, extra_proj]).
    n_iters=1 reduces to a single projection (the §6.2 single-step case).

    If `extra` is provided (e.g. the full train matrix for a downstream held-out
    probe), it is projected by the SAME sequence of projectors and returned as a
    final element. This is correct even for n_iters>1, where the cumulative
    application order matters and cannot be recovered from P_total.T alone.
    """
    d = h_train.shape[1]
    P_total = np.eye(d)
    htr, hte = h_train.copy(), h_test.copy()
    hex_ = extra.copy() if extra is not None else None

    w0, kappa_before = direction_from_probe(htr, y_train)
    if w0 is None:
        if extra is not None:
            return P_total, htr, hte, kappa_before, kappa_before, 0, hex_
        return P_total, htr, hte, kappa_before, kappa_before, 0

    removed = 0
    kappa_after = kappa_before
    for _ in range(n_iters):
        w, k = direction_from_probe(htr, y_train)
        if w is None:
            break
        P = projector(w)
        htr = htr @ P.T
        hte = hte @ P.T
        if hex_ is not None:
            hex_ = hex_ @ P.T
        P_total = P @ P_total
        removed += 1
        # re-probe the projected space to measure residual recoverability
        _, kappa_after = direction_from_probe(htr, y_train)
        # stop early if the attribute is no longer recoverable above chance
        if kappa_after < 0.02:
            break

    if extra is not None:
        return P_total, htr, hte, kappa_before, kappa_after, removed, hex_
    return P_total, htr, hte, kappa_before, kappa_after, removed


def round_qwk(y_true, y_pred):
    yp = np.clip(np.round(y_pred), 1, 6).astype(int)
    return cohen_kappa_score(y_true.astype(int), yp, weights="quadratic")


# ── Main ───────────────────────────────────────────────────────────────────
def run_attribute(demo_attr, models_dir, persuade_test, persuade_train,
                  tokenizer, outdir):
    is_race = demo_attr in RACE_ONE_VS_REST
    if is_race:
        race_col = "race_ethnicity"
        target_val = RACE_ONE_VS_REST[demo_attr]
        print(f"\n{'#'*60}\n# Attribute: {demo_attr}  ({target_val} vs rest)\n{'#'*60}")
    else:
        group_a, group_b = DEMO_PAIRS[demo_attr]
        print(f"\n{'#'*60}\n# Attribute: {demo_attr}  ({group_a} vs {group_b})\n{'#'*60}")

    rows = []          # per-essay projected predictions
    per_prompt = []    # per-prompt QWK + kappa summary
    proj_dump = {"train": {}, "test": {}}  # projected hidden states for rigorous probe
    prompts = sorted(persuade_train[PROMPT_COL].dropna().unique())

    for prompt in prompts:
        pt_path = os.path.join(models_dir, f"xlnet_{prompt.replace(' ', '_')}.pt")
        if not os.path.exists(pt_path):
            print(f"[{prompt}] no .pt, skipped")
            continue

        tr = persuade_train[persuade_train[PROMPT_COL] == prompt].reset_index(drop=True)
        te = persuade_test[persuade_test[PROMPT_COL] == prompt].reset_index(drop=True)

        # Restrict probe-training rows + build binary labels.
        if is_race:
            col_tr = tr[race_col].astype(str).str.lower().str.strip()
            # one-vs-rest: include all rows with a known (non-null) race value
            valid = ~col_tr.isin(["nan", "", "none", "<na>"])
            tr_mask = valid.values
        else:
            col_tr = tr[demo_attr].astype(str).str.lower().str.strip()
            tr_mask = col_tr.isin([group_a, group_b]).values
        if tr_mask.sum() < 20:
            print(f"[{prompt}] too few labeled train rows ({tr_mask.sum()}), skipped")
            continue

        model = load_score_model(pt_path)
        h_tr_all = extract_hidden(model, tr[TEXT_COL].tolist(), tokenizer)
        h_te_all = extract_hidden(model, te[TEXT_COL].tolist(), tokenizer)

        h_tr = h_tr_all[tr_mask]
        if is_race:
            y_tr = (col_tr[tr_mask] == target_val).astype(int).values  # target race = 1
        else:
            y_tr = (col_tr[tr_mask] == group_b).astype(int).values   # group_b = focal=1

        if len(np.unique(y_tr)) < 2:
            print(f"[{prompt}] only one class for {demo_attr}, skipped")
            continue
        if is_race and int((y_tr == 1).sum()) < 10:
            print(f"[{prompt}] too few {demo_attr} target rows "
                  f"({int((y_tr==1).sum())}), skipped")
            continue

        # Baseline (unprojected) test scores
        pred_base = model.score_from_hidden(
            torch.tensor(h_te_all, dtype=torch.float32, device=device)
        ).cpu().numpy()
        qwk_base = round_qwk(te[SCORE_COL].values, pred_base)

        # Estimate direction(s) on train, project train + test. Pass the FULL
        # train matrix (h_tr_all) as an extra carry-along so it is projected by
        # the exact same sequence of projectors (correct even for INLP_ITERS>1,
        # where projector order matters and P_total.T != product of P.T's).
        P, _, h_te_proj, k_before, k_after, n_dir, h_tr_all_proj = inlp_project(
            h_tr, y_tr, h_te_all, INLP_ITERS, extra=h_tr_all
        )

        # Re-score on projected test hidden states
        pred_proj = model.score_from_hidden(
            torch.tensor(h_te_proj, dtype=torch.float32, device=device)
        ).cpu().numpy()
        qwk_proj = round_qwk(te[SCORE_COL].values, pred_proj)

        # ── Accumulate projected hidden states for the rigorous "after" probe ──
        # We dump the FULL projected train + test matrices (all rows, all demo
        # columns) so the downstream probe applies its OWN label cleaning /
        # masking / encoding — identical methodology to the "before" probe on
        # raw states. Per-prompt arrays are stored under the prompt key.
        proj_dump["train"][prompt] = {
            "h": h_tr_all_proj.astype(np.float32),
            "essay_id": tr[ESSAY_ID].astype(str).values,
            **{c: tr[c].astype(str).values for c in ALL_DEMO_COLS if c in tr.columns},
            SCORE_COL: tr[SCORE_COL].values.astype(np.float32),
        }
        proj_dump["test"][prompt] = {
            "h": h_te_proj.astype(np.float32),
            "essay_id": te[ESSAY_ID].astype(str).values,
            **{c: te[c].astype(str).values for c in ALL_DEMO_COLS if c in te.columns},
            SCORE_COL: te[SCORE_COL].values.astype(np.float32),
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[{prompt}] QWK {qwk_base:.3f} -> {qwk_proj:.3f}  "
              f"probe_kappa {k_before:.3f} -> {k_after:.3f}  "
              f"({n_dir} dir removed, n_test={len(te)})")

        per_prompt.append({
            "prompt": prompt, "n_test": int(len(te)),
            "qwk_base": float(qwk_base), "qwk_proj": float(qwk_proj),
            "probe_kappa_before": float(k_before), "probe_kappa_after": float(k_after),
            "n_directions_removed": int(n_dir),
        })

        for i, (_, r) in enumerate(te.iterrows()):
            rows.append({
                "essay_id": r.get(ESSAY_ID),
                "prompt_name": prompt,
                "true_score": float(r[SCORE_COL]),
                "pred_score": float(pred_proj[i]),      # projected prediction
                **{c: r.get(c) for c in ALL_DEMO_COLS},
            })

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"test_predictions_proj_{demo_attr}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Dump projected hidden states for the rigorous held-out "after" probe ──
    # Flatten the nested dict into "split__prompt__field" keys (np.savez can't
    # nest). probe_projected_states.py reconstructs per-prompt train/test
    # matrices and runs the SAME probe methodology as probe_hidden_states.py.
    flat = {}
    for split in ("train", "test"):
        for prompt, d in proj_dump[split].items():
            safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace('"', "")
            for field, arr in d.items():
                flat[f"{split}__{safe_prompt}__{field}"] = arr
    # Keep a manifest of prompt names so the consumer can recover the mapping.
    flat["__prompts__"] = np.array(
        sorted({p for s in ("train", "test") for p in proj_dump[s]}), dtype=object
    )
    flat["__attr__"] = np.array(demo_attr, dtype=object)
    flat["__score_col__"] = np.array(SCORE_COL, dtype=object)
    flat["__demo_cols__"] = np.array(ALL_DEMO_COLS, dtype=object)
    npz_path = os.path.join(outdir, f"projected_states_{demo_attr}.npz")
    np.savez_compressed(npz_path, **flat)
    print(f"  -> dumped projected states: {npz_path}")

    qb = [p["qwk_base"] for p in per_prompt]
    qp = [p["qwk_proj"] for p in per_prompt]
    kb = [p["probe_kappa_before"] for p in per_prompt]
    ka = [p["probe_kappa_after"] for p in per_prompt]
    summary = {
        "attribute": demo_attr, "inlp_iters": INLP_ITERS,
        "macro_qwk_base": float(np.mean(qb)) if qb else None,
        "macro_qwk_proj": float(np.mean(qp)) if qp else None,
        "mean_probe_kappa_before": float(np.mean(kb)) if kb else None,
        "mean_probe_kappa_after":  float(np.mean(ka)) if ka else None,
        "n_prompts": len(per_prompt),
        "per_prompt": per_prompt,
    }
    json_path = os.path.join(outdir, f"projection_results_{demo_attr}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if not per_prompt:
        print(f"\n[{demo_attr}] NO prompts produced results — all skipped "
              f"(check group sizes / label matching). Wrote empty summary.")
        return summary

    print(f"\n[{demo_attr}] macro QWK {summary['macro_qwk_base']:.3f} -> "
          f"{summary['macro_qwk_proj']:.3f}   "
          f"probe kappa {summary['mean_probe_kappa_before']:.3f} -> "
          f"{summary['mean_probe_kappa_after']:.3f}")
    print(f"  -> {csv_path}\n  -> {json_path}")
    return summary


def load_persuade():
    def _load(split):
        # Try subdir layout first, then flat.
        sub = os.path.join(DATA_BASE, f"PERSUADE/{split}/persuade_corpus_2.0_{split}.csv")
        flat = os.path.join(DATA_BASE, f"PERSUADE/persuade_corpus_2.0_{split}.csv")
        path = sub if os.path.exists(sub) else flat
        df = pd.read_csv(path, low_memory=False)
        return df.drop_duplicates(subset=ESSAY_ID).reset_index(drop=True)

    tr, te = _load("train"), _load("test")
    for df in [tr, te]:
        for c in ALL_DEMO_COLS:
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower().str.strip().replace("nan", pd.NA)
    tr = tr[tr[TEXT_COL].notna() & tr[SCORE_COL].notna()].reset_index(drop=True)
    te = te[te[TEXT_COL].notna() & te[SCORE_COL].notna()].reset_index(drop=True)
    return tr, te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True,
                        help="Dir with per-prompt xlnet_<prompt>.pt files")
    parser.add_argument("--outdir", required=True,
                        help="Output dir for projected predictions + summary")
    parser.add_argument("--attrs", nargs="+",
                        default=[os.environ.get("DEMO_ATTR", "gender")],
                        help="Attributes to project (space-separated). "
                             "Default: $DEMO_ATTR or gender.")
    args = parser.parse_args()

    valid_attrs = set(DEMO_PAIRS) | set(RACE_ONE_VS_REST)
    bad = [a for a in args.attrs if a not in valid_attrs]
    if bad:
        raise ValueError(f"Unknown attrs {bad}. Choose from {sorted(valid_attrs)}")

    print(f"device={device}  models={args.models}  attrs={args.attrs}  "
          f"INLP_ITERS={INLP_ITERS}")
    persuade_train, persuade_test = load_persuade()
    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)

    all_summaries = {}
    for attr in args.attrs:
        all_summaries[attr] = run_attribute(
            attr, args.models, persuade_test, persuade_train, tokenizer, args.outdir
        )

    combined = os.path.join(args.outdir, "projection_summary_all.json")
    with open(combined, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nall attributes done -> {combined}")


if __name__ == "__main__":
    main()