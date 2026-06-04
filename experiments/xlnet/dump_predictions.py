"""
dump_predictions.py
Forward-pass a set of trained per-prompt .pt models and write a single
per-essay prediction CSV (test set), with all demographic columns attached.

This is the artifact that downstream analyses consume:
  - compute_weighted_smd.py  -> K&O-style weighted regression z-scores
                                 (needs per-essay model predictions)
  - orthogonal_projection.py -> baseline predictions to compare against
  - any per-essay fairness analysis

Works on both baseline (train_xlnet.py) and GRL (train_xlnet_grl.py) .pt files:
both store xlnet.* + regressor.* keys; GRL files additionally store demo_head.*
keys, which are ignored here (we only load the scoring path).

The per-prompt .pt filename convention is xlnet_<prompt with spaces->_>.pt,
mirroring how both training scripts save (quotes/?/etc. preserved as-is).

Usage:
    # baseline
    python dump_predictions.py \
        --models  ../../results/xlnet/baseline_replication \
        --out     ../../results/xlnet/baseline_replication/test_predictions.csv

    # GRL gender
    python dump_predictions.py \
        --models  ../../results/xlnet/grl_gender_l05 \
        --out     ../../results/xlnet/grl_gender_l05/test_predictions.csv

Paths default to repo-relative; override DATA_BASE via env var if needed.

Output CSV columns:
    essay_id, prompt_name, full_text_len, true_score, pred_score,
    gender, race_ethnicity, ell_status, economically_disadvantaged,
    student_disability_status
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel

warnings.filterwarnings("ignore")

# ── Paths / config ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_BASE  = os.environ.get("DATA_BASE", os.path.join(REPO_ROOT, "..", "DATA"))

MODEL_NAME = "xlnet-base-cased"
MAX_LEN    = int(os.environ.get("MAX_LEN", 2048))
ENCODER_BS = int(os.environ.get("ENCODER_BS", 8))

# Dataset switch: PERSUADE (default) or ASAP.
DATASET    = os.environ.get("DATASET", "PERSUADE").upper()
if DATASET not in ("PERSUADE", "ASAP"):
    raise ValueError("Set DATASET=PERSUADE or DATASET=ASAP")
if DATASET == "PERSUADE":
    SCORE_COL = "holistic_essay_score"
    TEST_REL  = "PERSUADE/test/persuade_corpus_2.0_test.csv"
else:  # ASAP
    SCORE_COL = "score"
    TEST_REL  = "ASAP/test/ASAP_2_Final_github_test.csv"
TEXT_COL   = "full_text"
PROMPT_COL = "prompt_name"
ESSAY_ID   = "essay_id"
DEMO_COLS  = ["gender", "race_ethnicity", "ell_status",
              "economically_disadvantaged", "student_disability_status"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Scoring model (score path only; matches both training scripts) ──────────
class XLNetRegressor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.xlnet     = XLNetModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.xlnet.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, -1, :]   # last token, same as training
        return self.regressor(self.dropout(cls)).squeeze(-1)


def load_score_model(pt_path):
    """
    Load a .pt into the scoring model. GRL checkpoints carry extra demo_head.*
    keys; strict=False lets us load just xlnet.* + regressor.* and ignore them.
    """
    model = XLNetRegressor().to(device)
    state = torch.load(pt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # 'unexpected' will list demo_head.* for GRL files — expected, harmless.
    # 'missing' should be empty; warn if not (means a key mismatch).
    real_missing = [k for k in missing if not k.startswith("demo_head")]
    if real_missing:
        print(f"    warn: missing score-path keys {real_missing[:4]}...")
    return model


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


@torch.no_grad()
def predict(model, texts, tokenizer):
    model.eval()
    ds = EssayDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=ENCODER_BS)
    preds = []
    for input_ids, attn, tok_type in dl:
        out = model(input_ids.to(device), attn.to(device), tok_type.to(device))
        preds.append(out.float().cpu().numpy())
    return np.concatenate(preds)


def main(models_dir, out_path):
    print(f"[dump] device={device}  models={models_dir}")

    persuade_test = pd.read_csv(
        os.path.join(DATA_BASE, TEST_REL),
        low_memory=False
    ).drop_duplicates(subset=ESSAY_ID).reset_index(drop=True)

    for c in DEMO_COLS:
        if c in persuade_test.columns:
            persuade_test[c] = (persuade_test[c].astype(str)
                                .str.lower().str.strip().replace("nan", pd.NA))

    persuade_test = persuade_test[
        persuade_test[TEXT_COL].notna() & persuade_test[SCORE_COL].notna()
    ].reset_index(drop=True)

    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)
    prompts = sorted(persuade_test[PROMPT_COL].dropna().unique())

    rows = []
    for prompt in prompts:
        pt_path = os.path.join(models_dir, f"xlnet_{prompt.replace(' ', '_')}.pt")
        if not os.path.exists(pt_path):
            print(f"[{prompt}] no .pt, skipped")
            continue

        te = persuade_test[persuade_test[PROMPT_COL] == prompt].reset_index(drop=True)
        if len(te) < 1:
            continue

        model = load_score_model(pt_path)
        preds = predict(model, te[TEXT_COL].tolist(), tokenizer)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for i, (_, r) in enumerate(te.iterrows()):
            rows.append({
                "essay_id":   r.get(ESSAY_ID),
                "prompt_name": prompt,
                "true_score": float(r[SCORE_COL]),
                "pred_score": float(preds[i]),
                **{c: r.get(c) for c in DEMO_COLS},
            })
        print(f"[{prompt}] n_test={len(te)} dumped")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True,
                        help="Dir containing per-prompt xlnet_<prompt>.pt files")
    parser.add_argument("--out", required=True,
                        help="Output CSV path for per-essay test predictions")
    args = parser.parse_args()
    main(args.models, args.out)