"""
mediation_analysis.py
Tests whether the model's demographic score gap is mediated by construct-relevant
surface features (length, complexity), per the paper's central claim.

For each attribute, fit two weighted regressions on the exact-matched sample:
    (1) pred_score ~ group                      -> raw demographic coefficient
    (2) pred_score ~ group + surface_features   -> controlled coefficient
If the group coefficient shrinks substantially from (1) to (2), the gap is
mediated by those features -- i.e. removing the demographic DIRECTION fails
because the scoring head reads the FEATURES, which survive. Reports the
proportion mediated = 1 - coef_controlled/coef_raw.

Surface features (all computable from text, no external deps):
    n_words, n_sentences, mean_word_len, flesch_kincaid (manual), type_token_ratio

Reuses the exact-matching weights from compute_weighted_smd.py so the mediation
is measured on the same matched sample as the bias estimate.

Usage:
    python mediation_analysis.py \
        --predictions ../../results/xlnet/baseline_replication/test_predictions.csv \
        --test-csv    /home/rl/CSEN364/PROJECT/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv

Output: prints a per-attribute table; writes mediation_results.json + .csv next
to the predictions file.
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCORE_COL = "holistic_essay_score"
TEXT_COL  = "full_text"
ESSAY_ID  = "essay_id"

DEMO_PAIRS = {
    "gender": ("f", "m"),
    "ell_status": ("no", "yes"),
    "economically_disadvantaged": (
        "not economically disadvantaged", "economically disadvantaged"),
    "student_disability_status": (
        "not identified as having disability", "identified as having disability"),
}

FEATURES = ["n_words", "n_sentences", "mean_word_len", "flesch_kincaid", "type_token_ratio"]


# ── Surface feature extraction ──────────────────────────────────────────────
_SENT_SPLIT = re.compile(r"[.!?]+")
_WORD       = re.compile(r"[A-Za-z]+")
_VOWEL_RUN  = re.compile(r"[aeiouy]+", re.IGNORECASE)


def count_syllables(word):
    # Cheap heuristic syllable count: vowel groups, min 1.
    groups = _VOWEL_RUN.findall(word)
    n = len(groups)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def extract_features(text):
    text = str(text)
    words = _WORD.findall(text)
    n_words = len(words)
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    n_sent = max(len(sents), 1)
    if n_words == 0:
        return dict(n_words=0, n_sentences=n_sent, mean_word_len=0.0,
                    flesch_kincaid=0.0, type_token_ratio=0.0)
    mean_word_len = float(np.mean([len(w) for w in words]))
    syll = sum(count_syllables(w) for w in words)
    # Flesch-Kincaid grade level
    fk = 0.39 * (n_words / n_sent) + 11.8 * (syll / n_words) - 15.59
    ttr = len(set(w.lower() for w in words)) / n_words
    return dict(n_words=n_words, n_sentences=n_sent, mean_word_len=mean_word_len,
                flesch_kincaid=float(fk), type_token_ratio=float(ttr))


# ── Exact-matching weights (same logic as compute_weighted_smd.py) ──────────
def exact_matching_weights(df, group_col, score_col, group_a, group_b):
    col = df[group_col].astype(str).str.lower().str.strip()
    w = pd.Series(0.0, index=df.index)
    for s in df.loc[col.isin([group_a, group_b]), score_col].dropna().unique():
        na = ((col == group_a) & (df[score_col] == s)).sum()
        nb = ((col == group_b) & (df[score_col] == s)).sum()
        if na == 0 or nb == 0:
            continue
        nmin = min(na, nb)
        w.loc[(col == group_a) & (df[score_col] == s)] = nmin / na
        w.loc[(col == group_b) & (df[score_col] == s)] = nmin / nb
    return w


def weighted_ols(X, y, w):
    """Return coefficient vector via weighted least squares (X includes intercept)."""
    W = np.diag(w)
    XtW = X.T @ W
    return np.linalg.solve(XtW @ X, XtW @ y)


def main(predictions_path, test_csv, control_features=None):
    # control_features: which surface features to control for. None = all FEATURES.
    # Lets you test robustness of (possibly suppression) results to collinearity,
    # e.g. --features n_words controls for length alone.
    active = control_features if control_features else FEATURES
    print(f"controlling for features: {active}")

    preds = pd.read_csv(predictions_path, low_memory=False)
    # predictions CSV uses 'true_score'; rename to the score col used here
    if "true_score" in preds.columns and SCORE_COL not in preds.columns:
        preds = preds.rename(columns={"true_score": SCORE_COL})

    test = pd.read_csv(test_csv, low_memory=False).drop_duplicates(subset=ESSAY_ID)
    test = test[[ESSAY_ID, TEXT_COL]]

    df = preds.merge(test, on=ESSAY_ID, how="left")
    n_missing = df[TEXT_COL].isna().sum()
    if n_missing:
        print(f"warn: {n_missing} essays had no text after join; dropping them")
        df = df[df[TEXT_COL].notna()].reset_index(drop=True)

    print(f"computing surface features for {len(df)} essays...")
    feats = df[TEXT_COL].apply(extract_features).apply(pd.Series)
    df = pd.concat([df, feats], axis=1)

    # standardize features (z-score) so coefficients are comparable
    for f in active:
        mu, sd = df[f].mean(), df[f].std()
        df[f + "_z"] = (df[f] - mu) / (sd if sd > 0 else 1.0)
    feat_z = [f + "_z" for f in active]

    results = {}
    rows = []
    for attr, (ga, gb) in DEMO_PAIRS.items():
        col = df[attr].astype(str).str.lower().str.strip()
        sub = df[col.isin([ga, gb])].copy()
        if len(sub) < 30:
            print(f"[{attr}] too few rows, skipped")
            continue

        w = exact_matching_weights(sub, attr, SCORE_COL, ga, gb).loc[sub.index].values
        keep = w > 0
        sub, w = sub[keep], w[keep]
        if len(sub) < 30:
            print(f"[{attr}] too few matched rows, skipped")
            continue

        scol = sub[attr].astype(str).str.lower().str.strip()
        group = (scol == gb).astype(int).values   # focal = group_b = 1
        y = sub["pred_score"].values
        sd_y = y.std() if y.std() > 0 else 1.0

        # (1) raw: pred ~ group
        X1 = np.column_stack([np.ones(len(sub)), group])
        b1 = weighted_ols(X1, y, w)
        coef_raw = b1[1]

        # (2) controlled: pred ~ group + features
        X2 = np.column_stack([np.ones(len(sub)), group, sub[feat_z].values])
        b2 = weighted_ols(X2, y, w)
        coef_ctrl = b2[1]

        prop_mediated = 1.0 - (coef_ctrl / coef_raw) if coef_raw != 0 else float("nan")

        results[attr] = {
            "n_matched": int(len(sub)),
            "coef_raw": round(float(coef_raw), 4),
            "coef_controlled": round(float(coef_ctrl), 4),
            "smd_raw": round(float(coef_raw / sd_y), 4),
            "smd_controlled": round(float(coef_ctrl / sd_y), 4),
            "prop_mediated": round(float(prop_mediated), 4),
        }
        rows.append({"attribute": attr, **results[attr]})
        print(f"[{attr:30s}] raw coef={coef_raw:+.4f}  controlled={coef_ctrl:+.4f}  "
              f"mediated={prop_mediated:.1%}  (n={len(sub)})")

    out_dir = os.path.dirname(os.path.abspath(predictions_path))
    payload = {"control_features": active, "results": results}
    with open(os.path.join(out_dir, "mediation_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "mediation_results.csv"), index=False)
    print(f"\nsaved mediation_results.json + .csv to {out_dir}")
    print("\nInterpretation: prop_mediated = fraction of the raw demographic gap")
    print("explained by surface features. High +ve = feature-mediated; negative")
    print("= suppression (controlling for features REVEALS a larger gap).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--features", nargs="+", default=None,
                    help=f"Subset of features to control for (default all). "
                         f"Choices: {FEATURES}. E.g. --features n_words to control "
                         f"for length alone (collinearity robustness check).")
    args = ap.parse_args()
    if args.features:
        bad = [f for f in args.features if f not in FEATURES]
        if bad:
            raise ValueError(f"Unknown features {bad}. Choose from {FEATURES}")
    main(args.predictions, args.test_csv, args.features)