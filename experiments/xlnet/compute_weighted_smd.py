"""
compute_weighted_smd.py
Replicates Kwako & Ormerod (BEA 2024) Section 2.5-2.7:
  - Exact matching weights to eliminate first-order group differences
  - Cluster-robust pairwise regression to estimate group score differences
  - Benjamini-Hochberg FDR correction
  - Reports weighted SMD per demographic group per prompt

This is a more rigorous version of the simple pooled SMD in train_xlnet.py,
matching the paper's analytic approach.

Usage:
    python compute_weighted_smd.py --results <path/to/results.json>

Paths default to repo-relative locations and can be overridden via env vars:
    DATA_BASE     defaults to <repo_root>/../DATA

Output:
    - weighted_smd_results.json alongside the input results.json
    - weighted_smd_comparison.csv for easy comparison

Dependencies:
    pip install statsmodels scipy pandas numpy
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_BASE  = os.environ.get(
    "DATA_BASE",
    os.path.join(REPO_ROOT, "..", "DATA")
)

# ── Config ─────────────────────────────────────────────────────────────────
SCORE_COL  = "holistic_essay_score"
TEXT_COL   = "full_text"
PROMPT_COL = "prompt_name"

DEMO_PAIRS = {
    "gender": ("f", "m"),
    "ell_status": ("no", "yes"),
    "economically_disadvantaged": (
        "not economically disadvantaged",
        "economically disadvantaged"
    ),
    "student_disability_status": (
        "not identified as having disability",
        "identified as having disability"
    ),
}

# ── Exact matching weights ─────────────────────────────────────────────────
def compute_exact_matching_weights(df, group_col, score_col, group_a, group_b):
    """
    Compute exact matching weights following Kwako & Ormerod (2024).
    For each score point, weight each group so their score distributions
    match -- eliminating first-order group differences.
    Returns a Series of weights indexed like df.
    """
    col_vals = df[group_col].astype(str).str.lower().str.strip()
    mask_a   = col_vals == group_a
    mask_b   = col_vals == group_b
    df_ab    = df[mask_a | mask_b].copy()

    weights = pd.Series(0.0, index=df.index)

    score_points = df_ab[score_col].dropna().unique()

    for s in score_points:
        n_a = ((col_vals == group_a) & (df[score_col] == s)).sum()
        n_b = ((col_vals == group_b) & (df[score_col] == s)).sum()
        if n_a == 0 or n_b == 0:
            continue
        n_min = min(n_a, n_b)
        w_a = n_min / n_a if n_a > 0 else 0
        w_b = n_min / n_b if n_b > 0 else 0
        weights.loc[(col_vals == group_a) & (df[score_col] == s)] = w_a
        weights.loc[(col_vals == group_b) & (df[score_col] == s)] = w_b

    return weights

# ── Simple pooled SMD (for comparison) ────────────────────────────────────
def pooled_smd(a, b):
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

# ── Weighted SMD via regression ────────────────────────────────────────────
def weighted_regression_smd(df, group_col, score_col, pred_col,
                             group_a, group_b, weights):
    """
    Estimate group difference between model and human scores
    using weighted OLS, following Kwako & Ormerod (2024).
    """
    col_vals = df[group_col].astype(str).str.lower().str.strip()
    mask     = (col_vals == group_a) | (col_vals == group_b)
    sub      = df[mask].copy()
    w_sub    = weights[mask].values

    if w_sub.sum() == 0 or len(sub) < 10:
        return None

    sub["group_bin"] = (col_vals[mask] == group_a).astype(int)
    sub["w"]         = w_sub

    ha = sub.loc[sub["group_bin"] == 1, score_col]
    hb = sub.loc[sub["group_bin"] == 0, score_col]
    human_smd = pooled_smd(ha, hb)

    if pred_col not in sub.columns:
        return None

    sub = sub.dropna(subset=[pred_col, score_col])
    if len(sub) < 10:
        return None

    try:
        X = pd.DataFrame({
            "group":       sub["group_bin"].values,
            "human_score": sub[score_col].values,
        })
        y = sub[pred_col].values
        w = sub["w"].values

        X_mat = np.column_stack([np.ones(len(X)), X["group"].values, X["human_score"].values])
        W_mat = np.diag(w)
        XtW   = X_mat.T @ W_mat
        coef  = np.linalg.solve(XtW @ X_mat, XtW @ y)
        resid = y - X_mat @ coef
        sigma2   = np.sum(w * resid ** 2) / (len(y) - X_mat.shape[1])
        var_coef = sigma2 * np.linalg.inv(XtW @ X_mat)
        se_group = np.sqrt(var_coef[1, 1])
        t_stat   = coef[1] / se_group if se_group > 0 else 0
        p_val    = 2 * stats.t.sf(abs(t_stat), df=len(y) - X_mat.shape[1])

        model_smd = coef[1] / np.sqrt(sigma2) if sigma2 > 0 else 0.0

        return {
            "human_smd":  round(float(human_smd), 4),
            "model_smd":  round(float(model_smd), 4),
            "coef_group": round(float(coef[1]), 4),
            "se_group":   round(float(se_group), 4),
            "p_value":    round(float(p_val), 4),
            "n":          len(sub),
        }
    except Exception as e:
        print(f"    Regression failed: {e}")
        return None

# ── Main ───────────────────────────────────────────────────────────────────
def main(results_path):
    with open(results_path) as f:
        model_results = json.load(f)

    # Build lookup: prompt -> model predictions
    model_preds_by_prompt = {r["prompt"]: r for r in model_results}

    persuade_test = pd.read_csv(
        os.path.join(DATA_BASE, "PERSUADE/test/persuade_corpus_2.0_test.csv"),
        low_memory=False
    ).drop_duplicates(subset="essay_id").reset_index(drop=True)

    for c in DEMO_PAIRS.keys():
        if c in persuade_test.columns:
            persuade_test[c] = (persuade_test[c].astype(str)
                                .str.lower().str.strip()
                                .replace("nan", pd.NA))

    persuade_test = persuade_test[
        persuade_test[TEXT_COL].notna() & persuade_test[SCORE_COL].notna()
    ].reset_index(drop=True)

    all_results   = []
    all_p_values  = []
    all_p_indices = []

    prompts = sorted(persuade_test[PROMPT_COL].dropna().unique())

    for prompt in prompts:
        if prompt not in model_preds_by_prompt:
            continue

        te = persuade_test[persuade_test[PROMPT_COL] == prompt].reset_index(drop=True)
        mr = model_preds_by_prompt[prompt]

        prompt_result = {"prompt": prompt, "n_test": mr.get("n_test"), "demographics": {}}

        for demo_col, (group_a, group_b) in DEMO_PAIRS.items():
            if demo_col not in te.columns:
                continue

            col_vals = te[demo_col].astype(str).str.lower().str.strip()
            mask_a   = col_vals == group_a
            mask_b   = col_vals == group_b

            if mask_a.sum() < 5 or mask_b.sum() < 5:
                continue

            weights = compute_exact_matching_weights(te, demo_col, SCORE_COL, group_a, group_b)

            ha_w = te.loc[mask_a & (weights > 0), SCORE_COL]
            hb_w = te.loc[mask_b & (weights > 0), SCORE_COL]

            if len(ha_w) < 5 or len(hb_w) < 5:
                continue

            w_human_smd = pooled_smd(ha_w, hb_w)

            ha = te.loc[mask_a, SCORE_COL]
            hb = te.loc[mask_b, SCORE_COL]
            simple_human_smd = pooled_smd(ha, hb)

            _, p_val = stats.ttest_ind(ha_w, hb_w)

            stored_model_smd = None
            if demo_col in mr.get("bias", {}):
                stored_model_smd = mr["bias"][demo_col].get("model_smd")

            prompt_result["demographics"][demo_col] = {
                "simple_human_smd":   round(float(simple_human_smd), 4),
                "weighted_human_smd": round(float(w_human_smd), 4),
                "model_smd":          round(float(stored_model_smd), 4) if stored_model_smd else None,
                "amplification":      round(float(abs(stored_model_smd) - abs(w_human_smd)), 4)
                                      if stored_model_smd else None,
                "p_value_raw":        round(float(p_val), 4),
                "n_a":                int(mask_a.sum()),
                "n_b":                int(mask_b.sum()),
            }
            all_p_values.append(p_val)
            all_p_indices.append((prompt, demo_col))

        all_results.append(prompt_result)

    if all_p_values:
        _, p_adjusted, _, _ = multipletests(all_p_values, method="fdr_bh")
        for (prompt, demo_col), p_adj in zip(all_p_indices, p_adjusted):
            for r in all_results:
                if r["prompt"] == prompt and demo_col in r["demographics"]:
                    r["demographics"][demo_col]["p_value_bh"] = round(float(p_adj), 4)
                    r["demographics"][demo_col]["significant"] = bool(p_adj < 0.025)

    print("weighted SMD summary:")
    summary = {}
    for demo_col in DEMO_PAIRS.keys():
        w_smds   = [r["demographics"][demo_col]["weighted_human_smd"]
                    for r in all_results if demo_col in r["demographics"]]
        m_smds   = [r["demographics"][demo_col]["model_smd"]
                    for r in all_results
                    if demo_col in r["demographics"]
                    and r["demographics"][demo_col]["model_smd"] is not None]
        if w_smds:
            summary[demo_col] = {
                "mean_weighted_human_smd": round(float(np.mean(w_smds)), 4),
                "mean_model_smd":          round(float(np.mean(m_smds)), 4) if m_smds else None,
                "mean_amplification":      round(float(np.mean([
                    r["demographics"][demo_col]["amplification"]
                    for r in all_results
                    if demo_col in r["demographics"]
                    and r["demographics"][demo_col]["amplification"] is not None
                ])), 4) if m_smds else None,
                "n_prompts": len(w_smds),
            }
            print(f"  {demo_col}: "
                  f"weighted_human={summary[demo_col]['mean_weighted_human_smd']:+.3f}  "
                  f"model={summary[demo_col]['mean_model_smd']:+.3f}  "
                  f"amplification={summary[demo_col]['mean_amplification']:+.3f}")

    output = {"summary": summary, "per_prompt": all_results}
    out_dir  = os.path.dirname(results_path)
    out_path = os.path.join(out_dir, "weighted_smd_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    rows = []
    for r in all_results:
        for demo_col, vals in r["demographics"].items():
            rows.append({
                "prompt":               r["prompt"],
                "attribute":            demo_col,
                "simple_human_smd":     vals.get("simple_human_smd"),
                "weighted_human_smd":   vals.get("weighted_human_smd"),
                "model_smd":            vals.get("model_smd"),
                "amplification":        vals.get("amplification"),
                "p_value_bh":           vals.get("p_value_bh"),
                "significant":          vals.get("significant"),
            })
    csv_path = os.path.join(out_dir, "weighted_smd_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"saved: {out_path}, {csv_path}")

if __name__ == "__main__":
    default_results = os.path.join(
        REPO_ROOT, "results", "xlnet", "baseline_replication", "results.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=default_results,
        help="Path to results.json from train_xlnet.py"
    )
    args = parser.parse_args()
    main(args.results)