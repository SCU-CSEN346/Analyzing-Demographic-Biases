# compare_bias_results.py
# Calculates bias metrics on debiased predictions and compares them to the baseline

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import cohen_kappa_score

DEMO_COLUMNS = [
    "gender", 
    "race_ethnicity", 
    "ell_status", 
    "economically_disadvantaged", 
    "student_disability_status",
    "economic_disadvantage" # Fallback for old persuade
]

def run_bias_analysis(preds_path, base_preds_path, meta_path, is_persuade):
    print(f"Loading predictions from {preds_path}...")
    preds_df = pd.read_csv(preds_path)
    base_preds_df = pd.read_csv(base_preds_path)
    meta_df = pd.read_csv(meta_path, low_memory=False)
    
    if is_persuade:
        df = pd.merge(preds_df, meta_df, left_on="essay_id", right_on="essay_id_comp")
        base_df = pd.merge(base_preds_df, meta_df, left_on="essay_id", right_on="essay_id_comp")
        df["raw_score_diff"] = df["roberta_score"] - df["human_score"]
        mean_diff = df["raw_score_diff"].mean()
        std_diff = df["raw_score_diff"].std()
        df["score_diff"] = (df["raw_score_diff"] - mean_diff) / std_diff
    else:
        df = pd.merge(preds_df, meta_df, on="essay_id")
        base_df = pd.merge(base_preds_df, meta_df, on="essay_id")
        df["raw_score_diff"] = df["roberta_score"] - df["human_score"]
        df["score_diff"] = df.groupby("set")["raw_score_diff"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
    results = []
    
    for column in DEMO_COLUMNS:
        if column not in df.columns:
            continue
            
        subset = df.dropna(subset=[column, "prompt_name", "score_diff"]).copy()
        groups = subset[column].unique()
        if len(groups) < 2:
            continue
            
        reference_group = groups[0]
        
        for focal_group in groups[1:]:
            group_mask = subset[column].isin([reference_group, focal_group])
            test_df = subset[group_mask].copy()
            
            if test_df.empty:
                continue
                
            # Calculate QWK for this specific focal group
            focal_mask_debiased = (df[column] == focal_group) & df["roberta_score"].notna() & df["human_score"].notna()
            focal_debiased = df[focal_mask_debiased]
            
            focal_mask_base = (base_df[column] == focal_group) & base_df["roberta_score"].notna() & base_df["human_score"].notna()
            focal_base = base_df[focal_mask_base]
            
            debiased_qwk = 0.0
            if len(focal_debiased) > 0:
                debiased_qwk = cohen_kappa_score(
                    focal_debiased["human_score"].astype(int), 
                    np.round(focal_debiased["roberta_score"]).astype(int), 
                    weights="quadratic"
                )
                
            base_qwk = 0.0
            if len(focal_base) > 0:
                base_qwk = cohen_kappa_score(
                    focal_base["human_score"].astype(int), 
                    np.round(focal_base["roberta_score"]).astype(int), 
                    weights="quadratic"
                )

            test_df["is_focal"] = (test_df[column] == focal_group).astype(int)
            score_counts = test_df.groupby(["human_score", "is_focal"]).size().unstack(fill_value=0)
            
            def calculate_weight(row):
                score = row['human_score']
                is_focal = row['is_focal']
                if score not in score_counts.index:
                    return 0.0
                focal_count = score_counts.loc[score, 1]
                ref_count = score_counts.loc[score, 0]
                if focal_count == 0 or ref_count == 0:
                    return 0.0
                if is_focal == 1:
                    return 1.0
                else:
                    return focal_count / ref_count

            test_df['weights'] = test_df.apply(calculate_weight, axis=1)
            test_df = test_df[test_df['weights'] > 0]
            
            if test_df.empty:
                continue

            X = sm.add_constant(test_df["is_focal"])
            y = test_df["score_diff"]
            weights = test_df["weights"]
            
            try:
                wls_model = sm.WLS(y, X, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': test_df['prompt_name']})
                z_score = wls_model.tvalues['is_focal']
                p_value = wls_model.pvalues['is_focal']
                
                results.append({
                    "Category": column,
                    "Reference Group": reference_group,
                    "Focal Group": focal_group,
                    "Z-Score (Debiased)": round(z_score, 3),
                    "Raw P-Value": p_value,
                    "QWK (Base)": round(base_qwk, 3),
                    "QWK (Debiased)": round(debiased_qwk, 3)
                })
            except Exception as e:
                pass

    if not results:
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    reject, pvals_corrected, _, _ = multipletests(results_df["Raw P-Value"], alpha=0.05, method='fdr_bh')
    results_df["P-Value (Debiased)"] = np.round(pvals_corrected, 4)
    results_df["Sig (Debiased)"] = reject
    
    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    args = parser.parse_args()

    is_persuade = args.dataset == "persuade"
    
    if is_persuade:
        preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/debiased_predictions/persuade_debiased_roberta_predictions.csv"
        base_preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/predictions/persuade_roberta_predictions.csv"
        meta_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv"
        baseline_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/results/roberta/bias_analysis/roberta_bias_results_persuade.csv"
    else:
        preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/debiased_predictions/asap_debiased_roberta_predictions.csv"
        base_preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/predictions/asap_roberta_predictions.csv"
        meta_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv"
        baseline_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/results/roberta/bias_analysis/roberta_bias_results_asap.csv"

    # 1. Run bias analysis on debiased predictions
    debiased_df = run_bias_analysis(preds_path, base_preds_path, meta_path, is_persuade)
    if debiased_df.empty:
        print("No results generated for debiased model.")
        return
        
    # 2. Load baseline results
    try:
        baseline_df = pd.read_csv(baseline_path)
    except FileNotFoundError:
        print(f"Could not find baseline results at {baseline_path}")
        return

    # 3. Merge and Compare
    baseline_df = baseline_df.rename(columns={
        "Z-Score (Bias)": "Z-Score (Base)",
        "Adjusted P-Value": "P-Value (Base)",
        "Statistically Significant": "Sig (Base)"
    })
    
    merge_cols = ["Category", "Reference Group", "Focal Group"]
    comparison_df = pd.merge(baseline_df, debiased_df, on=merge_cols, how="inner")
    
    # Reorder columns
    display_cols = [
        "Category", "Focal Group", 
        "QWK (Base)", "QWK (Debiased)",
        "Z-Score (Base)", "Z-Score (Debiased)", 
        "Sig (Base)", "Sig (Debiased)"
    ]
    
    print("\n" + "="*100)
    print(f"  BIAS & QWK COMPARISON: BASELINE vs ADVERSARIAL DEBIASING ({args.dataset.upper()})")
    print("="*100)
    print(comparison_df[display_cols].to_string(index=False))
    print("="*100)
    
    out_file = f"comparison_bias_{args.dataset}.csv"
    comparison_df.to_csv(out_file, index=False)
    print(f"Saved full comparison to {out_file}")

if __name__ == "__main__":
    main()
