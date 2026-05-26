# per_prompt_bias_analysis.py
# Calculates bias metrics on a per-prompt basis and aggregates Z magnitude and variance.

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm

DEMO_COLUMNS = [
    "gender", 
    "race_ethnicity", 
    "ell_status", 
    "economically_disadvantaged", 
    "student_disability_status",
    "economic_disadvantage" # Fallback for old persuade
]

def run_per_prompt_analysis(preds_path, base_preds_path, meta_path, is_persuade):
    print(f"Loading predictions from {preds_path}...")
    preds_df = pd.read_csv(preds_path)
    base_preds_df = pd.read_csv(base_preds_path)
    meta_df = pd.read_csv(meta_path, low_memory=False)
    
    id_col = "essay_id_comp" if is_persuade else "essay_id"
    df = pd.merge(preds_df, meta_df, left_on="essay_id", right_on=id_col)
    base_df = pd.merge(base_preds_df, meta_df, left_on="essay_id", right_on=id_col)
    
    df["raw_score_diff"] = df["roberta_score"] - df["human_score"]
    base_df["raw_score_diff"] = base_df["roberta_score"] - base_df["human_score"]
    
    prompts = df["prompt_name"].dropna().unique()
    results = []
    
    for prompt in prompts:
        prompt_df = df[df["prompt_name"] == prompt].copy()
        prompt_base_df = base_df[base_df["prompt_name"] == prompt].copy()
        
        # Standardize within the prompt
        prompt_df["score_diff"] = (prompt_df["raw_score_diff"] - prompt_df["raw_score_diff"].mean()) / (prompt_df["raw_score_diff"].std() + 1e-9)
        prompt_base_df["score_diff"] = (prompt_base_df["raw_score_diff"] - prompt_base_df["raw_score_diff"].mean()) / (prompt_base_df["raw_score_diff"].std() + 1e-9)
        
        for column in DEMO_COLUMNS:
            if column not in prompt_df.columns:
                continue
                
            subset = prompt_df.dropna(subset=[column, "score_diff"]).copy()
            subset_base = prompt_base_df.dropna(subset=[column, "score_diff"]).copy()
            
            groups = subset[column].unique()
            if len(groups) < 2:
                continue
                
            reference_group = groups[0]
            
            for focal_group in groups[1:]:
                # ----- Debiased WLS -----
                group_mask = subset[column].isin([reference_group, focal_group])
                test_df = subset[group_mask].copy()
                
                z_debiased = np.nan
                if not test_df.empty:
                    test_df["is_focal"] = (test_df[column] == focal_group).astype(int)
                    score_counts = test_df.groupby(["human_score", "is_focal"]).size().unstack(fill_value=0)
                    
                    def calc_weight(row, sc):
                        s = row['human_score']
                        foc = row['is_focal']
                        if s not in sc.index: return 0.0
                        fc = sc.loc[s, 1]
                        rc = sc.loc[s, 0]
                        if fc == 0 or rc == 0: return 0.0
                        return 1.0 if foc == 1 else fc / rc

                    test_df['weights'] = test_df.apply(lambda r: calc_weight(r, score_counts), axis=1)
                    test_df = test_df[test_df['weights'] > 0]
                    
                    if not test_df.empty and len(test_df["is_focal"].unique()) > 1:
                        X = sm.add_constant(test_df["is_focal"])
                        try:
                            wls = sm.WLS(test_df["score_diff"], X, weights=test_df["weights"]).fit(cov_type='HC3')
                            z_debiased = wls.tvalues['is_focal']
                        except: pass

                # ----- Baseline WLS -----
                group_mask_base = subset_base[column].isin([reference_group, focal_group])
                test_base_df = subset_base[group_mask_base].copy()
                
                z_base = np.nan
                if not test_base_df.empty:
                    test_base_df["is_focal"] = (test_base_df[column] == focal_group).astype(int)
                    score_counts_base = test_base_df.groupby(["human_score", "is_focal"]).size().unstack(fill_value=0)
                    
                    test_base_df['weights'] = test_base_df.apply(lambda r: calc_weight(r, score_counts_base), axis=1)
                    test_base_df = test_base_df[test_base_df['weights'] > 0]
                    
                    if not test_base_df.empty and len(test_base_df["is_focal"].unique()) > 1:
                        X_base = sm.add_constant(test_base_df["is_focal"])
                        try:
                            wls_base = sm.WLS(test_base_df["score_diff"], X_base, weights=test_base_df["weights"]).fit(cov_type='HC3')
                            z_base = wls_base.tvalues['is_focal']
                        except: pass
                
                if not np.isnan(z_base) or not np.isnan(z_debiased):
                    results.append({
                        "Prompt": prompt,
                        "Category": column,
                        "Reference Group": reference_group,
                        "Focal Group": focal_group,
                        "Z-Score (Base)": z_base,
                        "Z-Score (Debiased)": z_debiased
                    })
                    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    args = parser.parse_args()

    is_persuade = args.dataset == "persuade"
    
    if is_persuade:
        preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/debiased_predictions/persuade_debiased_roberta_predictions.csv"
        base_preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/predictions/persuade_roberta_predictions.csv"
        meta_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv"
    else:
        preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/debiased_predictions/asap_debiased_roberta_predictions.csv"
        base_preds_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/predictions/asap_roberta_predictions.csv"
        meta_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv"

    print(f"--- Running Per-Prompt Bias Analysis on {args.dataset.upper()} ---")
    
    raw_df = run_per_prompt_analysis(preds_path, base_preds_path, meta_path, is_persuade)
    if raw_df.empty:
        print("No valid per-prompt results generated.")
        return
        
    raw_out = f"per_prompt_raw_zscores_{args.dataset}.csv"
    raw_df.to_csv(raw_out, index=False)
    print(f"Saved raw per-prompt Z-scores to {raw_out}")
    
    # Calculate Magnitude |Z|
    raw_df["Abs Z (Base)"] = raw_df["Z-Score (Base)"].abs()
    raw_df["Abs Z (Debiased)"] = raw_df["Z-Score (Debiased)"].abs()
    
    # Aggregate
    agg_df = raw_df.groupby(["Category", "Reference Group", "Focal Group"]).agg(
        Num_Prompts=("Prompt", "count"),
        Mean_Abs_Z_Base=("Abs Z (Base)", "mean"),
        Var_Z_Base=("Z-Score (Base)", "var"),
        Mean_Abs_Z_Debiased=("Abs Z (Debiased)", "mean"),
        Var_Z_Debiased=("Z-Score (Debiased)", "var")
    ).reset_index()
    
    # Round for display
    for col in ["Mean_Abs_Z_Base", "Var_Z_Base", "Mean_Abs_Z_Debiased", "Var_Z_Debiased"]:
        agg_df[col] = agg_df[col].round(3)
        
    agg_out = f"per_prompt_aggregated_bias_{args.dataset}.csv"
    agg_df.to_csv(agg_out, index=False)
    
    print("\n" + "="*100)
    print(f"  PER-PROMPT Z-SCORE MAGNITUDE & VARIANCE ({args.dataset.upper()})")
    print("="*100)
    
    display_cols = ["Category", "Focal Group", "Num_Prompts", "Mean_Abs_Z_Base", "Mean_Abs_Z_Debiased", "Var_Z_Base", "Var_Z_Debiased"]
    print(agg_df[display_cols].to_string(index=False))
    print("="*100)
    print(f"Saved aggregated per-prompt statistics to {agg_out}")

if __name__ == "__main__":
    main()
