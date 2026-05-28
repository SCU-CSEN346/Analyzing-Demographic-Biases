"""
Compare QWK, Z-Score (bias), and Z-variance before and after orthogonal projection.
"""

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import cohen_kappa_score
import torch
from torch.utils.data import DataLoader

# Import dependencies from existing scripts
from train_debiased_adversarial import (
    AdversarialEssayDataset,
    load_data,
    DEMO_COLUMNS,
    PERSUADE_TRAIN, PERSUADE_TEST,
    ASAP_TRAIN, ASAP_TEST,
    BATCH_SIZE, device
)
from train_evaluate_roberta import RobertaForEssayScoring


def get_predictions(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)


def compute_bias_wls(df, demo_columns, id_col="essay_id"):
    """
    Calculate Z-scores (bias) for each demographic group using Weighted Least Squares.
    Returns a dataframe of results and the variance of the Z-scores.
    """
    results = []
    
    # Required columns: prompt_name/prompt_id for clustering
    cluster_col = "prompt_name" if "prompt_name" in df.columns else "prompt_id"
    
    for column in demo_columns:
        if column not in df.columns:
            continue
            
        # Drop rows where this specific demographic is missing
        # For our datasets, missing demographics were mapped to -1 during loading,
        # but here we'll map them back to strings or use the raw dataframe if provided.
        subset = df.dropna(subset=[column, cluster_col, "score_diff"]).copy()
        # Also drop if it's the missing marker "-1" (if using mapped demographics)
        subset = subset[subset[column] != -1]
        
        groups = subset[column].unique()
        if len(groups) < 2:
            continue
            
        reference_group = groups[0]
        
        for focal_group in groups[1:]:
            group_mask = subset[column].isin([reference_group, focal_group])
            test_df = subset[group_mask].copy()
            
            if test_df.empty:
                continue
                
            test_df["is_focal"] = (test_df[column] == focal_group).astype(int)
            
            # Exact matching weights calculation
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
            
            # WLS with Cluster-Robust Standard Errors
            # Handle possible single-prompt datasets
            try:
                wls_model = sm.WLS(y, X, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': test_df[cluster_col]})
                z_score = wls_model.tvalues['is_focal']
            except Exception as e:
                # Fallback to non-clustered standard errors if clustering fails
                wls_model = sm.WLS(y, X, weights=weights).fit()
                z_score = wls_model.tvalues['is_focal']
            
            results.append({
                "Category": column,
                "Reference Group": reference_group,
                "Focal Group": focal_group,
                "Z-Score": z_score,
            })
            
    if not results:
        return pd.DataFrame(), 0.0
        
    results_df = pd.DataFrame(results)
    
    # Calculate Z-variance (variance of all computed Z-scores)
    # Using absolute z-scores or just variance of the values. 
    # Usually, we look at the variance or mean of absolute Z-scores to quantify overall bias.
    # Variance of the Z-scores:
    z_var = results_df["Z-Score"].var(ddof=0)
    
    return results_df, z_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    parser.add_argument("--demo", default="gender", choices=DEMO_COLUMNS,
                        help="The demographic attribute the model was projected on.")
    args = parser.parse_args()

    is_persuade = args.dataset == "persuade"
    train_path = PERSUADE_TRAIN if is_persuade else ASAP_TRAIN
    test_path  = PERSUADE_TEST if is_persuade else ASAP_TEST
    
    label = "PERSUADE" if is_persuade else "ASAP"
    base_model_path = f"best_roberta_{args.dataset}.pt"
    proj_model_path = f"best_roberta_{args.dataset}_projected_{args.demo}.pt"

    print("\n" + "="*80, flush=True)
    print(f"  Dataset: {label} | Projection Demo: {args.demo}", flush=True)
    print("="*80, flush=True)

    # 1. Load the original test dataset dataframe to get metadata
    if is_persuade:
        test_df = pd.read_csv(test_path, low_memory=False).dropna(subset=["holistic_essay_score", "full_text"])
        test_df = test_df.drop_duplicates(subset=["essay_id_comp"])
        id_col = "essay_id_comp"
        score_col = "holistic_essay_score"
    else:
        test_df = pd.read_csv(test_path, low_memory=False).dropna(subset=["score", "full_text"])
        test_df = test_df.drop_duplicates(subset=["essay_id"])
        id_col = "essay_id"
        score_col = "score"
        if "prompt_name" not in test_df.columns and "prompt" in test_df.columns:
            test_df["prompt_name"] = test_df["prompt"]

    test_df["human_score"] = test_df[score_col]

    # 2. Load PyTorch test dataset for inference
    (train_texts, train_scores, train_demo, 
     test_texts, test_scores, test_demo, mappings) = load_data(train_path, test_path, is_persuade)
    
    def df_to_demo(texts, demo_dict):
        df = pd.DataFrame({"text": texts})
        for col in DEMO_COLUMNS:
            df[col] = demo_dict[col]
        return {col: df[col].tolist() for col in DEMO_COLUMNS}

    test_demo_dict = df_to_demo(test_texts, test_demo)
    test_dataset  = AdversarialEssayDataset(test_texts,  test_scores,  test_demo_dict)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # 3. Base Model Inference
    print("Running Base Model Inference...", flush=True)
    base_model = RobertaForEssayScoring().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_preds = get_predictions(base_model, test_loader)
    test_df["base_pred"] = base_preds
    
    # 4. Projected Model Inference
    print("Running Projected Model Inference...", flush=True)
    proj_model = RobertaForEssayScoring().to(device)
    try:
        proj_model.load_state_dict(torch.load(proj_model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Projected model not found at {proj_model_path}")
        return

    proj_preds = get_predictions(proj_model, test_loader)
    test_df["proj_pred"] = proj_preds

    # 5. Calculate overall QWK
    all_labels_int = np.array(test_scores).astype(int)
    base_preds_rounded = np.round(base_preds).astype(int)
    proj_preds_rounded = np.round(proj_preds).astype(int)
    
    base_qwk = cohen_kappa_score(all_labels_int, base_preds_rounded, weights="quadratic")
    proj_qwk = cohen_kappa_score(all_labels_int, proj_preds_rounded, weights="quadratic")

    # 6. WLS Bias Analysis for both models
    print("\nComputing Z-Scores using WLS...", flush=True)
    
    # Calculate score diffs and standardize for BASE model
    test_df["base_raw_diff"] = test_df["base_pred"] - test_df["human_score"]
    test_df["score_diff"] = (test_df["base_raw_diff"] - test_df["base_raw_diff"].mean()) / test_df["base_raw_diff"].std()
    base_results_df, base_z_var = compute_bias_wls(test_df, DEMO_COLUMNS, id_col=id_col)
    
    # Calculate score diffs and standardize for PROJ model
    test_df["proj_raw_diff"] = test_df["proj_pred"] - test_df["human_score"]
    test_df["score_diff"] = (test_df["proj_raw_diff"] - test_df["proj_raw_diff"].mean()) / test_df["proj_raw_diff"].std()
    proj_results_df, proj_z_var = compute_bias_wls(test_df, DEMO_COLUMNS, id_col=id_col)

    # 7. Print Comparison Table
    print("\n" + "-"*80)
    print("  OVERALL PERFORMANCE")
    print("-"*80)
    print(f"  Base QWK:            {base_qwk:.4f}")
    print(f"  Post-Projection QWK: {proj_qwk:.4f}")
    print(f"  QWK Change:          {proj_qwk - base_qwk:.4f}")

    print("\n" + "-"*80)
    print("  DEMOGRAPHIC BIAS (Z-SCORES)")
    print("-"*80)
    
    # Merge results
    if not base_results_df.empty and not proj_results_df.empty:
        merged_df = pd.merge(
            base_results_df, proj_results_df, 
            on=["Category", "Reference Group", "Focal Group"], 
            suffixes=("_Base", "_Proj")
        )
        merged_df["Z-Score_Diff"] = merged_df["Z-Score_Proj"] - merged_df["Z-Score_Base"]
        merged_df["Abs_Z_Reduction"] = merged_df["Z-Score_Base"].abs() - merged_df["Z-Score_Proj"].abs()

        # Format output
        for _, row in merged_df.iterrows():
            print(f"[{row['Category']}] {row['Focal Group']} vs {row['Reference Group']}:")
            print(f"  Base Z-Score: {row['Z-Score_Base']:>7.3f}")
            print(f"  Proj Z-Score: {row['Z-Score_Proj']:>7.3f} (Abs Reduction: {row['Abs_Z_Reduction']:>7.3f})")
            print("")
            
        print("-"*80)
        print("  AGGREGATE BIAS METRICS")
        print("-"*80)
        print(f"  Base Z-Variance: {base_z_var:.4f}")
        print(f"  Proj Z-Variance: {proj_z_var:.4f}")
        print(f"  Z-Var Change:    {proj_z_var - base_z_var:.4f}")

        import os
        out_dir = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/results/roberta"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"orthogonal_bias_{args.dataset}_{args.demo}.csv")
        
        # Add summary stats to dataframe for completeness
        merged_df["Base_QWK"] = base_qwk
        merged_df["Proj_QWK"] = proj_qwk
        merged_df["Base_Z_Var"] = base_z_var
        merged_df["Proj_Z_Var"] = proj_z_var
        
        merged_df.to_csv(out_file, index=False)
        print(f"\nSaved CSV results to {out_file}")

    else:
        print("Could not compute Z-Scores (insufficient matched data).")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
