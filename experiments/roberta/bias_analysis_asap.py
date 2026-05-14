import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------
# 1. Load, Merge, and Standardize Data
# ---------------------------------------------------------
print("Loading data...")
preds_df = pd.read_csv("/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/experiments/roberta/predictions/asap_roberta_predictions.csv")
meta_df = pd.read_csv("/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv", low_memory=False)

# Merge the predictions with the demographic metadata
df = pd.merge(preds_df, meta_df, on="essay_id")

# Calculate the Raw Error (Model Score - Human Score)
df["raw_score_diff"] = df["roberta_score"] - df["score"]

# Convert the raw differences into standardized Z-Scores
mean_diff = df["raw_score_diff"].mean()
std_diff = df["raw_score_diff"].std()
# Standardize score_diff WITHIN each prompt/set to account for different ASAP scales
df["score_diff"] = df.groupby("set")["raw_score_diff"].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
)
# ---------------------------------------------------------
# 2. Propensity Weighting (Neutralizing Baselines)
# ---------------------------------------------------------
# For this replication, we assign sample weights to neutralize 
# varying lengths/baselines. (A simplified inverse probability weight)
# We will use a standard weight of 1.0 for the baseline calculation.
# Define the demographic groups we want to test
# Ensure these match the exact column names in your PERSUADE CSV
demographic_columns = [
    "gender", 
    "race_ethnicity", 
    "ell_status", 
    "economically_disadvantaged", # Updated name
    "student_disability_status"
]

# ---------------------------------------------------------
# 3. WLS Regression & Cluster-Robust Errors
# ---------------------------------------------------------
results = []

print("Running WLS Regressions...")
for column in demographic_columns:
    if column not in df.columns:
        continue
        
    # Drop rows where this specific demographic is missing
    subset = df.dropna(subset=[column, "prompt_name", "score_diff"]).copy()
    
    # Get unique groups (e.g., Male, Female)
    groups = subset[column].unique()
    if len(groups) < 2:
        continue
        
    # Compare each group against a reference group (e.g., Female vs Male)
    reference_group = groups[0]
    
    for focal_group in groups[1:]:
        # Create a dummy variable: 1 if focal group, 0 if reference
        group_mask = subset[column].isin([reference_group, focal_group])
        test_df = subset[group_mask].copy()
        
        # Safety Check: Skip if the dataframe is empty after filtering
        if test_df.empty:
            print(f"Skipping {focal_group} vs {reference_group} (No data)")
            continue
            
        test_df["is_focal"] = (test_df[column] == focal_group).astype(int)
        
        # --- EXACT MATCHING WEIGHTS CALCULATION ---
        # Count how many students from each group received each specific human score
        score_counts = test_df.groupby(["human_score", "is_focal"]).size().unstack(fill_value=0)
        
        def calculate_weight(row):
            score = row['score']
            is_focal = row['is_focal']
            
            # If the score doesn't exist in our counts, skip it
            if score not in score_counts.index:
                return 0.0
            
            focal_count = score_counts.loc[score, 1]
            ref_count = score_counts.loc[score, 0]
            
            # If a score tier is missing one of the groups, we cannot exact match.
            if focal_count == 0 or ref_count == 0:
                return 0.0
            
            # Focal group members get a baseline weight of 1.0
            if is_focal == 1:
                return 1.0
            # Reference group is mathematically weighted to mirror the focal group's distribution
            else:
                return focal_count / ref_count

        # Apply the exact matching weights
        test_df['weights'] = test_df.apply(calculate_weight, axis=1)
        
        # Filter out any rows that couldn't be matched (weight = 0)
        test_df = test_df[test_df['weights'] > 0]
        
        # Second Safety Check: Ensure we still have data after exact matching
        if test_df.empty:
            print(f"Skipping {focal_group} vs {reference_group} (No exact matches found)")
            continue
        # ------------------------------------------

        # Set up the WLS Regression
        X = sm.add_constant(test_df["is_focal"])
        y = test_df["score_diff"]
        weights = test_df["weights"]
        
        # Run WLS and apply Cluster-Robust Standard Errors (clustered by essay prompt)
        wls_model = sm.WLS(y, X, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': test_df['prompt_name']})
        
        # Extract the Z-Score and p-value for the 'is_focal' variable
        z_score = wls_model.tvalues['is_focal']
        p_value = wls_model.pvalues['is_focal']
        
        results.append({
            "Category": column,
            "Reference Group": reference_group,
            "Focal Group": focal_group,
            "Z-Score (Bias)": round(z_score, 3),
            "Raw P-Value": p_value
        })

# ---------------------------------------------------------
# 4. Benjamini-Hochberg (B-H) Adjustment
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

# Apply B-H correction to control the False Discovery Rate across all tests
reject, pvals_corrected, _, _ = multipletests(results_df["Raw P-Value"], alpha=0.05, method='fdr_bh')

results_df["Adjusted P-Value"] = np.round(pvals_corrected, 4)
results_df["Statistically Significant"] = reject

# ---------------------------------------------------------
# 5. Output and Save Final Table
# ---------------------------------------------------------
print("\n" + "="*80)
print("  PHASE 3: DEMOGRAPHIC FAIRNESS ANALYSIS (RoBERTa)")
print("="*80)

# Define the final columns for clarity
output_columns = [
    "Category", 
    "Reference Group",
    "Focal Group", 
    "Z-Score (Bias)", 
    "Adjusted P-Value", 
    "Statistically Significant"
]

# Print to console
print(results_df[output_columns].to_string(index=False))
print("="*80)

# Save the results to a CSV file
output_path = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/results/roberta/roberta_bias_results_asap.csv"

# We use index=False so pandas doesn't write the row numbers into the file
results_df.to_csv(output_path, index=False)

print(f"\nSuccess! Results saved locally to: {output_path}")