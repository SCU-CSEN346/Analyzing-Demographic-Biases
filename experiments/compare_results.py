"""
compare_results.py
Compares our XLNet replication results against Kwako & Ormerod (BEA 2024) Table 2.
Outputs a side-by-side table of QWK, Exact Acc, and gender SMD per prompt.
"""

import json
import numpy as np
import pandas as pd

RESULTS_PATH = "/home/rl/CSEN364/PROJECT/Analyzing-Demographic-Biases/results/xlnet_replication/results.json"

# ── Kwako & Ormerod (BEA 2024) Table 2 reference numbers ──────────────────
# Prompt numbers map to prompt names in PERSUADE order
REFERENCE = {
    "Phones and driving":                      {"qwk": 0.781, "smd": -0.066, "acc": 0.683, "n": 464},
    "Exploring Venus":                         {"qwk": 0.856, "smd":  0.003, "acc": 0.677, "n": 923},
    "Community service":                       {"qwk": 0.800, "smd": -0.109, "acc": 0.693, "n": 773},
    "Seeking multiple opinions":               {"qwk": 0.674, "smd": -0.312, "acc": 0.429, "n": 7},
    "Facial action coding system":             {"qwk": 0.865, "smd": -0.116, "acc": 0.696, "n": 1062},
    "Distance learning":                       {"qwk": 0.875, "smd":  0.042, "acc": 0.697, "n": 656},
    "Summer projects":                         {"qwk": 0.813, "smd": -0.051, "acc": 0.634, "n": 872},
    "Cell phones at school":                   {"qwk": 0.800, "smd": -0.021, "acc": 0.717, "n": 824},
    "Car-free cities":                         {"qwk": 0.796, "smd": -0.087, "acc": 0.616, "n": 973},
    "Grades for extracurricular activities":   {"qwk": 0.779, "smd": -0.025, "acc": 0.699, "n": 808},
    "The Face on Mars":                        {"qwk": 0.818, "smd":  0.063, "acc": 0.658, "n": 764},
    "Does the electoral college work?":        {"qwk": 0.863, "smd": -0.011, "acc": 0.649, "n": 228},
    "Driverless cars":                         {"qwk": 0.774, "smd":  0.215, "acc": 0.621, "n": 496},
    "Mandatory extracurricular activities":    {"qwk": 0.815, "smd":  0.163, "acc": 0.659, "n": 824},
    '"A Cowboy Who Rode the Waves"':           {"qwk": 0.755, "smd": -0.040, "acc": 0.691, "n": 682},
}

REFERENCE_OVERALL = {"qwk": 0.864, "smd": -0.010, "acc": 0.672, "n": 10356}

# ── Load our results ───────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    our_results = json.load(f)

rows = []
for r in our_results:
    prompt = r["prompt"]
    ref    = REFERENCE.get(prompt)
    our_qwk  = r.get("qwk")
    our_acc  = r.get("exact_acc")
    our_gender_smd = r.get("bias", {}).get("gender", {}).get("model_smd")
    our_n    = r.get("n_test")

    rows.append({
        "Prompt":          prompt,
        "Ours QWK":        our_qwk,
        "Ref QWK":         ref["qwk"] if ref else None,
        "QWK Δ":           round(our_qwk - ref["qwk"], 4) if (our_qwk and ref) else None,
        "Ours Acc":        our_acc,
        "Ref Acc":         ref["acc"] if ref else None,
        "Acc Δ":           round(our_acc - ref["acc"], 4) if (our_acc and ref) else None,
        "Ours Gender SMD": our_gender_smd,
        "Ref Gender SMD":  ref["smd"] if ref else None,
        "N (ours)":        our_n,
        "N (ref)":         ref["n"] if ref else None,
    })

df = pd.DataFrame(rows)

# ── Overall row ────────────────────────────────────────────────────────────
our_qwks = [r["qwk"] for r in our_results if r.get("qwk")]
our_accs = [r["exact_acc"] for r in our_results if r.get("exact_acc")]
our_gender_smds = [r["bias"]["gender"]["model_smd"]
                   for r in our_results
                   if r.get("bias", {}).get("gender", {}).get("model_smd") is not None]

overall_row = {
    "Prompt":          "OVERALL (macro avg)",
    "Ours QWK":        round(np.mean(our_qwks), 4),
    "Ref QWK":         REFERENCE_OVERALL["qwk"],
    "QWK Δ":           round(np.mean(our_qwks) - REFERENCE_OVERALL["qwk"], 4),
    "Ours Acc":        round(np.mean(our_accs), 4),
    "Ref Acc":         REFERENCE_OVERALL["acc"],
    "Acc Δ":           round(np.mean(our_accs) - REFERENCE_OVERALL["acc"], 4),
    "Ours Gender SMD": round(np.mean(our_gender_smds), 4) if our_gender_smds else None,
    "Ref Gender SMD":  REFERENCE_OVERALL["smd"],
    "N (ours)":        sum(r.get("n_test") or 0 for r in our_results),
    "N (ref)":         REFERENCE_OVERALL["n"],
}
df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

# ── Print ──────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")

print("\n" + "="*120)
print("REPLICATION COMPARISON: Ours vs Kwako & Ormerod (BEA 2024)")
print("="*120)
print(df.to_string(index=False))

# ── Summary stats ──────────────────────────────────────────────────────────
qwk_deltas = df["QWK Δ"].dropna()
acc_deltas  = df["Acc Δ"].dropna()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  QWK  — Mean Δ: {qwk_deltas.mean():+.4f}  "
      f"Min Δ: {qwk_deltas.min():+.4f}  Max Δ: {qwk_deltas.max():+.4f}")
print(f"  Acc  — Mean Δ: {acc_deltas.mean():+.4f}  "
      f"Min Δ: {acc_deltas.min():+.4f}  Max Δ: {acc_deltas.max():+.4f}")

# Save CSV
out_path = "/home/rl/CSEN364/PROJECT/Analyzing-Demographic-Biases/results/xlnet_replication/comparison_vs_paper.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")