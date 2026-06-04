"""
probe_projected_states.py
Computes the RIGOROUS held-out "after" probe kappa on PROJECTED hidden states.

Pairs with:
  - orthogonal_projection.py : dumps projected_states_<attr>.npz (full projected
                               train + test hidden states + demographics + ids)
  - probe_hidden_states.py   : the rigorous probe. We IMPORT its train_probe,
                               evaluate_probe, safe_stratified_split, build_tasks
                               and constants, so the methodology here is byte-for-
                               byte identical to the "before" column. The ONLY
                               difference is the input: cached projected states
                               instead of freshly extracted raw states.

This closes the §6.3 loop: the "before" kappa comes from probe_hidden_states.py
on raw baseline states; the "after" kappa comes from THIS script on the projected
states. Both use one probe methodology, on held-out test, with the empty-string
+ NaN-aggregation + dev-leak fixes already in probe_hidden_states.py.

Usage:
    python probe_projected_states.py \
        --npz ../../results/xlnet/projection_baseline/projected_states_gender.npz \
        --out ../../results/xlnet/projection_baseline/after_probe_gender.json

    # all attributes in a dir:
    for f in .../projected_states_*.npz; do
        python probe_projected_states.py --npz "$f" --out "${f%.npz}_afterprobe.json"
    done

Output JSON mirrors the per-attribute "after" kappa, per prompt + aggregate, so
it drops straight into the §6.3 table alongside the recomputed "before".
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

# Import the rigorous probe's EXACT routines + constants. Importing the module
# is side-effect-safe (main() is __main__-guarded; it only makes an OUT_DIR and
# prints a device line at import).
from probe_hidden_states import (
    train_probe, evaluate_probe, safe_stratified_split, build_tasks,
    BINARY_COLS, RANDOM_SEED,
)

warnings.filterwarnings("ignore")


def load_npz(npz_path):
    """Reconstruct nested {split: {prompt: {field: array}}} from the flat npz."""
    z = np.load(npz_path, allow_pickle=True)
    attr = str(z["__attr__"])
    score_col = str(z["__score_col__"])
    demo_cols = list(z["__demo_cols__"])
    prompts = list(z["__prompts__"])

    data = {"train": {}, "test": {}}
    for key in z.files:
        if key.startswith("__"):
            continue
        split, safe_prompt, field = key.split("__", 2)
        data[split].setdefault(safe_prompt, {})[field] = z[key]
    return data, attr, score_col, demo_cols, prompts


def rows_to_df(prompt_dict, demo_cols, score_col):
    """Rebuild a per-prompt dataframe of demographics so build_tasks works
    exactly as in probe_hidden_states.py (same masking + label encoding)."""
    n = prompt_dict["h"].shape[0]
    df = pd.DataFrame(index=range(n))
    for c in demo_cols:
        if c in prompt_dict:
            # Re-apply the SAME cleaning as probe_hidden_states.py main():
            # lower/strip already done at dump time via astype(str); map all
            # null-like tokens to NA so empty strings don't form a 3rd class.
            s = pd.Series(prompt_dict[c]).astype(str).str.lower().str.strip()
            df[c] = s.replace(["nan", "", "none", "<na>"], pd.NA)
    df[score_col] = prompt_dict[score_col]
    return df


def main(npz_path, out_path):
    data, attr, score_col, demo_cols, prompts = load_npz(npz_path)
    print(f"[after-probe] attr={attr}  prompts={len(prompts)}  npz={npz_path}")

    all_results = []
    for safe_prompt in sorted(set(data["train"]) & set(data["test"])):
        tr_d = data["train"][safe_prompt]
        te_d = data["test"][safe_prompt]

        h_train = tr_d["h"]
        h_test = te_d["h"]
        tr_df = rows_to_df(tr_d, demo_cols, score_col)
        te_df = rows_to_df(te_d, demo_cols, score_col)

        if len(tr_df) < 20 or len(te_df) < 5:
            continue

        # build_tasks emits (task_name, label_fn) for every demographic column.
        # We keep only the projected attribute's task (the others weren't
        # projected out of these states, so their "after" kappa is meaningless).
        tasks = build_tasks(tr_df, te_df)
        tasks = [(name, fn) for (name, fn) in tasks if name == attr]
        if not tasks:
            continue

        prompt_results = {"prompt": safe_prompt,
                          "n_train": int(len(tr_df)),
                          "n_test": int(len(te_df)),
                          "demographics": {}}

        for task_name, label_fn in tasks:
            tr_mask, y_tr, classes = label_fn(tr_df)
            te_mask, y_te, _ = label_fn(te_df)
            n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
            if n_tr < 20 or n_te < 5:
                continue
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            h_tr_valid = h_train[tr_mask]
            h_te_valid = h_test[te_mask]
            num_classes = int(max(y_tr.max(), y_te.max()) + 1)

            # Identical dev-split logic to probe_hidden_states.py (no test leak).
            h_tr_split, h_dev_split, y_tr_split, y_dev_split = safe_stratified_split(
                h_tr_valid, y_tr, test_size=0.1, seed=RANDOM_SEED
            )
            no_early_stop = False
            if len(np.unique(y_dev_split)) < 2:
                h_tr_split, h_dev_split, y_tr_split, y_dev_split = safe_stratified_split(
                    h_tr_valid, y_tr, test_size=0.2, seed=RANDOM_SEED
                )
            if len(np.unique(y_dev_split)) < 2:
                h_tr_split, y_tr_split = h_tr_valid, y_tr
                h_dev_split, y_dev_split = h_tr_valid, y_tr
                no_early_stop = True

            if no_early_stop:
                probe, _ = train_probe(h_tr_split, y_tr_split,
                                       h_dev_split, y_dev_split, num_classes)
                dev_kappa = float("nan")
            else:
                probe, dev_kappa = train_probe(h_tr_split, y_tr_split,
                                               h_dev_split, y_dev_split, num_classes)
            test_kappa = evaluate_probe(probe, h_te_valid, y_te)

            print(f"  [{safe_prompt:34s}] after_kappa={test_kappa:+.3f} "
                  f"(n_tr={n_tr} n_te={n_te})")
            prompt_results["demographics"][task_name] = {
                "test_kappa": round(test_kappa, 4),
                "dev_kappa": (round(dev_kappa, 4) if not np.isnan(dev_kappa) else None),
                "n_classes": num_classes,
                "n_train": n_tr, "n_test": n_te, "classes": classes,
            }
        all_results.append(prompt_results)

    # Aggregate (NaN-safe, same as fixed probe_hidden_states.py)
    kappas_raw = [r["demographics"][attr]["test_kappa"]
                  for r in all_results if attr in r["demographics"]]
    kappas = [k for k in kappas_raw if k is not None and not np.isnan(k)]
    summary = {
        "attribute": attr,
        "mean_after_kappa": round(float(np.mean(kappas)), 4) if kappas else None,
        "median_after_kappa": round(float(np.median(kappas)), 4) if kappas else None,
        "min_after_kappa": round(float(np.min(kappas)), 4) if kappas else None,
        "max_after_kappa": round(float(np.max(kappas)), 4) if kappas else None,
        "n_prompts": len(kappas),
        "n_dropped_nan": len(kappas_raw) - len(kappas),
        "per_prompt": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[after-probe] {attr}: mean held-out after-kappa = "
          f"{summary['mean_after_kappa']}  (n={summary['n_prompts']})")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="projected_states_<attr>.npz from orthogonal_projection.py")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()
    main(args.npz, args.out)