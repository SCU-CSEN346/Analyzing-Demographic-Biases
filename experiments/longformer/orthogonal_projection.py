"""
Post-hoc orthogonal projection debiasing for Longformer AES.
Loads a trained .pt model, extracts hidden states per demographic group,
computes the bias direction via PCA, projects it out of the regression weights,
and re-evaluates.

Usage:
  python orthogonal_projection.py --dataset persuade --demo gender
  python orthogonal_projection.py --dataset asap --demo gender
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score

from longformer import (
    LongformerForEssayScoring, EssayDataset,
    load_persuade, load_asap,
    compute_smd, evaluate,
    PERSUADE_TRAIN, PERSUADE_TEST,
    ASAP_TRAIN, ASAP_TEST,
    DEMO_COLS, BATCH_SIZE, device
)


def extract_hidden_states(model, loader):
    model.eval()
    all_hidden, all_demo, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            outputs = model.longformer(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                global_attention_mask=batch["global_attention_mask"].to(device)
            )
            cls = outputs.last_hidden_state[:, 0, :]
            all_hidden.append(cls.cpu().numpy())
            all_demo.append(batch["demo_labels"].numpy())
            all_labels.append(batch["labels"].numpy())

    return (
        np.concatenate(all_hidden),
        np.concatenate(all_demo),
        np.concatenate(all_labels)
    )


def compute_bias_direction(hidden_states, demo_labels):
    group0 = hidden_states[demo_labels == 0]
    group1 = hidden_states[demo_labels == 1]
    diff   = group1.mean(axis=0) - group0.mean(axis=0)
    # PCA on the difference to get the principal bias direction
    pca    = PCA(n_components=1)
    pca.fit(np.stack([group0.mean(axis=0), group1.mean(axis=0)]))
    return pca.components_[0]


def project_out(weight, bias_direction):
    bias_dir = torch.tensor(bias_direction, dtype=weight.dtype, device=weight.device)
    bias_dir = bias_dir / bias_dir.norm()
    # Remove component along bias direction from each row
    proj = weight - (weight @ bias_dir).unsqueeze(1) * bias_dir.unsqueeze(0)
    return proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    parser.add_argument("--demo",    default="gender",
                        choices=["gender", "race", "ell", "ses", "disability"])
    args = parser.parse_args()

    if args.dataset == "persuade":
        _, _, train_texts, train_scores, \
        test_texts, test_scores, train_demo, test_demo = load_persuade(
            PERSUADE_TRAIN, PERSUADE_TEST, demo_col=args.demo
        )
        model_path = "best_longformer_persuade.pt"
        label      = "PERSUADE"
    else:
        _, _, train_texts, train_scores, \
        test_texts, test_scores, train_demo, test_demo = load_asap(
            ASAP_TRAIN, ASAP_TEST, demo_col=args.demo
        )
        model_path = "best_longformer_asap.pt"
        label      = "ASAP"

    model = LongformerForEssayScoring(debias=False).to(device)
    model.load_state_dict(torch.load(model_path))

    # Use train set to extract hidden states and compute bias direction
    train_dataset = EssayDataset(train_texts, train_scores, train_demo)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset  = EssayDataset(test_texts,  test_scores,  test_demo)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n--- {label} Before Projection ---", flush=True)
    evaluate(model, test_loader, verbose=True)

    print(f"\nExtracting hidden states...", flush=True)
    hidden, demo_labels, _ = extract_hidden_states(model, train_loader)

    print(f"Computing bias direction...", flush=True)
    bias_dir = compute_bias_direction(hidden, demo_labels)

    print(f"Projecting regression weights...", flush=True)
    with torch.no_grad():
        model.regressor.weight.data = project_out(
            model.regressor.weight.data, bias_dir
        )

    print(f"\n--- {label} After Projection ---", flush=True)
    evaluate(model, test_loader, verbose=True)

    out_path = model_path.replace(".pt", "_projected.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved projected model to {out_path}", flush=True)


if __name__ == "__main__":
    main()