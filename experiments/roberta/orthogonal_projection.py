"""
Post-hoc orthogonal projection debiasing for RoBERTa AES.
Loads a trained base model, extracts CLS hidden states per demographic group,
computes the bias direction, projects it out of the regression weights,
and re-evaluates. Saves the projected model as a new checkpoint.

Usage:
  python orthogonal_projection.py --dataset persuade --demo gender
  python orthogonal_projection.py --dataset asap --demo race_ethnicity
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score

# Import utilities from adversarial script
from train_debiased_adversarial import (
    AdversarialEssayDataset,
    load_data,
    DEMO_COLUMNS,
    PERSUADE_TRAIN, PERSUADE_TEST,
    ASAP_TRAIN, ASAP_TEST,
    BATCH_SIZE, device, tokenizer
)

# Import model architecture
from train_evaluate_roberta import RobertaForEssayScoring, evaluate


def extract_hidden_states(model, loader, demo_attr):
    """
    Run the model over the dataset and collect CLS embeddings,
    demographic labels for the target attribute, and essay scores.
    """
    model.eval()
    all_hidden, all_demo, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # RoBERTa CLS token is at index 0
            cls = outputs.last_hidden_state[:, 0, :]
            
            all_hidden.append(cls.cpu().numpy())
            all_demo.append(batch[demo_attr].numpy())
            all_labels.append(batch["labels"].numpy())

    return (
        np.concatenate(all_hidden),
        np.concatenate(all_demo),
        np.concatenate(all_labels)
    )


def compute_bias_direction(hidden_states, demo_labels):
    """
    Estimate the demographic bias direction by finding the principal
    component of the group mean embeddings.
    Ignores missing demographics (-1).
    """
    valid_mask = demo_labels != -1
    valid_hidden = hidden_states[valid_mask]
    valid_labels = demo_labels[valid_mask]

    unique_groups = np.unique(valid_labels)
    if len(unique_groups) < 2:
        raise ValueError("Not enough demographic groups found to compute bias direction.")

    # Calculate the mean embedding for each demographic group
    group_means = []
    for group_idx in unique_groups:
        group_emb = valid_hidden[valid_labels == group_idx]
        group_means.append(group_emb.mean(axis=0))

    group_means = np.stack(group_means)
    
    # Run PCA to find the principal component of variation across group means
    pca = PCA(n_components=1)
    pca.fit(group_means)
    return pca.components_[0]


def project_out(weight, bias_direction):
    """
    Remove the component of the regression weights that lies along
    the demographic bias direction using orthogonal projection.
    """
    bias_dir = torch.tensor(bias_direction, dtype=weight.dtype, device=weight.device)
    bias_dir = bias_dir / bias_dir.norm()
    proj = weight - (weight @ bias_dir).unsqueeze(1) * bias_dir.unsqueeze(0)
    return proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    parser.add_argument("--demo", default="gender", choices=DEMO_COLUMNS,
                        help="Target demographic attribute to debias.")
    args = parser.parse_args()

    is_persuade = args.dataset == "persuade"
    train_path = PERSUADE_TRAIN if is_persuade else ASAP_TRAIN
    test_path  = PERSUADE_TEST if is_persuade else ASAP_TEST
    
    label = "PERSUADE" if is_persuade else "ASAP"
    model_path = f"best_roberta_{args.dataset}.pt"

    print("\n" + "="*60, flush=True)
    print(f"  Dataset: {label} (Orthogonal Projection: {args.demo})", flush=True)
    print("="*60, flush=True)

    # Load data
    (train_texts, train_scores, train_demo, 
     test_texts, test_scores, test_demo, mappings) = load_data(train_path, test_path, is_persuade)
     
    print(f"  Demo encoding ({args.demo}): {mappings.get(args.demo, {})}", flush=True)

    # Convert to DataFrame to match dataset class requirements
    def df_to_demo(texts, demo_dict):
        df = pd.DataFrame({"text": texts})
        for col in DEMO_COLUMNS:
            df[col] = demo_dict[col]
        return {col: df[col].tolist() for col in DEMO_COLUMNS}

    train_demo_dict = df_to_demo(train_texts, train_demo)
    test_demo_dict = df_to_demo(test_texts, test_demo)

    train_dataset = AdversarialEssayDataset(train_texts, train_scores, train_demo_dict)
    test_dataset  = AdversarialEssayDataset(test_texts,  test_scores,  test_demo_dict)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = RobertaForEssayScoring().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"\n--- {label} Before Projection ---", flush=True)
    evaluate(model, test_loader, verbose=True)

    print(f"\nExtracting hidden states on train set...", flush=True)
    hidden, demo_labels, _ = extract_hidden_states(model, train_loader, args.demo)

    print(f"Computing bias direction for {args.demo}...", flush=True)
    bias_dir = compute_bias_direction(hidden, demo_labels)

    print(f"Projecting regression weights...", flush=True)
    with torch.no_grad():
        model.regressor.weight.data = project_out(
            model.regressor.weight.data, bias_dir
        )

    print(f"\n--- {label} After Projection ---", flush=True)
    evaluate(model, test_loader, verbose=True)

    out_path = f"best_roberta_{args.dataset}_projected_{args.demo}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved projected model to {out_path}", flush=True)


if __name__ == "__main__":
    main()
