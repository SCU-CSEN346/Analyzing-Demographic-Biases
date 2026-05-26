"""
Demographic probing for Longformer AES models.
Pre-extracts CLS embeddings once per model checkpoint, then trains a
linear classification head on those embeddings to predict demographic
attributes. Reports Cohen's kappa per demographic.

Pre-extracting features means we only run Longformer once per checkpoint
instead of once per epoch, making probing ~50x faster.

Usage:
  python probe.py --dataset persuade --model pt/best_longformer_persuade_base.pt
  python probe.py --dataset asap     --model pt/best_longformer_asap_base.pt
  python probe.py --dataset persuade --model pt/best_longformer_persuade_grl.pt
  python probe.py --dataset persuade --model pt/best_longformer_persuade_projected_base.pt
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

from transformers import LongformerModel, LongformerTokenizer
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME    = "allenai/longformer-base-4096"
MAX_LENGTH    = 1024
BATCH_SIZE    = 4
EXTRACT_BATCH = 16          # larger batch during feature extraction (no grad needed)
MAX_EPOCHS    = 50
PATIENCE      = 5
LEARNING_RATE = 1e-3        # standard linear probing LR; 5e-6 (paper's LR) didn't converge
RANDOM_STATE  = 42

PERSUADE_TRAIN = "PERSUADE/persuade_corpus_2.0_train.csv"
PERSUADE_TEST  = "PERSUADE/persuade_corpus_2.0_test.csv"
ASAP_TRAIN     = "ASAP/ASAP_2_Final_github_train.csv"
ASAP_TEST      = "ASAP/ASAP_2_Final_github_test.csv"

PROBE_COLS = {
    "persuade": {
        "gender":     "gender",
        "race":       "race_ethnicity",
        "ell":        "ell_status",
        "ses":        "economically_disadvantaged",
        "disability": "student_disability_status"
    },
    "asap": {
        "gender":     "gender",
        "race":       "race_ethnicity",
        "ell":        "ell_status",
        "ses":        "economically_disadvantaged",
        "disability": "student_disability_status"
    }
}

# Race is probed as one-vs-rest binary to match Table 4 in Kwako & Ormerod (2024)
RACE_BINARY_GROUPS = [
    "White",
    "Hispanic/Latino",
    "Black/African American",
    "Asian/Pacific Islander"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)


# ------------------------------------------------------------------
# Tokenization with caching
# ------------------------------------------------------------------

def tokenize_and_cache(df, cache_path):
    # Tokenization is slow, so we cache the result to disk.
    # All probe runs on the same dataset reuse the same cache file.
    if os.path.exists(cache_path):
        print(f"  Loading cached tokens from {cache_path}", flush=True)
        return torch.load(cache_path)
    print(f"  Tokenizing {len(df)} essays...", flush=True)
    encoded = tokenizer(
        df["full_text"].tolist(),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    # Longformer needs global attention on [CLS] for sequence classification
    global_attention_mask = torch.zeros_like(encoded["input_ids"])
    global_attention_mask[:, 0] = 1
    encoded["global_attention_mask"] = global_attention_mask
    torch.save(encoded, cache_path)
    print(f"  Cached to {cache_path}", flush=True)
    return encoded


# ------------------------------------------------------------------
# Feature extraction (called once per model checkpoint per split)
# ------------------------------------------------------------------

class _TokenDataset(Dataset):
    """Minimal dataset used only during bulk feature extraction."""
    def __init__(self, encoded):
        self.input_ids             = encoded["input_ids"]
        self.attention_mask        = encoded["attention_mask"]
        self.global_attention_mask = encoded["global_attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":             self.input_ids[idx],
            "attention_mask":        self.attention_mask[idx],
            "global_attention_mask": self.global_attention_mask[idx],
        }


def extract_features(longformer, encoded, cache_path):
    # Run the frozen Longformer over all essays once and save the CLS
    # embeddings. Subsequent probes for different demographics reuse
    # these embeddings without touching Longformer again.
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")

    print(f"  Extracting features for {encoded['input_ids'].shape[0]} essays...", flush=True)
    loader  = DataLoader(_TokenDataset(encoded), batch_size=EXTRACT_BATCH, shuffle=False)
    all_cls = []
    longformer.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            outputs = longformer(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                global_attention_mask=batch["global_attention_mask"].to(device)
            )
            all_cls.append(outputs.last_hidden_state[:, 0, :].cpu())
            if i % 50 == 0:
                print(f"    batch {i}/{len(loader)}", flush=True)

    features = torch.cat(all_cls, dim=0)
    torch.save(features, cache_path)
    print(f"  Cached features to {cache_path}", flush=True)
    return features


# ------------------------------------------------------------------
# Dataset for linear probe (works on pre-extracted embeddings)
# ------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, features, indices, labels):
        # Index into the full feature matrix with the relevant row indices
        self.features = features[indices]
        self.labels   = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ------------------------------------------------------------------
# Linear probe head
# ------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(x)


# ------------------------------------------------------------------
# Load checkpoint — extract only the Longformer backbone
# ------------------------------------------------------------------

def load_longformer(model_path):
    state_dict = torch.load(model_path, map_location=device)
    # Strip the "longformer." prefix added by LongformerForEssayScoring
    longformer_state = {
        k.replace("longformer.", ""): v
        for k, v in state_dict.items()
        if k.startswith("longformer.")
    }
    longformer = LongformerModel.from_pretrained(MODEL_NAME)
    longformer.load_state_dict(longformer_state)
    longformer.to(device).eval()
    for p in longformer.parameters():
        p.requires_grad = False
    return longformer


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_persuade(train_path, test_path, test_mode=False):
    train_df = pd.read_csv(train_path, low_memory=False).dropna(
        subset=["holistic_essay_score", "full_text"]
    ).drop_duplicates(subset=["essay_id_comp"]).reset_index(drop=True)
    test_df = pd.read_csv(test_path, low_memory=False).dropna(
        subset=["holistic_essay_score", "full_text"]
    ).drop_duplicates(subset=["essay_id_comp"]).reset_index(drop=True)
    if test_mode:
        train_df = train_df.head(200)
        test_df  = test_df.head(50)
    return train_df, test_df


def load_asap(train_path, test_path, test_mode=False):
    train_df = pd.read_csv(train_path, low_memory=False).dropna(
        subset=["score", "full_text"]
    ).drop_duplicates(subset=["essay_id"]).reset_index(drop=True)
    test_df = pd.read_csv(test_path, low_memory=False).dropna(
        subset=["score", "full_text"]
    ).drop_duplicates(subset=["essay_id"]).reset_index(drop=True)
    if test_mode:
        train_df = train_df.head(200)
        test_df  = test_df.head(50)
    return train_df, test_df


# ------------------------------------------------------------------
# Probing — no Longformer calls here, just linear head on embeddings
# ------------------------------------------------------------------

def run_probe(train_features, train_indices, train_labels,
              test_features,  test_indices,  test_labels,
              num_classes, label_name):

    # Weight classes inversely by frequency to handle imbalanced demographics
    # (e.g. ELL ~9%, disability ~10% in PERSUADE)
    label_counts  = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = torch.tensor(1.0 / (label_counts + 1e-6), dtype=torch.float).to(device)

    train_dataset = EmbeddingDataset(train_features, train_indices, train_labels)
    test_dataset  = EmbeddingDataset(test_features,  test_indices,  test_labels)

    dev_size   = max(1, int(0.1 * len(train_dataset)))
    train_size = len(train_dataset) - dev_size
    train_sub, dev_sub = random_split(
        train_dataset, [train_size, dev_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )

    train_loader = DataLoader(train_sub,    batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_sub,      batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    hidden_size = train_features.shape[1]
    probe       = LinearProbe(hidden_size, num_classes).to(device)
    optimizer   = Adam(probe.parameters(), lr=LEARNING_RATE)
    loss_fn     = nn.CrossEntropyLoss(weight=class_weights)

    best_dev_kappa   = -1.0
    best_probe_state = None
    no_improve       = 0

    for epoch in range(MAX_EPOCHS):
        probe.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(probe(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

        # Evaluate on dev set and use kappa for early stopping
        probe.eval()
        dev_preds, dev_true = [], []
        with torch.no_grad():
            for features, labels in dev_loader:
                preds = probe(features.to(device)).argmax(dim=1)
                dev_preds.extend(preds.cpu().numpy())
                dev_true.extend(labels.numpy())

        if len(set(dev_true)) < 2:
            continue

        dev_kappa = cohen_kappa_score(dev_true, dev_preds)
        print(f"  [{label_name}] Epoch {epoch+1:02d} | Dev κ: {dev_kappa:.4f}", flush=True)

        if dev_kappa > best_dev_kappa:
            best_dev_kappa   = dev_kappa
            best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
            no_improve       = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}.", flush=True)
                break

    if best_probe_state:
        probe.load_state_dict(best_probe_state)
    probe.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            preds = probe(features.to(device)).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.numpy())

    if len(set(test_true)) < 2:
        print(f"  {label_name}: insufficient class diversity in test set", flush=True)
        return None

    kappa = cohen_kappa_score(test_true, test_preds)
    print(f"  {label_name}: κ = {kappa:.4f}", flush=True)
    return kappa


def probe_demographic(train_df, test_df,
                      train_features, test_features,
                      col, demo_name):
    results = {}

    if demo_name == "race":
        # One-vs-rest binary probe for each racial group
        for group in RACE_BINARY_GROUPS:
            tr = train_df.dropna(subset=[col])
            te = test_df.dropna(subset=[col])
            label = f"race ({group})"
            kappa = run_probe(
                train_features, tr.index.tolist(), (tr[col] == group).astype(int).tolist(),
                test_features,  te.index.tolist(), (te[col] == group).astype(int).tolist(),
                num_classes=2, label_name=label
            )
            results[label] = kappa
    else:
        tr = train_df.dropna(subset=[col])
        te = test_df.dropna(subset=[col])
        le = LabelEncoder()
        tr_labels = le.fit_transform(tr[col]).tolist()
        te_labels = le.transform(
            te[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
        ).tolist()
        kappa = run_probe(
            train_features, tr.index.tolist(), tr_labels,
            test_features,  te.index.tolist(), te_labels,
            num_classes=len(le.classes_), label_name=demo_name
        )
        results[demo_name] = kappa

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    parser.add_argument("--model",   required=True)
    parser.add_argument("--demo",    default=None,
                        choices=["gender", "race", "ell", "ses", "disability"])
    parser.add_argument("--test",    action="store_true",
                        help="Run on a small subset for quick debugging.")
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"  Probing: {args.model}", flush=True)
    print(f"  Dataset: {args.dataset.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    if args.dataset == "persuade":
        train_df, test_df = load_persuade(PERSUADE_TRAIN, PERSUADE_TEST, test_mode=args.test)
    else:
        train_df, test_df = load_asap(ASAP_TRAIN, ASAP_TEST, test_mode=args.test)

    # Tokenize once per dataset and cache — shared across all model checkpoints
    os.makedirs("cache", exist_ok=True)
    slug         = args.dataset
    train_tokens = tokenize_and_cache(train_df, f"cache/{slug}_train_tokens.pt")
    test_tokens  = tokenize_and_cache(test_df,  f"cache/{slug}_test_tokens.pt")

    # Extract CLS features once per model checkpoint and cache
    model_slug      = os.path.splitext(os.path.basename(args.model))[0]
    train_feat_path = f"cache/{slug}_{model_slug}_train_features.pt"
    test_feat_path  = f"cache/{slug}_{model_slug}_test_features.pt"

    longformer     = load_longformer(args.model)
    train_features = extract_features(longformer, train_tokens, train_feat_path)
    test_features  = extract_features(longformer, test_tokens,  test_feat_path)

    # Free GPU memory — Longformer is no longer needed for probe training
    del longformer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    demo_cols = PROBE_COLS[args.dataset]
    if args.demo:
        demo_cols = {args.demo: demo_cols[args.demo]}

    all_results = {}
    for demo_name, col in demo_cols.items():
        print(f"\n  -- {demo_name} --", flush=True)
        results = probe_demographic(
            train_df, test_df,
            train_features, test_features,
            col, demo_name
        )
        all_results.update(results)

    print(f"\n{'='*60}", flush=True)
    print(f"  Summary: {args.model}", flush=True)
    print(f"{'='*60}", flush=True)
    for k, v in all_results.items():
        if v is not None:
            print(f"  {k:40s} κ = {v:.4f}", flush=True)


if __name__ == "__main__":
    main()