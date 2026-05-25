import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import cohen_kappa_score

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME    = "roberta-base"
MAX_LENGTH    = 512 
BATCH_SIZE    = 8
MAX_EPOCHS    = 3 # Reduced epochs for full fine-tuning (usually 3-5 is enough)
LEARNING_RATE = 2e-5 # Standard LR for fine-tuning RoBERTa
RANDOM_STATE  = 42

PERSUADE_TRAIN = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/train/persuade_corpus_2.0_train.csv"
PERSUADE_TEST  = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

DEMO_COLUMNS_TO_CHECK = [
    "gender",
    "race_ethnicity",
    "ell_status", 
    "economically_disadvantaged", 
    "student_disability_status"
]

print("Scanning dataset for prompts and demographics...", flush=True)

# Use `usecols` to only load the columns we care about. 
cols_to_load = ["prompt_name"] + DEMO_COLUMNS_TO_CHECK
df_meta = pd.read_csv(PERSUADE_TRAIN, usecols=lambda c: c in cols_to_load, low_memory=False)

# 1. Auto-extract all unique prompts
PROMPTS = df_meta["prompt_name"].dropna().unique().tolist()
print(f"--> Found {len(PROMPTS)} unique prompts.", flush=True)

# 2. Auto-build demographic mappings
DEMOGRAPHICS = {}
for col in DEMO_COLUMNS_TO_CHECK:
    if col in df_meta.columns:
        # Get unique values, dropping NaNs so 'nan' doesn't become a classification category
        unique_vals = df_meta[col].dropna().unique()
        
        # Skip if there's only 1 category (nothing to classify)
        if len(unique_vals) > 1:
            # Dynamically create mapping, e.g., {"White": 0, "Black": 1, "Asian": 2}
            DEMOGRAPHICS[col] = {val: idx for idx, val in enumerate(unique_vals)}

print(f"--> Mapped {len(DEMOGRAPHICS)} demographic features.", flush=True)

# Free up memory before PyTorch starts
del df_meta

# ------------------------------------------------------------------
# Automated Data Loader
# ------------------------------------------------------------------
class DemographicDataset(Dataset):
    def __init__(self, texts, labels):
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def get_dataloaders(train_path, test_path, prompt_name, target_demo, label_map):
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df  = pd.read_csv(test_path, low_memory=False)
    
    # Filter by prompt
    train_df = train_df[train_df["prompt_name"] == prompt_name]
    test_df  = test_df[test_df["prompt_name"] == prompt_name]
    
    # Map demographics and drop NaNs for this specific trait
    train_df["label"] = train_df[target_demo].map(label_map)
    test_df["label"]  = test_df[target_demo].map(label_map)
    train_df = train_df.dropna(subset=["label", "full_text"])
    test_df  = test_df.dropna(subset=["label", "full_text"])

    if len(train_df) == 0 or len(test_df) == 0:
        return None, None # Skip if no data for this combo

    train_dataset = DemographicDataset(train_df["full_text"].tolist(), train_df["label"].tolist())
    test_dataset  = DemographicDataset(test_df["full_text"].tolist(),  test_df["label"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train_and_evaluate(model, train_loader, test_loader):
    # Optimize ALL parameters for the upper bound (training from scratch/fine-tuning)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate once at the end
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    return cohen_kappa_score(all_labels, all_preds)

# ------------------------------------------------------------------
# Main Batch Execution
# ------------------------------------------------------------------
def main():
    results = []

    for prompt in PROMPTS:
        for demo_name, demo_map in DEMOGRAPHICS.items():
            print(f"\n--- Upper Bound Training: {prompt} | {demo_name} ---")
            
            train_loader, test_loader = get_dataloaders(
                PERSUADE_TRAIN, PERSUADE_TEST, prompt, demo_name, demo_map
            )
            
            if train_loader is None:
                print(f"Skipping (Not enough data for {demo_name} on {prompt})")
                continue

            num_classes = len(set(demo_map.values()))
            
            # Initialize a FRESH, completely separate model for each specific demographic
            print(f"Initializing fresh RobertaForSequenceClassification for {demo_name}...")
            model = RobertaForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=num_classes
            ).to(device)
            
            kappa = train_and_evaluate(model, train_loader, test_loader)
            print(f"Resulting Kappa: {kappa:.4f}")
            
            results.append({
                "Prompt": prompt,
                "Demographic": demo_name,
                "Kappa": kappa
            })

    # Save everything to a clean CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("upper_bound_results_matrix.csv", index=False)
    print("\nAll upper bound training complete! Results saved to 'upper_bound_results_matrix.csv'.")
    print(results_df)

if __name__ == "__main__":
    main()
