# RoBERTa Hidden-State Probing (Phase 4)
# SCU CSEN 346 - Rina Li, Tom Ngo, Karthik Tamil

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import cohen_kappa_score

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME    = "roberta-base"
MAX_LENGTH    = 512 
BATCH_SIZE    = 8
MAX_EPOCHS    = 20
PATIENCE      = 3
LEARNING_RATE = 5e-6
RANDOM_STATE  = 42

PERSUADE_TRAIN = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/train/persuade_corpus_2.0_train.csv"
PERSUADE_TEST  = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv"
TRAINED_GRADER_PATH = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/best_roberta_asap.pt"

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
# This prevents pandas from loading the massive "full_text" column into memory during the scan.
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
# Original Grader Architecture (Needed to load the weights)
# ------------------------------------------------------------------
class RobertaForDemographicProbing(nn.Module):
    def __init__(self, trained_roberta_body, num_classes):
        super().__init__()
        self.roberta = trained_roberta_body
        for param in self.roberta.parameters():
            param.requires_grad = False # Keep brain frozen
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ------------------------------------------------------------------
# Automated Data Loader
# ------------------------------------------------------------------
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

    train_dataset = ProbingDataset(train_df["full_text"].tolist(), train_df["label"].tolist())
    test_dataset  = ProbingDataset(test_df["full_text"].tolist(),  test_df["label"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train_and_evaluate(model, train_loader, test_loader):
    optimizer = AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    model.train()
    for epoch in range(MAX_EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

    # Evaluate once at the end
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    return cohen_kappa_score(all_labels, all_preds)

# ------------------------------------------------------------------
# Main Batch Execution
# ------------------------------------------------------------------
def main():
    print("Loading Frozen RoBERTa Body once...")
    grader = RobertaForEssayScoring().to(device)
    grader.load_state_dict(torch.load(TRAINED_GRADER_PATH, map_location=device))
    frozen_body = grader.roberta 
    
    results = []

    for prompt in PROMPTS:
        for demo_name, demo_map in DEMOGRAPHICS.items():
            print(f"\n--- Probing: {prompt} | {demo_name} ---")
            
            train_loader, test_loader = get_dataloaders(
                PERSUADE_TRAIN, PERSUADE_TEST, prompt, demo_name, demo_map
            )
            
            if train_loader is None:
                print(f"Skipping (Not enough data for {demo_name} on {prompt})")
                continue

            num_classes = len(set(demo_map.values()))
            
            # Initialize a FRESH model for each run using the SAME frozen body
            model = RobertaForDemographicProbing(frozen_body, num_classes).to(device)
            
            kappa = train_and_evaluate(model, train_loader, test_loader)
            print(f"Resulting Kappa: {kappa:.4f}")
            
            results.append({
                "Prompt": prompt,
                "Demographic": demo_name,
                "Kappa": kappa
            })

    # Save everything to a clean CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("probing_results_matrix.csv", index=False)
    print("\nAll probing complete! Results saved to 'probing_results_matrix.csv'.")
    print(results_df)

if __name__ == "__main__":
    main()