# RoBERTa Hidden-State Probing (Phase 4)
# SCU CSEN 346 - Rina Li, Tom Ngo, Karthik Tamil

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import cohen_kappa_score
import time

from train_evaluate_roberta import RobertaForEssayScoring

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
TRAINED_GRADER_PATH = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/best_roberta_persuade.pt"

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

# ------------------------------------------------------------------
# Architecture for Probing
# ------------------------------------------------------------------
class DemographicProbe(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(x)

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def extract_embeddings(texts, model, batch_size=BATCH_SIZE):
    """Passes text through the frozen RoBERTa body and returns the CLS embeddings."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            cls_output = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_output.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def train_and_evaluate(model, train_loader, test_loader):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    model.train()
    for epoch in range(MAX_EPOCHS):
        for batch_emb, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_emb.to(device))
            loss = loss_fn(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()

    # Evaluate once at the end
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_emb, batch_labels in test_loader:
            logits = model(batch_emb.to(device))
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    return cohen_kappa_score(all_labels, all_preds)

# ------------------------------------------------------------------
# Main Batch Execution
# ------------------------------------------------------------------
def main():
    print("Loading datasets into memory once...", flush=True)
    # Load entire CSVs into memory once
    full_train_df = pd.read_csv(PERSUADE_TRAIN, low_memory=False)
    full_test_df = pd.read_csv(PERSUADE_TEST, low_memory=False)
    
    # Auto-extract all unique prompts
    PROMPTS = full_train_df["prompt_name"].dropna().unique().tolist()
    print(f"--> Found {len(PROMPTS)} unique prompts.", flush=True)
    
    # Auto-build demographic mappings
    DEMOGRAPHICS = {}
    for col in DEMO_COLUMNS_TO_CHECK:
        if col in full_train_df.columns:
            unique_vals = full_train_df[col].dropna().unique()
            if len(unique_vals) > 1:
                DEMOGRAPHICS[col] = {val: idx for idx, val in enumerate(unique_vals)}
                
    print(f"--> Mapped {len(DEMOGRAPHICS)} demographic features.", flush=True)

    print("Loading Trained Grader Checkpoint...", flush=True)
    grader = RobertaForEssayScoring().to(device)
    grader.load_state_dict(torch.load(TRAINED_GRADER_PATH, map_location=device))
    frozen_body = grader.roberta 
    
    results = []

    for prompt in PROMPTS:
        print(f"\n======================================", flush=True)
        print(f"Processing Prompt: {prompt}", flush=True)
        
        # Filter texts for this prompt and drop missing texts
        prompt_train_df = full_train_df[full_train_df["prompt_name"] == prompt].dropna(subset=["full_text"]).copy()
        prompt_test_df = full_test_df[full_test_df["prompt_name"] == prompt].dropna(subset=["full_text"]).copy()
        
        if len(prompt_train_df) == 0 or len(prompt_test_df) == 0:
            print(f"Skipping {prompt}: Not enough data.", flush=True)
            continue
            
        print(f"Extracting embeddings for {len(prompt_train_df)} train and {len(prompt_test_df)} test essays...", flush=True)
        start_time = time.time()
        train_texts = prompt_train_df["full_text"].tolist()
        test_texts = prompt_test_df["full_text"].tolist()
        
        # Compute the RoBERTa embeddings exactly once for the prompt
        train_embeddings = extract_embeddings(train_texts, frozen_body, batch_size=BATCH_SIZE)
        test_embeddings = extract_embeddings(test_texts, frozen_body, batch_size=BATCH_SIZE)
        
        print(f"Embeddings extracted in {time.time() - start_time:.2f}s", flush=True)
        
        for demo_name, demo_map in DEMOGRAPHICS.items():
            print(f"\n--- Probing: {prompt} | {demo_name} ---", flush=True)
            
            # Map labels onto the filtered dataframe
            prompt_train_df["label"] = prompt_train_df[demo_name].map(demo_map)
            prompt_test_df["label"]  = prompt_test_df[demo_name].map(demo_map)
            
            # Mask out missing labels
            train_mask = prompt_train_df["label"].notna()
            test_mask = prompt_test_df["label"].notna()
            
            if not train_mask.any() or not test_mask.any():
                print(f"Skipping (Not enough data for {demo_name} on {prompt})", flush=True)
                continue
                
            # Filter embeddings and labels using the mask
            train_mask_np = train_mask.to_numpy(dtype=bool)
            test_mask_np = test_mask.to_numpy(dtype=bool)
            
            cur_train_emb = train_embeddings[train_mask_np]
            cur_train_lbl = torch.tensor(prompt_train_df["label"][train_mask_np].values, dtype=torch.long)
            
            cur_test_emb = test_embeddings[test_mask_np]
            cur_test_lbl = torch.tensor(prompt_test_df["label"][test_mask_np].values, dtype=torch.long)
            
            train_loader = DataLoader(TensorDataset(cur_train_emb, cur_train_lbl), batch_size=BATCH_SIZE, shuffle=True)
            test_loader  = DataLoader(TensorDataset(cur_test_emb, cur_test_lbl), batch_size=BATCH_SIZE, shuffle=False)
            
            num_classes = len(set(demo_map.values()))
            
            # Initialize the linear probe
            model = DemographicProbe(frozen_body.config.hidden_size, num_classes).to(device)
            
            kappa = train_and_evaluate(model, train_loader, test_loader)
            print(f"Resulting Kappa: {kappa:.4f}", flush=True)
            
            results.append({
                "Prompt": prompt,
                "Demographic": demo_name,
                "Kappa": kappa
            })

    # Save everything to a clean CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("probing_results_matrix.csv", index=False)
    print("\nAll probing complete! Results saved to 'probing_results_matrix.csv'.", flush=True)
    print(results_df, flush=True)

if __name__ == "__main__":
    main()