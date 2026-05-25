# Adversarial Training for Automated Essay Scoring
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
ADV_ALPHA     = 1.0 # Adversarial loss weight

PERSUADE_TRAIN = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/train/persuade_corpus_2.0_train.csv"
PERSUADE_TEST  = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv"
ASAP_TRAIN     = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/train/ASAP_2_Final_github_train.csv"
ASAP_TEST      = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

DEMO_COLUMNS = [
    "gender",
    "race_ethnicity",
    "ell_status", 
    "economically_disadvantaged", 
    "student_disability_status"
]

# ------------------------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class AdversarialEssayDataset(Dataset):
    def __init__(self, texts, scores, demographics):
        print(f"  Tokenizing {len(texts)} essays...", flush=True)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        self.input_ids      = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.scores         = torch.tensor(list(scores), dtype=torch.float)
        
        # demographics is a dict of lists { "gender": [0, 1, -1...], ... }
        # -1 indicates missing data
        self.demographics = {
            k: torch.tensor(v, dtype=torch.long) for k, v in demographics.items()
        }
        print("  Tokenization complete.", flush=True)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.scores[idx]
        }
        for k in self.demographics:
            item[k] = self.demographics[k][idx]
        return item

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
def extract_demographics(df, mappings=None):
    if mappings is None:
        mappings = {}
        # Build mappings from train data
        for col in DEMO_COLUMNS:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                mappings[col] = {val: i for i, val in enumerate(unique_vals)}
            else:
                mappings[col] = {}

    demo_data = {col: [] for col in DEMO_COLUMNS}
    for _, row in df.iterrows():
        for col in DEMO_COLUMNS:
            val = row.get(col)
            if pd.isna(val) or val not in mappings.get(col, {}):
                demo_data[col].append(-1) # Missing
            else:
                demo_data[col].append(mappings[col][val])
                
    return demo_data, mappings

def load_data(train_path, test_path, is_persuade=True):
    id_col = "essay_id_comp" if is_persuade else "essay_id"
    score_col = "holistic_essay_score" if is_persuade else "score"
    
    train_df = pd.read_csv(train_path, low_memory=False).dropna(subset=[score_col, "full_text"])
    test_df = pd.read_csv(test_path, low_memory=False).dropna(subset=[score_col, "full_text"])
    
    train_df = train_df.drop_duplicates(subset=[id_col])
    test_df  = test_df.drop_duplicates(subset=[id_col])

    train_demo, mappings = extract_demographics(train_df)
    test_demo, _ = extract_demographics(test_df, mappings)

    return (
        train_df["full_text"].tolist(), train_df[score_col].tolist(), train_demo,
        test_df["full_text"].tolist(),  test_df[score_col].tolist(),  test_demo,
        mappings
    )

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class RobertaForAdversarialEssayScoring(nn.Module):
    def __init__(self, num_classes_dict, model_name=MODEL_NAME):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        
        self.grl = GradientReversalLayer(alpha=ADV_ALPHA)
        
        self.adversaries = nn.ModuleDict()
        for demo_attr, num_classes in num_classes_dict.items():
            if num_classes > 1:
                self.adversaries[demo_attr] = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Essay Scoring Head
        score_preds = self.regressor(cls_output).squeeze(-1)
        
        # Adversarial Heads
        reversed_cls = self.grl(cls_output)
        demo_logits = {}
        for demo_attr, adv_head in self.adversaries.items():
            demo_logits[demo_attr] = adv_head(reversed_cls)
            
        return score_preds, demo_logits

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train(model, train_loader, dev_loader, optimizer, save_path, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    best_dev_loss     = float("inf")
    epochs_no_improve = 0
    
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # Ignore missing demographic labels

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_adv = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            score_preds, demo_logits = model(input_ids, attention_mask)
            
            # Essay Scoring Loss
            mse_loss = mse_loss_fn(score_preds, labels)
            
            # Adversarial Demographic Loss
            adv_loss = 0
            for demo_attr in demo_logits:
                demo_labels = batch[demo_attr].to(device)
                adv_loss += ce_loss_fn(demo_logits[demo_attr], demo_labels)
                
            loss = mse_loss + adv_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            if isinstance(adv_loss, torch.Tensor):
                total_adv += adv_loss.item()

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Total: {loss.item():.4f} (MSE: {mse_loss.item():.4f}, Adv: {total_adv/(batch_idx+1):.4f})", flush=True)

        dev_loss, dev_qwk = evaluate(model, dev_loader, verbose=False)
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss:.4f} | Dev Loss (MSE): {dev_loss:.4f} | Dev QWK: {dev_qwk:.4f}", flush=True)

        if dev_loss < best_dev_loss:
            best_dev_loss     = dev_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best model.", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.", flush=True)
                break

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def compute_smd(preds, labels):
    diff = np.array(preds) - np.array(labels)
    std = diff.std()
    return diff.mean() / std if std != 0 else 0

def evaluate(model, loader, verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    loss_fn    = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            score_preds, _ = model(input_ids, attention_mask)
            
            loss = loss_fn(score_preds, labels)
            total_loss += loss.item()
            all_preds.extend(score_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds_rounded = np.round(all_preds).astype(int)
    all_labels_int    = np.array(all_labels).astype(int)

    qwk   = cohen_kappa_score(all_labels_int, all_preds_rounded, weights="quadratic")
    exact = np.mean(all_preds_rounded == all_labels_int)
    smd   = compute_smd(all_preds, all_labels)

    if verbose:
        print(f"  MSE:             {total_loss:.4f}", flush=True)
        print(f"  QWK:             {qwk:.4f}", flush=True)
        print(f"  Exact Agreement: {exact:.4f}", flush=True)
        print(f"  SMD:             {smd:.4f}", flush=True)

    return total_loss, qwk

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["persuade", "asap"], required=True)
    args = parser.parse_args()

    print("\n" + "="*60, flush=True)
    print(f"  Dataset: {args.dataset.upper()} (Adversarial Debiasing)", flush=True)
    print("="*60, flush=True)

    is_persuade = args.dataset == "persuade"
    train_path = PERSUADE_TRAIN if is_persuade else ASAP_TRAIN
    test_path  = PERSUADE_TEST if is_persuade else ASAP_TEST
    
    (train_texts, train_scores, train_demo, 
     test_texts, test_scores, test_demo, mappings) = load_data(train_path, test_path, is_persuade)
     
    save_path = f"best_debiased_roberta_{args.dataset}.pt"
    
    # Filter out empty demographic mappings
    num_classes_dict = {k: len(v) for k, v in mappings.items() if len(v) > 1}
    print(f"Adversarial Targets: {list(num_classes_dict.keys())}", flush=True)

    # Convert to DataFrame for train/dev split
    train_df = pd.DataFrame({"text": train_texts, "score": train_scores})
    for col in DEMO_COLUMNS:
        train_df[col] = train_demo[col]
        
    dev_df = train_df.sample(frac=0.1, random_state=RANDOM_STATE)
    train_df = train_df.drop(dev_df.index)

    print(f"  Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_texts)}", flush=True)

    # Reconstruct demographic dicts
    def df_to_demo(df):
        return {col: df[col].tolist() for col in DEMO_COLUMNS}
        
    train_dataset = AdversarialEssayDataset(train_df["text"].tolist(), train_df["score"].tolist(), df_to_demo(train_df))
    dev_dataset   = AdversarialEssayDataset(dev_df["text"].tolist(),   dev_df["score"].tolist(), df_to_demo(dev_df))
    test_dataset  = AdversarialEssayDataset(test_texts,                test_scores,              test_demo)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = RobertaForAdversarialEssayScoring(num_classes_dict).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, dev_loader, optimizer, save_path=save_path)

    model.load_state_dict(torch.load(save_path))
    print(f"\n--- {args.dataset.upper()} Test Results ---", flush=True)
    evaluate(model, test_loader, verbose=True)

if __name__ == "__main__":
    main()
