# RoBERTa for Automated Essay Scoring
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
MAX_LENGTH    = 512 # RoBERTa has a strict 512 max coPntext limit
BATCH_SIZE    = 8
MAX_EPOCHS    = 20
PATIENCE      = 3
LEARNING_RATE = 5e-6
RANDOM_STATE  = 42

PERSUADE_TRAIN = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/train/persuade_corpus_2.0_train.csv"
PERSUADE_TEST  = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv"
ASAP_TRAIN     = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/train/ASAP_2_Final_github_train.csv"
ASAP_TEST      = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class EssayDataset(Dataset):
    def __init__(self, texts, scores):
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
        print("  Tokenization complete.", flush=True)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.scores[idx]
        }


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

def load_persuade(train_path, test_path):
    train_df = pd.read_csv(train_path, low_memory=False).dropna(
        subset=["holistic_essay_score", "full_text"]
    )
    test_df = pd.read_csv(test_path, low_memory=False).dropna(
        subset=["holistic_essay_score", "full_text"]
    )
    train_df = train_df.drop_duplicates(subset=["essay_id_comp"])
    test_df  = test_df.drop_duplicates(subset=["essay_id_comp"])

    return (
        train_df["full_text"].tolist(), train_df["holistic_essay_score"].tolist(),
        test_df["full_text"].tolist(),  test_df["holistic_essay_score"].tolist()
    )


def load_asap(train_path, test_path):
    train_df = pd.read_csv(train_path, low_memory=False).dropna(
        subset=["score", "full_text"]
    )
    test_df = pd.read_csv(test_path, low_memory=False).dropna(
        subset=["score", "full_text"]
    )
    train_df = train_df.drop_duplicates(subset=["essay_id"])
    test_df  = test_df.drop_duplicates(subset=["essay_id"])

    return (
        train_df["full_text"].tolist(), train_df["score"].tolist(),
        test_df["full_text"].tolist(),  test_df["score"].tolist()
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class RobertaForEssayScoring(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.roberta   = RobertaModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # RoBERTa uses <s> at index 0 for sequence-level representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(model, train_loader, dev_loader, optimizer, loss_fn,
          save_path, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    best_dev_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            loss = loss_fn(preds, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        dev_loss, dev_qwk = evaluate(model, dev_loader, verbose=False)
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss:.4f} | "
              f"Dev Loss: {dev_loss:.4f} | Dev QWK: {dev_qwk:.4f}", flush=True)

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

def evaluate(model, loader, verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    loss_fn    = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            preds = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            loss = loss_fn(preds, batch["labels"].to(device))
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    all_preds_rounded = np.round(all_preds).astype(int)
    all_labels_int    = np.array(all_labels).astype(int)

    qwk   = cohen_kappa_score(all_labels_int, all_preds_rounded, weights="quadratic")
    exact = np.mean(all_preds_rounded == all_labels_int)

    if verbose:
        print(f"  MSE:             {total_loss:.4f}", flush=True)
        print(f"  QWK:             {qwk:.4f}", flush=True)
        print(f"  Exact Agreement: {exact:.4f}", flush=True)

    return total_loss, qwk


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["persuade", "asap"],
        required=True,
        help="Dataset to train and test on."
    )
    args = parser.parse_args()

    print("\n" + "="*60, flush=True)
    print(f"  Dataset: {args.dataset.upper()} (RoBERTa Baseline)", flush=True)
    print("="*60, flush=True)

    if args.dataset == "persuade":
        train_texts, train_scores, test_texts, test_scores = load_persuade(PERSUADE_TRAIN, PERSUADE_TEST)
        save_path = "best_roberta_persuade.pt"
        label     = "PERSUADE"
    else:
        train_texts, train_scores, test_texts, test_scores = load_asap(ASAP_TRAIN, ASAP_TEST)
        save_path = "best_roberta_asap.pt"
        label     = "ASAP"

    train_df = pd.DataFrame({"text": train_texts, "score": train_scores})
    dev_df   = train_df.sample(frac=0.1, random_state=RANDOM_STATE)
    train_df = train_df.drop(dev_df.index)

    print(f"  Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_texts)}", flush=True)

    train_dataset = EssayDataset(train_df["text"].tolist(), train_df["score"].tolist())
    dev_dataset   = EssayDataset(dev_df["text"].tolist(),   dev_df["score"].tolist())
    test_dataset  = EssayDataset(test_texts,                test_scores)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model     = RobertaForEssayScoring().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.MSELoss()

    train(model, train_loader, dev_loader, optimizer, loss_fn, save_path=save_path)

    model.load_state_dict(torch.load(save_path))
    print(f"\n--- {label} Test Results ---", flush=True)
    evaluate(model, test_loader, verbose=True)


if __name__ == "__main__":
    main()