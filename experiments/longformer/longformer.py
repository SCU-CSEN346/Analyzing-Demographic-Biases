"""
Longformer for Automated Essay Scoring
SCU CSEN 346 - Rina Li, Tom Ngo, Karthik Tamil

Usage:
  python longformer.py --dataset persuade
  python longformer.py --dataset asap
  python longformer.py --dataset persuade --skip-train
  python longformer.py --dataset persuade --per-prompt
  python longformer.py --dataset persuade --per-prompt --debias --demo gender
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.autograd import Function

from transformers import LongformerModel, LongformerTokenizer
from sklearn.metrics import cohen_kappa_score


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME    = "allenai/longformer-base-4096"
MAX_LENGTH    = 1024
BATCH_SIZE    = 4
MAX_EPOCHS    = 20
PATIENCE      = 3
LEARNING_RATE = 5e-6
RANDOM_STATE  = 42
GRL_LAMBDA    = 0.01

PERSUADE_TRAIN = "PERSUADE/persuade_corpus_2.0_train.csv"
PERSUADE_TEST  = "PERSUADE/persuade_corpus_2.0_test.csv"
ASAP_TRAIN     = "ASAP/ASAP_2_Final_github_train.csv"
ASAP_TEST      = "ASAP/ASAP_2_Final_github_test.csv"

DEMO_COLS = {
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)


# ------------------------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------------------------

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=GRL_LAMBDA):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambda_)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class EssayDataset(Dataset):
    def __init__(self, texts, scores, demo_labels=None):
        print(f"  Tokenizing {len(texts)} essays...", flush=True)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        self.input_ids             = encoded["input_ids"]
        self.attention_mask        = encoded["attention_mask"]
        # Longformer requires global attention on [CLS] token for regression
        self.global_attention_mask = torch.zeros_like(encoded["input_ids"])
        self.global_attention_mask[:, 0] = 1
        self.scores      = torch.tensor(list(scores), dtype=torch.float)
        self.demo_labels = torch.tensor(list(demo_labels), dtype=torch.long) \
            if demo_labels is not None else None
        print("  Tokenization complete.", flush=True)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {
            "input_ids":             self.input_ids[idx],
            "attention_mask":        self.attention_mask[idx],
            "global_attention_mask": self.global_attention_mask[idx],
            "labels":                self.scores[idx]
        }
        if self.demo_labels is not None:
            item["demo_labels"] = self.demo_labels[idx]
        return item


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

def encode_binary(series):
    vals = series.dropna().unique()
    if len(vals) != 2:
        raise ValueError(f"Expected binary column, got {vals}")
    mapping = {vals[0]: 0, vals[1]: 1}
    return series.map(mapping), mapping


def load_persuade(train_path, test_path, demo_col=None):
    train_df = pd.read_csv(train_path, low_memory=False).dropna(
        subset=["holistic_essay_score", "full_text"]
    )
    test_df = pd.read_csv(test_path, low_memory=False).dropna(
        subset=["holistic_essay_score", "full_text"]
    )
    train_df = train_df.drop_duplicates(subset=["essay_id_comp"])
    test_df  = test_df.drop_duplicates(subset=["essay_id_comp"])

    train_demo = test_demo = None
    if demo_col:
        col = DEMO_COLS["persuade"][demo_col]
        train_df = train_df.dropna(subset=[col])
        test_df  = test_df.dropna(subset=[col])
        train_demo, mapping = encode_binary(train_df[col])
        test_demo, _        = encode_binary(test_df[col])
        train_demo = train_demo.tolist()
        test_demo  = test_demo.tolist()
        print(f"  Demo encoding ({col}): {mapping}", flush=True)

    return (
        train_df, test_df,
        train_df["full_text"].tolist(), train_df["holistic_essay_score"].tolist(),
        test_df["full_text"].tolist(),  test_df["holistic_essay_score"].tolist(),
        train_demo, test_demo
    )


def load_asap(train_path, test_path, demo_col=None):
    train_df = pd.read_csv(train_path, low_memory=False).dropna(
        subset=["score", "full_text"]
    )
    test_df = pd.read_csv(test_path, low_memory=False).dropna(
        subset=["score", "full_text"]
    )
    train_df = train_df.drop_duplicates(subset=["essay_id"])
    test_df  = test_df.drop_duplicates(subset=["essay_id"])

    train_demo = test_demo = None
    if demo_col:
        col = DEMO_COLS["asap"][demo_col]
        train_df = train_df.dropna(subset=[col])
        test_df  = test_df.dropna(subset=[col])
        train_demo, mapping = encode_binary(train_df[col])
        test_demo, _        = encode_binary(test_df[col])
        train_demo = train_demo.tolist()
        test_demo  = test_demo.tolist()
        print(f"  Demo encoding ({col}): {mapping}", flush=True)

    return (
        train_df, test_df,
        train_df["full_text"].tolist(), train_df["score"].tolist(),
        test_df["full_text"].tolist(),  test_df["score"].tolist(),
        train_demo, test_demo
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class LongformerForEssayScoring(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_demo_classes=2, debias=False):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        hidden_size     = self.longformer.config.hidden_size
        self.regressor  = nn.Linear(hidden_size, 1)
        self.debias     = debias
        if debias:
            self.grl             = GradientReversal(lambda_=GRL_LAMBDA)
            self.demo_classifier = nn.Linear(hidden_size, num_demo_classes)

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs    = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        score_pred = self.regressor(cls_output).squeeze(-1)

        if self.debias:
            reversed_cls = self.grl(cls_output)
            demo_logits  = self.demo_classifier(reversed_cls)
            return score_pred, demo_logits

        return score_pred


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(model, train_loader, dev_loader, optimizer, loss_fn,
          save_path, max_epochs=MAX_EPOCHS, patience=PATIENCE, debias=False):
    best_dev_loss     = float("inf")
    epochs_no_improve = 0
    demo_loss_fn      = nn.CrossEntropyLoss() if debias else None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if debias:
                score_pred, demo_logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    global_attention_mask=batch["global_attention_mask"].to(device)
                )
                scoring_loss = loss_fn(score_pred, batch["labels"].to(device))
                adv_loss     = demo_loss_fn(demo_logits, batch["demo_labels"].to(device))
                loss         = scoring_loss + GRL_LAMBDA * adv_loss
            else:
                score_pred = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    global_attention_mask=batch["global_attention_mask"].to(device)
                )
                loss = loss_fn(score_pred, batch["labels"].to(device))

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

def compute_smd(preds, labels):
    diff = np.array(preds) - np.array(labels)
    return diff.mean() / diff.std()


def evaluate(model, loader, verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    loss_fn    = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                global_attention_mask=batch["global_attention_mask"].to(device)
            )
            preds = out[0] if isinstance(out, tuple) else out
            loss  = loss_fn(preds, batch["labels"].to(device))
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

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
# Per-Prompt Runner
# ------------------------------------------------------------------

def run_per_prompt(train_df, test_df, score_col, prompt_col,
                   save_dir, demo_col=None, skip_train=False, debias=False):
    os.makedirs(save_dir, exist_ok=True)
    prompts  = train_df[prompt_col].unique()
    all_qwk  = []

    for prompt in prompts:
        print(f"\n  Prompt: {prompt}", flush=True)
        tr = train_df[train_df[prompt_col] == prompt].dropna(subset=[score_col])
        te = test_df[test_df[prompt_col] == prompt].dropna(subset=[score_col])

        if len(tr) < 10 or len(te) < 5:
            print(f"  Skipping {prompt} (too few samples).", flush=True)
            continue

        train_demo = test_demo = None
        if demo_col:
            tr = tr.dropna(subset=[demo_col]).copy()
            te = te.dropna(subset=[demo_col]).copy()
            if tr[demo_col].nunique() != 2:
                print(f"  Skipping demo for {prompt} (non-binary).", flush=True)
            else:
                vals    = tr[demo_col].dropna().unique()
                mapping = {vals[0]: 0, vals[1]: 1}
                tr["demo_encoded"] = tr[demo_col].map(mapping)
                te["demo_encoded"] = te[demo_col].map(mapping)
                train_demo = tr["demo_encoded"].tolist()
                test_demo  = te["demo_encoded"].tolist()

        dev_tr   = tr.sample(frac=0.1, random_state=RANDOM_STATE)
        train_tr = tr.drop(dev_tr.index)

        train_dataset = EssayDataset(
            train_tr["full_text"].tolist(),
            train_tr[score_col].tolist(),
            train_demo[:len(train_tr)] if train_demo else None
        )
        dev_dataset = EssayDataset(
            dev_tr["full_text"].tolist(),
            dev_tr[score_col].tolist()
        )
        test_dataset = EssayDataset(
            te["full_text"].tolist(),
            te[score_col].tolist(),
            test_demo if test_demo else None
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        prompt_slug = str(prompt).replace(" ", "_").replace("/", "-")[:40]
        save_path   = os.path.join(save_dir, f"longformer_{prompt_slug}.pt")

        model = LongformerForEssayScoring(debias=debias).to(device)

        if skip_train and os.path.exists(save_path):
            print(f"  Loading {save_path}", flush=True)
            model.load_state_dict(torch.load(save_path))
        else:
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            loss_fn   = nn.MSELoss()
            train(model, train_loader, dev_loader, optimizer, loss_fn,
                  save_path=save_path, debias=debias)
            model.load_state_dict(torch.load(save_path))

        print(f"  Results for: {prompt}", flush=True)
        _, qwk = evaluate(model, test_loader, verbose=True)
        all_qwk.append(qwk)

    macro_avg = np.mean(all_qwk)
    print(f"\n{'='*60}", flush=True)
    print(f"  Macro-Average QWK: {macro_avg:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["persuade", "asap"], required=True)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--per-prompt", action="store_true")
    parser.add_argument("--debias",     action="store_true")
    parser.add_argument("--demo",       default="gender",
                        choices=["gender", "race", "ell", "ses", "disability"])
    args = parser.parse_args()

    print("\n" + "="*60, flush=True)
    print(f"  Dataset: {args.dataset.upper()}", flush=True)
    if args.per_prompt:
        print(f"  Mode: per-prompt", flush=True)
    if args.debias:
        print(f"  Debiasing: GRL (target={args.demo})", flush=True)
    print("="*60, flush=True)

    if args.dataset == "persuade":
        train_df, test_df, train_texts, train_scores, \
        test_texts, test_scores, train_demo, test_demo = load_persuade(
            PERSUADE_TRAIN, PERSUADE_TEST,
            demo_col=args.demo if args.debias else None
        )
        prompt_col = "prompt_name"
        score_col  = "holistic_essay_score"
        save_path  = "best_longformer_persuade.pt"
        save_dir   = "checkpoints/persuade"
        label      = "PERSUADE"
    else:
        train_df, test_df, train_texts, train_scores, \
        test_texts, test_scores, train_demo, test_demo = load_asap(
            ASAP_TRAIN, ASAP_TEST,
            demo_col=args.demo if args.debias else None
        )
        prompt_col = "set"
        score_col  = "score"
        save_path  = "best_longformer_asap.pt"
        save_dir   = "checkpoints/asap"
        label      = "ASAP"

    if args.per_prompt:
        demo_col = DEMO_COLS[args.dataset].get(args.demo) if args.debias else None
        run_per_prompt(
            train_df, test_df, score_col, prompt_col,
            save_dir=save_dir,
            demo_col=demo_col,
            skip_train=args.skip_train,
            debias=args.debias
        )
        return

    # --- Single model (all prompts combined) ---
    model = LongformerForEssayScoring(debias=args.debias).to(device)

    if args.skip_train and os.path.exists(save_path):
        print(f"  Found {save_path}, skipping training.", flush=True)
        test_dataset = EssayDataset(test_texts, test_scores, test_demo)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model.load_state_dict(torch.load(save_path))
        print(f"\n--- {label} Test Results ---", flush=True)
        evaluate(model, test_loader, verbose=True)
        return

    df = pd.DataFrame({"text": train_texts, "score": train_scores})
    if train_demo:
        df["demo"] = train_demo
    dev_df       = df.sample(frac=0.1, random_state=RANDOM_STATE)
    train_df_sub = df.drop(dev_df.index)

    print(f"  Train: {len(train_df_sub)} | Dev: {len(dev_df)} | Test: {len(test_texts)}", flush=True)

    train_dataset = EssayDataset(
        train_df_sub["text"].tolist(),
        train_df_sub["score"].tolist(),
        train_df_sub["demo"].tolist() if train_demo else None
    )
    dev_dataset  = EssayDataset(dev_df["text"].tolist(), dev_df["score"].tolist())
    test_dataset = EssayDataset(test_texts, test_scores, test_demo)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.MSELoss()

    train(model, train_loader, dev_loader, optimizer, loss_fn,
          save_path=save_path, debias=args.debias)

    model.load_state_dict(torch.load(save_path))
    print(f"\n--- {label} Test Results ---", flush=True)
    evaluate(model, test_loader, verbose=True)


if __name__ == "__main__":
    main()