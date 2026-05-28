# inference_longformer.py
# Generates prediction CSVs from saved Longformer checkpoints for bias analysis.

import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import LongformerTokenizer
from longformer import LongformerForEssayScoring, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, device

tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)

DATASETS = [
    {
        "name":        "PERSUADE",
        "test_path":   "PERSUADE/persuade_corpus_2.0_test.csv",
        "id_col":      "essay_id_comp",
        "text_col":    "full_text",
        "score_col":   "holistic_essay_score",
        "model_paths": {
            "base": "pt/best_longformer_persuade_base.pt",
            "grl":  "pt/best_longformer_persuade_grl.pt"
        },
        "output_names": {
            "base": "persuade_longformer_predictions.csv",
            "grl":  "persuade_debiased_longformer_predictions.csv"
        }
    },
    {
        "name":        "ASAP",
        "test_path":   "ASAP/ASAP_2_Final_github_test.csv",
        "id_col":      "essay_id",
        "text_col":    "full_text",
        "score_col":   "score",
        "model_paths": {
            "base": "pt/best_longformer_asap_base.pt",
            "grl":  "pt/best_longformer_asap_grl.pt"
        },
        "output_names": {
            "base": "asap_longformer_predictions.csv",
            "grl":  "asap_debiased_longformer_predictions.csv"
        }
    }
]


def generate_predictions(model_type):
    for ds in DATASETS:
        print(f"\n--- Generating Predictions for {ds['name']} ({model_type.upper()}) ---")

        df = pd.read_csv(ds["test_path"], low_memory=False).dropna(
            subset=[ds["score_col"], ds["text_col"]]
        ).drop_duplicates(subset=[ds["id_col"]])

        texts        = df[ds["text_col"]].tolist()
        essay_ids    = df[ds["id_col"]].tolist()
        human_scores = df[ds["score_col"]].tolist()

        model_path  = ds["model_paths"][model_type]
        output_name = ds["output_names"][model_type]

        # Load with debias=True for GRL checkpoints so the state dict matches
        model = LongformerForEssayScoring(debias=(model_type == "grl")).to(device)
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Could not load {model_path}: {e}")
            continue

        model.eval()
        predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Scoring {ds['name']}"):
                batch_texts = texts[i : i + BATCH_SIZE]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)

                global_attention_mask = torch.zeros_like(encoded["input_ids"])
                global_attention_mask[:, 0] = 1

                out = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    global_attention_mask=global_attention_mask
                )
                # Handle both standard and debiased model outputs
                preds = out[0] if isinstance(out, tuple) else out
                predictions.extend(preds.cpu().numpy())

        output_df = pd.DataFrame({
            "essay_id":         essay_ids,
            "human_score":      human_scores,
            "longformer_score": predictions
        })

        output_df.to_csv(output_name, index=False)
        print(f"Saved {len(output_df)} predictions to {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "grl"], required=True,
                        help="Which checkpoint to use for inference.")
    args = parser.parse_args()
    generate_predictions(args.model)