# inference.py
# Generates prediction CSVs for Bias Analysis

import torch
import pandas as pd
from tqdm import tqdm # Run: pip install tqdm
from transformers import RobertaTokenizer
from roberta import RobertaForEssayScoring, MODEL_NAME, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BATCH_SIZE = 16 # Inference takes less memory, we can safely double this
tokenizer  = RobertaTokenizer.from_pretrained(MODEL_NAME)

DATASETS = [
    {
        "name": "PERSUADE",
        "test_path": "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv",
        "model_path": "best_roberta_persuade.pt",
        "id_col": "essay_id_comp",
        "text_col": "full_text",
        "score_col": "holistic_essay_score",
        "output_name": "persuade_roberta_predictions.csv"
    },
    {
        "name": "ASAP",
        "test_path": "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv",
        "model_path": "best_roberta_asap.pt",
        "id_col": "essay_id",
        "text_col": "full_text",
        "score_col": "score",
        "output_name": "asap_roberta_predictions.csv"
    }
]

# ------------------------------------------------------------------
# Inference Loop
# ------------------------------------------------------------------
def generate_predictions():
    for ds in DATASETS:
        print(f"\n--- Generating Predictions for {ds['name']} ---")
        
        # 1. Load Data
        df = pd.read_csv(ds["test_path"], low_memory=False).dropna(subset=[ds["score_col"], ds["text_col"]])
        df = df.drop_duplicates(subset=[ds["id_col"]])
        
        texts = df[ds["text_col"]].tolist()
        essay_ids = df[ds["id_col"]].tolist()
        human_scores = df[ds["score_col"]].tolist()
        
        # 2. Load Model
        model = RobertaForEssayScoring().to(device)
        model.load_state_dict(torch.load(ds["model_path"]))
        model.eval()
        
        predictions = []
        
        # 3. Predict in Batches
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
                
                preds = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
                predictions.extend(preds.cpu().numpy())
                
        # 4. Save to CSV
        output_df = pd.DataFrame({
            "essay_id": essay_ids,
            "human_score": human_scores,
            "roberta_score": predictions
        })
        
        output_df.to_csv(ds["output_name"], index=False)
        print(f"Saved {len(output_df)} predictions to {ds['output_name']}")

if __name__ == "__main__":
    generate_predictions()