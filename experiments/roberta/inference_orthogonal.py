# inference_orthogonal.py
# Generates prediction CSVs from the orthogonally projected models for Bias Analysis

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer
from train_evaluate_roberta import RobertaForEssayScoring, MODEL_NAME, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BATCH_SIZE = 16 
tokenizer  = RobertaTokenizer.from_pretrained(MODEL_NAME)

DEMO_COLUMNS = [
    "gender",
    "race_ethnicity",
    "ell_status", 
    "economically_disadvantaged", 
    "student_disability_status"
]

OUTPUT_DIR = "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/results/roberta/inference_dataset/orthogonal_projection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "PERSUADE",
        "dataset_key": "persuade",
        "test_path": "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv",
        "id_col": "essay_id_comp",
        "text_col": "full_text",
        "score_col": "holistic_essay_score"
    },
    {
        "name": "ASAP",
        "dataset_key": "asap",
        "test_path": "/WAVE/users2/unix/pngo2/Analyzing-Demographic-Biases/DATA/ASAP/test/ASAP_2_Final_github_test.csv",
        "id_col": "essay_id",
        "text_col": "full_text",
        "score_col": "score"
    }
]

# ------------------------------------------------------------------
# Inference Loop
# ------------------------------------------------------------------
def generate_predictions():
    for ds in DATASETS:
        print(f"\n--- Loading Dataset for {ds['name']} ---")
        
        # 1. Load Data
        df = pd.read_csv(ds["test_path"], low_memory=False).dropna(subset=[ds["score_col"], ds["text_col"]])
        df = df.drop_duplicates(subset=[ds["id_col"]])
        
        texts = df[ds["text_col"]].tolist()
        essay_ids = df[ds["id_col"]].tolist()
        human_scores = df[ds["score_col"]].tolist()
        
        for demo in DEMO_COLUMNS:
            print(f"\n--> Running Inference for {ds['name']} (OrthProj: {demo})")
            model_path = f"best_roberta_{ds['dataset_key']}_projected_{demo}.pt"
            output_name = os.path.join(OUTPUT_DIR, f"{ds['dataset_key']}_ortho_{demo}_roberta_predictions.csv")
            
            # 2. Load Model
            model = RobertaForEssayScoring().to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"Could not load {model_path}: {e}")
                continue
                
            model.eval()
            
            predictions = []
            
            # 3. Predict in Batches
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Scoring {ds['name']} ({demo})"):
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
            
            output_df.to_csv(output_name, index=False)
            print(f"Saved {len(output_df)} predictions to {output_name}")

if __name__ == "__main__":
    generate_predictions()
