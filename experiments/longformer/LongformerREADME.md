# Analyzing and Mitigating Demographic Biases in Transformer-Based AES
SCU CSEN 346 — Rina Li, Tom Ngo, Karthik Tamil

Code and experiments for replicating and extending the bias analysis of Kwako & Ormerod (2024)
across XLNet, RoBERTa, and Longformer on the PERSUADE 2.0 and ASAP 2.0 corpora.

---

## Repository Structure

```
longformer/
├── longformer.py               # Longformer training, evaluation, GRL debiasing
├── orthogonal_projection.py    # Post-hoc orthogonal projection debiasing
├── probe.py                    # Demographic probing (frozen backbone + linear head)
│
├── PERSUADE/
│   ├── persuade_corpus_2.0_train.csv
│   └── persuade_corpus_2.0_test.csv
│
├── ASAP/
│   ├── ASAP_2_Final_github_train.csv
│   ├── ASAP_2_Final_github_test.csv
│   └── ASAP2_train_sourcetexts.csv
│
├── pt/                         # Saved model checkpoints
│   ├── best_longformer_persuade_base.pt
│   ├── best_longformer_persuade_grl.pt
│   ├── best_longformer_persuade_projected_base.pt
│   ├── best_longformer_persuade_projected_grl.pt
│   ├── best_longformer_asap_base.pt
│   ├── best_longformer_asap_grl.pt
│   ├── best_longformer_asap_projected_base.pt
│   └── best_longformer_asap_projected_grl.pt
│
├── cache/                      # Auto-generated — tokenization and feature caches
├── checkpoints/                # Auto-generated — per-prompt model checkpoints
│   ├── persuade/
│   └── asap/
├── logs/                       # SLURM job output logs
└── slurm/                      # Batch scripts
```

---

## Installation

### On WAVE HPC

```bash
module purge
module load Anaconda3

conda create --prefix /WAVE/projects/CSEN-346-Sp26/Analyzing-Demographic-Biases/conda-envs/aes_env python=3.10 -y
source activate /WAVE/projects/CSEN-346-Sp26/Analyzing-Demographic-Biases/conda-envs/aes_env

pip install transformers==4.40.0 torch==2.0.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
pip install pandas scikit-learn numpy
```

### Local

```bash
pip install transformers torch pandas scikit-learn numpy
```

---

## Dataset Setup

Download the datasets and place them in the correct folders:

- **PERSUADE 2.0:** https://github.com/scrosseye/persuade_corpus_2.0
- **ASAP 2.0:** https://github.com/scrosseye/ASAP_2.0

```bash
mkdir -p PERSUADE ASAP pt cache logs checkpoints/persuade checkpoints/asap
```

Move files into place:
```bash
mv persuade_corpus_2.0_train.csv PERSUADE/
mv persuade_corpus_2.0_test.csv  PERSUADE/
mv ASAP_2_Final_github_train.csv ASAP/
mv ASAP_2_Final_github_test.csv  ASAP/
mv ASAP2_train_sourcetexts.csv   ASAP/
```

---

## Running Experiments

All commands below use the full Python path for WAVE. Replace with `python` if running locally.

```bash
PYTHON=/WAVE/projects/CSEN-346-Sp26/Analyzing-Demographic-Biases/conda-envs/aes_env/bin/python
```

### 1. Base Scoring (all prompts combined)

```bash
$PYTHON longformer.py --dataset persuade
$PYTHON longformer.py --dataset asap
```

Saves to: `pt/best_longformer_persuade_base.pt`, `pt/best_longformer_asap_base.pt`

---

### 2. Base Scoring (per-prompt, mirrors XLNet setup)

PERSUADE has 15 prompts so it is split into two jobs to fit within the 48hr wall time limit:

```bash
# Prompts 0-7
$PYTHON longformer.py --dataset persuade --per-prompt --prompt-start 0 --prompt-end 8

# Prompts 8-15
$PYTHON longformer.py --dataset persuade --per-prompt --prompt-start 8

# ASAP (7 prompts, fits in one job)
$PYTHON longformer.py --dataset asap --per-prompt
```

Saves per-prompt checkpoints to: `checkpoints/persuade/` and `checkpoints/asap/`

To skip training and re-evaluate from saved checkpoints:
```bash
$PYTHON longformer.py --dataset persuade --per-prompt --skip-train
```

---

### 3. GRL Adversarial Debiasing

```bash
$PYTHON longformer.py --dataset persuade --debias --demo gender
$PYTHON longformer.py --dataset asap     --debias --demo gender
```

Saves to: `pt/best_longformer_persuade_grl.pt`, `pt/best_longformer_asap_grl.pt`

Available `--demo` options: `gender`, `race`, `ell`, `ses`, `disability`

---

### 4. Orthogonal Projection

Must be run after base scoring (step 1). Loads the base checkpoint, projects out the
demographic bias direction, and saves a new checkpoint.

```bash
$PYTHON orthogonal_projection.py --dataset persuade --demo gender
$PYTHON orthogonal_projection.py --dataset asap     --demo gender
```

Saves to: `pt/best_longformer_persuade_projected_base.pt`, `pt/best_longformer_asap_projected_base.pt`

---

### 5. Demographic Probing

Probing tests whether demographic information is linearly recoverable from the model's
hidden states. Run on all four checkpoint types (base, GRL, projected base, projected GRL).

```bash
# Base
$PYTHON probe.py --dataset persuade --model pt/best_longformer_persuade_base.pt
$PYTHON probe.py --dataset asap     --model pt/best_longformer_asap_base.pt

# GRL
$PYTHON probe.py --dataset persuade --model pt/best_longformer_persuade_grl.pt
$PYTHON probe.py --dataset asap     --model pt/best_longformer_asap_grl.pt

# Projected (base)
$PYTHON probe.py --dataset persuade --model pt/best_longformer_persuade_projected_base.pt
$PYTHON probe.py --dataset asap     --model pt/best_longformer_asap_projected_base.pt

# Projected (GRL)
$PYTHON probe.py --dataset persuade --model pt/best_longformer_persuade_projected_grl.pt
$PYTHON probe.py --dataset asap     --model pt/best_longformer_asap_projected_grl.pt
```

To run a single demographic only:
```bash
$PYTHON probe.py --dataset asap --model pt/best_longformer_asap_base.pt --demo gender
```

To run a quick test on a small subset:
```bash
$PYTHON probe.py --dataset asap --model pt/best_longformer_asap_base.pt --test
```

---

## SLURM (WAVE HPC)

Submit all probe jobs simultaneously:
```bash
for f in slurm/run_probe_*.sh; do sed -i 's/\r//' $f && sbatch $f; done
```

Check job status:
```bash
squeue --user=$USER
```

Monitor a running job:
```bash
tail -f logs/<job_name>_<jobid>.out
```

Recommended partitions: `gpu` (4 nodes, 48hr limit) or `condo` (bio01, bio03 have 8x Ampere GPUs).

---

## Expected Output

### Scoring (`longformer.py`)
```
Epoch 01 | Train Loss: 1724.54 | Dev Loss: 121.81 | Dev QWK: 0.8503
...
--- PERSUADE Test Results ---
  MSE:             781.9488
  QWK:             0.8509
  Exact Agreement: 0.6586
  SMD:             0.1358
```

### Probing (`probe.py`)
```
============================================================
  Summary: pt/best_longformer_persuade_base.pt
============================================================
  gender                                   κ = 0.2607
  race (White)                             κ = 0.3039
  ell                                      κ = 0.4861
  ses                                      κ = 0.3334
  disability                               κ = 0.3142
```

κ < 0.2: near-chance encoding | 0.2 ≤ κ < 0.4: minimal | κ ≥ 0.4: meaningful encoding
