# Testing Fairness Interventions in Automated Essay Scoring

**CSEN 364 · Santa Clara University · Authors: Rina Li, Tom Ngo, Karthik Tamil · Advisor: Dr. Oana Ignat**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[Interactive demo](https://huggingface.co/spaces/RinaL/Analyzing-Demographic-Biases-Demo) · [Paper (PDF)](paper/AES-Paper.pdf) · [Poster (PDF)](paper/AES-Poster.pdf)

---

## Summary

Transformer-based Automated Essay Scoring (AES) systems can amplify demographic differences from training data — over-predicting essays from advantaged writers and under-predicting from marginalized ones. Two debiasing methods are commonly proposed: **adversarial debiasing** via Gradient Reversal Layer (GRL), and **orthogonal projection** of demographic directions from hidden states.

We tested both methods across three transformer architectures (XLNet, RoBERTa, Longformer) on PERSUADE 2.0 and ASAP 2.0.

**Finding: across three models and two methods, no approach reduced demographic bias while preserving scoring quality.** Either bias did not move (XLNet, all methods), or apparent bias reduction coincided with QWK loss (RoBERTa GRL), or projection collapsed the scoring pipeline (RoBERTa and Longformer projection: QWK 0.84 → 0.04 and 0.87 → 0.07). The success quadrant — bias reduced AND quality preserved — was empty on every attribute.

Bias is not a removable signal layered on top of scoring. It rides on construct-relevant features (length, complexity, lexical diversity) the model legitimately uses for scoring. Representation-side interventions cannot fix prediction-side bias of this type.

This project extends [Kwako & Ormerod (BEA 2024)](https://aclanthology.org/2024.bea-1.7/), who measured bias amplification in XLNet on PERSUADE.

---

## Interactive demo

Pick an essay, see baseline and post-GRL predictions across all three models with an aggregate SMD chart:

**[https://huggingface.co/spaces/RinaL/Analyzing-Demographic-Biases-Demo](https://huggingface.co/spaces/RinaL/Analyzing-Demographic-Biases-Demo)**

Source in `demo/`.

---

## Results

### Scoring quality (baseline QWK)

| | XLNet | RoBERTa | Longformer |
|---|---|---|---|
| PERSUADE | 0.755 | 0.837 | 0.865 |
| ASAP | 0.738 | — | — |

### GRL — bias change (PERSUADE)

Per-prompt mean |t| = |coef/SE| for the group coefficient regressing predicted score on group + true score, base → post. Lower magnitude = bias closer to zero. Negative Δ = bias reduced.

| Attribute | XLNet | RoBERTa | Longformer |
|---|---|---|---|
| Gender | 1.82 → 1.72 (−0.11) | 4.58 → 4.21 (−0.37) | 5.55 → 5.43 (−0.12) |
| Race (Black–White) | 2.96 → 2.74 (−0.22) | 6.29 → 4.51 (−1.78) | 6.04 → 8.05 (+2.00) |
| ELL | 7.43 → 7.23 (−0.20) | 7.22 → 5.12 (−2.11) | 6.86 → 6.29 (−0.57) |
| Economic | 4.08 → 4.09 (+0.01) | 8.06 → 6.51 (−1.55) | 8.91 → 10.56 (+1.65) |
| Disability | 3.58 → 3.05 (−0.53) | 4.83 → 4.24 (−0.59) | 4.78 → 5.01 (+0.23) |
| **ΔQWK** | **+0.023** | **−0.025** | **+0.085** |

- XLNet GRL: bias unchanged, QWK preserved
- RoBERTa GRL: bias reduced, QWK dropped
- Longformer GRL: bias backfires on race / economic / disability, QWK rose

### Orthogonal projection (PERSUADE)

| Model | QWK base → projected | Outcome |
|---|---|---|
| XLNet | 0.755 → 0.755 | preserved; demographic encoding (κ) unchanged |
| RoBERTa | 0.837 → 0.043 | model collapsed |
| Longformer | 0.865 → 0.069 | model collapsed |

Projection collapsed RoBERTa and Longformer because in their CLS representations the demographic direction overlaps with the scoring direction. Removing one removes the other.

### Mediation (PERSUADE)

Bias change after controlling for surface features (length, sentence count, word length, Flesch-Kincaid, lexical diversity). Negative numbers mean the gap closed; positive means it widened.

| Attribute | Length only | All features | Pattern |
|---|---|---|---|
| Gender | +68% | +99% | Gap fully explained by writing features |
| ELL | −66% | −118% | Gap **widens** when features held constant |
| Economic | −29% | −18% | Gap widens |
| Disability | −51% | −57% | Gap widens |

Gender bias is entirely length-mediated. ELL, economic, and disability gaps remain after controlling for surface features — the residual penalty is distributed and non-linear, not erasable by removing one direction.

---

## Approach

### Cross-model bias analysis

We replicate and extend [Kwako & Ormerod (BEA 2024)](https://aclanthology.org/2024.bea-1.7/) — who measured bias amplification in XLNet on PERSUADE — by testing **three architectures** (XLNet, RoBERTa, Longformer) on **two corpora** (PERSUADE 2.0, ASAP 2.0), across **five demographic attributes** (gender, race, ELL, economic disadvantage, disability).

### Mitigation methods

1. **Adversarial debiasing (GRL)** — joint multi-attribute gradient reversal with a 2-layer MLP adversary per attribute, λ_max = 0.5, γ = 10. The adversary tries to predict demographic group from the encoder's hidden state; gradient reversal pushes the encoder to make this prediction harder. ([Zhang et al., 2018](https://dl.acm.org/doi/10.1145/3278721.3278779); [Han et al., 2021](https://aclanthology.org/2021.findings-acl.41/))

2. **Orthogonal projection** — at inference time, projects the hidden representation onto the orthogonal complement of the demographic direction (estimated from PCA of group-mean differences) before scoring. ([Bolukbasi et al., 2016](https://proceedings.neurips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html))

### Evaluation

- **Quality**: Quadratic Weighted Kappa (QWK) — model–human agreement
- **Bias**: weighted standardized regression z, per-prompt — does the model widen the human-score gap?
- **Representation**: probing κ — is demographic information still recoverable from frozen hidden states?

---

## Datasets

| Dataset | Size | Demographic columns | Source |
|---|---|---|---|
| [PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) | 25,992 essays, 15 prompts | gender, race, ELL, SES, disability | Public |
| [ASAP 2.0](https://github.com/scrosseye/ASAP_2.0) | 24,728 essays, 7 prompts | gender, race, ELL, SES, disability | Public |

Data is not included in this repository. Download via:

```bash
bash data/download_data.sh
```

Creates `DATA/{ASAP,PERSUADE}/{train,test}/`.

---

## Repository structure

```
.
├── data/                  # Data download script
│   └── download_data.sh
├── experiments/
│   ├── xlnet/             # Fine-tuning, GRL, projection, probing — XLNet
│   ├── roberta/           # Same — RoBERTa
│   └── longformer/        # Same — Longformer
├── results/
│   ├── xlnet/             # Per-corpus results JSON, predictions CSVs, probing
│   ├── roberta/
│   └── longformer/
├── demo/                  # Snapshot of Hugging Face Space (Gradio app + curated CSV)
├── paper/                 # Paper PDF + poster PDF
└── README.md
```

---

## Reproducibility

To reproduce XLNet PERSUADE baseline:

```bash
DATASET=PERSUADE DATA_BASE=./DATA python experiments/xlnet/train_xlnet.py

DATASET=PERSUADE DATA_BASE=./DATA python experiments/xlnet/dump_predictions.py \
    --models results/xlnet/baseline_replication/checkpoints \
    --out results/xlnet/baseline_replication/test_predictions.csv

DATASET=PERSUADE DATA_BASE=./DATA python experiments/xlnet/compute_weighted_smd.py \
    --results results/xlnet/baseline_replication/results.json \
    --predictions results/xlnet/baseline_replication/test_predictions.csv \
    --out-dir results/xlnet/baseline_replication \
    --label baseline
```

For ASAP: replace `DATASET=PERSUADE` with `DATASET=ASAP`. For GRL: use `train_xlnet_grl_joint.py` with `PATIENCE=6`. RoBERTa and Longformer use parallel scripts in `experiments/roberta/` and `experiments/longformer/`.

Hardware: experiments ran on RTX 4090 and H100. Single-GPU XLNet PERSUADE baseline: ~6–8 h. GRL joint training with PATIENCE=6: ~8–10 h per corpus.

---

## Related work

| Paper | Models | Datasets |
|---|---|---|
| [Kwako & Ormerod (2024)](https://aclanthology.org/2024.bea-1.7/) | XLNet | PERSUADE 2.0 |
| [Ormerod & Kehat (2025)](https://aclanthology.org/2025.aimecon-main.5) | DeBERTa, XLNet, Longformer, Mamba | ASAP 2.0 |
| [Abdullah et al. / FairGrade (2026)](https://sci-cult.net/index.php/cult/article/view/3738/2210) | BERT + GRL | ASAP, TOEFL11 |
| [Fan & Yun (2026)](https://arxiv.org/abs/2601.16724) | DeBERTa-v3 + contrastive learning | ASAP 2.0, ELLIPSE |

---

## Contributions

**Rina Li** — XLNet implementation (fine-tuning, GRL, orthogonal projection, hidden states probing); cross-model analysis pipeline; demo design and deployment; results aggregation; paper related work and evaluation sections.

**Tom Ngo** — RoBERTa implementation (fine-tuning, GRL, orthogonal projection, hidden states probing); data preparation and tokenization; paper introduction, model architecture, and methods sections.

**Karthik Tamil** — Longformer implementation (fine-tuning, GRL, orthogonal projection, hidden states probing); HuggingFace dataset preparation; paper abstract and dataset sections; results section.

All authors contributed to cross-model analysis, paper writing, and findings interpretation.

---

## Citation

```bibtex
@unpublished{li-ngo-tamil-2026-aes-bias,
  author    = {Rina Li and Tom Ngo and Karthik Tamil},
  title     = {Testing Fairness Interventions in Automated Essay Scoring},
  year      = {2026},
  school    = {Santa Clara University},
  note      = {Course project, CSEN 364}
}
```

---

## License

MIT — see `LICENSE`.

## Contact

[eli3@scu.edu](mailto:eli3@scu.edu) · [pngo2@scu.edu](mailto:pngo2@scu.edu) · [ktamil@scu.edu](mailto:ktamil@scu.edu)