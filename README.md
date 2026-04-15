# Analyzing and Mitigating Demographic Biases in Transformer-Based AES

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> 🚧 **This repository is under active development.** Code, results, and model weights will be added as the project progresses.

Transformer-based Automated Essay Scoring (AES) systems risk inheriting and amplifying the demographic biases present in their pretraining data. When deployed at scale, these biases can systematically disadvantage students by race, gender, or language background — with real consequences for grades and future opportunities.

This project extends [Kwako & Ormerod (BEA 2024)](https://aclanthology.org/2024.bea-1.7/), who showed that XLNet fine-tuned on PERSUADE amplifies marginal human score differences across race and gender, and that demographic information is implicitly encoded in model hidden states. We make two contributions: a **cross-model, cross-corpus bias analysis** and a **fairness-aware AES model** incorporating in-training debiasing techniques.

---

## Available Models

> 🚧 Model weights coming soon. The table below will be updated upon release.

| Model | Base | Mitigation Strategy | PERSUADE 2.0 QWK | ASAP 2.0 QWK |
|---|---|---|---|---|
| `[MODEL NAME]-GRL` | DeBERTa-v3 | Adversarial (GRL) | — | — |
| `[MODEL NAME]-OrthoProj` | DeBERTa-v3 | Orthogonal Projection | — | — |
| Baseline (no debiasing) | DeBERTa-v3 | — | — | — |

<!-- TODO: Fill in QWK scores and model names after experiments complete -->

---

## Quick Usage

> 🚧 Installation and usage instructions coming soon.

```python
# Placeholder — will be updated once model code is finalized
from aes_debiasing import load_model

model = load_model('[MODEL NAME]-GRL')
score = model.predict("Essay text goes here...")
print(score)  # Returns holistic score
```

---

## Approach

### Contribution 1 — Cross-Model Bias Analysis
We replicate the Kwako & Ormerod bias analysis pipeline across BERT, XLNet, and DeBERTa on both **PERSUADE 2.0** and **ASAP 2.0**, assessing whether demographic scoring disparities are systemic across transformer architectures or artifacts of specific model-data combinations.

### Contribution 2 — In-Training Bias Mitigation
We propose **[MODEL NAME]**, which incorporates two fairness-aware training strategies:

- **Adversarial debiasing via Gradient Reversal Layer (GRL)** — penalizes the model for encoding demographic information in its representations ([Zhang et al., 2018](https://dl.acm.org/doi/10.1145/3278721.3278779); [Han et al., 2021](https://aclanthology.org/2021.findings-acl.41/))
- **Orthogonal projection on demographic subspaces** — geometrically removes demographic signals from hidden representations ([Bolukbasi et al., 2016](https://proceedings.neurips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html))

Both interventions target the root cause of bias — the implicit encoding of demographic features within the scoring model — rather than applying post-hoc score corrections.

---

## Datasets

This project uses ASAP 2.0 and PERSUADE 2.0. Data is not included in the repo.

| Dataset | Size | Demographic Metadata | Access |
|---|---|---|---|
| [PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) | 25,000+ essays | Race, gender, ELL, SES | Public |
| [ASAP 2.0](https://github.com/scrosseye/ASAP_2.0) | ~25,000 essays | Race, gender, ELL | Public |

To download:

```bash
bash data/download_data.sh
```

This will create a `DATA/` directory (gitignored) with the following structure:
- `DATA/ASAP/train/` and `DATA/ASAP/test/`
- `DATA/PERSUADE/train/` and `DATA/PERSUADE/test/`

---

## Evaluation

Model performance is measured using the three standard AES metrics from [Williamson et al. (2012)](https://doi.org/10.1111/j.1745-3992.2011.00223.x):

- **QWK** — Quadratic Weighted Kappa
- **SMD** — Standardized Mean Difference
- **Exact Agreement**

Fairness is evaluated using demographic parity and equalized odds across race and gender subgroups.

---

## Results

> 🚧 Experiments in progress. Results table will be populated here.

<!-- TODO: Add results table comparing baseline vs GRL vs OrthoProj across subgroups -->

---

## Baselines

| Paper | Model | Dataset |
|---|---|---|
| [Ormerod & Kehat (2025)](https://aclanthology.org/2025.aimecon-main.5) | DeBERTa, XLNet, Longformer, Mamba | ASAP 2.0 |
| [Abdullah et al. / FairGrade (2026)](https://sci-cult.net/index.php/cult/article/view/3738/2210) | BERT + GRL | ASAP, TOEFL11 |
| [Fan & Yun (2026)](https://arxiv.org/abs/2601.16724) | DeBERTa-v3 + contrastive learning | ASAP 2.0, ELLIPSE |

---

## Repository Structure

```
.
├── data/               # Data loading and preprocessing scripts
├── models/             # Model implementations
├── experiments/        # Training and evaluation scripts
├── results/            # Output scores and fairness metrics
└── paper/              # LaTeX source (Overleaf)
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{li-ngo-tamil-2026-aes-bias,
  author  = {Rina Li and Tom Ngo and Karthik Tamil},
  title   = {Analyzing and Mitigating Demographic Biases in Transformer-Based Automated Essay Scoring},
  year    = {2026},
  school  = {Santa Clara University}
}
```

---

## Contact

For questions about this repository, please open an issue. For other inquiries, contact:
[eli3@scu.edu](mailto:eli3@scu.edu) · [pngo2@scu.edu](mailto:pngo2@scu.edu) · [ktamil@scu.edu](mailto:ktamil@scu.edu)