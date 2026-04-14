# Analyzing and Mitigating Demographic Biases in Transformer-Based Automated Essay Scoring
by Karthik Tamil, Rina Li, Tom Ngo

## Model Description:
Recent advances in Large Language Models (LLMs) have enabled strong performance across many NLP domains, including text classification. Automated Essay Scoring (AES) can be approached as a text classification task well-suited for these models. However, LLMs are pretrained on large human-written corpora that reflect real-world societal inequities, making them susceptible to inheriting and amplifying demographic biases during fine-tuning and inference. Deploying LLMs with such biases at scale for AES can detrimentally impact grades and subsequently harm the future prospects of numerous students, especially those within impacted communities \citep{kwako-ormerod-2024-language, hardt2016equality}. It is therefore important to ensure that LLMs used for AES are ethical---that grading biases are minimized, or eliminated if possible, before these systems are widely deployed in education.

Prior work has begun to examine demographic bias in AES. \citet{kwako-ormerod-2024-language} demonstrated that XLNet fine-tuned on the PERSUADE corpus amplifies marginal human score differences across race and gender, favoring White students over Black students and female students over male students, and found that demographic information is implicitly embedded in the model's hidden states. Concurrent work by \citet{schaller-etal-2024-fairness} extended these findings to German-language essays, while \citet{yamashita2025racial} identified racial and ethnic bias in GPT-4o's zero-shot scoring on ELL essays. Despite this growing body of evidence, two critical gaps remain. First, existing bias analyses are largely confined to single models and single datasets, leaving open the question of whether observed bias patterns are systemic across transformer architectures or artifacts of specific model-data combinations. Second, and more importantly, no prior AES work has experimentally validated bias mitigation techniques---the original paper itself stops at measurement, explicitly deferring debiasing to future work.

In this paper, we extend \citet{kwako-ormerod-2024-language} along both of these dimensions. We first replicate their bias analysis pipeline across a broader set of transformer-based models on both the PERSUADE 2.0 and ASAP 2.0 corpora, allowing us to assess whether demographic scoring disparities are model-specific or systemic. We then investigate two bias mitigation strategies at the training stage: adversarial debiasing via a Gradient Reversal Layer \citep{zhang2018mitigating, han-etal-2021-decoupling}, which penalizes the scoring model for encoding demographic information, and orthogonal projection on demographic subspaces \citep{bolukbasi2016man}, which geometrically removes demographic signals from the model's hidden representation. Together, rather than applying post-hoc corrections, these interventions target the source of bias identified by \citet{kwako-ormerod-2024-language}---the implicit encoding of demographic features in the scoring model.

## Installation Instructions:
To be added.
## Usage Instructions: 
To be added.
## Expected Output: 
To be added.
## Member Contributions: 
Equal contribution. Detailed contributions to be added.

Testing
