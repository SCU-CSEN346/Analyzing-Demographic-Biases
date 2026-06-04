---
title: Analyzing Demographic Biases Demo
emoji: 📝
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Analyzing Demographic Biases — Interactive Demo

An interactive companion to our poster on demographic bias in automated essay
scoring (AES). Pick one of 16 curated student essays and see how three
transformer models (XLNet, RoBERTa, Longformer) score it — before and after
**GRL** adversarial debiasing — next to the human (teacher) score. An aggregate
chart shows the standardized score gap (SMD) per demographic attribute across
the full test set. The takeaway: across all three models, GRL fails to reduce
demographic bias without sacrificing scoring quality.

All predictions are precomputed in `demo_predictions.csv` (no live model
inference), so the Space runs comfortably on free-tier hardware.

**Authors:** Rina Li, Tom Ngo, Karthik Tamil
**Advisor:** Dr. Oana Ignat · Santa Clara University
**Code:** https://github.com/SCU-CSEN346/Analyzing-Demographic-Biases
