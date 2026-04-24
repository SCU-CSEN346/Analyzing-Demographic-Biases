#!/bin/bash
# Run from repo root: bash data/download_data.sh
# Prerequisites: gdown (pip install gdown), unzip, git

set -e  # exit on any error

BASE="$(cd "$(dirname "$0")/.." && pwd)/DATA"
echo "Setting up data in: $BASE"

mkdir -p "$BASE/ASAP/train" "$BASE/ASAP/test"
mkdir -p "$BASE/PERSUADE/train" "$BASE/PERSUADE/test"

# ── ASAP 2.0 ──────────────────────────────────────────────────────────────
echo "Cloning ASAP 2.0..."
git clone https://github.com/scrosseye/ASAP_2.0 "$BASE/ASAP/tmp"

echo "Unzipping ASAP files (password: asap2_test)..."
unzip -P asap2_test "$BASE/ASAP/tmp/ASAP_2_Final_github_test.zip"  -d "$BASE/ASAP/test/"
unzip -P asap2_test "$BASE/ASAP/tmp/ASAP_2_Final_github_train.zip" -d "$BASE/ASAP/train/"
rm -rf "$BASE/ASAP/tmp"

# ── PERSUADE 2.0 ───────────────────────────────────────────────────────────
echo "Downloading PERSUADE 2.0 (requires gdown)..."
gdown "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135" -O "$BASE/PERSUADE/train/persuade_corpus_2.0_train.csv"

gdown "1K1SIJiG-2zWgMlTzxQeYOcLwOsFaVel1" -O "$BASE/PERSUADE/test/persuade_corpus_2.0_test.zip"
echo "Unzipping PERSUADE test (password: persuade_test)..."
unzip -P persuade_test "$BASE/PERSUADE/test/persuade_corpus_2.0_test.zip" -d "$BASE/PERSUADE/test/"
rm "$BASE/PERSUADE/test/persuade_corpus_2.0_test.zip"

echo "Done. Data ready in $BASE"