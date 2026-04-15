# data/download_data.sh
#!/bin/bash
mkdir -p DATA/ASAP/train DATA/ASAP/test
mkdir -p DATA/PERSUADE/train DATA/PERSUADE/test

# ASAP 2.0 — from original authors
git clone https://github.com/scrosseye/ASAP_2.0 DATA/ASAP/tmp
mv DATA/ASAP/tmp/*.csv DATA/ASAP/train/
rm -rf DATA/ASAP/tmp

# PERSUADE 2.0 — from Google Drive
pip install gdown
gdown "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135" -O DATA/PERSUADE/train/persuade_corpus_2.0_train.csv
gdown "1K1SIJiG-2zWgMlTzxQeYOcLwOsFaVel1" -O DATA/PERSUADE/test/persuade_corpus_2.0_test.csv

echo "Done. Data ready in DATA/"
