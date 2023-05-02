#!/bin/bash

git clone https://github.com/InseeFrLab/formation-mlops.git
cd formation-mlops
pip install -r requirements.txt
python -m nltk.downloader stopwords
