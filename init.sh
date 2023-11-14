#!/bin/bash
GIT_REPO=formation-mlops
GIT_USERNAME=InseeFrLab

git clone https://github.com/$GIT_USERNAME/$GIT_REPO.git
cd $GIT_REPO
pip install -r requirements.txt
python -m nltk.downloader stopwords
