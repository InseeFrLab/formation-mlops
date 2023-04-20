#!/bin/bash

cd content
quarto render index.qmd
python3 -m http.server 5000
