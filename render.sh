#!/bin/bash

cd slides
quarto render index.qmd
python3 -m http.server 5000
