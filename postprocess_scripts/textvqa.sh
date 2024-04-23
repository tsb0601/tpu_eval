#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/eval_textvqa.py \
    --annotation-file $1/textvqa/TextVQA_0.5.1_val.json \
    --result-file $2