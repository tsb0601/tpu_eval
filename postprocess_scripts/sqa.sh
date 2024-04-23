#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/eval_science_qa.py \
    --base-dir $1/scienceqa \
    --result-file $2 \
    --output-file ./results/scienceqa/answers/$3\_output.jsonl \
    --output-result ./results/scienceqa/answers/$3\_result.json