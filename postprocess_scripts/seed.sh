#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/convert_seed_for_submission.py \
    --annotation-file $1/seed_bench/SEED-Bench.json \
    --result-file $2 \
    --result-upload-file ./results/seed_bench/answers_upload/$3.jsonl