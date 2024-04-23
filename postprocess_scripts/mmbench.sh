#!/bin/bash

SPLIT="mmbench_dev_20230712"

# $1: benchmark folder
# $2: answers file
# $3: output name
# $4: split

mkdir -p ./results/mmbench/answers_upload/$4

python3 postprocess_scripts/convert_mmbench_for_submission.py \
    --annotation-file $1/mmbench/$4.tsv \
    --result-dir ./results/mmbench/answers/$4 \
    --upload-dir ./results/mmbench/answers_upload/$4 \
    --experiment $3
