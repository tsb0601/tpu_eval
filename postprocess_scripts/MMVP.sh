#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/eval_MMVP.py \
    --annotation-file $1/MMVP/test_questions.jsonl \
    --result-file $2