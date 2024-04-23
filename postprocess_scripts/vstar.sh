#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name


python3 postprocess_scripts/eval_vstar.py \
    --annotation-file $1/vstar/test_questions.jsonl \
    --result-file $2