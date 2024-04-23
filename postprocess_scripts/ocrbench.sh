#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name


python postprocess_scripts/eval_ocrbench.py \
    --annotation-file $1/OCRBench/test_questions.jsonl \
    --result-file $2