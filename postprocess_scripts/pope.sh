#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/eval_pope.py \
    --annotation-dir $1/pope/coco \
    --question-file $1/pope/llava_pope_test.jsonl \
    --result-file $2