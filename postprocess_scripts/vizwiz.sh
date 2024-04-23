#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/convert_vizwiz_for_submission.py \
    --annotation-file $1/vizwiz/llava_test.jsonl \
    --result-file $2 \
    --result-upload-file ./results/vizwiz/answers_upload/$3.jsonl