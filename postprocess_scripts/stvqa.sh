#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python postprocess_scripts/convert_stvqa_for_submission.py \
    --result-file $2 \
    --result-upload-file ./results/stvqa/answers_upload/$3.json