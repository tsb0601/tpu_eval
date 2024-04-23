#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

python3 postprocess_scripts/convert_doc_info_for_submission.py \
    --result-file $2 \
    --result-upload-file ./results/docvqa/answers_upload/$3.json