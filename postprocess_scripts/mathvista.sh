#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name
# $4: openai_api_key

python3 postprocess_scripts/convert_mathvista_for_eval.py \
    --output_file $2 \
    --api-key $4

python3 postprocess_scripts/eval_mathvista.py \
    --output_file $2 \
    --gt_file $1/mathvista/annot_testmini.json