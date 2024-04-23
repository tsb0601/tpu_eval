#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

mkdir -p ./results/mm-vet/results

python3 postprocess_scripts/convert_mmvet_for_eval.py \
    --src ./results/mm-vet/answers/$3.jsonl \
    --dst ./results/mm-vet/results/$3.json