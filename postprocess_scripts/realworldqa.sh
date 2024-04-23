#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name


python3 postprocess_scripts/eval_realworldqa.py \
    --result-file $2