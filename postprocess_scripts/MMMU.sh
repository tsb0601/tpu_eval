#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

 

MMMUDIR="$1/MMMU/eval"

cp $2 $MMMUDIR/tmp.json

cd $MMMUDIR
python3 main_eval_only.py --output_path tmp.json