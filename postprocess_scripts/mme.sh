#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name

cp $2 $1/MME/answers/$3.jsonl

cd $1/MME 

python3 convert_answer_to_mme.py --experiment $3

cd eval_tool

python3 calculation.py --results_dir answers/$3