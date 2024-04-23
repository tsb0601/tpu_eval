#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name
# $4: split

SPLIT="llava_vqav2_mscoco_test-dev2015"

mkdir -p ./results/vqav2/answers_upload/$4

python3 $1/vqav2 --result $2 --dst ./results/vqav2/answers_upload/$4/$3.json