#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name


GQADIR="$1/gqa/data"

python3 postprocess_scripts/convert_gqa_for_eval.py --src $2 --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python3 eval.py --tier testdev_balanced