#!/bin/bash

# $1: benchmark folder
# $2: answers file
# $3: output name
# $4: openai_api_key

mkdir -p ./results/llava-bench-in-the-wild/reviews

python3 postprocess_scripts/eval_gpt_review_bench.py \
    --question $1/llava-bench-in-the-wild/questions.jsonl \
    --context $1/llava-bench-in-the-wild/context.jsonl \
    --rule $1/llava-bench-in-the-wild/rule.json \
    --answer-list \
        $1/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $2 \
    --output \
        ./results/llava-bench-in-the-wild/reviews/$3.jsonl \
    --api-key $4

python3 postprocess_scripts/summarize_gpt_review.py -f ./results/llava-bench-in-the-wild/reviews/$3.jsonl