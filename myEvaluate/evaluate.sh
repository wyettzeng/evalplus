#!/bin/bash

MAX_RETRY=3
models=(
    "codellama_instruct_7b"
    "mistral_instruct_v3_7b"
    "llama3_instruct_8b"
    "qwen_coder_2.5_7b"
    "nxcode_cq_orpo_7b"
    "code_qwen_v1.5_7b"
    # "deepseek_coder_v2_16b"
)
model_name=${models[$1]}

ct=0
until python myEvaluate/evaluate.py --model_name=$model_name --n_samples=1
do
    ct=$((ct + 1))
    echo "failed ${ct} times"
    if [ $ct -ge $MAX_RETRY ]; then
        break
    fi
done

ct=0
until python myEvaluate/evaluate.py --model_name=$model_name --n_samples=16
do
    ct=$((ct + 1))
    echo "failed ${ct} times"
    if [ $ct -ge $MAX_RETRY ]; then
        break
    fi
done