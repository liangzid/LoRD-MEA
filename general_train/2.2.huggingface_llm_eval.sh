#!/bin/bash
######################################################################
#2.2.HUGGINGFACE_LLM_EVAL ---

# Evaluate the results with huggingface's LLM leaderboard!

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  9 June 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="1"
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/general_train/ckpts/boring_test/"
# export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
# export from_path="meta-llama/Llama-2-13b-chat-hf"
export from_path="Vezora/Mistral-22B-v0.1"
# export from_path="Vezora/Mistral-22B-v0.1"
export pmp=$from_path
export eval=${HOME}/anaconda3/envs/align/bin/lm_eval

# export train_taskls=("LoRD-VIII")
export train_task="LoRD-VI"
# # export save_path="${POD_save_dir}NewTemperature${train_task}NewLoss"
# export save_path="${POD_save_dir}NewTemperatureNewTau22B${train_task}NewLoss"

# export ckpt_ls=("${POD_save_dir}NewTemperatureNewTau13BvanillaNewLoss___finally")
# export ckpt_ls=("${POD_save_dir}NewTemperatureNewTau13BLoRD-VIINewLoss___period500" "${POD_save_dir}NewTemperatureNewTau13BvanillaNewLoss___finally")
# export ckpt_ls=("${POD_save_dir}NewTemperatureNewTau22BLoRD-VIIINewLoss___period500")
export ckpt_ls=("${POD_save_dir}NewTemperatureNewTau22BLoRD-IXNewLoss___period500")
# export ckpt_ls=("${POD_save_dir}NewTemperatureNewTau22BvanillaNewLoss___finally" "${POD_save_dir}NewTemperatureNewTau22BLoRD-VIIINewLoss___period500")

for fmp in ${ckpt_ls[*]}
do


echo "================================================================"
echo "EVALUATION MODEL: pretrained: ${pmp} lora: ${fmp}"
echo "EVALUATION TASKS: ${qas}"
echo "================================================================"

evaltasks=arc_challenge
export fewshot_number=25
$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks ${evaltasks}\
    --num_fewshot=${fewshot_number}\
    --device cuda\
    --batch_size auto

export evaltasks=hellaswag
export fewshot_number=10
$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks ${evaltasks}\
    --num_fewshot=${fewshot_number}\
    --device cuda\
    --batch_size auto

export evaltasks=truthfulqa
$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks ${evaltasks}\
    --device cuda\
    --batch_size auto

sleep 120

export evaltasks=mmlu
export fewshot_number=5
$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks ${evaltasks}\
    --num_fewshot=${fewshot_number}\
    --device cuda\
    --batch_size auto:5

export evaltasks=winogrande
export fewshot_number=5
$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks ${evaltasks}\
    --num_fewshot=${fewshot_number}\
    --device cuda\
    --batch_size auto

export evaltasks=gsm8k
export fewshot_number=5
$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks ${evaltasks}\
    --num_fewshot=${fewshot_number}\
    --device cuda\
    --batch_size auto

done


# # export pmp="meta-llama/Meta-Llama-3-8B-Instruct"
# # export pmp="meta-llama/Llama-2-13b-chat-hf"
# export pmp="Vezora/Mistral-22B-v0.1"

# echo "================================================================"
# echo "EVALUATION MODEL: pretrained: ${pmp}"
# echo "EVALUATION TASKS: ${qas}"
# echo "================================================================"

# export evaltasks=arc_challenge
# export fewshot_number=25
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks ${evaltasks}\
#     --num_fewshot=${fewshot_number}\
#     --device cuda\
#     --batch_size auto

# export evaltasks=hellaswag
# export fewshot_number=10
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks ${evaltasks}\
#     --num_fewshot=${fewshot_number}\
#     --device cuda\
#     --batch_size auto

# export evaltasks=truthfulqa
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks ${evaltasks}\
#     --device cuda\
#     --batch_size auto

# sleep 120

# export evaltasks=mmlu
# export fewshot_number=5
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks ${evaltasks}\
#     --num_fewshot=${fewshot_number}\
#     --device cuda\
#     --batch_size auto

# export evaltasks=winogrande
# export fewshot_number=5
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks ${evaltasks}\
#     --num_fewshot=${fewshot_number}\
#     --device cuda\
#     --batch_size auto

# export evaltasks=gsm8k
# export fewshot_number=5
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks ${evaltasks}\
#     --num_fewshot=${fewshot_number}\
#     --device cuda\
#     --batch_size auto


echo "RUNNING 2.2.huggingface_llm_eval.sh DONE."
# 2.2.huggingface_llm_eval.sh ends here
