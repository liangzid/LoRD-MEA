#!/bin/bash
######################################################################
#2.1.EVALUTATE_LLMS ---

# Evaluate LLMs with the EAI's harness

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created: 26 April 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# export CUDA_VISIBLE_DEVICES="4,5"
# export CUDA_VISIBLE_DEVICES="6,7"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export save_dir="${root_dir}/general_train/ckpts/boring_test/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"


# export qas=openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq
# export qas=arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k
export qas=arc_challenge,hellaswag,winogrande,gsm8k
# export qas=piqa
export eval=${HOME}/anaconda3/envs/align/bin/lm_eval
export pmp=meta-llama/Meta-Llama-3-8B-Instruct

# LoRD-VI Inference.
export task_ls=("liangzid/claude3_short256")

# export fmp="${root_dir}general_train/ckpts/shorttext/lordvi-explore0.80.9___period300"
# export fmp=${save_dir}longtext2491liangzid/claude3_short256vanilla2121256256___finally
export fmp="${save_dir}NewTemperatureNewLoss___period100"

$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks $qas\
    --device cuda\
    --batch_size auto:4

# ------------------------------------------------------------------
# # Now for long text models
# export fmp="${root_dir}general_train/ckpts/longtext/longtext30001liangzid/claude3_chat3.3kvanilla11212561024___finally"
# # export fmp="${root_dir}general_train/ckpts/longtext/longtext30001liangzid/claude3_chat3.3kLoRD-VI11212561024___3000"

# echo "================================================================"
# echo "EVALUATION MODEL: pretrained: ${pmp} lora: ${fmp}"
# echo "EVALUATION TASKS: ${qas}"
# echo "================================================================"

# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
#     --tasks $qas\
#     --device cuda\
#     --batch_size auto:4


# ## Vanilla LoRA TRAIN on LLAMA3 8B
# export fmp="${save_dir}BORING_TEST___2491liangzid/claude3_short256vanilla3321641024___finally/"
# export fmp=${save_dir}longtext2491liangzid/claude3_short256vanilla3321256256___finally

# echo "================================================================"
# echo "EVALUATION MODEL: pretrained: ${pmp} lora: ${fmp}"
# echo "EVALUATION TASKS: ${qas}"
# echo "================================================================"

# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
#     --tasks $qas\
#     --device cuda\
#     --batch_size auto


# # ORIGINAL LLAMA3 8B EVAL
# $eval --model hf \
#     --model_args pretrained=${pmp},parallelize=True\
#     --tasks $qas\
#     --device cuda\
#     --batch_size auto


echo "RUNNING 2.1.evalutate_llms.sh DONE."
# 2.1.evalutate_llms.sh ends here
