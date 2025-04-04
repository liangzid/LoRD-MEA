#!/bin/bash
######################################################################
#EVAL_MATH_LMEVAL --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2025, ZiLiang, all rights reserved.
# Created:  2 April 2025
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=${HOME}/anaconda3/envs/robench/bin/python3
export root_dir="${HOME}/alignmentExtraction/"

export pretrained_path="meta-llama/Meta-Llama-3-8B"
# export peft=${root_dir}"math_ckpts/math_reasoningopenai/gsm8k641LoRD-VI___period512/"
export peft=${root_dir}"math_ckpts/math_reasoningopenai/gsm8k641LoRD-VI___period256/"
# export peft=${root_dir}"math_ckpts/math_reasoningopenai/gsm8k641LoRD-VI___period512/"

# lm-eval --model hf\
# 	--model_args pretrained=${pretrained_path},peft=${peft_path} \
# 	--tasks gsm8k \
# 	--num_fewshot 0 \
# 	--device cuda \
# 	--limit 500 \
# 	--batch_size 32

# echo "================================================================"

lm-eval --model hf\
	--model_args pretrained=${pretrained_path} \
	--tasks gsm8k \
	--num_fewshot 0 \
	--device cuda \
	--limit 500 \
	--batch_size 32



echo "RUNNING eval_math_lmeval.sh DONE."
# eval_math_lmeval.sh ends here


