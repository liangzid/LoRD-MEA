#!/bin/bash
######################################################################
#1.1.RUN_SFT_TRAINING --- 

# SFT training

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 26 February 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"

export python=${HOME}/anaconda3/envs/align/bin/python3

export CUDA_VISIBLE_DEVICES="0,1,2,3"


export root_dir="${HOME}/alignmentExtraction/"
export SFT_save_dir="${root_dir}SFT_SAVE_CKPTs/"
export from_path="openai-community/gpt2-xl"
export save_path="${SFT_save_dir}TheFirstTimeAttempts/policy-"
# export v_from_path="openai-community/gpt2_large"
export v_from_path="${root_dir}/reward_modeling_anthropic_hh/checkpoint-30000"
export v_save_path="${POD_save_dir}TheFirstTimeAttempts/v-"


$python sft_myself.py\
	--device="cuda:0" \
	--epoch=3 \
	--acc_step=1 \
	--log_step=1 \
	--save_step=100000 \
	--LR="3e-5" \
	--batch_size=1 \
	--task="none set yet" \
	--max_length=1024 \
	--from_path=$from_path \
	--save_path=$save_path 


echo "RUNNING 1.1.run_SFT_training.sh DONE."
# 1.1.run_SFT_training.sh ends here
