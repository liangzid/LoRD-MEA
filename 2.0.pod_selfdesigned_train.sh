#!/bin/bash
######################################################################
#2.0.POD_SELFDESIGNED_TRAIN ---

# POD training for my self-designed models.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 27 February 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"

export python=${HOME}/anaconda3/envs/align/bin/python3

# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"

export msl=256
export task="cola"
export save_path="${POD_save_dir}pod_style_test_fast${msl}${task}"

$python pod_train.py\
	--device="cuda" \
	--epoch=2 \
	--period_num=3 \
	--acc_step=1 \
	--log_step=1 \
	--save_step=100000 \
	--LR="3e-5" \
	--beta=1.0 \
	--temperature=1.0 \
	--batch_size=1 \
	--task="none set yet" \
	--max_length=$msl \
	--dataset_task=$task \
	--from_path=$from_path \
	--save_path=$save_path

echo "RUNNING 2.0.pod_selfdesigned_train.sh DONE."
# 2.0.pod_selfdesigned_train.sh ends here
