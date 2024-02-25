#!/bin/bash
######################################################################
#1.0.RUN_POD_TRAINING ---

# RUN TRAINING on POD algorithms.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 25 February 2024
######################################################################

echo "HOME: ${HOME}"

export python=${HOME}/anaconda3/envs/align/bin/python3

export CUDA_VISIBLE_DEVICES="1,2,3"


export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/"
export from_path="facebook/opt-350m"
export save_path="${POD_save_dir}TheFirstTimeAttempts/policy-"
export v_from_path="facebook/opt-350m"
export v_save_path="${POD_save_dir}TheFirstTimeAttempts/v-"


$python rlhf_train.py\
	--device="cuda:0" \
	--epoch=1 \
	--period_num=1 \
	--acc_step=1 \
	--log_step=1 \
	--save_step=100000 \
	--LR="3e-5" \
	--lambdaa=0.95 \
	--lambda1=1.0 \
	--lambda2=1.0 \
	--batch_size=1 \
	--task="none set yet" \
	--max_length=1024 \
	--from_path=$from_path \
	--save_path=$save_path \
	--v_from_path=$v_from_path \
	--v_save_path=$v_save_path 


echo "RUNNING 1.0.run_POD_training.sh DONE."
# 1.0.run_POD_training.sh ends here
