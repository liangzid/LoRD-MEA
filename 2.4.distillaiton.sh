#!/bin/bash
######################################################################
#2.4.DISTILLAITON --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  5 March 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################


echo "HOME: ${HOME}"

export python=${HOME}/anaconda3/envs/align/bin/python3

# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/vary_trainNum/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"

export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")

export task="cola"
# export train_task="lord"
# export train_task="kd"
export train_task="vanilla"
export epoch=3
export period=10
export beta=1.0
export temperature=2.0
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=1

export train_num=16

# export train_task="kd"
export save_path="${POD_save_dir}${task}${train_num}_3Epoch${train_task}_${msl}${task}"

$python lord_train.py\
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--acc_step=1 \
	--log_step=50 \
	--save_step=100000 \
	--train_num=$train_num \
	--LR="3e-5" \
	--beta=$beta \
	--temperature=$temperature \
	--batch_size=$batch_size \
	--task=$train_task \
	--use_old_logits=$use_old_logits\
	--use_vic_logits=$use_vic_logits\
	--use_kld=$use_kld\
	--max_length=$msl \
	--dataset_task=$task \
	--from_path=$from_path \
	--save_path=$save_path











echo "RUNNING 2.4.distillaiton.sh DONE."
# 2.4.distillaiton.sh ends here
