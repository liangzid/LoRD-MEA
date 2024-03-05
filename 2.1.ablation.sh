#!/bin/bash
######################################################################
#2.1.ABLATION ---

# Ablation Study of the paper.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  4 March 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"

export python=${HOME}/anaconda3/envs/align/bin/python3

# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/AblationExperiment/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"

export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")

export task="cola"
export train_task="lord"
# export train_task="kd"
export epoch=2
export period=3
export beta=1.0
export temperature=0.8
export batch_size=1

export use_old_logits=0
export use_vic_logits=1
export use_kld=1
export train_num=100

export save_path="${POD_save_dir}Methods011${train_task}_${msl}${task}"

$python pod_train.py\
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--acc_step=1 \
	--log_step=1 \
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


export use_old_logits=1
export use_vic_logits=1
export use_kld=0

export save_path="${POD_save_dir}Methods111${train_task}_${msl}${task}"

$python pod_train.py\
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--acc_step=1 \
	--log_step=1 \
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

export use_old_logits=1
export use_vic_logits=0
export use_kld=0

export save_path="${POD_save_dir}Methods100${train_task}_${msl}${task}"

$python pod_train.py\
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--acc_step=1 \
	--log_step=1 \
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

export use_old_logits=0
export use_vic_logits=1
export use_kld=0

export save_path="${POD_save_dir}Methods010${train_task}_${msl}${task}"

$python pod_train.py\
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--acc_step=1 \
	--log_step=1 \
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


echo "RUNNING 2.1.ablation.sh DONE."
# 2.1.ablation.sh ends here
