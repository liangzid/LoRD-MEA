#!/bin/bash
######################################################################
#BASH_TEMP ---

# TEMP SCRIPTS for sequential RUNNING.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  6 March 2024
######################################################################

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/vary_period0306"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"
export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
export task_ls=("cs-en" "du-en" "fi-en" "ro-en" "ru-en" "tr-en")
# export task="cs-en"
export task="cs-en"
# export task="sum"
# export train_task="lord"
# export train_task="Complex-lord"
# export train_task="reinforce-lord"
# export train_task="kd"

export train_task="Very--Complex-lord"
# export train_task="Black--Very--Complex-lord"

export epoch=3
export period=3
export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export train_num=100

# export train_task="kd"
export save_path="${POD_save_dir}${task}/${train_task}_${msl}${task}_test"

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
	--use_entropy=$use_entropy\
	--max_length=$msl \
	--dataset_task=$task \
	--from_path=$from_path \
	--save_path=$save_path

echo "EVERYTHING DONE."
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"


echo "RUNNING bash_temp.sh DONE."
# bash_temp.sh ends here
