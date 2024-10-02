#!/bin/bash
######################################################################
#2.2.VARY_PERIOD --- 

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
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
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"

export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")

export task="cola"
# echo "Also, determine to use the `complex-lord`, or the `lord`. Which one is the best."
# echo "Test complex-version first."

export train_task="Complex-lord"
# export train_task="lord"
export epoch=3
export period=10
export beta=0.5
export temperature=2.0
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export train_num=100

# export train_task="kd"
export save_path="${POD_save_dir}vary_period_complex/${train_task}${msl}${task}${use_old_logits}${use_vic_logits}${use_kld}${use_entropy}"

$python lord_train.py\
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



echo "RUNNING 2.2.vary_period.sh DONE."
# 2.2.vary_period.sh ends here
