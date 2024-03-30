#!/bin/bash
######################################################################
#4.0.RUN_LORD_II --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 27 March 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################


echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export root_dir="${HOME}/alignmentExtraction/"
export save_dir="${root_dir}lordii_ckpt/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"
export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
export task_ls=("cs-en" "du-en" "fi-en" "ro-en" "ru-en" "tr-en")
# export task="cs-en"
export task="cs-en"
export train_task="LoRD-II"

# export epoch=1
# export period=2
# export sub_set_num=2
# export sub_stage_num=10
# export train_num=16

export epoch=1
export period=3
export sub_set_num=16
export sub_stage_num=15
export train_num=100


export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export tau1=0.99
export tau2=0.998

export max_new_tokens=64

# export train_task="kd"
export save_path="${save_dir}${task}/${train_task}${sub_set_num}${sub_stage_num}${msl}${task}${max_new_tokens}__long_stage_style_ckpt"

$python lord_train.py\
	--tau1=$tau1 \
	--tau2=$tau2 \
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--sub_set_num=$sub_set_num \
	--sub_stage_num=$sub_stage_num \
	--acc_step=1 \
	--log_step=1 \
	--save_step=100000 \
	--train_num=$train_num \
	--max_new_tokens=$max_new_tokens\
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

# $python wmt_process.py


echo "RUNNING 4.0.run_lord_ii.sh DONE."
# 4.0.run_lord_ii.sh ends here
