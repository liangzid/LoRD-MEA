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
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")

export task_ls=("cs-en" "du-en" "fi-en" "ro-en" "ru-en" "tr-en")

export task="cs-en"
# export task="cola"
# export task="sum"
# export train_task="lord"
export train_task="Complex-lord"
# export train_task="reinforce-lord"
# export train_task="kd"
export epoch=3
export period=9
export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export train_num=20

# export train_task="kd"
export save_path="${POD_save_dir}${task}/${train_task}_${msl}${task}"

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

# for task in ${task_ls[*]}
# do
#     # export task="cola"
#     export save_path="${POD_save_dir}pod_style_test_fast${msl}${task}"
#     echo "task: $task"

#     $python lord_train.py\
# 	    --device="cuda" \
# 	    --epoch=2 \
# 	    --period_num=3 \
# 	    --acc_step=1 \
# 	    --log_step=1 \
# 	    --save_step=100000 \
# 	    --LR="3e-5" \
# 	    --beta=1.0 \
# 	    --temperature=1.0 \
# 	    --batch_size=1 \
# 	    --task="none set yet" \
# 	    --max_length=$msl \
# 	    --dataset_task=$task \
# 	    --from_path=$from_path \
# 	    --save_path=$save_path
# done



echo "RUNNING 2.0.pod_selfdesigned_train.sh DONE."
# 2.0.pod_selfdesigned_train.sh ends here
