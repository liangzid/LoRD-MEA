#!/bin/bash
######################################################################
#3.2.GLUE_EXPERIMENTS ---

# scripts of the GLUE experiments.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 12 March 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}GLUE_ckpts/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"
export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
# export train_taskls=("Complex-lord" "vanilla" "kd" "black--Complex-lord")
export train_taskls=("black--Complex-lord")
# export task_ls=("cs-en" "du-en" "fi-en" "ro-en" "ru-en" "tr-en")
# export train_task="Complex-lord"
# export train_task="reinforce-lord"
# export train_task="kd"
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

for task in ${task_ls[*]}
do
    for train_task in ${train_taskls[*]}
    do
	export save_path="${POD_save_dir}${task}${train_task}${msl}${train_num}"

	echo "++++++++++${train_task}-----${task}++++++++++"
	echo "Save path: ${save_path}."

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

    done
done


echo "RUNNING 3.2.glue_experiments.sh DONE."
# 3.2.glue_experiments.sh ends here
