#!/bin/bash
######################################################################
#3.3.WMT16_EXPERIMENTS ---

# RUNNING WMT16 experiments

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 12 March 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}wmt2b_ckpts/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"
export msl=256
export task_ls=("cs-en" "de-en" "fi-en" "ro-en" "ru-en" "tr-en")
# export task_ls=("de-en")
export train_taskls=("vanilla" "kd")
# export train_taskls=("black--Complex-lord")
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
export max_new_tokens=64

export train_times=("1" "2" "3" "4" "5")

for train_time in ${train_times[*]}
do
    for task in ${task_ls[*]}
    do
	for train_task in ${train_taskls[*]}
	do
	    export save_path="${POD_save_dir}${task}${train_task}${msl}${train_num}__${train_time}"

	    echo "++++++++++${train_task}-----${task}++++++++++"
	    echo "++++++++++train_time--${train_time}++++++++++"
	    echo "Save path: ${save_path}."

	    $python pod_train.py\
		    --device="cuda" \
		    --epoch=$epoch \
		    --period_num=$period \
		    --acc_step=1 \
		    --log_step=1 \
		    --save_step=100000 \
		    --train_num=$train_num \
		    --max_new_tokens=${max_new_tokens} \
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
done


echo "RUNNING 3.3.WMT16_experiments.sh DONE."
# 3.3.WMT16_experiments.sh ends here
