#!/bin/bash
######################################################################
#2.0.1.LORA_TRAIN ---

# test the train procedure of the LoRA model.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 17 April 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="5,6,7"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}LoRA-LoRD-ckpts"
export from_path="google/gemma-7b"
export msl=256
export TRAIN_NUMS=("32")
export train_times=("1")
# export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
# export task_ls=("cs-en" "de-en" "fi-en" "ro-en" "ru-en" "tr-en")
# export task_ls=("piqa" "truthful_qa" "allenai/ai2_arc")
# export task_ls=("allenai/ai2_arc")
export task_ls=("piqa")
# export train_taskls=("vanilla")
# export train_taskls=("LoRD-IV")
export train_taskls=("Complex-lord")

# ## ====================TO DEBUG====================
# export epoch=1
# export period=2
# export beta=1.0
# export temperature=2
# export batch_size=1
# ## ====================TO DEBUG====================

export use_lora=1

export epoch=3
export period=3

export sub_set_num=8
export sub_stage_num=16
# export train_num=32
export max_new_tokens=64

# export epoch=3
# export period=3
# export beta=1.0
# export temperature=2
# export batch_size=1

export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

# export train_num=100
export max_new_tokens=64


echo "To run this script, you should decide your LoRD method: which one is best? should you use Complex-lord, lord, or reinforce-lord?"

for train_num in ${TRAIN_NUMS[*]}
do
    for train_time in ${train_times[*]}
    do
	for task in ${task_ls[*]}
	do
	    for train_task in ${train_taskls[*]}
	    do
		echo "====================================================="
		echo "+++++++train_num: ${train_num}+++++++"
		echo "+++++++train_time: ${train_time}+++++++"
		echo "+++++++task: ${task}+++++++"
		echo "+++++++train_task: ${train_task}+++++++"
		echo "====================================================="

		export save_path="${POD_save_dir}varyTrainNum___${train_num}${train_time}${task}${train_task}${epoch}${period}${temperature}${batch_size}${max_new_tokens}${msl}"

		$python ${root_dir}lord_train.py\
		    --task=$train_task \
		    --device="cuda" \
		    --epoch=$epoch \
		    --period_num=$period \
		    --acc_step=1 \
		    --log_step=50 \
		    --save_step=100000 \
		    --train_num=$train_num \
		    --max_new_tokens=$max_new_tokens \
		    --LR="3e-5" \
		    --beta=$beta \
		    --temperature=$temperature \
		    --batch_size=$batch_size \
		    --use_old_logits=$use_old_logits\
		    --use_vic_logits=$use_vic_logits\
		    --use_kld=$use_kld\
		    --max_length=$msl \
		    --dataset_task=$task \
		    --from_path=$from_path \
		    --save_path=$save_path
		echo "DONE FOR ONE TRAIN NUMBERS...."
	    done
	done
    done
done


echo "RUNNING 2.0.1.lora_train.sh DONE."
# 2.0.1.lora_train.sh ends here
