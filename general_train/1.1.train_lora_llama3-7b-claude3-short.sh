#!/bin/bash
######################################################################
#1.1.TRAIN_LORA_LLAMA3-7B-CLAUDE3-SHORT --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 25 April 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="4,5,6"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/general_train/ckpts/boring_test/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export msl=1024
export TRAIN_NUMS=(249)
export train_times=(1)
export task_ls=("liangzid/claude3_short256")
# export train_taskls=("vanilla")
# export train_taskls=("LoRD-II")
export train_taskls=("LoRD-II" "LoRD-IV")

# ## ====================TO DEBUG====================
# export epoch=1
# export period=2
# export beta=1.0
# export temperature=2
# export batch_size=1
# ## ====================TO DEBUG====================

export is_black_box=1
export use_lora=1

export epoch=1
export period=1
export sub_set_num=128
export sub_stage_num=4
export max_new_tokens=1024

export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

# export train_num=100
export max_new_tokens=64

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

		export save_path="${POD_save_dir}BORING_TEST___${train_num}${train_time}${task}${train_task}${epoch}${period}${temperature}${batch_size}${max_new_tokens}${msl}"

		$python ${root_dir}lord_train.py\
		    --use_lora=$use_lora \
		    --from_path=$from_path \
		    --is_black_box=$is_black_box \
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
		    --save_path=$save_path
		echo "DONE FOR ONE TRAIN NUMBERS...."
	    done
	done
    done
done











echo "RUNNING 1.1.train_lora_llama3-7b-claude3-short.sh DONE."
# 1.1.train_lora_llama3-7b-claude3-short.sh ends here
