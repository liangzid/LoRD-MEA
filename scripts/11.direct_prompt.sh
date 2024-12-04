#!/bin/bash
######################################################################
#11.DIRECT_PROMPT ---

# Add the script of a direct prompt training.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  3 December 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="0"
# export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/MethodDirectPrompt/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export TRAIN_NUMS=(256)
# export TRAIN_NUMS=(512)
# export TRAIN_NUMS=(64 128 256 1024)
export train_times=(1)
# export train_times=(1)
export msl=256
export task_ls=("de-en")
export train_taskls=("simPO")
# export temperature=0.8
export temperature=1.3

export is_black_box=1
export use_lora=1
export max_new_tokens=64
export batch_size=1

# Useless settigns:
export epoch=1
export period=1
export sub_set_num=1
export sub_stage_num=256
export infer_batch_size=1
export beta=-1
export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0
export tau1=0.80
export tau2=0.85

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

		export save_path="${POD_save_dir}WMTTT0519${task}${train_num}${train_time}${train_task}temperature${temperature}"

		proxychains $python ${root_dir}direct_prompt_train.py\
		    --use_lora=$use_lora \
		    --from_path=$from_path \
		    --is_black_box=$is_black_box \
		    --sub_set_num=$sub_set_num \
		    --sub_stage_num=$sub_stage_num\
		    --infer_batch_size=$infer_batch_size\
		    --tau1=$tau1 \
		    --tau2=$tau2 \
		    --task=$train_task \
		    --device="cuda" \
		    --epoch=$epoch \
		    --period_num=$period \
		    --acc_step=1 \
		    --log_step=50 \
		    --train_num=$train_num \
		    --max_new_tokens=$max_new_tokens \
		    --LR="3e-5" \
		    --save_step=$sub_stage_num \
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

echo "RUNNING 11.direct_prompt.sh DONE."
# 11.direct_prompt.sh ends here
