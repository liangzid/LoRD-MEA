#!/bin/bash
######################################################################
#8.2.FIDELITY_VISUALIZE ---

# Victim Models X Local Models
# Their fidelity.

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created:  4 July 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2"
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/cross_fidelity/"
# export train_num=(2)
export train_num=64
export train_time=(1)
export msl=140
export task_ls=("e2e_nlg")
# export train_taskls=("vanilla" "LoRD-VI")
export train_taskls=("LoRD-VI")
# export train_taskls=("vanilla")
# export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
# export from_path_ls=("microsoft/Phi-3-mini-4k-instruct" "Qwen/Qwen2-7B-Instruct" "facebook/opt-6.7b" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct")
export victim_path_ls=("gpt-4o" "gpt-4" "gpt-3.5-turbo")
# export victim_path_ls=("gpt-3.5-turbo")

export from_path_ls=($1)
export CUDA_VISIBLE_DEVICES=$2

export is_black_box=1
export use_lora=1

export epoch=1
export period=1

export sub_set_num=1
export sub_stage_num=512
export max_new_tokens=64
export infer_batch_size=1
export batch_size=1

export beta=-1
export temperature=-1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export use_pure_blackbox=1

# export tau1=0.85
export tau1=0.80
export tau2=0.85

for from_path in ${from_path_ls[*]}
do
    for victim_path in ${victim_path_ls[*]}
    do
	for task in ${task_ls[*]}
	do
	    for train_task in ${train_taskls[*]}
	    do
		echo "====================================================="
		echo "+++++++from path: ${from_path}+++++++"
		echo "+++++++victim path: ${victim_path}+++++++"
		echo "+++++++train_num: ${train_num}+++++++"
		echo "+++++++train_time: ${train_time}+++++++"
		echo "+++++++task: ${task}+++++++"
		echo "+++++++train_task: ${train_task}+++++++"
		echo "====================================================="

		export save_path="${POD_save_dir}text2sql${task}${train_task}${victim_path}${from_path}"

		$python ${root_dir}lord_train.py\
		    --use_lora=$use_lora \
		    --from_path=$from_path \
		    --victim_path=$victim_path \
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
		    --use_pure_blackbox=$use_pure_blackbox\
		    --use_kld=$use_kld\
		    --max_length=$msl \
		    --dataset_task=$task \
		    --save_path=$save_path
		echo "DONE FOR ONE TRAIN NUMBERS...."
	    done
	done
    done
done

# $python ${root_dir}data2text_process.py


echo "RUNNING 8.2.fidelity_visualize.sh DONE."
# 8.2.fidelity_visualize.sh ends here
