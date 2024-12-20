#!/bin/bash
######################################################################
#9.ABLATION_P_Y_VIC --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 24 November 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2"
export CUDA_VISIBLE_DEVICES="0"
# export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/ablation_ckpts/"
# export from_path="google/gemma-2-2b-it"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
# export from_path="microsoft/Phi-3-mini-4k-instruct"
# export from_path="Qwen/Qwen2-7B-Instruct"
# export from_path="microsoft/Phi-3-small-8k-instruct"
# export from_path="mistralai/Mistral-7B-Instruct-v0.3"
export TRAIN_NUMS=(16)
# export TRAIN_NUMS=(64 128 256)
export train_times=(1)
# export train_times=(1)
export msl=256
export task_ls=("de-en")
# export task_ls=("wmt_mix")
# export task_ls=("cs-en")
# export train_taskls=("LoRD-VI" "vanilla")
# export train_taskls=("LoRD-VI" "vanilla")
# export train_taskls=("w.y_vic" "usey+" "w.KL")
# export train_taskls=("LoRD-VI")
export train_taskls=("w.o.Sigmoid")
# export train_taskls=("simPO2")
# export train_taskls=("kd")
# export train_taskls=("LoRD-II")

export is_black_box=1
export use_lora=1

# export epoch=3
# export period=3

export epoch=2
export period=1

export sub_set_num=1
export sub_stage_num=256
export max_new_tokens=64
export infer_batch_size=1
export batch_size=1

export beta=-1
export temperature=-1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

# export tau1=0.85
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

		export save_path="${POD_save_dir}WMTTT0519${task}${train_num}${train_time}${train_task}"

		$python ${root_dir}lord_train.py\
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


# $python ${root_dir}wmt_process.py








echo "RUNNING 9.ablation_P_y_vic.sh DONE."
# 9.ablation_P_y_vic.sh ends here
