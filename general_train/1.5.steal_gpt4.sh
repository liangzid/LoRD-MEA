#!/bin/bash
######################################################################
#1.5.STEAL_GPT4 ---

# Steal GPT-4.

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created: 31 May 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1,2"
export CUDA_VISIBLE_DEVICES="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/general_train/ckpts/steal_gpt4/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export pmp=$from_path
export TRAIN_NUMS=(1000)
export train_times=(1)
# export msl=256
export task_ls=("teknium/GPT4-LLM-Cleaned")
export msl=2048
# export train_taskls=("vanilla")
export epoch=2
export train_taskls=("LoRD-VII")
# export epoch=1
# export train_taskls=("LoRD-II")

export is_black_box=1
export use_lora=1

export period=1
export sub_set_num=1
export sub_stage_num=2000
export max_new_tokens=1000
export infer_batch_size=1
export batch_size=1

export beta=1.0

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export tau1=0.8
export tau2=0.9
export tau_delta=-0.1
export save_step=1000
export temperature=1.0

# export train_num=100

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

		export save_path="${POD_save_dir}llama3"

		$python ${root_dir}lord_train.py\
		    --use_lora=$use_lora \
		    --from_path=$from_path \
		    --is_black_box=$is_black_box \
		    --sub_set_num=$sub_set_num \
		    --sub_stage_num=$sub_stage_num\
		    --infer_batch_size=$infer_batch_size\
		    --T=$temperature\
		    --tau1=$tau1 \
		    --tau2=$tau2 \
		    --task=$train_task \
		    --device="cuda" \
		    --epoch=$epoch \
		    --period_num=$period \
		    --acc_step=1 \
		    --save_step=$save_step \
		    --log_step=50 \
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

# export qas=openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq
# export qas=arc_challenge,hellaswag,winogrande,gsm8k
export qas=arc_challenge,hellaswag,winogrande
export eval=${HOME}/anaconda3/envs/align/bin/lm_eval
export fmp="${save_path}___period2000/"
# export fmp="${save_path}___finally/"

echo "================================================================"
echo "EVALUATION MODEL: pretrained: ${pmp} lora: ${fmp}"
echo "EVALUATION TASKS: ${qas}"
echo "================================================================"

$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks $qas\
    --device cuda\
    --batch_size auto:4
	    done
	done
    done
done


echo "RUNNING 1.5.steal_gpt4.sh DONE."
# 1.5.steal_gpt4.sh ends here
