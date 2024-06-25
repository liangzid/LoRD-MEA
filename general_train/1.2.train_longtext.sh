#!/bin/bash
######################################################################
#1.2.TRAIN_LONGTEXT --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  5 June 2024
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
export POD_save_dir="${root_dir}/general_train/ckpts/boring_test/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export pmp=$from_path
export TRAIN_NUMS=(3000)
export train_times=(1)
export task_ls=("liangzid/claude3_chat3.3k")
export msl=1500
export train_taskls=("LoRD-VII")
export epoch=1

export is_black_box=1
export use_lora=1

export period=1
export sub_set_num=1
export sub_stage_num=6000
# export sub_stage_num=1000
export max_new_tokens=512
# export max_new_tokens=512
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
export save_step=3000
export temperature=1.0
export lambda1=0.5

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

		export save_path="${POD_save_dir}longtext${train_task}NewLoss"

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
		    --lambda1=$lambda1 \
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

	    done
	done
    done
done


bash ${root_dir}/general_train/2.2.huggingface_llm_eval.sh

echo "RUNNING 1.2.train_longtext.sh DONE."
# 1.2.train_longtext.sh ends here
