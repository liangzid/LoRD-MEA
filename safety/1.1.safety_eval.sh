#!/bin/bash
######################################################################
#1.1.SAFETY_EVAL ---

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 28 June 2024
######################################################################

echo "First we execute STEALING..."

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="1"
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/safety_ckpts/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export TRAIN_NUMS=(64)
# export TRAIN_NUMS=(2)
# export TRAIN_NUMS=(2 4 8 16 32 64 128 256 512 1024 2048)
# export train_times=(2 3 4 5)
export train_times=(1)
export msl=256
# export task_ls=("PKU-Alignment/PKU-SafeRLHF")
export task_ls=("thu-coai/diasafety")
export train_taskls=("vanilla" "LoRD-VI")
# export train_taskls=("vanilla")
# export train_taskls=("LoRD-VI")

export is_black_box=1
export use_lora=1

export epoch=1
export period=1

export sub_set_num=1
export max_new_tokens=64
export infer_batch_size=1
export batch_size=1

export beta=-1
export temperature=1

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

	if [[ "${train_num}" == "1024" ]]; then
            export sub_stage_num=${train_num}
	else
            export sub_stage_num=512
	fi
	if [[ "${train_num}" == "2048" ]]; then
            export sub_stage_num=${train_num}
	fi

		echo "====================================================="
		echo "+++++++train_num: ${train_num}+++++++"
		echo "+++++++train_time: ${train_time}+++++++"
		echo "+++++++task: ${task}+++++++"
		echo "+++++++train_task: ${train_task}+++++++"
		echo "====================================================="

		export save_path="${POD_save_dir}SAFETY${task}${train_num}${train_time}${train_task}"

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

echo "Then Evaluating with P-API......"

$python ${root_dir}safety_process.py

# echo "Code Related Inference Experiments......"
# $python ${root_dir}code_process.py

echo "RUNNING 1.1.safety_eval.sh DONE."
# 1.1.safety_eval.sh ends here
