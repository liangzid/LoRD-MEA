#!/bin/bash
######################################################################
#1.4.HYPER_SEARCH --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 23 May 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1,2"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_USE_CUDA_DSA="1"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/general_train/ckpts/shorttext/"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export TRAIN_NUMS=(249)
export train_times=(1)

export msl=256
export task_ls=("liangzid/claude3_short256")

# export msl=1024
# export task_ls=("liangzid/claude3_chat3.3k")

# export train_taskls=("vanilla")
# export train_taskls=("LoRD-II")
# export train_taskls=("LoRD-V" "LoRD-VI")
# export train_taskls=("vanilla" "LoRD-VI")
export train_taskls=("LoRD-VI")

export is_black_box=1
export use_lora=1

# export epoch=3
# export period=3
# export epoch=1
# export period=5
                    prefix = f"./text2sql_ckpts/text2sql---{task}{train_num}{itime}{m}_res.json"

export epoch=1
export period=1
export sub_set_num=1
export sub_stage_num=500
export max_new_tokens=256
export infer_batch_size=1
export batch_size=1

export beta=1.0
export temperature=2

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export tau1=0.5
export tau2=0.8
export tau_delta=-0.1
export save_step=100

export tau1_ls=(0.8 0.9 1.0)
# export tau2_ls=(0.3 0.4 0.5 0.6 0.7 0.8)
export tau2_ls=(1.0)

for tau1 in ${tau1_ls[*]}
do
    for tau2 in ${tau2_ls[*]}
    do

for train_num in ${TRAIN_NUMS[*]}
do
    for train_time in ${train_times[*]}
    do
	for task in ${task_ls[*]}
	do
	    for train_task in ${train_taskls[*]}
	    do
		echo "====================================================="
		echo "++tau1: ${tau1} ${tau2}++"
		echo "+++++++train_num: ${train_num}+++++++"
		echo "+++++++train_time: ${train_time}+++++++"
		echo "+++++++task: ${task}+++++++"
		echo "+++++++train_task: ${train_task}+++++++"
		echo "====================================================="

		export save_path="${POD_save_dir}lordvi-explore${tau1}${tau2}"

		$python ${root_dir}lord_train.py\
		    --use_lora=$use_lora \
		    --from_path=$from_path \
		    --is_black_box=$is_black_box \
		    --sub_set_num=$sub_set_num \
		    --sub_stage_num=$sub_stage_num\
		    --infer_batch_size=$infer_batch_size\
		    --tau1=$tau1 \
		    --tau2=$tau2 \
		    --tau_delta=$tau_delta \
		    --task=$train_task \
		    --device="cuda" \
		    --epoch=$epoch \
		    --period_num=$period \
		    --acc_step=1 \
		    --log_step=50 \
		    --train_num=$train_num \
		    --max_new_tokens=$max_new_tokens \
		    --LR="3e-5" \
		    --save_step=${save_step} \
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


export qas=openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq
export eval=${HOME}/anaconda3/envs/align/bin/lm_eval
export pmp=meta-llama/Meta-Llama-3-8B-Instruct
export fmp="${save_path}___period500"

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
    done
done


echo "RUNNING 1.4.hyper_search.sh DONE."
# 1.4.hyper_search.sh ends here
