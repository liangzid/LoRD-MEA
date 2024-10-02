#!/bin/bash
######################################################################
#1.3.SHORT_PRETRAIN --- 

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
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

# for train_num in ${TRAIN_NUMS[*]}
# do
#     for train_time in ${train_times[*]}
#     do
# 	for task in ${task_ls[*]}
# 	do
# 	    for train_task in ${train_taskls[*]}
# 	    do
# 		echo "====================================================="
# 		echo "+++++++train_num: ${train_num}+++++++"
# 		echo "+++++++train_time: ${train_time}+++++++"
# 		echo "+++++++task: ${task}+++++++"
# 		echo "+++++++train_task: ${train_task}+++++++"
# 		echo "====================================================="

# 		# export save_path="${POD_save_dir}longtext${train_num}${train_time}${task}${train_task}${epoch}${period}${temperature}${batch_size}${max_new_tokens}${msl}"
# 		# export save_path="${POD_save_dir}longtext${train_num}${train_time}${task}${train_task}${epoch}${period}${temperature}${batch_size}${max_new_tokens}${msl}"
# 		export save_path="${POD_save_dir}lordvi-explore"

# 		$python ${root_dir}lord_train.py\
# 		    --use_lora=$use_lora \
# 		    --from_path=$from_path \
# 		    --is_black_box=$is_black_box \
# 		    --sub_set_num=$sub_set_num \
# 		    --sub_stage_num=$sub_stage_num\
# 		    --infer_batch_size=$infer_batch_size\
# 		    --tau1=$tau1 \
# 		    --tau2=$tau2 \
# 		    --tau_delta=$tau_delta \
# 		    --task=$train_task \
# 		    --device="cuda" \
# 		    --epoch=$epoch \
# 		    --period_num=$period \
# 		    --acc_step=1 \
# 		    --log_step=50 \
# 		    --train_num=$train_num \
# 		    --max_new_tokens=$max_new_tokens \
# 		    --LR="3e-5" \
# 		    --save_step=${save_step} \
# 		    --beta=$beta \
# 		    --temperature=$temperature \
# 		    --batch_size=$batch_size \
# 		    --use_old_logits=$use_old_logits\
# 		    --use_vic_logits=$use_vic_logits\
# 		    --use_kld=$use_kld\
# 		    --max_length=$msl \
# 		    --dataset_task=$task \
# 		    --save_path=$save_path
# 		echo "DONE FOR ONE TRAIN NUMBERS...."
# 	    done
# 	done
#     done
# done

export python=${HOME}/anaconda3/envs/align/bin/python3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_USE_CUDA_DSA="1"
export from_path="meta-llama/Meta-Llama-3-8B-Instruct"

# export qas=openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq
export qas=arc_easy,hellaswag,mmlu,truthfulqa,winogrande,gsm8k
# export qas=piqa
export eval=${HOME}/anaconda3/envs/align/bin/lm_eval
export pmp=meta-llama/Meta-Llama-3-8B-Instruct

## ------------------------------------------------------------------
# Now for long text models
# export fmp="${root_dir}general_train/ckpts/longtext/longtext30001liangzid/claude3_chat3.3kvanilla11212561024___finally"
# export fmp="${root_dir}general_train/ckpts/longtext/longtext30001liangzid/claude3_chat3.3kLoRD-VI11212561024___3000"

export fmp="${POD_save_dir}lordvi-explore___period500"

echo "================================================================"
echo "EVALUATION MODEL: pretrained: ${pmp} lora: ${fmp}"
echo "EVALUATION TASKS: ${qas}"
echo "================================================================"

$eval --model hf \
    --model_args pretrained=${pmp},parallelize=True,peft=${fmp}\
    --tasks $qas\
    --device cuda\
    --batch_size auto:4



echo "RUNNING 1.3.short_pretrain.sh DONE."
# 1.3.short_pretrain.sh ends here
