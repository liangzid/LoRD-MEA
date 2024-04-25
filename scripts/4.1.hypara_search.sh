#!/bin/bash
######################################################################
#4.1.HYPARA_SEARCH ---

# Searching appropriate hyper-parameters

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 30 March 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export root_dir="${HOME}/alignmentExtraction/"
export save_dir="${root_dir}lordii_ckpt/"
export from_path="google/gemma-2b"
export msl=256
export task="cs-en"
export train_task="LoRD-II"

export epoch=1
export periods=(1 3 5)
export sub_set_nums=(4 8 16 32 64)
export sub_stage_num=15
export train_num=100

export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export tau1=0.99
export tau2=0.998

export max_new_tokens=64


for period in ${periods[*]}
do
    for sub_set_num in ${sub_set_nums[*]}
    do

	echo "+++++PERIOD: ${period}+++++"
	echo "++subsetnum: ${sub_set_num}++"

	export save_path="${save_dir}PERIOD-SEARCH-LORDII/${train_task}${period}${sub_set_num}${sub_stage_num}${msl}${task}${max_new_tokens}__long_stage_style_ckpt"

	$python lord_train.py\
		--tau1=$tau1 \
		--tau2=$tau2 \
		--device="cuda" \
		--epoch=$epoch \
		--period_num=$period \
		--sub_set_num=$sub_set_num \
		--sub_stage_num=$sub_stage_num \
		--acc_step=1 \
		--log_step=1 \
		--save_step=100000 \
		--train_num=$train_num \
		--max_new_tokens=$max_new_tokens\
		--LR="3e-5" \
		--beta=$beta \
		--temperature=$temperature \
		--batch_size=$batch_size \
		--task=$train_task \
		--use_old_logits=$use_old_logits\
		--use_vic_logits=$use_vic_logits\
		--use_kld=$use_kld\
		--use_entropy=$use_entropy\
		--max_length=$msl \
		--dataset_task=$task \
		--from_path=$from_path \
		--save_path=$save_path
	
    done
done

echo "EVERYTHING DONE."
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

# $python wmt_process_hyper_search.py\
# 	--search_type="period"

export tau1="empty"
export tau2="empty"

export tau1s=(0.98 0.98 0.98 1.00 1.00 1.00)
export tau2s=(0.95 0.99 1.00 0.95 0.99 0.999)

export period="3"
export sub_set_num="16"

length=${#tau1s[@]}

for (( i=0; i<${length}; i++ )); do
    echo "+++++tau1: ${tau1s[$i]}+++++"
    echo "+++++tau2: ${tau2s[$i]}+++++"

    export save_path="${save_dir}tau-SEARCH-LORDII/${tau1}${tau2}${train_task}${sub_set_num}${sub_stage_num}${msl}${task}${max_new_tokens}__long_stage_style_ckpt"

    $python lord_train.py\
	    --tau1=$tau1 \
	    --tau2=$tau2 \
	    --device="cuda" \
	    --epoch=$epoch \
	    --period_num=$period \
	    --sub_set_num=$sub_set_num \
	    --sub_stage_num=$sub_stage_num \
	    --acc_step=1 \
	    --log_step=1 \
	    --save_step=100000 \
	    --train_num=$train_num \
	    --max_new_tokens=$max_new_tokens\
	    --LR="3e-5" \
	    --beta=$beta \
	    --temperature=$temperature \
	    --batch_size=$batch_size \
	    --task=$train_task \
	    --use_old_logits=$use_old_logits\
	    --use_vic_logits=$use_vic_logits\
	    --use_kld=$use_kld\
	    --use_entropy=$use_entropy\
	    --max_length=$msl \
	    --dataset_task=$task \
	    --from_path=$from_path \
	    --save_path=$save_path
	

done


echo "EVERYTHING DONE."
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"




echo "RUNNING 4.1.hypara_search.sh DONE."
# 4.1.hypara_search.sh ends here
