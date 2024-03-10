#!/bin/bash
######################################################################
#2.5.WMT_VARYING_METHODS ---

# WMT dataset experiments, to vary different methods.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 10 March 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="1,2,3,0"

export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}wmt_ckpt/"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"

export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")

export task_ls=("cs-en" "du-en" "fi-en" "ro-en" "ru-en" "tr-en")

export task="cs-en"
# export train_task="lord"
# export train_task="Complex-lord"
# export train_task="reinforce-lord"
# export train_task="kd"
export epoch=3
export period=3
export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export train_num=100

export train_task_ls=("kd" "vanilla" "Complex-lord")

for train_task in ${train_task_ls[*]}
do
    echo "+++++>>>> NOW RUNNING ${train_task} <<<<+++++"
    export save_path="${POD_save_dir}${train_task}${msl}${task}${train_num}"
    $python pod_train.py\
	    --device="cuda" \
	    --epoch=$epoch \
	    --period_num=$period \
	    --acc_step=1 \
	    --log_step=1 \
	    --save_step=100000 \
	    --train_num=$train_num \
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

echo "RUNNING 2.5.wmt_varying_methods.sh DONE."
# 2.5.wmt_varying_methods.sh ends here
