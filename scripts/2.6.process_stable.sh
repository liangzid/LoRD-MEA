#!/bin/bash
######################################################################
#2.6.PROCESS_STABLE ---


# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created: 23 March 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "This file is used to tracking the stablity of model during stealing."

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}tracking_process_stable"
# export from_path="openai-community/gpt2-xl"
export from_path="google/gemma-2b"
export msl=256
export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
export task_ls=("cs-en" "du-en" "fi-en" "ro-en" "ru-en" "tr-en")
# export task="cs-en"
export task="cs-en"
# export task="sum"
# export train_task="lord"
export train_task="Complex-lord"
# export train_task="reinforce-lord"
# export train_task="kd"

# export train_task="Very--Complex-lord"
# export train_task="nolog--Complex-lord"
# export train_task="ComplexV3"
# export train_task="Black--Very--Complex-lord"

export epoch=3
export period=3
export beta=1.0
export temperature=2
export batch_size=1

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

# export train_num=100
export train_num=100


# export tasks=("Complex-lord" "vanilla")
export tasks=("vanilla")

for train_task in ${tasks[*]}
do
    # export train_task="kd"
    export save_path="${POD_save_dir}${task}/${train_task}_${msl}${task}_step"
    $python lord_train.py\
	    --device="cuda" \
	    --epoch=$epoch \
	    --period_num=$period \
	    --acc_step=1 \
	    --log_step=1 \
	    --save_step=32 \
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

echo "EVERYTHING DONE."
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"



echo "RUNNING 2.6.process_stable.sh DONE."
# 2.6.process_stable.sh ends here
