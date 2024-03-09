#!/bin/bash
######################################################################
#3.1.VARY_TRAIN_NUM ---

# Experiments on varying sample numbers of training dataset.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  8 March 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}POD_SAVE_CKPTs/vary_trainNum"


export TRAIN_NUMS=("1" "2" "4" "8" "16" "32" "64" "100" "256" "300")

echo "To run this script, you should decide your LoRD method: which one is best? should you use Complex-lord, lord, or reinforce-lord?"


for train_num in ${TRAIN_NUMS[*]}
do
    # echo "====================================================="
    # echo "+++++++USING TRAIN SET NUM: ${TRAIN_NUM}+++++++"
    # echo "+++++++TRAIN WITH: vanilla supervised training+++++++"
    # export from_path="google/gemma-2b"
    # export msl=256
    # export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
    # export task="cola"
    # export train_task="vanilla"
    # export epoch=3
    # export period=10
    # export beta=1.0
    # export temperature=2.0
    # export batch_size=1
    # export use_old_logits=1
    # export use_vic_logits=1
    # export use_kld=1

    # export save_path="${POD_save_dir}${task}${train_num}${train_task}_${msl}${task}"

    # $python pod_train.py\
    # 	--device="cuda" \
    # 	--epoch=$epoch \
    # 	--period_num=$period \
    # 	--acc_step=1 \
    # 	--log_step=50 \
    # 	--save_step=100000 \
    # 	--train_num=$train_num \
    # 	--LR="3e-5" \
    # 	--beta=$beta \
    # 	--temperature=$temperature \
    # 	--batch_size=$batch_size \
    # 	--task=$train_task \
    # 	--use_old_logits=$use_old_logits\
    # 	--use_vic_logits=$use_vic_logits\
    # 	--use_kld=$use_kld\
    # 	--max_length=$msl \
    # 	--dataset_task=$task \
    # 	--from_path=$from_path \
    # 	--save_path=$save_path

    # echo "====================================================="
    # echo "+++++++USING TRAIN SET NUM: ${TRAIN_NUM}+++++++"
    # echo "+++++++TRAIN WITH: kd supervised training+++++++"
    # export from_path="google/gemma-2b"
    # export msl=256
    # export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
    # export task="cola"
    # export train_task="kd"
    # export epoch=3
    # export period=10
    # export beta=1.0
    # export temperature=2.0
    # export batch_size=1
    # export use_old_logits=1
    # export use_vic_logits=1
    # export use_kld=1

    # export save_path="${POD_save_dir}${task}${train_num}${train_task}_${msl}${task}"

    # $python pod_train.py\
    # 	--device="cuda" \
    # 	--epoch=$epoch \
    # 	--period_num=$period \
    # 	--acc_step=1 \
    # 	--log_step=50 \
    # 	--save_step=100000 \
    # 	--train_num=$train_num \
    # 	--LR="3e-5" \
    # 	--beta=$beta \
    # 	--temperature=$temperature \
    # 	--batch_size=$batch_size \
    # 	--task=$train_task \
    # 	--use_old_logits=$use_old_logits\
    # 	--use_vic_logits=$use_vic_logits\
    # 	--use_kld=$use_kld\
    # 	--max_length=$msl \
    # 	--dataset_task=$task \
    # 	--from_path=$from_path \
    # 	--save_path=$save_path

    # echo "====================================================="
    # echo "+++++++USING TRAIN SET NUM: ${TRAIN_NUM}+++++++"
    # echo "+++++++TRAIN WITH: lord+++++++"
    # export from_path="google/gemma-2b"
    # export msl=256
    # export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
    # export task="cola"
    # export train_task="lord"
    # export epoch=3
    # export period=10
    # export beta=1.0
    # export temperature=2.0
    # export batch_size=1
    # export use_old_logits=1
    # export use_vic_logits=1
    # export use_kld=0
    # export use_entropy=0

    # export save_path="${POD_save_dir}${task}${train_num}${train_task}${msl}${task}"

    # $python pod_train.py\
    # 	--device="cuda" \
    # 	--epoch=$epoch \
    # 	--period_num=$period \
    # 	--acc_step=1 \
    # 	--log_step=50 \
    # 	--save_step=100000 \
    # 	--train_num=$train_num \
    # 	--LR="3e-5" \
    # 	--beta=$beta \
    # 	--temperature=$temperature \
    # 	--batch_size=$batch_size \
    # 	--task=$train_task \
    # 	--use_old_logits=$use_old_logits\
    # 	--use_vic_logits=$use_vic_logits\
    # 	--use_kld=$use_kld\
    # 	--max_length=$msl \
    # 	--dataset_task=$task \
    # 	--from_path=$from_path \
    # 	--save_path=$save_path


    echo "====================================================="
    echo "+++++++USING TRAIN SET NUM: ${TRAIN_NUM}+++++++"
    echo "+++++++TRAIN WITH: Complex-lord+++++++"
    export from_path="google/gemma-2b"
    export msl=256
    export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
    export task="cola"
    export train_task="Complex-lord"
    export epoch=3
    export period=3
    export beta=1.0
    export temperature=2.0
    export batch_size=1
    export use_old_logits=1
    export use_vic_logits=1
    export use_kld=0
    export use_entropy=0

    export save_path="${POD_save_dir}${task}${train_num}${train_task}${msl}${task}"

    $python pod_train.py\
	--device="cuda" \
	--epoch=$epoch \
	--period_num=$period \
	--acc_step=1 \
	--log_step=50 \
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
	--max_length=$msl \
	--dataset_task=$task \
	--from_path=$from_path \
	--save_path=$save_path


    # echo "====================================================="
    # echo "+++++++USING TRAIN SET NUM: ${TRAIN_NUM}+++++++"
    # echo "+++++++TRAIN WITH: reinforce-lord+++++++"
    # export from_path="google/gemma-2b"
    # export msl=256
    # export task_ls=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
    # export task="cola"
    # export train_task="reinforce-lord"
    # export epoch=3
    # export period=10
    # export beta=1.0
    # export temperature=2.0
    # export batch_size=1
    # export use_old_logits=1
    # export use_vic_logits=1
    # export use_kld=0
    # export use_entropy=0

    # export save_path="${POD_save_dir}${task}${train_num}${train_task}${msl}${task}"

    # $python pod_train.py\
    # 	--device="cuda" \
    # 	--epoch=$epoch \
    # 	--period_num=$period \
    # 	--acc_step=1 \
    # 	--log_step=50 \
    # 	--save_step=100000 \
    # 	--train_num=$train_num \
    # 	--LR="3e-5" \
    # 	--beta=$beta \
    # 	--temperature=$temperature \
    # 	--batch_size=$batch_size \
    # 	--task=$train_task \
    # 	--use_old_logits=$use_old_logits\
    # 	--use_vic_logits=$use_vic_logits\
    # 	--use_kld=$use_kld\
    # 	--max_length=$msl \
    # 	--dataset_task=$task \
    # 	--from_path=$from_path \
    # 	--save_path=$save_path
    

    echo "DONE FOR ONE TRAIN NUMBERS...."
done























echo "RUNNING 3.1.vary_train_num.sh DONE."
# 3.1.vary_train_num.sh ends here
