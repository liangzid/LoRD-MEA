#!/bin/bash
######################################################################
#7.2.VARYTRAINNUM__WMT --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 14 June 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export root_dir="${HOME}/alignmentExtraction/"
export POD_save_dir="${root_dir}/NEW_VARYING_QUERYTIME_CKPTS/"
# export from_path="meta-llama/Meta-Llama-3-8B-Instruct"
export from_path="facebook/opt-6.7b"
# export TRAIN_NUMS=(64)
# export TRAIN_NUMS=(8 16 32 64 128 256)
# export cudals=(0 1 2 3 4 5)


export TRAIN_NUMS=($1)
export cudals=($2)

# export TRAIN_NUMS=(32)
# export cudals=(0)

# export train_times=(2 3 4 5)
export train_times=(1)
# export msl=256
export msl=140
# export task_ls=("cs-en" "de-en" "fi-en" "ro-en")
# export task_ls=("ru-en" "de-en")
export task_ls=("ru-en")
# export task_ls=("de-en")
export train_taskls=("vanilla" "LoRD-VI")
# export train_taskls=("LoRD-VIII" "vanilla")
# export train_taskls=("LoRD-VI")
# export train_taskls=("vanilla")

export is_black_box=1
export use_lora=1

# export epoch=3
# export period=3

export epoch=1
export period=1

export sub_set_num=1
export sub_stage_num=512
# export sub_stage_num=2048
# export max_new_tokens=64
export max_new_tokens=32
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



length=${#TRAIN_NUMS[@]}


for (( i=0; i<$length; i++ )); do
# for train_num in ${TRAIN_NUMS[*]}
# do
    
    export CUDA_VISIBLE_DEVICES=${cudals[$i]}
    export TORCH_USE_CUDA_DSA="1"
    export train_num=${TRAIN_NUMS[$i]}

    echo "train_num: ${train_num}\t device: ${cudals[$i]}"

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

	# if [[ "${train_num}" == "1024" ]]; then
	#     export sub_stage_num="1024"
	# fi

	# if [[ "${train_num}" == "2048" ]]; then
	#     export sub_stage_num="2048"
	# fi
        # export sub_stage_num=$((train_num * 4))
	echo "Sub-stage-num: ${sub_stage_num}"

		export save_path="${POD_save_dir}text2sql${task}${train_num}${train_time}${train_task}"

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


echo "RUNNING 7.2.varytrainnum__wmt.sh DONE."
# 7.2.varytrainnum__wmt.sh ends here
