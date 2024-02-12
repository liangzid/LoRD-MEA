#!/bin/bash
######################################################################
#0.2.0.RWM_TRAIN ---

# Reward Model Training

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 10 February 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/align/bin/python3

export bs_per=1
# export pretrained_model="facebook/opt-350m"
# export pretrained_model="microsoft/phi-1_5"
export pretrained_model="microsoft/phi-2"
# export pretrained_model="gpt2"

$python rewardmodel_train.py \
    --model_name_or_path=$pretrained_model \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=$bs_per \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=3e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=500 \
    --evaluation_strategy="steps" \
    --max_length=1024 

echo "RUNNING 0.2.0.rwm_train.sh DONE."
# 0.2.0.rwm_train.sh ends here
