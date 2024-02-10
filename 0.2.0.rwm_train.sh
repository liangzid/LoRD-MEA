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

export bs_per=4

$python rewardmodel_train.py \
    --model_name_or_path="facebook/opt-350m" \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=$bs_per \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \

echo "RUNNING 0.2.0.rwm_train.sh DONE."
# 0.2.0.rwm_train.sh ends here
