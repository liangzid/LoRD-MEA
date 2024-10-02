#!/bin/bash
######################################################################
#0.2.0.RWM_TRAIN ---

# Reward Model Training

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created: 10 February 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/align/bin/python3

export bs_per=1
# export pretrained_model="facebook/opt-350m"
# export pretrained_model="openai_community/gpt2_large"
export pretrained_model="google/gemma-2b"
# export pretrained_model="microsoft/phi-1_5"
# export pretrained_model="microsoft/phi-2"
# export pretrained_model="gpt2"

export CUDA_VISIBLE_DEVICES="1,2,3"
export reward_save_pth="rwd_ckpts/gemma2blarge_reward_anthropic_ckpt/"

$python rewardmodel_train.py \
	--do_eval=True\
	--do_train=True\
	--model_name_or_path=$pretrained_model \
	--output_dir=$reward_save_pth \
	--per_device_train_batch_size=$bs_per \
	--num_train_epochs=1 \
	--gradient_accumulation_steps=4 \
	--gradient_checkpointing=False \
	--learning_rate=1.41e-5 \
	--report_to="wandb" \
	--remove_unused_columns=False \
	--optim="adamw_torch" \
	--logging_steps=30000 \
	--evaluation_strategy="steps" \
	--max_length=1024 

echo "RUNNING 0.2.0.rwm_train.sh DONE."
# 0.2.0.rwm_train.sh ends here
