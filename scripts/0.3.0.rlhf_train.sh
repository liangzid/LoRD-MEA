#!/bin/bash
######################################################################
#0.3.0.RLHF_TRAIN ---

RLHF model training script.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 11 February 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/align/bin/python3

$python rlhf_train.py \
	--log_with=wandb\
	--model_name="gpt2"\
	--query_dataset=""


echo "RUNNING 0.3.0.rlhf_train.sh DONE."
# 0.3.0.rlhf_train.sh ends here
