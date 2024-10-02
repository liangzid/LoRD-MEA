#!/bin/bash
######################################################################
#META_7.2.2.VARYTRAINNUM_QA --- 

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created: 30 June 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export root_dir="${HOME}/alignmentExtraction/"


# export TRAIN_NUMS=(8 16 32 64 128 256 512 1024)
# export TRAIN_NUMS=(256 512 1024)
export TRAIN_NUMS=(256 512)
# export cudals=(2 3)
# export cudals=(0 1 2 3 4 5 6 7)
export cudals=(4 6)


length=${#TRAIN_NUMS[@]}

for (( i=0; i<$length; i++ )); do
    export trainnum=${TRAIN_NUMS[$i]}
    export cudanum=${cudals[$i]}

# bash ${root_dir}/scripts/7.2.2.varytrainnum_qa.sh ${trainnum} ${cudanum} > "0630_trainvaryingtrainnum${trainnum}${cudanum}.log"
nohup bash ${root_dir}/scripts/7.2.2.varytrainnum_qa.sh ${trainnum} ${cudanum} > "0630_trainvaryingtrainnum${trainnum}${cudanum}.log" &
done


# sleep 10800
# $python ${root_dir}qa_process.py

echo "RUNNING meta_7.2.2.varytrainnum_qa.sh DONE."
# meta_7.2.2.varytrainnum_qa.sh ends here
