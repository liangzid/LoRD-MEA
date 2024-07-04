#!/bin/bash
######################################################################
#META_7.2.3.VARY_TRAINNUM_GLUE ---

# GLUE vary train numbers experiments.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  2 July 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export root_dir="${HOME}/alignmentExtraction/"

# export TRAIN_NUMS=(8 16 32 64 128 256 512 1024)
export TRAIN_NUMS=(8 16 32 64 128)
# export TRAIN_NUMS=(256 512 1024)
# export cudals=(2 3)
export cudals=(0 1 2 3 4 5 6 7)
# export cudals=(0 1 2)


length=${#TRAIN_NUMS[@]}

for (( i=0; i<$length; i++ )); do
    export trainnum=${TRAIN_NUMS[$i]}
    export cudanum=${cudals[$i]}
# bash ${root_dir}/scripts/7.2.2.varytrainnum_qa.sh ${trainnum} ${cudanum} > "0630_trainvaryingtrainnum${trainnum}${cudanum}.log"
nohup bash ${root_dir}/scripts/7.2.3.glue_vary_trainnum.sh ${trainnum} ${cudanum} > "0702_trainvaryingtrainnum${trainnum}${cudanum}.log" &

done



echo "RUNNING meta_7.2.3.vary_trainnum_glue.sh DONE."
# meta_7.2.3.vary_trainnum_glue.sh ends here
