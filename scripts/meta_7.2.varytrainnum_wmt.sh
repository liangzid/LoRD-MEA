#!/bin/bash
######################################################################
#META_7.2.VARYTRAINNUM_WMT --- 

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created: 14 June 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################


echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export root_dir="${HOME}/alignmentExtraction/"


# export TRAIN_NUMS=(8 16 32 64 128 256 512 1024)
# export cudals=(1 1 1 1 1 1 1 1)
# export TRAIN_NUMS=(16)
# export cudals=(0 1 2 3 4 5 6)
# export TRAIN_NUMS=(8 16 32 64)
export TRAIN_NUMS=(128 256 512)
export cudals=(2 3 4 5)


length=${#TRAIN_NUMS[@]}

for (( i=0; i<$length; i++ )); do
    export trainnum=${TRAIN_NUMS[$i]}
    export cudanum=${cudals[$i]}

nohup bash ${root_dir}/scripts/7.2.varytrainnum__wmt.sh ${trainnum} ${cudanum} > "0629--trainvaryingtrainnum${trainnum}${cudanum}.log" &

done


# bash ${root_dir}/scripts/7.2.varytrainnum__wmt.sh 256 5 > "0615--trainvaryingtrainnum${trainnum}${cudanum}.log"

# bash ${root_dir}/scripts/7.2.varytrainnum__wmt.sh 512 5 > "0615--trainvaryingtrainnum${trainnum}${cudanum}.log"

# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlcs-en2561LoRD-VIII___period2048_____cs-en_wmt_infer_resjson"
# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlcs-en5121LoRD-VIII___period2048_____cs-en_wmt_infer_resjson"
# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlde-en2561LoRD-VIII___period2048_____de-en_wmt_infer_resjson"
# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlde-en5121LoRD-VIII___period2048_____de-en_wmt_infer_resjson"

# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlde-en321vanilla___finally_____de-en_wmt_infer_resjson"


# $python "${root_dir}wmt_process.py"

# rm "${root_dir}vary_query_num_overall_res_wmt16.json"

# $python "${root_dir}eval_vary_trainNum.py"

# $python ${root_dir}watermark/watermark_detect.py



# sleep 10800

# $python "${root_dir}wmt_process.py"

echo "RUNNING meta_7.2.varytrainnum_wmt.sh DONE."
# meta_7.2.varytrainnum_wmt.sh ends here
