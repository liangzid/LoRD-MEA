#!/bin/bash
######################################################################
#META_7.3.VARYSIZE --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 17 June 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export root_dir="${HOME}/alignmentExtraction/"

# export model_ckpts=("facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b" )
# export model_ckpts=("EleutherAI/pythia-410m" "EleutherAI/pythia-1.4b" "EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b")
export model_ckpts=("EleutherAI/pythia-410m" "EleutherAI/pythia-1.4b" "EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b")
# export model_ckpts=("EleutherAI/pythia-410m")


export cudals=(0 1 2 3 4 5 6)

length=${#model_ckpts[@]}

for (( i=0; i<$length; i++ )); do
    export model_ckpt=${model_ckpts[$i]}
    export cudanum=${cudals[$i]}
    export ckpt_part=$(echo "${model_ckpt}" | cut -d'/' -f2)
    echo "cuda: $cudanum"

# nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0617--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &
    nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0620--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &
done

# echo "sleep 3 hours Now."
# sleep 10800
# echo "sleep DONE."

# export model_ckpts=("facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b" )
# length=${#model_ckpts[@]}

# for (( i=0; i<$length; i++ )); do
#     export model_ckpt=${model_ckpts[$i]}
#     export cudanum=${cudals[$i]}
#     export ckpt_part=$(echo "${model_ckpt}" | cut -d'/' -f2)
#     echo "cuda: $cudanum"

# # nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0617--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &
#     nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0619--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &

# done


# bash ${root_dir}/scripts/7.2.varytrainnum__wmt.sh 256 5 > "0615--trainvaryingtrainnum${trainnum}${cudanum}.log"

# bash ${root_dir}/scripts/7.2.varytrainnum__wmt.sh 512 5 > "0615--trainvaryingtrainnum${trainnum}${cudanum}.log"

# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlcs-en2561LoRD-VIII___period2048_____cs-en_wmt_infer_resjson"
# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlcs-en5121LoRD-VIII___period2048_____cs-en_wmt_infer_resjson"
# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlde-en2561LoRD-VIII___period2048_____de-en_wmt_infer_resjson"
# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlde-en5121LoRD-VIII___period2048_____de-en_wmt_infer_resjson"

# rm "${root_dir}wmt_0613_dataset_res/__NEW_VARYING_QUERYTIME_CKPTS__text2sqlde-en321vanilla___finally_____de-en_wmt_infer_resjson"

# echo "sleep 3 hours Now."
# sleep 10800
# echo "sleep DONE."

# $python "${root_dir}wmt_process.py"

# rm "${root_dir}vary_query_num_overall_res_wmt16.json"
# $python "${root_dir}eval_vary_trainNum.py"


echo "RUNNING meta_7.3.varysize.sh DONE."
# meta_7.3.varysize.sh ends here
