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

# for (( i=0; i<$length; i++ )); do
#     export model_ckpt=${model_ckpts[$i]}
#     export cudanum=${cudals[$i]}
#     export ckpt_part=$(echo "${model_ckpt}" | cut -d'/' -f2)
#     echo "cuda: $cudanum"

# # nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0617--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &
#     nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0620--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &
# done

# echo "sleep 3 hours Now."
# # sleep 10800
# sleep 1800
# echo "sleep DONE."

# export model_ckpts=("facebook/opt-125m" "facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b" "facebook/opt-30b" "facebook.opt-66b")
# export model_ckpts=("facebook/opt-6.7b" "facebook/opt-13b" )
# length=${#model_ckpts[@]}

# for (( i=0; i<$length; i++ )); do
#     export model_ckpt=${model_ckpts[$i]}
#     export cudanum=${cudals[$i]}
#     export ckpt_part=$(echo "${model_ckpt}" | cut -d'/' -f2)
#     echo "cuda: $cudanum"

# # nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0617--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &
#     nohup bash ${root_dir}/scripts/7.3.varymodel_size.sh ${model_ckpt} ${cudanum} > "0619--TrainVaryingModelSize${ckpt_part}${cudanum}.log" &

# done


bash ${root_dir}/scripts/7.3.varymodel_size.sh "facebook/opt-30b" 1

# bash ${root_dir}/scripts/7.2.varytrainnum__wmt.sh 512 5 > "0615--trainvaryingtrainnum${trainnum}${cudanum}.log"


## remove all lord-vi results.
# rm -rf "${root_dir}wmt_0617_varymodelsize_dataset_res/*LoRD-VI*"
# rm -rf "${root_dir}wmt_0617_varymodelsize_dataset_res/*vanilla*"

# echo "sleep 3 hours Now."
# sleep 3600
# echo "sleep DONE."

$python "${root_dir}wmt_process.py"
# $python "${root_dir}text2sql_process.py"

# rm "${root_dir}vary_modelsize_overall_res_wmt16.json"
# $python "${root_dir}eval_vary_modelsize.py"


bash ${root_dir}/general_train/1.2.train_longtext.sh

echo "RUNNING meta_7.3.varysize.sh DONE."
# meta_7.3.varysize.sh ends here
