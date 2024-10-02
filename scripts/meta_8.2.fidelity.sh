#!/bin/bash
######################################################################
#META_8.2.FIDELITY --- 

# Author: Anonymous authors
# Copyright © 2024, Anonymous, all rights reserved.
# Created:  4 July 2024
######################################################################

######################### Commentary ##################################
##  
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export root_dir="${HOME}/alignmentExtraction/"

export from_path_ls=("microsoft/Phi-3-mini-4k-instruct" "Qwen/Qwen2-7B-Instruct" "facebook/opt-6.7b" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct")
# export from_path_ls=("microsoft/Phi-3-mini-4k-instruct")
export cudals=(0 1 2 3 5)


length=${#from_path_ls[@]}

for (( i=0; i<$length; i++ )); do
    export trainnum=${from_path_ls[$i]}
    export cudanum=${cudals[$i]}

nohup bash ${root_dir}/scripts/8.2.fidelity_visualize.sh ${trainnum} ${cudanum} > "0704--fidelity_varying${cudanum}.log" &

# bash ${root_dir}/scripts/8.2.fidelity_visualize.sh ${trainnum} ${cudanum}
done


# sleep 10800

# $python ${root_dir}/data2text_process.py


echo "RUNNING meta_8.2.fidelity.sh DONE."
# meta_8.2.fidelity.sh ends here
