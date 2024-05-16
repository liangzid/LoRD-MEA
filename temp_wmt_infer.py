"""
======================================================================
TEMP_WMT_INFER --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 16 May 2024
======================================================================
"""
# ----------------------- Code --------------------------------------
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from wmt_process import *

def evaluation_datas():
    base_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
    ckpt_ls=[
        # ["cs-en",base_model_name,],
        # ["de-en",base_model_name,],
        # ["fi-en",base_model_name,],
        ["cs-en","gpt-3.5-turbo-1106",],
        # ["de-en","gpt-3.5-turbo-1106",],
        # ["fi-en","gpt-3.5-turbo-1106",],
        ]
    res_dict = {}
    dir_p = "./wmt16_res/"
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    for task_ckpt in ckpt_ls:
        task, ckpt = task_ckpt
        res_pth = ckpt+f"___{task}_glue_infer_res"
        res_pth = res_pth.replace("/", "__").replace(".", "")
        res_pth += ".json"
        if not os.path.exists(dir_p+res_pth):
            res_ls = infer_wmt(ckpt,
                               task, dir_p+res_pth,
                               test_set_take_num=500,
                               mnt=64,
                               base_model_name=None)
        else:
            # from collections import OrderedDict
            with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                res_ls = json.load(f, object_pairs_hook=OrderedDict)

        scores = eval_wmt(res_ls)
        print(task, ckpt)
        print(scores)
        res_dict[task+"-----"+ckpt] = scores
    with open(dir_p+"wmt_inference_scores_overall.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)

if __name__=="__main__":
    evaluation_datas()
