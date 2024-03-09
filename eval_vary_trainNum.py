"""
======================================================================
EVAL_VARY_TRAINNUM ---

Evaluate the code of varying train nums.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  9 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
import json
from glue_process import infer_glue, eval_glue

def glue():
    train_numls=[1, 2, 4, 8, 16, 32, 64, 100, 256, 300, ]
    method_ls=[
        "vanilla",
        "kd",
        # "lord",
        # "Complex-lord",
        ]

    res_dict={}
    save_dir_p="./res_vary_trainnum_glue/"
    if not os.path.exists(save_dir_p):
        os.makedirs(save_dir_p)

    task="cola"
    for tn in train_numls:
        for m in method_ls:
            pth=f"./POD_SAVE_CKPTs/vary_trainNum{task}{tn}{m}_256{task}"
            pth+="___finally/"


            res_pth=pth+f"___{task}_glue_infer_res"
            res_pth=res_pth.replace("/","__").replace(".", "")
            res_pth+=".json"
            if not os.path.exists(save_dir_p+res_pth):
                res_ls=infer_glue(pth, task, save_dir_p+res_pth)
            else:
                with open(save_dir_p+res_pth,
                          'r',encoding='utf8') as f:
                    res_ls=json.load(f,object_pairs_hook=OrderedDict)

            scores=eval_glue(task, res_ls)
            print(task, pth)
            print(scores)
            res_dict[task+"-----"+pth]=scores


## running entry
if __name__=="__main__":
    glue()
    print("EVERYTHING DONE.")


