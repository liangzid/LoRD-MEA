"""
======================================================================
TEMP_QA_INFER ---

TEMPORAL INFERENCE FOR QA.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created: 23 April 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
import json

import random
from tqdm import tqdm

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process

import os
from collections import OrderedDict
from pprint import pprint

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from qa_process import *


def main3():
    taskls=["piqa","truthful_qa","allenai/ai2_arc",]
    score_dict={}
    for task in taskls:
        # dir_p = "./vary_train_num_qa_infers/"
        dir_p = "./qa_dataset_res/"
        res_dict = {}
        if not os.path.exists(dir_p):
            os.makedirs(dir_p)
        # ckpt="google/gemma-7b"
        ckpt="meta-llama/Meta-Llama-3-8B-Instruct"

        res_pth = ckpt + f"___{task}_qa_infer_res.json"
        res_pth = res_pth.replace("/", "__").replace(".", "")

        if not os.path.exists(dir_p + res_pth):
            print(dir_p+res_pth)
            print("file not exist.")
            res_ls = infer_qa(
                ckpt, task, dir_p + res_pth,
                # test_set_take_num=1000,
                test_set_take_num=500,
                mnt=32,
                # base_model_name=base_model,
            )
        else:
            print("directly loading")
            # from collections import OrderedDict
            with open(dir_p + res_pth, "r", encoding="utf8") as f:
                res_ls = json.load(
                    f, object_pairs_hook=OrderedDict)

        scores = eval_qaacc(task, res_ls)
        res_dict[task + "-----" + res_pth] = scores
        print(scores)
        score_dict[task]=scores
    print(score_dict)
    print("OVERALL Save DONE.")

def main2():
    base_model = "google/gemma-7b"


    task="piqa"

    dir_p = "./vary_train_num_qa_infers/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    ckpt="./LoRA-LoRD-ckptsvaryTrainNum___321piqaComplex-lord332164256___period2"
    res_pth = ckpt + f"___{task}_qa_infer_res.json"
    res_pth = res_pth.replace("/", "__").replace(".", "")

    if not os.path.exists(dir_p + res_pth):
        print(dir_p+res_pth)
        print("file not exist.")
        res_ls = infer_qa(
            ckpt, task, dir_p + res_pth,
            # test_set_take_num=1000,
            test_set_take_num=500,
            mnt=64,
            base_model_name=base_model,
        )
    else:
        print("directly loading")
        # from collections import OrderedDict
        with open(dir_p + res_pth, "r", encoding="utf8") as f:
            res_ls = json.load(
                f, object_pairs_hook=OrderedDict)

    scores = eval_qaacc(task, res_ls)
    res_dict[task + "-----" + res_pth] = scores
    print(scores)
    print("OVERALL Save DONE.")

def main():
    base_model = "google/gemma-7b"

    taskls = [
        "piqa",
        # "truthful_qa",
        # "allenai/ai2_arc",
    ]
    # mls = ["LoRD-II"]
    mls = ["LoRD-IV"]
    # mls=["vanilla"]
    # mls=["kd"]
    # mls=["vanilla", "kd", "LoRD-II", "LoRD-IV"]

    # mls = ["google/gemma-2b",]
    train_times = [
        "1",
        # "2",
        # "3",
    ]
    train_nums = ["4", "8", "16", "32", "64", "100", "256", "512"]
    # train_nums = ["4", "8", "16", "32",]
    train_nums = ["32",]
    period_nums = ["8"]

    dir_p = "./vary_train_num_qa_infers/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                print(f"Current task: {m}")
                for itime in train_times:
                    for periodnum in period_nums:
                        prefix = "./vArY_TrAiN_num_LoRA-LoRD-ckpts/"
                        if m == "google/gemma-2b" or\
                           m == "google/gemma-7b":
                            ckpt = m
                        elif m == "Complex-lord":
                            ckpt = (
                                prefix
                                + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___period2/"
                            )
                        elif "LoRD" in m:
                            ckpt = (
                                prefix
                                + f"varyTrainNum___{train_num}{itime}{task}{m}112164256___period{periodnum}/"
                            )
                        else:
                            ckpt = (
                                prefix
                                + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___finally"
                            )

                        if m == "google/gemma-2b" or\
                           m=="google/gemma-7b":
                            res_pth = ckpt + \
                                f"__{itime}_{task}_qa_infer_res.json"
                        else:
                            res_pth = ckpt + f"___{task}_qa_infer_res.json"

                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        if not os.path.exists(dir_p + res_pth):
                            print(dir_p+res_pth)
                            print("file not exist.")
                            res_ls = infer_qa(
                                ckpt, task, dir_p + res_pth,
                                # test_set_take_num=1000,
                                test_set_take_num=500,
                                mnt=64,
                                base_model_name=base_model,
                            )
                        else:
                            print("directly loading")
                            # from collections import OrderedDict
                            with open(dir_p + res_pth, "r", encoding="utf8") as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        scores = eval_qaacc(task, res_ls)
                        res_dict[task + "-----" + res_pth] = scores
                        print(scores)
    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)


## running entry
if __name__=="__main__":
    # main()
    # main2()
    main3()
    print("EVERYTHING DONE.")

