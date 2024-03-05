"""
======================================================================
SUM_PROCESS ---

Summerization dataset preperation script.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  4 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import torch
from datasets import load_dataset
from openai import OpenAI as oa
# import time
import json
from collections import OrderedDict
import os
from math import exp
import random

import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from training_data_collecting_openai import chatWithOpenAI_APIs
from training_data_collecting_openai import chatWithOpenAI__LogLogits

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process

def load_sum_datals(tokenizer,
                    task_name="sum",
                    train_num=100,
                    model_name="gpt-3.5-turbo-1106",
                    topk=5,
                    max_length=1024,
                    openai_tmp_save_pth="./wmt_data_saveto_"):

    lm_tokenizer=tokenizer

    
    V = lm_tokenizer.vocab_size
    dataset_name = "UCL-DARK/openai-tldr-filtered"
    trainset_text = load_dataset(dataset_name,
                                 split=f"train[:{train_num}]")
    inp_ls = []

    for item in trainset_text:
        subreddit=item["subreddit"]
        title=item["title"]
        post=item["post"]
        summary=item["summary"]

        text=f"Subreddit: {subreddit} Title: {title} Post: {post}"
        
        inp_ls.append(text)

    pp= "Please summerize the content given by user."
    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    openai_tmp_save_pth += f"SUMtask_{task_name}-trainNUM_{train_num}.pkl"

    return commonly_used_openai_post_process(
        openai_tmp_save_pth,
        inp_ls,
        pp,
        model_name,
        topk,
        max_length,
        p_idxls,
        V,
        lm_tokenizer,
        )




## running entry
if __name__=="__main__":
    # main()
    print("EVERYTHING DONE.")


