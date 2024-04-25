"""
======================================================================
TEXT2SQL_PROCESS ---

Process of `text2sql` datasets.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 26 March 2024
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


def load_text2sql_datals(tokenizer,
                         task_name="wikisql",
                         train_num=100,
                         model_name="gpt-3.5-turbo-1106",
                         topk=5,
                         max_length=512,
                         openai_tmp_save_pth="./STEALED_PKLS/wmt_data_saveto_"):

    lm_tokenizer = tokenizer

    V = lm_tokenizer.vocab_size
    tasks_we_used = [
        "wikisql",
        "spider",
    ]
    assert task_name in tasks_we_used
    dataset_name = task_name
    inp_ls = []
    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:
            question = item["question"]
            table = item["table"]
            sql = item["sql"]["human_readable"]
            text = f"Qestion: {question}\n\n Table: {table}."
            inp_ls.append(text)
    elif task_name == tasks_we_used[1]:

        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:
            query = item["query"]
            question = item["question"]
            text = f"Question: {question}"
            inp_ls.append(text)
    assert inp_ls != []

    pp = "TEXT-TO-SQL: please return to me the SQL sentence based on the text (i.e., Question) and the table information (Table) provided by the User. "

    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    openai_tmp_save_pth += f"T2SQLtask_{task_name}-trainNUM_{train_num}.pkl"

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


def infer_t2s(modelname, task_name, res_pth,
              test_set_take_num=100,
              mnt=32
              ):

    save_pth = res_pth

    tasks_we_used = [
        "wikisql",
        "spider",
    ]

    assert task_name in tasks_we_used

    task_seqlen_map = {
        "wikisql": 1024,
        "spider": 512,
    }

    model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=task_seqlen_map[task_name])

    gen_pipeline = model.text_gen

    prompt = "TEXT-TO-SQL: please return to me the SQL sentence based on the text (i.e., Question) and the table information (Table) provided by the User. "
    pp = prompt

    inp_ls = []

    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(task_name,
                                     split=f"test")\
            .shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)
        for item in trainset_text:
            question = item["question"]
            table = item["table"]
            sql = item["sql"]["human_readable"]
            text = f"Qestion: {question}\n\n Table: {table}."

            inp_ls.append((text, sql))

    elif task_name == tasks_we_used[1]:

        trainset_text = load_dataset(task_name,
                                     split=f"validation")\
            .shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)

        for item in trainset_text:
            question = item["question"]
            table = item["table"]
            sql = item["sql"]["human_readable"]
            text = f"Qestion: {question}\n\n Table: {table}."

            inp_ls.append((text, sql))
    assert inp_ls != []

    res_ls = []
    for d in tqdm(inp_ls):
        inps, summary = d
        final_inps = "Instruction: " + pp +\
            " User: "+inps+" Assistant: "
        res = gen_pipeline(final_inps,
                           max_new_tokens=mnt,)[0]["generated_text"]
        res = res.split(final_inps)[1]
        res_ls.append((res, summary))

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
