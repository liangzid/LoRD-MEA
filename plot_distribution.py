"""
======================================================================
PLOT_DISTRIBUTION ---

Plot the distribution of generation among different checkpoints.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 13 March 2024
======================================================================
"""

import os
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

# ------------------------ Code --------------------------------------

from math import exp
import pickle
# normal import
import torch.nn.functional as F
import torch
from glue_process import task_prompt_map
from gen_pipeline_open import InferObj
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel
from glue_process import load_glue_datals

import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict

from peft import PeftModel


def get_dist_mat(ckpt_pth, task_name,
                 pretrained_model=None,
                 select_num=3,
                 train_num=100, max_length=256,
                 max_new_tokens=3,
                 only_original=False,
                 dataset_name="text2sql",
                 using_test_set=0,
                 ):

    # model = InferObj(model_name=ckpt_pth,
    #                  device="auto",
    #                  max_length=2047)
    # gen_pipeline = model.text_gen

    if task_name in task_prompt_map:
        prompt = task_prompt_map

    if pretrained_model is None:
        lm = AutoModelForCausalLM.from_pretrained(
            ckpt_pth,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        lm_tokenizer = AutoTokenizer.from_pretrained(ckpt_pth)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_pth)
    else:
        base_model_name=pretrained_model 
        lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            # trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        lm = PeftModel.from_pretrained(lm, ckpt_pth,)
        lm_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # if lm_tokenizer.pad_token is None:
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    if dataset_name=="glue":
        raw_train_datals = load_glue_datals(
            tokenizer,
            task_name=task_name,
            train_num=train_num,
            max_length=max_length)

        p_ls, idx2ls, logits2ls, idx2_distls = raw_train_datals

    elif dataset_name=="qa":
        from qa_process import load_qa_datals
        raw_train_datals = load_qa_datals(
            tokenizer,
            task_name=task_name,
            train_num=train_num,
            max_length=max_length)
        p_ls, idx2ls, logits2ls, idx2_distls = raw_train_datals
    elif dataset_name=="text2sql":
        from text2sql_process import load_text2sql_datals
        raw_train_datals = load_text2sql_datals(
            tokenizer,
            task_name=task_name,
            train_num=train_num,
            max_length=max_length,
            is_test=using_test_set,
            )
        p_ls, idx2ls, logits2ls, idx2_distls = raw_train_datals

    matrix_ls = []
    for i in range(select_num):
        inps = p_ls[i]
        inps = inps.to("cuda").unsqueeze(0)
        idx2 = torch.tensor(idx2ls[i], dtype=torch.long)\
            .to("cuda").unsqueeze(0)
        idx2_dist = torch.tensor(idx2_distls[i], dtype=torch.long)\
            .to("cuda").unsqueeze(0)
        # genidx = lm.generate(inps,
        #                      do_sample=True,
        #                      max_length=max_length,
        #                      max_new_tokens=max_new_tokens)
        sl = inps.shape[1]
        with torch.no_grad():
            all_logits = lm(idx2).logits
        gen_logits = F.log_softmax(
            all_logits[:, :-1,],
            dim=-1)

        # 1, max_new_tokens, 5
        sampled_logits = torch.gather(gen_logits, 2, idx2_dist)
        sampled_logits = sampled_logits[:, sl-1:,]

        # max_new_tokens, 5
        sampled_logits = sampled_logits.squeeze(0).to("cpu").numpy()
        # print(sampled_logits)
        if not only_original:
            matrix_ls.append(sampled_logits[:5])
        else:
            ssll = sl
            per_data = logits2ls[i]
            sl = len(per_data)
            v = len(per_data[0])
            tmp_ts = torch.ones((sl, v), dtype=torch.float)
            for jjjj, per_token_logit in enumerate(per_data):
                tmp_ts[jjjj] = torch.tensor(per_token_logit,)
            original_logits = tmp_ts.numpy()[ssll-1:,][:5]
            matrix_ls.append(original_logits)

    return matrix_ls, idx2_distls


def visualize_heat(
        # lord_ckpt="./GLUE_ckpts/colaComplex-lord256100___period2/",
        # ce_ckpt="./GLUE_ckpts/colavanilla256100___finally/",
        # kd_ckpt="./POD_SAVE_CKPTs/vary_period0306cs-en/kd_256cs-en_newkd___finally/",

        lord_ckpt="./text2sql_ckpts/text2sqlspider161vanilla___finally/",
        ce_ckpt="./text2sql_ckpts/text2sqlspider161LoRD-VI___period256/",
        kd_ckpt=None,
        pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
        select_num=8,
        train_num=16,
        task_name="spider",
        save_path="distribute_heat_res.pdf",
        using_test_set=0,
        ):

    origin_mat, idx2distls = get_dist_mat(ckpt_pth=lord_ckpt,
                              pretrained_model=pretrained_model_pth,
                              task_name=task_name,
                              select_num=select_num,
                              train_num=train_num,
                              only_original=True,
                              using_test_set=using_test_set,
                              )
    lord_mat,_ = get_dist_mat(ckpt_pth=lord_ckpt,
                            pretrained_model=pretrained_model_pth,
                            task_name=task_name,
                            select_num=select_num,
                            train_num=train_num,
                            only_original=False,
                            using_test_set=using_test_set,
                            )
    ce_mat,_ = get_dist_mat(ckpt_pth=ce_ckpt,
                          pretrained_model=pretrained_model_pth,
                          task_name=task_name,
                          select_num=select_num,
                          train_num=train_num,
                          only_original=False,
                          using_test_set=using_test_set,
                          )
    init_mat,_ = get_dist_mat(ckpt_pth=pretrained_model_pth,
                          pretrained_model=None,
                          task_name=task_name,
                          select_num=select_num,
                          train_num=train_num,
                          only_original=False,
                          using_test_set=using_test_set,
                          )
    # kd_mat = get_dist_mat(ckpt_pth=kd_ckpt,
    #                       task_name="cola",
    #                       select_num=select_num,
    #                       train_num=train_num,
    #                       only_original=False,
    #                       )

    res_dict = OrderedDict({"Victim Model": origin_mat,
                            "Local Model": init_mat,
                            "LoRD": lord_mat,
                            "Cross-Entropy": ce_mat,
                            # "Distillation": kd_mat,
                            })

    res_dict = OrderedDict({"Victim model": origin_mat,
                            "Local Model": init_mat,
                            "LoRD": lord_mat,
                            "Cross-Entropy": ce_mat,
                            # "Distillation": kd_mat,
                            })
    with open("./3d_res.pkkl", 'wb') as f:
        pickle.dump(res_dict, f,)

    with open("./3d_res.pkkl", 'rb') as f:
        res_dict = pickle.load(f)


    xls = list(res_dict.keys())

    fig, axs = plt.subplots(4, 8, figsize=(35, 3.7*4))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)

    fs = 17
    for col in range(select_num):
        for row in range(4):
            axs[row, col].imshow(res_dict[xls[row]][col],
                                 cmap=plt.cm.Blues,
                                 )
            axs[row, col].set_xlabel("Top-5 Tokens", fontsize=fs)
            axs[row, col].set_ylabel("Generated Token", fontsize=fs)
            # print("Type of dist idxes: ",type(idx2distls[col]))
            # axs[row, col].set_xticklabels(idx2distls[col])

            text = f"Distribution of\n{xls[row]}'s {col+1} Sample."
            axs[row, col].title.set_text(text)
            axs[row, col].title.set_fontsize(fs)

    plt.savefig(save_path,
                pad_inches=0.1)
    print("SAVE DONE.")


def visualize_3d(
        lord_ckpt="./text2sql_ckpts/text2sqlwikisql161vanilla___finally/",
        ce_ckpt="./GLUE_ckpts/colavanilla256100___finally/",
        # kd_ckpt="./GLUE_ckpts/colakd256100___finally/",
        kd_ckpt="./POD_SAVE_CKPTs/vary_period0306cs-en/kd_256cs-en_newkd___finally/",
        pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
        select_num=8,
        train_num=16,
        task_name="spider",
        save_path="distribute_3d_res.pdf",
        using_test_set=0,
):

    origin_mat,_ = get_dist_mat(ckpt_pth=lord_ckpt,
                              pretrained_model=pretrained_model_pth,
                              task_name=task_name,
                              select_num=select_num,
                              train_num=train_num,
                              only_original=True,
                              using_test_set=using_test_set,
                              )
    lord_mat,_ = get_dist_mat(ckpt_pth=lord_ckpt,
                            pretrained_model=pretrained_model_pth,
                            task_name=task_name,
                            select_num=select_num,
                            train_num=train_num,
                            only_original=False,
                            using_test_set=using_test_set,
                            )
    ce_mat,_ = get_dist_mat(ckpt_pth=ce_ckpt,
                          pretrained_model=pretrained_model_pth,
                          task_name=task_name,
                          select_num=select_num,
                          train_num=train_num,
                          only_original=False,
                          using_test_set=using_test_set,
                          )

    init_mat,_ = get_dist_mat(ckpt_pth=pretrained_model_pth,
                          pretrained_model=None,
                          task_name=task_name,
                          select_num=select_num,
                          train_num=train_num,
                          only_original=False,
                          using_test_set=using_test_set,
                          )

    if kd_ckpt is not None:
        kd_mat,_ = get_dist_mat(ckpt_pth=kd_ckpt,
                            pretrained_model=pretrained_model_pth,
                            task_name=task_name,
                            select_num=select_num,
                            train_num=train_num,
                            only_original=False,
                            using_test_set=using_test_set,
                            )

    res_dict = OrderedDict({"Victim Model": origin_mat,
                            "Local Model": init_mat,
                            "LoRD": lord_mat,
                            "Cross-Entropy": ce_mat,
                            # "Distillation": kd_mat,
                            })
    with open("./3d_res.pkkl", 'wb') as f:
        pickle.dump(res_dict, f,)

    with open("./3d_res.pkkl", 'rb') as f:
        res_dict = pickle.load(f)

    xls = list(res_dict.keys())

    fig = plt.figure(figsize=(40, 3.7*4))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)

    fs = 13
    ii = 0
    for col in range(select_num):
        for row in range(4):
            axs = fig.add_subplot(4, 8, ii+1, projection="3d")
            ii += 1
            s1, s2 = res_dict[xls[row]][col].shape
            x = np.array([[x,]*s2 for x in range(s1)]).flatten()
            y = np.array(list(range(s1))*s2).flatten()
            dz = res_dict[xls[row]][col].flatten()
            dz = [exp(x) for x in dz]
            dx = dy = 1
            z = 0

            axs.bar3d(x, y, z, dx, dy, dz,
                      shade=True,
                      )

            axs.set_xlim(0, 5)
            axs.set_ylim(0, 5)
            axs.set_zlim(0, 1)

            axs.set_ylabel("Token Indexs", fontsize=fs)
            axs.set_zlabel("Generated Token", fontsize=fs)
            axs.set_zlabel("log Probality", fontsize=fs)

            text = f"Distribution of\n{xls[row]}'s {col+1} Sample."
            axs.title.set_text(text)
            axs.title.set_fontsize(fs)

    plt.savefig(save_path,
                pad_inches=0.1)
    print("SAVE DONE.")


if __name__ == "__main__":
    # visualize_heat()

    # visualize_heat(
    #     lord_ckpt="./text2sql_ckpts/text2sqlspider161vanilla___finally/",
    #     ce_ckpt="./text2sql_ckpts/text2sqlspider161LoRD-VI___period256/",
    #     kd_ckpt=None,
    #     pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
    #     select_num=8,
    #     train_num=16,
    #     task_name="spider",
    #     save_path="distribute_heat_res.pdf",)

    # visualize_heat(
    #     lord_ckpt="./text2sql_ckpts/text2sqlwikisql161vanilla___finally/",
    #     ce_ckpt="./text2sql_ckpts/text2sqlwikisql161LoRD-VI___period256/",
    #     kd_ckpt=None,
    #     pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
    #     select_num=8,
    #     train_num=16,
    #     task_name="wikisql",
    #     save_path="distribute_heat_res.pdf",)

    # visualize_heat(
    #     ce_ckpt="./text2sql_ckpts/text2sqlwikisql161vanilla___finally/",
    #     lord_ckpt="./text2sql_ckpts/text2sqlwikisql161LoRD-VI___period256/",
    #     kd_ckpt=None,
    #     pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
    #     select_num=8,
    #     train_num=16,
    #     task_name="wikisql",
    #     save_path="distribute_heat_res_test.pdf",
    #     using_test_set=1,
    #     )

    visualize_3d(
        ce_ckpt="./text2sql_ckpts/text2sqlwikisql161vanilla___finally/",
        lord_ckpt="./text2sql_ckpts/text2sqlwikisql161LoRD-VI___period256/",
        kd_ckpt=None,
        pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
        select_num=8,
        train_num=16,
        task_name="wikisql",
        save_path="distribute_heat_res_test.pdf",
        using_test_set=1,
        )

    # visualize_3d()
