"""
======================================================================
PLOT_DISTRIBUTION ---

Plot the distribution of generation among different checkpoints.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 13 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

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


def get_dist_mat(ckpt_pth, task_name,
                 select_num=3,
                 train_num=100, max_length=256,
                 max_new_tokens=3,
                 only_original=False,
                 ):

    # model = InferObj(model_name=ckpt_pth,
    #                  device="auto",
    #                  max_length=2047)
    # gen_pipeline = model.text_gen

    if task_name in task_prompt_map:
        prompt = task_prompt_map

    lm = AutoModelForCausalLM.from_pretrained(
        ckpt_pth,
        device_map="auto",
    )

    lm_tokenizer = AutoTokenizer.from_pretrained(ckpt_pth)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_pth)

    # if lm_tokenizer.pad_token is None:
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    raw_train_datals = load_glue_datals(
        tokenizer,
        task_name=task_name,
        train_num=train_num,
        max_length=max_length)

    p_ls, idx2ls, logits2ls, idx2_distls = raw_train_datals

    matrix_ls = []
    for i in range(select_num):
        inps = p_ls[i]
        inps = inps.to("cuda").unsqueeze(0)
        idx2 = torch.tensor(idx2ls[i], dtype=torch.long)\
            .to("cuda").unsqueeze(0)
        idx2_dist=torch.tensor(idx2_distls[i],dtype=torch.long)\
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
            ssll=sl
            per_data = logits2ls[i]
            sl = len(per_data)
            v = len(per_data[0])
            tmp_ts = torch.ones((sl, v), dtype=torch.float)
            for jjjj, per_token_logit in enumerate(per_data):
                tmp_ts[jjjj] = torch.tensor(per_token_logit,)
            original_logits = tmp_ts.numpy()[ssll-1:,][:5]
            matrix_ls.append(original_logits)

    return matrix_ls


def visualize_heat(
        lord_ckpt="./GLUE_ckpts/colaComplex-lord256100___period2/",
        ce_ckpt="./GLUE_ckpts/colavanilla256100___finally/",
        kd_ckpt="./GLUE_ckpts/colakd256100___finally/",
        select_num=8,
        save_path="distribute_heat_res.pdf",):

    origin_mat = get_dist_mat(ckpt_pth=lord_ckpt,
                              task_name="cola",
                              select_num=select_num,
                              train_num=100,
                              only_original=True,
                              )
    lord_mat = get_dist_mat(ckpt_pth=lord_ckpt,
                            task_name="cola",
                            select_num=select_num,
                            train_num=100,
                            only_original=False,
                            )
    ce_mat = get_dist_mat(ckpt_pth=ce_ckpt,
                          task_name="cola",
                          select_num=select_num,
                          train_num=100,
                          only_original=False,
                          )
    kd_mat = get_dist_mat(ckpt_pth=kd_ckpt,
                          task_name="cola",
                          select_num=select_num,
                          train_num=100,
                          only_original=False,
                          )

    res_dict = OrderedDict({"Victim model": origin_mat,
                            "LoRD": lord_mat,
                            "Cross-Entropy": ce_mat,
                            "Distillation": kd_mat,
                            })
    xls = list(res_dict.keys())

    fig, axs = plt.subplots(4, 8, figsize=(40, 3.7*4))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)

    fs = 17
    for col in range(select_num):
        for row in range(4):
            axs[row, col].imshow(res_dict[xls[row]][col],
                                 cmap=plt.cm.Blues,
                                 )
            axs[row, col].set_xlabel("Token Indexs", fontsize=fs)
            axs[row, col].set_ylabel("Generated Token", fontsize=fs)

            text = f"Distribution of\n{xls[row]}'s {col+1} Sample."
            axs[row, col].title.set_text(text)
            axs[row, col].title.set_fontsize(fs)

    plt.savefig(save_path,
                pad_inches=0.1)
    print("SAVE DONE.")


if __name__ == "__main__":
    visualize_heat()
