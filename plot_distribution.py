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
import scipy
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
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
                 use_opensource=0,
                 topk=5,
                 ):
    if task_name=="e2e_nlg":
        dataset_name="data2text"
    if task_name in ["de-en","ru-en"]:
        dataset_name="wmt16"

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
    elif dataset_name=="data2text":
        from data2text_process import load_data2text_datals
        raw_train_datals = load_data2text_datals(
            tokenizer,
            task_name=task_name,
            train_num=train_num,
            max_length=max_length,
            is_test=using_test_set,
            )
        p_ls, idx2ls, logits2ls, idx2_distls = raw_train_datals
    elif dataset_name=="wmt16":
        from wmt_process import load_wmt_datals
        raw_train_datals = load_wmt_datals(
            tokenizer,
            task_name=task_name,
            train_num=train_num,
            max_length=max_length,
            is_test=using_test_set,
            use_opensource=use_opensource,
            topk=topk,
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
            matrix_ls.append(sampled_logits[:topk])
        else:
            ssll = sl
            per_data = logits2ls[i]
            sl = len(per_data)
            v = len(per_data[0])
            tmp_ts = torch.ones((sl, v), dtype=torch.float)
            for jjjj, per_token_logit in enumerate(per_data):
                tmp_ts[jjjj] = torch.tensor(per_token_logit,)
            original_logits = tmp_ts.numpy()[ssll-1:,][:topk]
            matrix_ls.append(original_logits)

    return matrix_ls, idx2_distls


def visualize_heat(
        # lord_ckpt="./GLUE_ckpts/colaComplex-lord256100___period2/",
        # ce_ckpt="./GLUE_ckpts/colavanilla256100___finally/",
        # kd_ckpt="./POD_SAVE_CKPTs/vary_period0306cs-en/kd_256cs-en_newkd___finally/",

        ce_ckpt="./text2sql_ckpts/text2sqlspider161vanilla___finally/",
        lord_ckpt="./text2sql_ckpts/text2sqlspider161LoRD-VI___period256/",
        kd_ckpt=None,
        pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
        select_num=8,
        train_num=16,
        task_name="spider",
        save_path="distribute_heat_res.pdf",
        using_test_set=0,
        use_opensource=0,
        topk=5,
        ):

    origin_mat, idx2distls = get_dist_mat(ckpt_pth=lord_ckpt,
                              pretrained_model=pretrained_model_pth,
                              task_name=task_name,
                              select_num=select_num,
                              train_num=train_num,
                              only_original=True,
                              using_test_set=using_test_set,
                              use_opensource=use_opensource,
                              topk=topk,
                              )
    lord_mat,_ = get_dist_mat(ckpt_pth=lord_ckpt,
                            pretrained_model=pretrained_model_pth,
                            task_name=task_name,
                            select_num=select_num,
                            train_num=train_num,
                            only_original=False,
                            using_test_set=using_test_set,
                            use_opensource=use_opensource,
                            topk=topk,
                            )
    ce_mat,_ = get_dist_mat(ckpt_pth=ce_ckpt,
                          pretrained_model=pretrained_model_pth,
                          task_name=task_name,
                          select_num=select_num,
                          train_num=train_num,
                          only_original=False,
                          using_test_set=using_test_set,
                          use_opensource=use_opensource,
                          topk=topk,
                          )
    init_mat,_ = get_dist_mat(ckpt_pth=pretrained_model_pth,
                          pretrained_model=None,
                          task_name=task_name,
                          select_num=select_num,
                          train_num=train_num,
                          only_original=False,
                          using_test_set=using_test_set,
                          use_opensource=use_opensource,
                          topk=topk,
                          )
    # kd_mat = get_dist_mat(ckpt_pth=kd_ckpt,
    #                       task_name="cola",
    #                       select_num=select_num,
    #                       train_num=train_num,
    #                       only_original=False,
    #                       )

    # print(f"Shape of the distribution.")
    # print(f"Shape of the distribution: {len(origin_mat)}")
    # print(f"Shape of the distribution: {len(origin_mat[0])}")
    # print(f"Shape of the distribution: {origin_mat}")

    ikld,ie1,ie2,isc=distSim(origin_mat, init_mat) 
    lkld,le1,le2,lsc=distSim(origin_mat, lord_mat) 
    ckld,ce1,ce2,csc=distSim(origin_mat, ce_mat) 
    quantitive_res={
        "Victim Model":{"entropy":ie1,},

        "Local Model":{
            "KL":ikld,
            "Entropy":ie2,
            "Spearman Correlation":isc,
            },

        "LoRD":{
            "KL":lkld,
            "Entropy":le2,
            "Spearman Correlation":lsc,
            },

        "MLE":{
            "KL":ckld,
            "Entropy":ce2,
            "Spearman Correlation":csc,
            },
        }
    print(quantitive_res)
    print("---------------------------------------------------------")

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
    with open("./heat_res.pkkl", 'wb') as f:
        pickle.dump(res_dict, f,)

    with open("./heat_res.pkkl", 'rb') as f:
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

def visualize_heat_twolines(
        ce_ckpt="./text2sql_ckpts/text2sqlspider161vanilla___finally/",
        lord_ckpt="./text2sql_ckpts/text2sqlspider161LoRD-VI___period256/",
        kd_ckpt=None,
        pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
        select_num=8,
        train_num=16,
        task_name="spider",
        save_path="distribute_heat_res_overall.pdf",
        use_opensource=0,
        topk=5,
        ):
    # origin_mat, idx2distls = get_dist_mat(ckpt_pth=lord_ckpt,
    #                           pretrained_model=pretrained_model_pth,
    #                           task_name=task_name,
    #                           select_num=select_num,
    #                           train_num=train_num,
    #                           only_original=True,
    #                           using_test_set=0,
    #                           use_opensource=use_opensource,
    #                           topk=topk,
    #                           )
    # lord_mat,_ = get_dist_mat(ckpt_pth=lord_ckpt,
    #                         pretrained_model=pretrained_model_pth,
    #                         task_name=task_name,
    #                         select_num=select_num,
    #                         train_num=train_num,
    #                         only_original=False,
    #                         using_test_set=0,
    #                         use_opensource=use_opensource,
    #                         topk=topk,
    #                         )
    # ce_mat,_ = get_dist_mat(ckpt_pth=ce_ckpt,
    #                       pretrained_model=pretrained_model_pth,
    #                       task_name=task_name,
    #                       select_num=select_num,
    #                       train_num=train_num,
    #                       only_original=False,
    #                       using_test_set=0,
    #                       use_opensource=use_opensource,
    #                       topk=topk,
    #                       )
    # init_mat,_ = get_dist_mat(ckpt_pth=pretrained_model_pth,
    #                       pretrained_model=None,
    #                       task_name=task_name,
    #                       select_num=select_num,
    #                       train_num=train_num,
    #                       only_original=False,
    #                       using_test_set=0,
    #                       use_opensource=use_opensource,
    #                       topk=topk,
    #                       )

    # # print(f"Origin Mat's shape: {origin_mat.shape}.")
    # ikld,ie1,ie2,isc=distSim(origin_mat[2], init_mat[2]) 
    # lkld,le1,le2,lsc=distSim(origin_mat[2], lord_mat[2]) 
    # ckld,ce1,ce2,csc=distSim(origin_mat[2], ce_mat[2]) 
    # likld,_,_,lisc=distSim(init_mat[2], lord_mat[2]) 
    # cikld,_,_,cisc=distSim(init_mat[2], ce_mat[2]) 
    # quantitive_res={
    #     "Victim Model":{"entropy":ie1,},
    #     "Local Model":{
    #         "KL":ikld,
    #         "Entropy":ie2,
    #         "Spearman Correlation":isc,
    #         },
    #     "LoRD":{
    #         "KL":lkld,
    #         "Entropy":le2,
    #         "Spearman Correlation":lsc,
    #         },
    #     "MLE":{
    #         "KL":ckld,
    #         "Entropy":ce2,
    #         "Spearman Correlation":csc,
    #         },

    #     "LoRD-to-LocalModel":{
    #         "KL":likld,
    #         "Spearman Correlation":lisc,
    #         },
    #     "MLE-to-LocalModel":{
    #         "KL":cikld,
    #         "Spearman Correlation":cisc,
    #         },
    #     }
    # print(quantitive_res)
    # print("---------------------------------------------------------")

    # res_dict = OrderedDict({"Victim Model": origin_mat,
    #                         "Local Model": init_mat,
    #                         "LoRD": lord_mat,
    #                         "Cross-Entropy": ce_mat,
    #                         # "Distillation": kd_mat,
    #                         })
    # res_dict = OrderedDict({"Victim model": origin_mat,
    #                         "Local Model": init_mat,
    #                         "LoRD": lord_mat,
    #                         "Cross-Entropy": ce_mat,
    #                         # "Distillation": kd_mat,
    #                         })
    # with open("./heat_res.pkkl", 'wb') as f:
    #     pickle.dump(res_dict, f,)
    # with open("./heat_res.pkkl", 'rb') as f:
    #     res_tr_dict = pickle.load(f)

    # xls = list(res_tr_dict.keys())

    # origin_mat, idx2distls = get_dist_mat(ckpt_pth=lord_ckpt,
    #                           pretrained_model=pretrained_model_pth,
    #                           task_name=task_name,
    #                           select_num=select_num,
    #                           train_num=train_num,
    #                           only_original=True,
    #                           using_test_set=1,
    #                           use_opensource=use_opensource,
    #                           topk=topk,
    #                           )
    # lord_mat,_ = get_dist_mat(ckpt_pth=lord_ckpt,
    #                         pretrained_model=pretrained_model_pth,
    #                         task_name=task_name,
    #                         select_num=select_num,
    #                         train_num=train_num,
    #                         only_original=False,
    #                         using_test_set=1,
    #                         use_opensource=use_opensource,
    #                         topk=topk,
    #                         )
    # ce_mat,_ = get_dist_mat(ckpt_pth=ce_ckpt,
    #                       pretrained_model=pretrained_model_pth,
    #                       task_name=task_name,
    #                       select_num=select_num,
    #                       train_num=train_num,
    #                       only_original=False,
    #                       using_test_set=1,
    #                       use_opensource=use_opensource,
    #                       topk=topk,
    #                       )
    # init_mat,_ = get_dist_mat(ckpt_pth=pretrained_model_pth,
    #                       pretrained_model=None,
    #                       task_name=task_name,
    #                       select_num=select_num,
    #                       train_num=train_num,
    #                       only_original=False,
    #                       using_test_set=1,
    #                       use_opensource=use_opensource,
    #                       topk=topk,
    #                       )

    # ikld,ie1,ie2,isc=distSim(origin_mat[4], init_mat[4]) 
    # lkld,le1,le2,lsc=distSim(origin_mat[4], lord_mat[4]) 
    # ckld,ce1,ce2,csc=distSim(origin_mat[4], ce_mat[4]) 
    # likld,_,_,lisc=distSim(init_mat[4], lord_mat[4]) 
    # cikld,_,_,cisc=distSim(init_mat[4], ce_mat[4]) 
    # quantitive_res={
    #     "Victim Model":{"entropy":ie1,},
    #     "Local Model":{
    #         "KL":ikld,
    #         "Entropy":ie2,
    #         "Spearman Correlation":isc,
    #         },
    #     "LoRD":{
    #         "KL":lkld,
    #         "Entropy":le2,
    #         "Spearman Correlation":lsc,
    #         },
    #     "MLE":{
    #         "KL":ckld,
    #         "Entropy":ce2,
    #         "Spearman Correlation":csc,
    #         },

    #     "LoRD-to-LocalModel":{
    #         "KL":likld,
    #         "Spearman Correlation":lisc,
    #         },
    #     "MLE-to-LocalModel":{
    #         "KL":cikld,
    #         "Spearman Correlation":cisc,
    #         },
    #     }
    # print(quantitive_res)
    # print("---------------------------------------------------------")

    # res_dict = OrderedDict({"Victim Model": origin_mat,
    #                         "Local Model": init_mat,
    #                         "LoRD": lord_mat,
    #                         "Cross-Entropy": ce_mat,
    #                         # "Distillation": kd_mat,
    #                         })
    # res_dict = OrderedDict({"Victim model": origin_mat,
    #                         "Local Model": init_mat,
    #                         "LoRD": lord_mat,
    #                         "Cross-Entropy": ce_mat,
    #                         # "Distillation": kd_mat,
    #                         })
    # with open("./heat_res_overall.pkkl", 'wb') as f:
    #     pickle.dump({
    #         "tr":res_tr_dict,
    #         "te":res_dict,
    #         "x":xls,
    #         }, f,)

    with open("./heat_res_overall.pkkl", 'rb') as f:
        adict = pickle.load(f)
    xls=adict["x"]
    res_tr_dict=adict["tr"]
    res_te_dict=adict["te"]

    # for ky in res_tr_dict:
    #     res_tr_dict[ky]=np.exp(res_tr_dict[ky])
    # for ky in res_te_dict:
    #     res_te_dict[ky]=np.exp(res_te_dict[ky])
    # print(res_tr_dict)

    fig, axs = plt.subplots(2, 4, figsize=(15.5, 7.5))
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    # norm = mcolors.Normalize(vmin=0.1, vmax=1.1)
    norm = mcolors.Normalize(vmin=-5, vmax=0.0)
    fs = 24
    # cmap=plt.get_cmap("Blues")
    cmap=plt.get_cmap("GnBu")
    # interp="bilinear"
    interp="nearest"
    # interp="mitchell"
    for col in range(4):
        axs[0, col].imshow(res_tr_dict[xls[col]][2],
                                # cmap=plt.cm.Blues,
                                cmap=cmap,
                           norm=norm,
               interpolation=interp,
                                )
        if col==0:
            axs[0, col].set_ylabel("Generated Token\n(Train Set)",
                                   fontsize=fs)
        else:
            axs[0, col].set_yticklabels(["" for x in xls])
        axs[0, col].set_xticklabels(["" for x in xls])
        axs[0, col].set_xlabel(" ", fontsize=fs)
        # print("Type of dist idxes: ",type(idx2distls[col]))
        # axs[row, col].set_xticklabels(idx2distls[col])
        if col==3:
            text="MLE"
        else:
            text = f"{xls[col]}"
        axs[0, col].title.set_text(text)
        axs[0, col].title.set_fontsize(fs)

        axs[1, col].imshow(res_te_dict[xls[col]][4],
                                # cmap=plt.cm.Blues,
                                cmap=cmap,
                           norm=norm,
               interpolation=interp,
                           )
        axs[1, col].set_xlabel("Top-5 Tokens", fontsize=fs)
        if col==0:
            axs[1, col].set_ylabel("Generated Token\n(Test Set)",
                                   fontsize=fs)
        else:
            axs[1, col].set_yticklabels(["" for x in xls])
        axs[1, col].set_xticklabels(["" for x in xls])

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
    else:
        print("INFO: NOT prepared for KD.")

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
    for row in range(4):
        for col in range(select_num):
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


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def averaged_spearman(x1s,x2s):
    num_samples=x1s.shape[0]
    res_ls=[]
    for ns in range(num_samples):
        res=spearman_correlation(x1s[ns],x2s[ns])
        res_ls.append(res)
    return sum(res_ls)/len(res_ls)

import torch
def distSim(dist1,dist2):
    dist1=torch.tensor(dist1)
    dist2=torch.tensor(dist2)

    dist1=dist1.reshape(-1, 5)
    dist2=dist2.reshape(-1, 5)
    # print(dist1.shape, dist2.shape)

    exp_dist1=torch.exp(dist1)
    exp_dist2=torch.exp(dist2)

    kl_d=torch.nn.functional.kl_div(dist2, exp_dist1)
    entropy1=torch.\
        distributions.Categorical(
            exp_dist1).entropy()
    entropy2=torch.\
        distributions.Categorical(
            exp_dist2).entropy()
    entropy1=sum(entropy1)/len(entropy1)
    entropy2=sum(entropy2)/len(entropy2)
    spearman_c=averaged_spearman(dist1,dist2)

    # compute the KL divergence

    # dist1=[list(range(len(dist1[0]))) for x in dist1]

    # dist1=[np.argsort(-x) for x in dist1]
    # dist2=[np.argsort(-x) for x in dist2]
    # spearman_res=scipy.stats.spearmanr(dist1,dist2)
    # print(f"KL-D: {kl_d}")
    # print(f"Entropy 1: {entropy1}")
    # print(f"Entropy 2: {entropy2}")
    # print(f"SpearMan: {spearman_c}")
    # print(f"Spearman Results: {spearman_res}")
    # return kl_d,spearman_res
    return float(kl_d),float(entropy1),float(entropy2),float(spearman_c)

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
    #     save_path="distribute_heat_res.pdf",
    #     using_test_set=0,
    #     )

    # # visualize_heat()

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

    # visualize_3d(
    #     ce_ckpt="./text2sql_ckpts/text2sqlwikisql161vanilla___finally/",
    #     lord_ckpt="./text2sql_ckpts/text2sqlwikisql161LoRD-VI___period256/",
    #     kd_ckpt=None,
    #     pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
    #     select_num=8,
    #     train_num=16,
    #     task_name="wikisql",
    #     save_path="distribute_3d_res.pdf",
    #     using_test_set=1,
    #     )

    # visualize_3d()

    # visualize_heat(
    #     ce_ckpt="./visual_ckpts/wmte2e_nlg641vanilla___finally",
    #     lord_ckpt="./visual_ckpts/wmte2e_nlg641LoRD-VI___period512",
    #     kd_ckpt=None,
    #     pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
    #     select_num=8,
    #     train_num=16,
    #     task_name="e2e_nlg",
    #     save_path="distribute_heat_res.pdf",
    #     using_test_set=0,
    #     # is_opensource=0,
    #     topk=5,
    #     )

    # # visualize_heat()

    # visualize_heat(
    #     ce_ckpt="./visual_ckpts/wmte2e_nlg641vanilla___finally",
    #     lord_ckpt="./visual_ckpts/wmte2e_nlg641LoRD-VI___period512",
    #     kd_ckpt=None,
    #     pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
    #     select_num=8,
    #     train_num=16,
    #     task_name="e2e_nlg",
    #     save_path="distribute_heat_res_test.pdf",
    #     using_test_set=1,
    #     # is_opensource=0,
    #     topk=5,
    #     )

    visualize_heat_twolines(
        ce_ckpt="./visual_ckpts/wmte2e_nlg641vanilla___finally",
        lord_ckpt="./visual_ckpts/wmte2e_nlg641LoRD-VI___period512",
        kd_ckpt=None,
        pretrained_model_pth="meta-llama/Meta-Llama-3-8B-Instruct",
        select_num=8,
        train_num=16,
        task_name="e2e_nlg",
        save_path="distribute_heat_res2x4.pdf",
        # is_opensource=0,
        topk=5,
        )
