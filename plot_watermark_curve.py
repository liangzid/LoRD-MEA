"""
======================================================================
PLOT_WATERMARK_CURVE ---

Draw the curve of the watermarks.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 10 June 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from math import exp
import pickle
import torch.nn.functional as F
import torch
from glue_process import task_prompt_map
from gen_pipeline_open import InferObj

from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel
from glue_process import load_glue_datals

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
import sklearn
from collections import OrderedDict

from peft import PeftModel

def dictfindValue(data,key,dataset,method,lambda1):
    is_find=0
    for KEY in data.keys():
        if dataset in KEY and method in KEY and lambda1 in KEY:
            va=data[KEY][key]
            if key=="bertscore" or key=="rouge-l":
                va=va["f1"]
            is_find=1
            break
    if is_find==0:
        va=0.
    return va

def parse_json_file():
    lambda1_ls=[
        "00","02","04","06","08","10",
        ]
    dataset_ls=[
        "e2e_nlg","allenai/common_gen",
        "cs-en","de-en",
        ]
    d_label_ls={
        dataset_ls[0]:"E2E NLG",
        dataset_ls[1]:"CommonGen",
        dataset_ls[2]:"WMT (cs-en)",
        dataset_ls[3]:"WMT (de-en)",
        }
    dataset_ls=[
        # "e2e_nlg","allenai/common_gen",
        "cs-en","de-en",
        ]
    method_ls=[
        "vanilla","LoRD-VI","pretrained",
        ]
    method_label_ls={
        "vanilla":"MLE",
        "LoRD-VI":"LoRD",
        "pretrained":"Watermarked Victim Model",
        }
    key_ls=[
        "p_value","z_score","green_fraction","rouge-l",
        ]
    key_label_ls={
        key_ls[0]:"P-value(↑)",
        key_ls[1]:"Z-score(↓)",
        key_ls[2]:"WM Frac.(↓)",
        key_ls[3]:"Rouge-L(↑)",
        }

    # from collections import OrderedDict
    with open("./watermark_res/Overall__d2t_varytrain_num_inference_scores.json",
              'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)

    results=OrderedDict({})
    for key in key_ls:
        results[key_label_ls[key]]={}
        for dataset in dataset_ls:
            results[key_label_ls[key]][d_label_ls[dataset]]={}
            for method in method_ls:
                results[key_label_ls[key]][d_label_ls[dataset]]\
                    [method_label_ls[method]]=[]
                for lambda1 in lambda1_ls:
                    v=dictfindValue(data,key,dataset,method,lambda1)
                    results[key_label_ls[key]]\
                        [d_label_ls[dataset]]\
                        [method_label_ls[method]].append(v)
                if method=="vanilla" or method=="pretrained":
                    als=results[key_label_ls[key]]\
                        [d_label_ls[dataset]]\
                        [method_label_ls[method]]
                    vle=sum(als)/len(als)

                    results[key_label_ls[key]]\
                        [d_label_ls[dataset]]\
                        [method_label_ls[method]]=[vle for x in als]

    with open("./watermark_res/curve_results.json",
              'w',encoding='utf8') as f:
        json.dump(results,
                  f,ensure_ascii=False,indent=4)
    print("SAVE DONE.")
    return results


def main1():
    x_ls=[0.0,0.2,0.4,0.6,0.8,1.0]

    # pvaluels=OrderedDict({
    #     "cs-en":{
    #     "LoRD":[45.36,46.58,47.64,46.63,45.60,45.09,],
    #     "MLE":[45.28 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },

    #     "CommonGen":{
    #     "LoRD":[47.15,45.34,46.69,47.95,45.51,44.66],
    #     "MLE":[50.94 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },

    #     "E2E NLG":{
    #     "LoRD":[42.70,39.86,35.04,38.25,34.96,43.27,],
    #     "MLE":[37.99 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },
    #     })
    # zscorels=OrderedDict({
    #     "cs-en":{
    #     "LoRD":[0.21,0.14,0.10,0.14,0.18,0.20,],
    #     "MLE":[0.19 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },

    #     "CommonGen":{
    #     "LoRD":[0.11,0.17,0.13,0.09,0.19,0.21,],
    #     "MLE":[-0.02 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },
    #     "E2E NLG":{
    #     "LoRD":[0.28,0.39,0.52,0.43,0.55,0.25,],
    #     "MLE":[0.42 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },
    #     })
    # greenwordfracls=OrderedDict({
    #     "cs-en":{
    #     "LoRD":[0.27,0.27,0.26,0.26,0.27,0.27,],
    #     "MLE":[0.27 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },

    #     "CommonGen":{
    #     "LoRD":[0.26,0.27,0.26,0.26,0.27,0.28,],
    #     "MLE":[0.25 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },
    #     "E2E NLG":{
    #     "LoRD":[0.27,0.28,0.29,0.28,0.30,0.27],
    #     "MLE":[0.28 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },
    #     })

    # bleu1ls=OrderedDict({
    #     "E2E NLG":{
    #     "LoRD":[49.27,50.01,44.98,48.30,48.66,48.42,],
    #     "MLE":[0.28 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },

    #     "CommonGen":{
    #     "LoRD":[28.79,29.54,26.79,34.73,33.70,33.03,],
    #     "MLE":[22.80 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },

    #     "cs-en":{
    #     "LoRD":[53.64,53.78,49.77,55.12,54.95,47.74,],
    #     "MLE":[56.05 for x in x_ls],
    #     "Watermarked Victim Model":[0. for x in x_ls],
    #     },
    #     })

    # overall_data={
    #     "P-value":pvaluels,
    #     "Z-score":zscorels,
    #     "Watermark Frac.":greenwordfracls,
    #     "BLEU-1": bleu1ls,
    #     }
    
    overall_data=parse_json_file()

    # row_ls=["CommonGen", "E2E NLG",]
    # row_ls=["CommonGen", "cs-en",]
    row_ls=["WMT (cs-en)", "WMT (de-en)",]
    column_ls=["P-value(↑)",
               "Z-score(↓)",
               "WM Frac.(↓)",
               "Rouge-L(↑)",]

    method_ls=[
        "Watermarked Victim Model",
        "MLE",
        "LoRD",
        ]

    fig, axs = plt.subplots(2, 4, figsize=(20, 7.0))

    font_size = 21
    a=0.2
    lw=2.5
    marker = {
        method_ls[0]: "x",
        method_ls[1]: "s",
        method_ls[2]: "o",
    }
    model_color_dict = {
        method_ls[0]: "#eb3b5a",
        method_ls[1]: "#3867d6",
        method_ls[2]: "#eb3b5a",

    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0] : "#f78fb3",
        method_ls[1] : "#778beb",
        method_ls[2] : "#f78fb3",
    }

    model_line_style = {
        method_ls[0]: "-",
        method_ls[1]: "-.",
        method_ls[2]: "-",
    }

    method_ls=[
        # "Watermarked Victim Model",
        "MLE",
        "LoRD",
        ]

    for i_row, row in enumerate(row_ls):
        for i_col, col in enumerate(column_ls):
            data=overall_data[col][row]
            for method in method_ls:
                # print("data[method]",data[method])
                axs[i_row][i_col].plot(x_ls,
                                       data[method],
                                    label=method,
                                    linewidth=lw,
                                    marker=marker[method],
                                    markevery=1, markersize=15,
                                    markeredgewidth=lw,
                                    markerfacecolor='none',
                                    alpha=1.,
                                    linestyle=model_line_style[method],
                                    color=model_color_dict[method],
                                       )
                # axs[i_row][i_col].fill_between(x_ls,
                #                     ymin, ymax,
                #                     alpha=a,
                #                     linewidth=0.,
                #                     # alpha=1.0,
                #                     color=model_color_dict2[method])

            if i_row==0:
                xlabelname="WMT (cs-en)"
            else:
                xlabelname="WMT (de-en)\n$\lambda_1$"
            axs[i_row][i_col].set_xlabel(xlabelname,
                                         fontsize=font_size-3)
            axs[i_row][i_col].set_ylabel(col,
                                         fontsize=font_size-5)
            axs[i_row][i_col].set_xticks(x_ls, x_ls,
                                # rotation=48,
                                         size=font_size-4)
            axs[i_row][i_col].tick_params(axis='y',
                                    labelsize=font_size-6,
                                    rotation=65,
                                    width=2, length=2,
                                    pad=0, direction="in",
                                    which="both")
            
    font1 = {
        'weight': 'normal',
        'size': font_size-1,
    }

    plt.legend(loc=(-1.81, 2.60),
               prop=font1, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./watermarks.pdf",
                pad_inches=0.1)

## running entry
if __name__=="__main__":
    main1()
    print("EVERYTHING DONE.")


