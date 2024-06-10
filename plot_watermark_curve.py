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


def main1():
    x_ls=[0.0,0.2,0.4,0.6,0.8,1.0]
    pvaluels=OrderedDict({
        "CommonGen":{
        "LoRD":[42.15 for x in x_ls],
        "MLE":[3.73 for x in x_ls],
        "Watermarked Victim Model":[0. for x in x_ls],
        },
        "E2E NLG":{
        "LoRD":[32.98 for x in x_ls],
        "MLE":[21.40 for x in x_ls],
        "Watermarked Victim Model":[0. for x in x_ls],
        },
        })
    zscorels=OrderedDict({
        "CommonGen":{
        "LoRD":[0.19 for x in x_ls],
        "MLE":[1.78 for x in x_ls],
        "Watermarked Victim Model":[0. for x in x_ls],
        },
        "E2E NLG":{
        "LoRD":[0.44 for x in x_ls],
        "MLE":[0.79 for x in x_ls],
        "Watermarked Victim Model":[0. for x in x_ls],
        },
        })
    greenwordfracls=OrderedDict({
        "CommonGen":{
        "LoRD":[26.47 for x in x_ls],
        "MLE":[38.23 for x in x_ls],
        "Watermarked Victim Model":[0. for x in x_ls],
        },
        "E2E NLG":{
        "LoRD":[27.90 for x in x_ls],
        "MLE":[30.23 for x in x_ls],
        "Watermarked Victim Model":[0. for x in x_ls],
        },
        })

    overall_data={
        "P-value":pvaluels,
        "Z-score":zscorels,
        "Watermark Frac.":greenwordfracls,
        }

    row_ls=["CommonGen", "E2E NLG",]
    column_ls=["P-value", "Z-score", "Watermark Frac.",]

    method_ls=[
        "Watermarked Victim Model",
        "MLE",
        "LoRD",
        ]

    fig, axs = plt.subplots(2, 3, figsize=(15, 7.0))

    font_size = 21
    a=0.2
    lw=1.7
    marker = {
        method_ls[0]: "o",
        method_ls[1]: "s",
        method_ls[2]: "x",
    }
    model_color_dict = {
        method_ls[0]: "#eb3b5a",
        method_ls[1]: "#3867d6",
        method_ls[2]: "#3867d6",

    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0] : "#f78fb3",
        method_ls[1] : "#778beb",
        method_ls[2] : "#778beb",
    }

    model_line_style = {
        method_ls[0]: "-",
        method_ls[1]: "-.",
        method_ls[2]: "dotted",
    }

    for i_row, row in enumerate(row_ls):
        for i_col, col in enumerate(column_ls):
            data=overall_data[col][row]
            for method in method_ls:
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

            axs[i_row][i_col].set_xlabel("$\lambda_1$",
                                         fontsize=font_size)
            axs[i_row][i_col].set_ylabel(col,
                                         fontsize=font_size-5)
            axs[i_row][i_col].set_xticks(x_ls, x_ls,
                                rotation=48, size=font_size-4)
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

    plt.legend(loc=(-2.71, 2.60),
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


