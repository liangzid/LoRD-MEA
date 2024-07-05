"""
======================================================================
PLOT_FIDELITY ---

Draw the fidelity experiments.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  5 July 2024
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
from gen_pipeline_open import InferObj

from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
import sklearn
from collections import OrderedDict

from peft import PeftModel

def visualize_heat(
        ):

    # victim_ls=[
    #     "gpt-4o","gpt-4","gpt-3.5-turbo",
    #     ]
    victim_ls=[
        "GPT-4o","GPT-4","GPT-3.5",
        ]
    # local_ls=[
    #     "microsoft/Phi-3-mini-4k-instruct",
    #     "Qwen/Qwen2-7B-Instruct",
    #     "facebook/opt-6.7b",
    #     "mistralai/Mistral-7B-Instruct-v0.3",
    #     "meta-llama/Meta-Llama-3-8B-Instruct",
    #     ]
    local_ls=[
        "Phi-3",
        "Qwen2-7B",
        "OPT-6.7B",
        "Mistral-7B",
        "Llama3-8B",
        ]
    res_mat=np.random.random((len(local_ls),len(victim_ls),))
    
    # fig, axs = plt.subplots(1, 1, figsize=(9.5, 5.3))
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 9.5,))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)

    fs = 19
    axs[0].imshow(res_mat,
               cmap=plt.cm.Reds,
               interpolation="nearest",
               )
    axs[0].set_yticks([0, 1, 2, 3, 4,])
    axs[0].set_xticks([0, 1, 2])
    axs[0].set_yticklabels(local_ls, fontsize=fs-2, rotation=45,)
    axs[0].set_xticklabels(victim_ls, fontsize=fs-2,)
    axs[0].set_ylabel("Local Models", fontsize=fs)
    axs[0].set_xlabel("Victim Models", fontsize=fs)
    text = f"BLEU"
    axs[0].title.set_text(text)
    axs[0].title.set_fontsize(fs)

    gca=plt.gca()

    for (i,j), value in np.ndenumerate(res_mat):
        axs[0].text(j,i,"{:.2f}".format(value),
                 ha="center",
                 va="center",
                 color="white",
                 fontsize=fs+2,)

    ## second
    res_mat=np.random.random((len(local_ls),len(victim_ls),))
    axs[1].imshow(res_mat,
               cmap=plt.cm.Reds,
               interpolation="nearest",
               )
    axs[1].set_yticks([0, 1, 2, 3, 4,])
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_yticklabels(["" for x in local_ls],
                           fontsize=fs-2, rotation=90,)
    axs[1].set_xticklabels(victim_ls, fontsize=fs-2,)
    # axs[1].set_ylabel("Local Models", fontsize=fs)
    axs[1].set_xlabel("Victim Models", fontsize=fs)
    text = f"Rouge-L"
    axs[1].title.set_text(text)
    axs[1].title.set_fontsize(fs)

    for (i,j), value in np.ndenumerate(res_mat):
        axs[1].text(j,i,"{:.2f}".format(value),
                 ha="center",
                 va="center",
                 color="white",
                 fontsize=fs+2,)

    ## third
    res_mat=np.random.random((len(local_ls),len(victim_ls),))
    axs[2].imshow(res_mat,
               cmap=plt.cm.Reds,
               interpolation="nearest",
               )
    axs[2].set_yticks([0, 1, 2, 3, 4,])
    axs[2].set_xticks([0, 1, 2])
    axs[2].set_yticklabels(["" for x in local_ls],
                           fontsize=fs-2,
                           rotation=90,)
    axs[2].set_xticklabels(victim_ls, fontsize=fs-2,)
    # axs[1].set_ylabel("Local Models", fontsize=fs)
    axs[2].set_xlabel("Victim Models", fontsize=fs)
    text = f"BERTScore"
    axs[2].title.set_text(text)
    axs[2].title.set_fontsize(fs)

    for (i,j), value in np.ndenumerate(res_mat):
        axs[2].text(j,i,"{:.2f}".format(value),
                 ha="center",
                 va="center",
                 color="white",
                 fontsize=fs+2,)







    save_path="fidelity-results.pdf"
    plt.savefig(save_path,
                pad_inches=0.1)
    print("SAVE DONE.")

## running entry
if __name__=="__main__":
    # main()
    visualize_heat()
    print("EVERYTHING DONE.")


