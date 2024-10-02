"""
======================================================================
PLOT_FIDELITY ---

Draw the fidelity experiments.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
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

    color="white"
    interp="nearest"
    # interp="bilinear"
    
    victim_raw_ls=[
        "gpt-4o","gpt-4","gpt-3.5-turbo",
        ]
    victim_ls=[
        "GPT-4o","GPT-4","GPT-3.5",
        ]
    local_raw_ls=[
        "microsoft/Phi-3-mini-4k-instruct",
        "facebook/opt-6.7b",
        "Qwen/Qwen2-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        ]
    local_ls=[
        "Phi3-3.8B",
        "OPT-6.7B",
        "Qwen2-7B",
        "MistralV3-7B",
        "Llama3-8B",
        ]
    # res_mat=np.random.random((len(local_ls),len(victim_ls),))
    res_mat_dict={
        "BLEU":np.zeros(((len(local_ls),len(victim_ls),))),
        "Rouge-L":np.zeros(((len(local_ls),len(victim_ls),))),
        "BERTScore":np.zeros(((len(local_ls),len(victim_ls),))),
        }
    res_vic_dict={
        "BLEU":np.zeros(((len(victim_ls),))),
        "Rouge-L":np.zeros(((len(victim_ls),))),
        "BERTScore":np.zeros(((len(victim_ls),))),
        }

    with open("./d2t_cross_0704_dataset_res/Overall__d2t_varytrain_num_inference_scores.json",
              'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
        
    from pprint import pprint
    pprint(data)
    for i,local in enumerate(local_raw_ls):
        for j,victim in enumerate(victim_raw_ls):
            # ......
            keyName=f"e2e_nlg-----__cross_fidelity__text2sqle2e_nlgvanilla{victim}{local}___finally_____e2e_nlg_d2t_infer_resjson"
            keyName=keyName.replace("/","__").replace(".","")
            adict=data[keyName]

            bleu4_score=adict["bleu"]["4"]
            bertscore=adict["bertscore"]["f1"]
            rougel=adict["rouge-l"]["f1"]
            res_mat_dict["BLEU"][i,j]=bleu4_score
            res_mat_dict["Rouge-L"][i,j]=rougel
            res_mat_dict["BERTScore"][i,j]=bertscore

    with open("./d2t_cross_0704_dataset_res/victim_scores.json",
              'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
    # print(data)
    for i,victim in enumerate(victim_raw_ls):
        keyName=f"e2e_nlg-----{victim}___e2e_nlg_d2t_infer_resjson"
        keyName=keyName.replace("/","__").replace(".","")
        adict=data[keyName]

        bleu4_score=adict["bleu"]["4"]
        bertscore=adict["bertscore"]["f1"]
        rougel=adict["rouge-l"]["f1"]

        res_vic_dict["BLEU"][i]=bleu4_score
        res_vic_dict["Rouge-L"][i]=rougel
        res_vic_dict["BERTScore"][i]=bertscore

    pprint(res_mat_dict)
    pprint(res_vic_dict)
    for key in res_mat_dict:
        for i in range(len(local_ls)):
            for j in range(len(victim_ls)):
                res_mat_dict[key][i,j]/=res_vic_dict[key][j]
    
    # fig, axs = plt.subplots(1, 1, figsize=(9.5, 5.3))
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 9.5,))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)

    res_mat=res_mat_dict["BLEU"]
    fs = 26
    axs[0].imshow(res_mat,
               cmap=plt.cm.Reds,
               interpolation=interp,
               )
    axs[0].set_yticks([0, 1, 2, 3, 4,])
    axs[0].set_xticks([0, 1, 2])
    axs[0].set_yticklabels(local_ls, fontsize=fs, rotation=60,)
    axs[0].set_xticklabels(victim_ls, fontsize=fs,)
    axs[0].set_ylabel("Local Models", fontsize=fs)
    axs[0].set_xlabel("Victim Models", fontsize=fs)
    text = f"BLEU-4"
    axs[0].title.set_text(text)
    axs[0].title.set_fontsize(fs)

    gca=plt.gca()

    for (i,j), value in np.ndenumerate(res_mat):
        axs[0].text(j,i,"{:.2f}".format(value),
                 ha="center",
                 va="center",
                 color=color,
                 fontsize=fs+2,)

    ## second
    # res_mat=np.random.random((len(local_ls),len(victim_ls),))
    res_mat=res_mat_dict["Rouge-L"]
    axs[1].imshow(res_mat,
               cmap=plt.cm.Reds,
               interpolation=interp,
               )
    axs[1].set_yticks([0, 1, 2, 3, 4,])
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_yticklabels(["" for x in local_ls],
                           fontsize=fs, rotation=90,)
    axs[1].set_xticklabels(victim_ls, fontsize=fs,)
    # axs[1].set_ylabel("Local Models", fontsize=fs)
    axs[1].set_xlabel("Victim Models", fontsize=fs)
    text = f"Rouge-L (F1)"
    axs[1].title.set_text(text)
    axs[1].title.set_fontsize(fs)

    for (i,j), value in np.ndenumerate(res_mat):
        axs[1].text(j,i,"{:.2f}".format(value),
                 ha="center",
                 va="center",
                 color=color,
                 fontsize=fs,)

    ## third
    res_mat=res_mat_dict["BERTScore"]
    axs[2].imshow(res_mat,
               cmap=plt.cm.Reds,
               interpolation=interp,
               )
    axs[2].set_yticks([0, 1, 2, 3, 4,])
    axs[2].set_xticks([0, 1, 2])
    axs[2].set_yticklabels(["" for x in local_ls],
                           fontsize=fs-2,
                           rotation=90,)
    axs[2].set_xticklabels(victim_ls, fontsize=fs-2,)
    # axs[1].set_ylabel("Local Models", fontsize=fs)
    axs[2].set_xlabel("Victim Models", fontsize=fs)
    text = f"BERTScore (F1)"
    axs[2].title.set_text(text)
    axs[2].title.set_fontsize(fs)

    for (i,j), value in np.ndenumerate(res_mat):
        axs[2].text(j,i,"{:.2f}".format(value),
                 ha="center",
                 va="center",
                 color=color,
                 fontsize=fs,)


    save_path="fidelity-results.pdf"
    plt.savefig(save_path,
                pad_inches=0.1)
    print("SAVE DONE.")

## running entry
if __name__=="__main__":
    # main()
    visualize_heat()
    print("EVERYTHING DONE.")


