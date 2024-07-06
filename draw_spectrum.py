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


def draw_spectrum(
        ):
    fidelity_dict=OrderedDict({
        "Translation":{
            "Czech-to-English":0.891,
            "German-to-English":0.903,
            # "ru-en":0.0,
            "Finnish-to-English":0.871,
            },
        "Summarization":{
            "TLDR":0.978,
            "SamSUM":0.956,
            "CNN Daily Mail":0.677,
            },
        "Reasoning":{
            "PiQA":0.947,
            "TruthfulQA":0.986,
            },
        "Data-to-Text":{
            "E2E NLG":0.977,
            "CommonGen":0.958,
            },
        "Text-to-SQL":{
            "WikiSQL":0.951,
            "Spider":0.862,
            },
        })
    scoreup_dict=OrderedDict({
        "Translation":{
            "Czech-to-English":1.55,
            "German-to-English":1.64,
            # "ru-en":0.0,
            "Finnish-to-English":1.52,
            },
        "Summarization":{
            "TLDR":1.10,
            "CNN Daily Mail":1.02,
            "SamSUM":1.18,
            },
        "Reasoning":{
            "PiQA":1.29,
            "TruthfulQA":1.02,
            },
        "Data-to-Text":{
            "E2E NLG":1.40,
            "CommonGen":1.60,
            },
        "Text-to-SQL":{
            "WikiSQL":1.71,
            "Spider":1.48,
            },
        })

    marker_dict={
        "Spider":"o", "WikiSQL":"s", "TLDR":"^", "CNN Daily Mail":"v",
        "SamSUM":">", "E2E NLG":"d", "CommonGen":"D", "PiQA":"X",
        "TruthfulQA":"P",
        "Czech-to-English":"h",
        "German-to-English":"p",
        "Finnish-to-English":"H",
        }
    cls=[
        "#66C2A5",
        "#FC8D62",
        "#8DA0CB",
        "#E78AC3",
        "#A6D854",
        "#FFD92F",
        "#E5C494",
        "#B3B3B3",
        ]
    color_dict={
        "Spider":cls[0], "WikiSQL":cls[0],
        "TLDR":cls[1], "CNN Daily Mail":cls[1], "SamSUM":cls[1],
        "E2E NLG":cls[2], "CommonGen":cls[2],
        "PiQA":cls[3], "TruthfulQA":cls[3],
        "Czech-to-English":cls[4],
        "German-to-English":cls[4],
        "Finnish-to-English":cls[4],
        }
    
    plt.style.use("seaborn-v0_8-dark")
    # plt.style.use("Solarize_Light2")
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 4.3))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)

    fs = 13
    axs.grid(True)
    compact_dict=OrderedDict({})
    for domain_key in fidelity_dict.keys():
        for task_key in fidelity_dict[domain_key].keys():
            x=fidelity_dict[domain_key][task_key]
            y=scoreup_dict[domain_key][task_key]
            compact_dict[task_key]=[x,y]
    for task in compact_dict.keys():
        x=compact_dict[task][0]
        y=compact_dict[task][1]
        axs.scatter(
            x,y,
            s=240,
            c=color_dict[task],
            facecolors=color_dict[task],
            edgecolors=color_dict[task],
            linewidths=0.0,
            marker=marker_dict[task],
            alpha=1.0,
            label=task,
            )

    font1 = {
        'weight': 'normal',
        'size': 9,
    }
    plt.legend(loc=(0.02, 0.03),
               prop=font1,
               ncol=2, frameon=True,
               markerscale=0.5,
               handletextpad=0., handlelength=1.2)  # 设置信息框


    # axs.set_yticks([0, 1, 2, 3, 4,])
    # axs.set_xticks([0, 1, 2])
    # axs.set_yticklabels(local_ls, fontsize=fs-2, rotation=45,)
    # axs.set_xticklabels(victim_ls, fontsize=fs-2,)
    axs.set_ylabel("Performance-Up", fontsize=fs,
                   fontfamily='DejaVu Sans', fontweight='normal',
                   )
    axs.set_xlabel("Fidelity", fontsize=fs,
                   fontfamily='DejaVu Sans', fontweight='normal',
                   )
    axs.set_xlim(0.835,1.00)

    # for (i,j), value in np.ndenumerate(res_mat):
    #     axs[0].text(j,i,"{:.2f}".format(value),
    #              ha="center",
    #              va="center",
    #              color="white",
    #              fontsize=fs+2,)


    save_path="spectrum-results.pdf"
    plt.savefig(save_path,
                pad_inches=0.1)
    print("SAVE DONE.")

## running entry
if __name__=="__main__":
    # main()
    draw_spectrum()
    print("EVERYTHING DONE.")


