"""
======================================================================
EVAL_VARY_PERIOD ---

Experiments of RL-based extraction methods that foucsing on the period
numbers.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  9 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
import json
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
import sklearn
import sys
import numpy as np
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
from collections import OrderedDict

from glue_process import infer_glue, eval_glue

# results already obtained! In the `glue_res` directory.
def glue():
    periods=[0,1,2,3,4,5,6,7,8,9,]
    method_ls=[
        # "lord",
        "Complex-lord",
        ]

    res_dict={}
    save_dir_p="./res_vary_period_glue/"
    if not os.path.exists(save_dir_p):
        os.makedirs(save_dir_p)

    task="cola"
    for period in periods:
        for m in method_ls:
            pth=f"./POD_SAVE_CKPTs/vary_period_complex/{m}256{task}1100"
            pth+=f"___period{period}/"


            res_pth=pth+f"___{task}_glue_infer_res"
            res_pth=res_pth.replace("/","__").replace(".", "")
            res_pth+=".json"
            if not os.path.exists(save_dir_p+res_pth):
                res_ls=infer_glue(pth, task, save_dir_p+res_pth)
            else:
                with open(save_dir_p+res_pth,
                          'r',encoding='utf8') as f:
                    res_ls=json.load(f,object_pairs_hook=OrderedDict)

            scores=eval_glue(task, res_ls)
            print(task, pth)
            print(scores)
            res_dict[task+"-----"+pth]=scores


def draw_curve():

    periods=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]
    method_ls=[
        # "lord",
        "Complex-lord",
        ]

    res_dict={}

    task="cola"
    dir_p="./glue_res/"
    prefix="./POD_SAVE_CKPTs/vary_period_complex/"

    marker = {
        "Complex-lord": "o",
        "lord": "s",
        }
    model_color_dict = {
        # "E": "#FF0202",
        # "I": "#008000",
        # "E": "red",
        # "I": "green",
        "Complex-lord": "#eb3b5a",
        "lord": "#3867d6",
        # "E":"#2d98da",
        }

    model_color_dict2 = {
        # "E": "#FF0202",
        # "I": "#008000",
        # "E": (252/255, 224/255, 225/255),
        # "I": (194/255, 232/255, 247/255),
        "Complex-lord" : "#f78fb3",
        "lord" : "#778beb",
    }

    a=0.4
    lw=1.7
    model_line_style = {
        "Complex-lord": "-",
        "lord": "-.",
    }
    font_size = 21

    paths_dict={}
    adict={}
    pdict={}
    rdict={}
    fdict={}
    # using existing results of the paths.
    for m in method_ls:
        paths_dict[m]=[]
        adict[m]=[]
        pdict[m]=[]
        rdict[m]=[]
        fdict[m]=[]
        for p in periods:
            pth=prefix+f"{m}256cola1100___period{p}"
            paths_dict[m].append(pth)

            res_pth=pth+f"___{task}_glue_infer_res.json"
            res_pth=res_pth.replace("/","__").replace(".", "")
            if not os.path.exists(dir_p+res_pth):
                res_ls=infer_glue(pth, task, dir_p+res_pth)
            else:
                # from collections import OrderedDict
                with open(dir_p+res_pth, 'r',encoding='utf8') as f:
                    res_ls=json.load(f,object_pairs_hook=OrderedDict)

            scores=eval_glue(task, res_ls)
            adict[m].append(scores[0])
            pdict[m].append(scores[1])
            rdict[m].append(scores[2])
            fdict[m].append(scores[3])

    res_dict=OrderedDict({"Accuracy":adict,
              "Precision":pdict,
              "Recall":rdict,
              "F1 Score":fdict,
              })
    fig, axs=plt.subplots(1,4,figsize=(20, 4.65))

    
    for i,ylabel in enumerate(list(res_dict.keys())):
        y_dict=res_dict[ylabel]
        for method in y_dict.keys():
            yls=y_dict[method]
            axs[i].plot(
                periods,
                yls,
                label=method,
                linewidth=lw,

                marker=marker[method],
                markevery=1, markersize=15,
                markeredgewidth=lw,
                markerfacecolor='none',
                alpha=1.,
                linestyle=model_line_style[method],
                color=model_color_dict[method]
                )

            # axs[0][i_n].fill_between(periods,
            # ymin, ymax,
            #                          alpha=a,
            #                          linewidth=0.,
            #                          # alpha=1.0,
            #                          color=model_color_dict2[mode])
            axs[i].set_xlabel("Training Periods",
                                   fontsize=font_size)
            axs[i].set_ylabel(ylabel,
                                   fontsize=font_size-5)
            axs[i].set_xticks(periods, periods,
                                   rotation=48, size=font_size-4)
            axs[i].tick_params(axis='y',
                                    labelsize=font_size-6,
                                    rotation=65,
                                    width=2, length=2,
                                    pad=0, direction="in",
                                    which="both")
    font_legend = {
        'weight': 'normal',
        'size': font_size-1,
    }
    plt.legend(loc=(-2.71, 2.60),
               prop=font_legend, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_periods.pdf",
                pad_inches=0.1)
    print("Save DONE.")

            

## running entry
if __name__=="__main__":
    # glue()
    draw_curve()
    print("EVERYTHING DONE.")

