"""
======================================================================
EVAL_VARY_TRAINNUM ---

Evaluate the code of varying train nums.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  9 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
import json
from glue_process import infer_glue, eval_glue
from collections import OrderedDict

from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
import sklearn
import sys
import numpy as np
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from wmt_process import eval_varying_train_num


def wmt_curve_trainNums():
    method_ls = [
        "vanilla",
        "kd",
        # "lord",
        # "Complex-lord",
    ]
    marker = {
        "vanilla": "s",
        "kd": "D",
        "Complex-lord": "o",
        "lord": "*",
    }
    model_color_dict = {
        "vanilla": "#4a148c",
        "kd": "#469de9",
        "Complex-lord": "#eb3b5a",
        "lord": "#3867d6",
    }

    model_color_dict2 = {
        "vanilla": "#9c27b0",
        "kd": "#98c8f3",
        "Complex-lord": "#f78fb3",
        "lord": "#778beb",
    }

    taskls = ["cs-en", "de-en", "fi-en",]
    train_times = ["1", "2", "3",]
    train_nums = ["4", "8", "16", "32", "64", "100", "256", "512"]

    res_dict = eval_varying_train_num()

    a = 0.4
    lw = 1.7
    model_line_style = {
        "vanilla": "-.",
        "kd": "-.",
        "Complex-lord": "-",
        "lord": "-",
    }
    font_size = 21

    paths_dict = {}
    adict = {}
    pdict = {}
    rdict = {}
    fdict = {}
    # using existing results of the paths.
    for m in method_ls:
        paths_dict[m] = []
        adict[m] = []
        pdict[m] = []
        rdict[m] = []
        fdict[m] = []
        for tn in train_numls:
            if m in ["vanilla", "kd"]:
                pth = prefix+f"{task}{tn}{m}_256{task}___"
                pth += "finally"
            else:
                pth = prefix+f"{task}{tn}{m}256{task}___"
                pth += "period2"
            paths_dict[m].append(pth)

            res_pth = pth+f"___{task}_glue_infer_res"
            res_pth = res_pth.replace("/", "__").replace(".", "")
            res_pth += ".json"

            if not os.path.exists(dir_p+res_pth):
                res_ls = infer_glue(pth, task, dir_p+res_pth)
            else:
                with open(dir_p+res_pth,
                          'r', encoding='utf8') as f:
                    res_ls = json.load(f, object_pairs_hook=OrderedDict)

            scores = eval_glue(task, res_ls)
            adict[m].append(scores[0])
            pdict[m].append(scores[1])
            rdict[m].append(scores[2])
            fdict[m].append(scores[3])

    res_dict = OrderedDict({"Accuracy": adict,
                           "Precision": pdict,
                            "Recall": rdict,
                            "F1 Score": fdict,
                            })
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.65))

    for i, ylabel in enumerate(list(res_dict.keys())):
        y_dict = res_dict[ylabel]
        for method in y_dict.keys():
            yls = y_dict[method]
            axs[i].set_xscale("log")
            axs[i].plot(
                train_numls,
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
            axs[i].set_xticks(train_numls, train_numls,
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
    plt.legend(loc=(-1.91, 1.0),
               prop=font_legend, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_trainNum_wmt16.pdf",
                pad_inches=0.1)
    print("Save DONE.")
    pass


def glue_curve_trainNums():
    train_numls = [1, 2, 4, 8, 16, 32, 64, 100, 256, 300, ]
    method_ls = [
        "vanilla",
        "kd",
        # "lord",
        "Complex-lord",
    ]

    res_dict = {}
    task = "cola"

    dir_p = "./res_vary_trainnum_glue/"
    prefix = "./POD_SAVE_CKPTs/vary_trainNum"

    marker = {
        "vanilla": "s",
        "kd": "D",
        "Complex-lord": "o",
        "lord": "*",
    }
    model_color_dict = {
        "vanilla": "#4a148c",
        "kd": "#469de9",
        "Complex-lord": "#eb3b5a",
        "lord": "#3867d6",
    }

    model_color_dict2 = {
        "vanilla": "#9c27b0",
        "kd": "#98c8f3",
        "Complex-lord": "#f78fb3",
        "lord": "#778beb",
    }

    a = 0.4
    lw = 1.7
    model_line_style = {
        "vanilla": "-.",
        "kd": "-.",
        "Complex-lord": "-",
        "lord": "-",
    }
    font_size = 21

    paths_dict = {}
    adict = {}
    pdict = {}
    rdict = {}
    fdict = {}
    # using existing results of the paths.
    for m in method_ls:
        paths_dict[m] = []
        adict[m] = []
        pdict[m] = []
        rdict[m] = []
        fdict[m] = []
        for tn in train_numls:
            if m in ["vanilla", "kd"]:
                pth = prefix+f"{task}{tn}{m}_256{task}___"
                pth += "finally"
            else:
                pth = prefix+f"{task}{tn}{m}256{task}___"
                pth += "period2"
            paths_dict[m].append(pth)

            res_pth = pth+f"___{task}_glue_infer_res"
            res_pth = res_pth.replace("/", "__").replace(".", "")
            res_pth += ".json"

            if not os.path.exists(dir_p+res_pth):
                res_ls = infer_glue(pth, task, dir_p+res_pth)
            else:
                with open(dir_p+res_pth,
                          'r', encoding='utf8') as f:
                    res_ls = json.load(f, object_pairs_hook=OrderedDict)

            scores = eval_glue(task, res_ls)
            adict[m].append(scores[0])
            pdict[m].append(scores[1])
            rdict[m].append(scores[2])
            fdict[m].append(scores[3])

    res_dict = OrderedDict({"Accuracy": adict,
                           "Precision": pdict,
                            "Recall": rdict,
                            "F1 Score": fdict,
                            })
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.65))

    for i, ylabel in enumerate(list(res_dict.keys())):
        y_dict = res_dict[ylabel]
        for method in y_dict.keys():
            yls = y_dict[method]
            axs[i].set_xscale("log")
            axs[i].plot(
                train_numls,
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
            axs[i].set_xticks(train_numls, train_numls,
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
    plt.legend(loc=(-1.91, 1.0),
               prop=font_legend, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_trainNum.pdf",
                pad_inches=0.1)
    print("Save DONE.")


# running entry
if __name__ == "__main__":
    # glue()
    curve_trainNums()
    print("EVERYTHING DONE.")
