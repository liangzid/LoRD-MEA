"""
======================================================================
EVAL_VARY_TRAINNUM ---

Evaluate the code of varying train nums.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  9 March 2024
======================================================================
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

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
from wmt_process import eval_wmt, infer_wmt
from qa_process import infer_qa, eval_qaacc


def qa_curve_trainNums():
    method_ls = [
        "vanilla",
        # "kd",
        # "google/gemma-2b",
        # "lord",
        # "Complex-lord",
        "LoRD-VI",
        # "LoRD-IV",
    ]

    marker = {
        "vanilla": "s",
        # "kd": "D",
        # "Complex-lord": "o",
        # "LoRD-II": "v",
        # "LoRD-IV": "^",
        "LoRD-VI": "o",
        # "google/gemma-2b": "*",
    }

    model_color_dict = {
        "vanilla": "#4a148c",
        "LoRD-VI": "#eb3b5a",
        # "kd": "#469de9",
        # "Complex-lord": "#eb3b5a",
        # "LoRD-II": "#eb3b5a",
        # "LoRD-IV": "#eb3b5a",
        # "google/gemma-2b": "#3867d6",
    }

    model_color_dict2 = {
        "vanilla": "#9c27b0",
        "LoRD-VI": "#f78fb3",
        # "kd": "#98c8f3",
        # "Complex-lord": "#f78fb3",
        # "LoRD-II": "#f78fb3",
        # "LoRD-IV": "#f78fb3",
        # "google/gemma-2b": "#778beb",
    }

    taskls = [
        "piqa",
        "truthful_qa",
        # "allenai/ai2_arc",
    ]

    train_times = [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]

    train_nums = [
        # "4",
        "8", "16", "32", "64", "128", "256", "512",
                  "1024",
                  ]

    a = 0.4
    lw = 1.7
    model_line_style = {
        "vanilla": "-.",
        "LoRD-VI": "-",
        # "kd": "-.",
        # "Complex-lord": "-",
        # "LoRD-II": "-",
        # "LoRD-IV": "-",
        # "google/gemma-2b": "-",
    }
    font_size = 21

    results_dict = {}
    acc_dict = {}
    p_dict = {}
    r_dict = {}
    f1_dict = {}

    infer_save_pth = "./vary_query_num_overall_res_qa.json"

    if not os.path.exists(infer_save_pth):
        dir_p = "./qa_0630_dataset_res/"
        # using existing results of the paths.
        prefix = "./NEW_VARYING_QUERYTIME_CKPTS/text2sql"
        for task in taskls:
            results_dict[task] = {}
            acc_dict[task] = {}
            p_dict[task] = {}
            r_dict[task] = {}
            f1_dict[task] = {}

            for m in method_ls:
                results_dict[task][m] = {}
                acc_dict[task][m] = {}
                p_dict[task][m] = {}
                r_dict[task][m] = {}
                f1_dict[task][m] = {}
                for tn in train_nums:
                    results_dict[task][m][tn] = {}
                    acc_dict[task][m][tn] = {}
                    p_dict[task][m][tn] = {}
                    r_dict[task][m][tn] = {}
                    f1_dict[task][m][tn] = {}
                    for train_time in train_times:
                        results_dict[task][m][tn][train_time] = {}
                        acc_dict[task][m][tn][train_time] = {}
                        p_dict[task][m][tn][train_time] = {}
                        r_dict[task][m][tn][train_time] = {}
                        f1_dict[task][m][tn][train_time] = {}

                        if m == "google/gemma-2b":
                            pth = m
                        else:
                            pth = prefix + \
                                f"{task}{tn}{train_time}{m}___"

                            if m in [
                                "vanilla",
                                "kd",
                            ]:
                                pth += "finally/"
                            elif tn=="1024":
                                pth += "period1024/"
                            elif tn=="2048":
                                pth += "period2048/"
                            else:
                                pth += "period512/"

                        if m == "google/gemma-2b":
                            res_pth = pth + \
                                f"__{train_time}_{task}_qa_infer_res.json"
                        else:
                            res_pth = pth + f"___{task}_qa_infer_res.json"
                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        print(f"Targeted found pth: {dir_p+res_pth}.")
                        if not os.path.exists(dir_p + res_pth):
                            print("ERORR..")
                            return -10000
                        else:
                            # from collections import OrderedDict
                            with open(dir_p + res_pth,
                                      "r", encoding="utf8") as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        ss = eval_qaacc(task, res_ls)
                        results_dict[task][m][tn][train_time] = ss
                        acc_dict[task][m][tn][train_time] = ss[0]
                        p_dict[task][m][tn][train_time] = ss[1]
                        r_dict[task][m][tn][train_time] = ss[2]
                        f1_dict[task][m][tn][train_time] = ss[3]
        with open(infer_save_pth, "w", encoding="utf8") as f:
            json.dump(
                [results_dict, acc_dict, p_dict, r_dict, f1_dict],
                f,
                ensure_ascii=False,
                indent=4,
            )
    else:
        # from collections import OrderedDict
        with open(infer_save_pth, "r", encoding="utf8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            results_dict, acc_dict, p_dict, r_dict, f1_dict = data

    res_dict = {}
    for task in taskls:
        res_dict[task] = {}
        res_dict[task]["Accurary"] = acc_dict[task]
        res_dict[task]["Precision"] = p_dict[task]
        res_dict[task]["Recall"] = r_dict[task]
        res_dict[task]["F1"] = f1_dict[task]

    fig, axs = plt.subplots(2, 4, figsize=(20.8, 9.04))

    for i, task in enumerate(list(res_dict.keys())):
        for j, metricName in enumerate(list(res_dict[task].keys())):
            adict = res_dict[task][metricName]
            for method in adict.keys():
                ylss = []
                for tn in train_nums:
                    templs = []
                    for tt in train_times:
                        templs.append(adict[method][tn][tt])
                    ylss.append(templs)
                ylss = np.array(ylss)

                xls = [float(x) for x in train_nums]

                ymeanls = np.mean(ylss, axis=1)
                # ymaxls = np.max(ylss, axis=0)
                # yminls = np.min(ylss, axis=0)
                ystdls = np.std(ylss, axis=1)

                # print(f"train nums: {train_nums}.")
                # print(f"y-mean-ls: {ymeanls}.")

                axs[i][j].set_xscale("log")
                axs[i][j].plot(
                    xls,
                    ymeanls,
                    label=method,
                    linewidth=lw,
                    marker=marker[method],
                    markevery=1,
                    markersize=15,
                    markeredgewidth=lw,
                    markerfacecolor="none",
                    alpha=1.0,
                    linestyle=model_line_style[method],
                    color=model_color_dict[method],
                )

                axs[i][j].fill_between(
                    xls,
                    ymeanls - ystdls,
                    ymeanls + ystdls,
                    alpha=a,
                    linewidth=0.0,
                    color=model_color_dict2[method],
                )

                axs[i][j].set_xlabel("# Training Samples", fontsize=font_size)
                axs[i][j].set_ylabel(metricName, fontsize=font_size - 5)
                axs[i][j].set_xticks(xls, xls, rotation=48, size=font_size - 4)
                axs[i][j].tick_params(
                    axis="y",
                    labelsize=font_size - 6,
                    rotation=65,
                    width=2,
                    length=2,
                    pad=0,
                    direction="in",
                    which="both",
                )
    font_legend = {
        "weight": "normal",
        "size": font_size - 1,
    }
    plt.legend(
        loc=(-1.91, 4.2),
        prop=font_legend,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_trainNum_QAs.pdf", pad_inches=0.1)
    print("Save QAs-varing trainnum experiments DONE.")


def wmt_curve_trainNums():
    method_ls = [
        "vanilla",
        "LoRD-VI",
        # "kd",
        # "google/gemma-2b",
        # "lord",
        # "Complex-lord",
    ]
    marker = {
        "vanilla": "s",
        # "kd": "D",
        # "google/gemma-2b": "*",
        "LoRD-VI": "o",
    }
    model_color_dict = {
        "vanilla": "#4a148c",
        # "kd": "#469de9",
        "LoRD-VI": "#eb3b5a",
        # "google/gemma-2b": "#3867d6",
    }

    model_color_dict2 = {
        "vanilla": "#9c27b0",
        # "kd": "#98c8f3",
        "LoRD-VI": "#f78fb3",
        # "google/gemma-2b": "#778beb",
    }

    taskls = [
        "ru-en",
        "de-en",
    ]
    train_times = [
        "1",
        # "2",
        # "3",
    ]
    train_nums = ["8", "16", "32", "64", "128", "256", "512", "1024",]

    a = 0.4
    lw = 1.7
    model_line_style = {
        "vanilla": "-.",
        "kd": "-.",
        "LoRD-VI": "-",
        "google/gemma-2b": "-",
    }
    font_size = 21

    results_dict = {}
    bs_p_dict = {}
    bs_r_dict = {}
    bs_f1_dict = {}
    rg_f1_dict = {}
    bl_1_dict = {}

    infer_save_pth = "./vary_query_num_overall_res_wmt16.json"
    # infer_save_pth = "./wmt_0613_dataset_res/Overall__wmt_varytrain_num_inference_scores.json"

    if not os.path.exists(infer_save_pth):
        # dir_p = "./vary_train_num_WMT16_infers/"
        dir_p = "./wmt_0613_dataset_res/"
        # using existing results of the paths.
        prefix = "./NEW_VARYING_QUERYTIME_CKPTS/text2sql"
        for task in taskls:
            results_dict[task] = {}
            bs_p_dict[task] = {}
            bs_r_dict[task] = {}
            bs_f1_dict[task] = {}
            rg_f1_dict[task] = {}
            bl_1_dict[task] = {}
            for m in method_ls:
                results_dict[task][m] = {}
                bs_p_dict[task][m] = {}
                bs_r_dict[task][m] = {}
                bs_f1_dict[task][m] = {}
                rg_f1_dict[task][m] = {}
                bl_1_dict[task][m] = {}
                for tn in train_nums:
                    results_dict[task][m][tn] = {}
                    bs_p_dict[task][m][tn] = {}
                    bs_r_dict[task][m][tn] = {}
                    bs_f1_dict[task][m][tn] = {}
                    rg_f1_dict[task][m][tn] = {}
                    bl_1_dict[task][m][tn] = {}
                    for train_time in train_times:
                        results_dict[task][m][tn][train_time] = {}
                        results_dict[task][m][tn][train_time] = {}
                        bs_p_dict[task][m][tn][train_time] = {}
                        bs_r_dict[task][m][tn][train_time] = {}
                        bs_f1_dict[task][m][tn][train_time] = {}
                        rg_f1_dict[task][m][tn][train_time] = {}
                        bl_1_dict[task][m][tn][train_time] = {}

                        if m == "google/gemma-2b":
                            pth = m
                        else:
                            pth = prefix + \
                                f"{task}{tn}{train_time}{m}___"

                            if m in [
                                "vanilla",
                                "kd",
                            ]:
                                pth += "finally/"
                            elif tn=="1024":
                                pth += "period1024/"
                            elif tn=="2048":
                                pth += "period2048/"
                            else:
                                pth += "period512/"

                        if m == "google/gemma-2b":
                            res_pth = pth + \
                                f"__{train_time}_{task}_wmt_infer_res.json"
                        else:
                            res_pth = pth + f"___{task}_wmt_infer_res.json"
                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        print(f"Targeted found pth: {dir_p+res_pth}.")
                        if not os.path.exists(dir_p + res_pth):
                            print("ERORR..")
                            return -10000
                            # res_ls = infer_wmt(
                            #     pth,
                            #     task,
                            #     dir_p + res_pth,
                            #     test_set_take_num=100,
                            #     mnt=64,
                            # )
                        else:
                            # from collections import OrderedDict
                            with open(dir_p + res_pth, "r",
                                      encoding="utf8") as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        ss = eval_wmt(res_ls)
                        results_dict[task][m][tn][train_time] = ss
                        bs_p_dict[task][m][tn][train_time] = ss["bertscore"]["p"]
                        bs_r_dict[task][m][tn][train_time] = ss["bertscore"]["r"]
                        bs_f1_dict[task][m][tn][train_time] = ss["bertscore"]["f1"]
                        rg_f1_dict[task][m][tn][train_time] = ss["rouge-l"]["f1"]
                        bl_1_dict[task][m][tn][train_time] = ss["bleu"]["1"]
        with open(infer_save_pth, "w", encoding="utf8") as f:
            json.dump(
                [results_dict, bs_p_dict, bs_r_dict,
                    bs_f1_dict, rg_f1_dict, bl_1_dict],
                f,
                ensure_ascii=False,
                indent=4,
            )
    else:
        # from collections import OrderedDict
        with open(infer_save_pth, "r", encoding="utf8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            results_dict, bs_p_dict, bs_r_dict, bs_f1_dict, rg_f1_dict, bl_1_dict = data

    res_dict = {}
    for task in taskls:
        res_dict[task] = {}
        res_dict[task]["BERTScore Precision"] = bs_p_dict[task]
        res_dict[task]["BERTScore Recall"] = bs_r_dict[task]
        res_dict[task]["BERTScore F1"] = bs_f1_dict[task]
        res_dict[task]["Rouge-L F1"] = rg_f1_dict[task]
        res_dict[task]["BLEU"] = bl_1_dict[task]

    fig, axs = plt.subplots(2, 5, figsize=(26, 9.37))

    for i, task in enumerate(list(res_dict.keys())):
        for j, metricName in enumerate(list(res_dict[task].keys())):
            adict = res_dict[task][metricName]
            for method in adict.keys():
                ylss = []
                for tn in train_nums:
                    templs = []
                    for tt in train_times:
                        templs.append(adict[method][tn][tt])
                    ylss.append(templs)
                ylss = np.array(ylss)

                xls = [float(x) for x in train_nums]

                ymeanls = np.mean(ylss, axis=1)
                # ymaxls = np.max(ylss, axis=0)
                # yminls = np.min(ylss, axis=0)
                ystdls = np.std(ylss, axis=1)

                # print(f"train nums: {train_nums}.")
                # print(f"y-mean-ls: {ymeanls}.")

                axs[i][j].set_xscale("log")
                axs[i][j].plot(
                    xls,
                    ymeanls,
                    label=method,
                    linewidth=lw,
                    marker=marker[method],
                    markevery=1,
                    markersize=15,
                    markeredgewidth=lw,
                    markerfacecolor="none",
                    alpha=1.0,
                    linestyle=model_line_style[method],
                    color=model_color_dict[method],
                )

                # axs[i][j].fill_between(
                #     xls,
                #     ymeanls - ystdls,
                #     ymeanls + ystdls,
                #     alpha=a,
                #     linewidth=0.0,
                #     color=model_color_dict2[method],
                # )

                axs[i][j].set_xlabel("# Training Samples", fontsize=font_size)
                axs[i][j].set_ylabel(metricName, fontsize=font_size - 5)
                axs[i][j].set_xticks(xls, xls, rotation=48, size=font_size - 4)
                axs[i][j].tick_params(
                    axis="y",
                    labelsize=font_size - 6,
                    rotation=65,
                    width=2,
                    length=2,
                    pad=0,
                    direction="in",
                    which="both",
                )
    font_legend = {
        "weight": "normal",
        "size": font_size - 1,
    }
    plt.legend(
        loc=(-3.21, 2.6),
        prop=font_legend,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_trainNum_wmt16.pdf", pad_inches=0.1)
    print("Save wmt16-varing trainnum experiments DONE.")
    pass


def code_curve_trainNums():
    method_ls = [
        "vanilla",
        "LoRD-VI",
        # "kd",
        # "google/gemma-2b",
        # "lord",
        # "Complex-lord",
    ]
    marker = {
        "vanilla": "s",
        # "kd": "D",
        # "google/gemma-2b": "*",
        "LoRD-VI": "o",
    }
    model_color_dict = {
        "vanilla": "#4a148c",
        # "kd": "#469de9",
        "LoRD-VI": "#eb3b5a",
        # "google/gemma-2b": "#3867d6",
    }

    model_color_dict2 = {
        "vanilla": "#9c27b0",
        # "kd": "#98c8f3",
        "LoRD-VI": "#f78fb3",
        # "google/gemma-2b": "#778beb",
    }

    taskls = [
        "deepmind/code_contests",
    ]
    train_times = [
        "1",
        # "2",
        # "3",
    ]
    train_nums = [
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "256",
        "512",
        "1024",
        # "2048",
        ]

    a = 0.4
    lw = 1.7
    model_line_style = {
        "vanilla": "-.",
        "kd": "-.",
        "LoRD-VI": "-",
        "google/gemma-2b": "-",
    }
    font_size = 21

    results_dict = {}
    bs_p_dict = {}
    bs_r_dict = {}
    bs_f1_dict = {}
    rg_f1_dict = {}
    bl_1_dict = {}

    infer_save_pth = "./vary_query_num_overall_res_code.json"
    # infer_save_pth = "./wmt_0613_dataset_res/Overall__wmt_varytrain_num_inference_scores.json"

    if not os.path.exists(infer_save_pth):
        # dir_p = "./vary_train_num_WMT16_infers/"
        dir_p = "./code_0627_dataset_res/"
        # using existing results of the paths.
        prefix = "./codee_ckpts/CODEEE"
        for task in taskls:
            results_dict[task] = {}
            bs_p_dict[task] = {}
            bs_r_dict[task] = {}
            bs_f1_dict[task] = {}
            rg_f1_dict[task] = {}
            bl_1_dict[task] = {}
            for m in method_ls:
                results_dict[task][m] = {}
                bs_p_dict[task][m] = {}
                bs_r_dict[task][m] = {}
                bs_f1_dict[task][m] = {}
                rg_f1_dict[task][m] = {}
                bl_1_dict[task][m] = {}
                for tn in train_nums:
                    results_dict[task][m][tn] = {}
                    bs_p_dict[task][m][tn] = {}
                    bs_r_dict[task][m][tn] = {}
                    bs_f1_dict[task][m][tn] = {}
                    rg_f1_dict[task][m][tn] = {}
                    bl_1_dict[task][m][tn] = {}
                    for train_time in train_times:
                        results_dict[task][m][tn][train_time] = {}
                        results_dict[task][m][tn][train_time] = {}
                        bs_p_dict[task][m][tn][train_time] = {}
                        bs_r_dict[task][m][tn][train_time] = {}
                        bs_f1_dict[task][m][tn][train_time] = {}
                        rg_f1_dict[task][m][tn][train_time] = {}
                        bl_1_dict[task][m][tn][train_time] = {}

                        if m == "google/gemma-2b":
                            pth = m
                        else:
                            pth = prefix + \
                                f"{task}{tn}{train_time}{m}___"

                            if m in [
                                "vanilla",
                                "kd",
                            ]:
                                pth += "finally/"
                            elif tn=="1024":
                                pth += "period1024/"
                            elif tn=="2048":
                                pth += "period2048/"
                            else:
                                pth += "period512/"

                        if m == "google/gemma-2b":
                            res_pth = pth + \
                                f"__{train_time}_{task}_code_infer_res.json"
                        else:
                            res_pth = pth + f"___{task}_code_infer_res.json"
                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        print(f"Targeted found pth: {dir_p+res_pth}.")
                        if not os.path.exists(dir_p + res_pth):
                            print("ERORR..")
                            return -10000
                            # res_ls = infer_wmt(
                            #     pth,
                            #     task,
                            #     dir_p + res_pth,
                            #     test_set_take_num=100,
                            #     mnt=64,
                            # )
                        else:
                            # from collections import OrderedDict
                            with open(dir_p + res_pth, "r",
                                      encoding="utf8") as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        ss = eval_code(res_ls)
                        results_dict[task][m][tn][train_time] = ss
                        bs_p_dict[task][m][tn][train_time] = ss["bertscore"]["p"]
                        bs_r_dict[task][m][tn][train_time] = ss["bertscore"]["r"]
                        bs_f1_dict[task][m][tn][train_time] = ss["bertscore"]["f1"]
                        rg_f1_dict[task][m][tn][train_time] = ss["rouge-l"]["f1"]
                        bl_1_dict[task][m][tn][train_time] = ss["bleu"]["1"]
        with open(infer_save_pth, "w", encoding="utf8") as f:
            json.dump(
                [results_dict, bs_p_dict, bs_r_dict,
                    bs_f1_dict, rg_f1_dict, bl_1_dict],
                f,
                ensure_ascii=False,
                indent=4,
            )
    else:
        # from collections import OrderedDict
        with open(infer_save_pth, "r", encoding="utf8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            results_dict, bs_p_dict, bs_r_dict, bs_f1_dict, rg_f1_dict, bl_1_dict = data

    res_dict = {}
    for task in taskls:
        res_dict[task] = {}
        res_dict[task]["BERTScore Precision"] = bs_p_dict[task]
        res_dict[task]["BERTScore Recall"] = bs_r_dict[task]
        res_dict[task]["BERTScore F1"] = bs_f1_dict[task]
        res_dict[task]["Rouge-L F1"] = rg_f1_dict[task]
        res_dict[task]["BLEU"] = bl_1_dict[task]

    fig, axs = plt.subplots(1, 5, figsize=(26, 4.67))

    for i, task in enumerate(list(res_dict.keys())):
        for j, metricName in enumerate(list(res_dict[task].keys())):
            adict = res_dict[task][metricName]
            for method in adict.keys():
                ylss = []
                for tn in train_nums:
                    templs = []
                    for tt in train_times:
                        templs.append(adict[method][tn][tt])
                    ylss.append(templs)
                ylss = np.array(ylss)

                xls = [float(x) for x in train_nums]

                ymeanls = np.mean(ylss, axis=1)
                # ymaxls = np.max(ylss, axis=0)
                # yminls = np.min(ylss, axis=0)
                ystdls = np.std(ylss, axis=1)

                # print(f"train nums: {train_nums}.")
                # print(f"y-mean-ls: {ymeanls}.")

                axs[j].set_xscale("log")
                axs[j].plot(
                    xls,
                    ymeanls,
                    label=method,
                    linewidth=lw,
                    marker=marker[method],
                    markevery=1,
                    markersize=15,
                    markeredgewidth=lw,
                    markerfacecolor="none",
                    alpha=1.0,
                    linestyle=model_line_style[method],
                    color=model_color_dict[method],
                )

                # axs[i][j].fill_between(
                #     xls,
                #     ymeanls - ystdls,
                #     ymeanls + ystdls,
                #     alpha=a,
                #     linewidth=0.0,
                #     color=model_color_dict2[method],
                # )

                axs[j].set_xlabel("# Training Samples", fontsize=font_size)
                axs[j].set_ylabel(metricName, fontsize=font_size - 5)
                axs[j].set_xticks(xls, xls, rotation=48, size=font_size - 4)
                axs[j].tick_params(
                    axis="y",
                    labelsize=font_size - 6,
                    rotation=65,
                    width=2,
                    length=2,
                    pad=0,
                    direction="in",
                    which="both",
                )
    font_legend = {
        "weight": "normal",
        "size": font_size - 1,
    }
    plt.legend(
        loc=(-3.21, 2.6),
        prop=font_legend,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_trainNum_code.pdf", pad_inches=0.1)
    print("Save CODE-varing trainnum experiments DONE.")
    pass


def glue_curve_trainNums():
    train_numls = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        100,
        256,
        300,
    ]
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
                pth = prefix + f"{task}{tn}{m}_256{task}___"
                pth += "finally"
            else:
                pth = prefix + f"{task}{tn}{m}256{task}___"
                pth += "period2"
            paths_dict[m].append(pth)

            res_pth = pth + f"___{task}_glue_infer_res"
            res_pth = res_pth.replace("/", "__").replace(".", "")
            res_pth += ".json"

            if not os.path.exists(dir_p + res_pth):
                res_ls = infer_glue(pth, task, dir_p + res_pth)
            else:
                with open(dir_p + res_pth, "r", encoding="utf8") as f:
                    res_ls = json.load(f, object_pairs_hook=OrderedDict)

            scores = eval_glue(task, res_ls)
            adict[m].append(scores[0])
            pdict[m].append(scores[1])
            rdict[m].append(scores[2])
            fdict[m].append(scores[3])

    res_dict = OrderedDict(
        {
            "Accuracy": adict,
            "Precision": pdict,
            "Recall": rdict,
            "F1 Score": fdict,
        }
    )
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
                markevery=1,
                markersize=15,
                markeredgewidth=lw,
                markerfacecolor="none",
                alpha=1.0,
                linestyle=model_line_style[method],
                color=model_color_dict[method],
            )

            # axs[0][i_n].fill_between(periods,
            # ymin, ymax,
            #                          alpha=a,
            #                          linewidth=0.,
            #                          # alpha=1.0,
            #                          color=model_color_dict2[mode])
            axs[i].set_xlabel("Training Periods", fontsize=font_size)
            axs[i].set_ylabel(ylabel, fontsize=font_size - 5)
            axs[i].set_xticks(train_numls, train_numls,
                              rotation=48, size=font_size - 4)
            axs[i].tick_params(
                axis="y",
                labelsize=font_size - 6,
                rotation=65,
                width=2,
                length=2,
                pad=0,
                direction="in",
                which="both",
            )
    font_legend = {
        "weight": "normal",
        "size": font_size - 1,
    }
    plt.legend(
        loc=(-1.91, 1.0),
        prop=font_legend,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    plt.savefig("./vary_trainNum.pdf", pad_inches=0.1)
    print("Save DONE.")


# running entry
if __name__ == "__main__":
    # glue()
    # wmt_curve_trainNums()
    # code_curve_trainNums()
    qa_curve_trainNums()
    print("EVERYTHING DONE.")
