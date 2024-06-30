"""
======================================================================
EVAL_VARY_MODELSIZE ---

Evaluate the methods with different model sizes.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 19 June 2024
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


def wmt_curve_trainNums(overall_name="wmt16", taskls=["cs-en","de-en",]):
    method_ls = [
        "vanilla",
        # "LoRD-VIII",
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
        "LoRD-VIII": "o",
        "LoRD-VI": "o",
    }
    model_color_dict = {
        "vanilla": "#4a148c",
        # "kd": "#469de9",
        "LoRD-VIII": "#eb3b5a",
        "LoRD-VI": "#eb3b5a",
        # "google/gemma-2b": "#3867d6",
    }
    model_color_dict2 = {
        "vanilla": "#9c27b0",
        # "kd": "#98c8f3",
        "LoRD-VIII": "#f78fb3",
        "LoRD-VI": "#f78fb3",
        # "google/gemma-2b": "#778beb",
    }
    train_times = [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    # train_nums = ["8", "16", "32", "64", "128", "256", "512"]
    # train_nums = ["64"]
    train_nums = ["16"]
    base1_pth_ls=[
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
        
        ]

    base2_pth_ls=[
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        # "facebook/opt-30b",
        ]
    # base_pth_ls=[x for x in base1_pth_ls]
    # base_pth_ls.extend(base2_pth_ls)
    base_pth_ls=base2_pth_ls

    x1_ls=[0.41,1.4,2.8,6.9,]
    x2_ls=[0.12,0.35,1.3,2.7,6.7,13]
    # x2_ls=[0.12,0.35,1.3,2.7,6.7,13,30]
    # xls=[0.3, 1.0, 2.5, 4.0, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0]
    xls=x2_ls

    a = 0.4
    lw = 1.7
    model_line_style = {
        "vanilla": "-.",
        "kd": "-.",
        "LoRD-VIII": "-",
        "LoRD-VI": "-",
        "google/gemma-2b": "-",
    }
    font_size = 21

    results_dict = {}
    bs_p_dict = {}
    bs_r_dict = {}
    bs_f1_dict = {}
    rg_p_dict = {}
    rg_r_dict = {}
    rg_f1_dict = {}
    bl_1_dict = {}
    bl_4_dict = {}

    # infer_save_pth = "./vary_modelsize_overall_res_wmt16.json"
    infer_save_pth = f"./vary_modelsize_overall_res_{overall_name}.json"

    if not os.path.exists(infer_save_pth):
        dir_p = "./wmt_0617_varymodelsize_dataset_res/"
        # using existing results of the paths.
        prefix = "./SCALE_VARYING_CKPTS/text2sql"
        for task in taskls:
            results_dict[task] = {}
            bs_p_dict[task] = {}
            bs_r_dict[task] = {}
            bs_f1_dict[task] = {}
            rg_p_dict[task] = {}
            rg_r_dict[task] = {}
            rg_f1_dict[task] = {}
            bl_1_dict[task] = {}
            bl_4_dict[task] = {}
            for m in method_ls:
                results_dict[task][m] = {}
                bs_p_dict[task][m] = {}
                bs_r_dict[task][m] = {}
                bs_f1_dict[task][m] = {}
                rg_p_dict[task][m] = {}
                rg_r_dict[task][m] = {}
                rg_f1_dict[task][m] = {}
                bl_1_dict[task][m] = {}
                bl_4_dict[task][m] = {}
                for tn in train_nums:
                    results_dict[task][m][tn] = {}
                    bs_p_dict[task][m][tn] = {}
                    bs_r_dict[task][m][tn] = {}
                    bs_f1_dict[task][m][tn] = {}
                    rg_p_dict[task][m][tn] = {}
                    rg_r_dict[task][m][tn] = {}
                    rg_f1_dict[task][m][tn] = {}
                    bl_1_dict[task][m][tn] = {}
                    bl_4_dict[task][m][tn] = {}
                    for base_pth in base_pth_ls:
                        results_dict[task][m][tn][base_pth] = {}
                        bs_p_dict[task][m][tn][base_pth] = {}
                        bs_r_dict[task][m][tn][base_pth] = {}
                        bs_f1_dict[task][m][tn][base_pth] = {}
                        rg_p_dict[task][m][tn][base_pth] = {}
                        rg_r_dict[task][m][tn][base_pth] = {}
                        rg_f1_dict[task][m][tn][base_pth] = {}
                        bl_1_dict[task][m][tn][base_pth] = {}
                        bl_4_dict[task][m][tn][base_pth] = {}
                        for train_time in train_times:
                            results_dict[task][m][tn][base_pth][train_time] = {}
                            results_dict[task][m][tn][base_pth][train_time] = {}
                            bs_p_dict[task][m][tn][base_pth][train_time] = {}
                            bs_r_dict[task][m][tn][base_pth][train_time] = {}
                            bs_f1_dict[task][m][tn][base_pth][train_time] = {}
                            rg_p_dict[task][m][tn][base_pth][train_time] = {}
                            rg_r_dict[task][m][tn][base_pth][train_time] = {}
                            rg_f1_dict[task][m][tn][base_pth][train_time] = {}
                            bl_1_dict[task][m][tn][base_pth][train_time] = {}
                            bl_4_dict[task][m][tn][base_pth][train_time] = {}

                            if m == "google/gemma-2b":
                                pth = m
                            else:
                                pth = prefix + \
                                    f"{base_pth}{task}{tn}{train_time}{m}___"

                                if m in [
                                    "vanilla",
                                    "kd",
                                ]:
                                    pth += "finally/"
                                elif tn=="256" or tn=="512":
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
                            results_dict[task][m][tn][base_pth][train_time] = ss
                            bs_p_dict[task][m][tn][base_pth][train_time] = ss["bertscore"]["p"]
                            bs_r_dict[task][m][tn][base_pth][train_time] = ss["bertscore"]["r"]
                            bs_f1_dict[task][m][tn][base_pth][train_time] = ss["bertscore"]["f1"]
                            rg_p_dict[task][m][tn][base_pth][train_time] = ss["rouge-l"]["p"]
                            rg_r_dict[task][m][tn][base_pth][train_time] = ss["rouge-l"]["r"]
                            rg_f1_dict[task][m][tn][base_pth][train_time] = ss["rouge-l"]["f1"]
                            bl_1_dict[task][m][tn][base_pth][train_time] = ss["bleu"]["1"]
                            bl_4_dict[task][m][tn][base_pth][train_time] = ss["bleu"]["4"]
        with open(infer_save_pth, "w", encoding="utf8") as f:
            json.dump(
                [results_dict, bs_p_dict, bs_r_dict,
                    bs_f1_dict, rg_p_dict, rg_r_dict, rg_f1_dict,
                 bl_1_dict, bl_4_dict, ],
                f,
                ensure_ascii=False,
                indent=4,
            )
    else:
        # from collections import OrderedDict
        with open(infer_save_pth, "r", encoding="utf8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            results_dict, bs_p_dict, bs_r_dict, bs_f1_dict, rg_p_dict, rg_r_dict, rg_f1_dict, bl_1_dict, bl_4_dict = data

    res_dict = {}
    for task in taskls:
        res_dict[task] = {}
        res_dict[task]["BLEU-1"] = bl_1_dict[task]
        # res_dict[task]["BLEU-4"] = bl_4_dict[task]
        res_dict[task]["Rouge-L Recall"] = rg_r_dict[task]
        # res_dict[task]["Rouge-L F1"] = rg_f1_dict[task]
        res_dict[task]["BERTScore Recall"] = bs_r_dict[task]
        res_dict[task]["BERTScore F1"] = bs_f1_dict[task]

    fig, axs = plt.subplots(2, 4, figsize=(21, 9.37))

    for i, task in enumerate(list(res_dict.keys())):
        for j, metricName in enumerate(list(res_dict[task].keys())):
            adict = res_dict[task][metricName]
            for method in adict.keys():
                ylss = []
                for tn in train_nums:
                    for base_pth in base_pth_ls:
                        templs = []
                        for tt in train_times:
                            templs.append(adict[method][tn][base_pth][tt])
                        ylss.append(templs)
                ylss = np.array(ylss)

                ymeanls = np.mean(ylss, axis=1)
                # ymaxls = np.max(ylss, axis=0)
                # yminls = np.min(ylss, axis=0)
                ystdls = np.std(ylss, axis=1)
                # y1meanls=ymeanls[:len(base1_pth_ls)]
                y2meanls=ymeanls

                # print(f"train nums: {train_nums}.")
                # print(f"y-mean-ls: {ymeanls}.")

                # axs[i][j].set_xscale("log")
                # axs[i][j].plot(
                #     x1_ls,
                #     y1meanls,
                #     label=method,
                #     linewidth=lw,
                #     marker=marker[method],
                #     markevery=1,
                #     markersize=15,
                #     markeredgewidth=lw,
                #     markerfacecolor="none",
                #     alpha=1.0,
                #     linestyle=model_line_style[method],
                #     color=model_color_dict[method],
                # )

                axs[i][j].set_xscale("log")
                axs[i][j].plot(
                    x2_ls,
                    y2meanls,
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
                    x2_ls,
                    ymeanls - ystdls,
                    ymeanls + ystdls,
                    alpha=a,
                    linewidth=0.0,
                    color=model_color_dict2[method],
                )

                axs[i][j].set_xlabel("# Model Parameters (Billion)", fontsize=font_size)
                axs[i][j].set_ylabel(metricName, fontsize=font_size - 2)
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
        loc=(-1.91, 2.6),
        prop=font_legend,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()

    # plt.savefig("./vary_modelsize_wmt16.pdf", pad_inches=0.1)
    # print("Save wmt16-varing trainnum experiments DONE.")
    plt.savefig(f"./vary_modelsize_{overall_name}.pdf", pad_inches=0.1)
    print(f"Save {overall_name}-varing trainnum experiments DONE.")
    pass


## running entry
if __name__=="__main__":
    # main()
    # wmt_curve_trainNums()

    # wmt_curve_trainNums(overall_name="t2s",
    #                     taskls=["wikisql","spider",])

    wmt_curve_trainNums(overall_name="mix",
                        # taskls=["ru-en","de-en",],
                        # taskls=["ru-en","de-en",],
                        taskls=["ru-en","de-en",],
                        )
    print("EVERYTHING DONE.")


