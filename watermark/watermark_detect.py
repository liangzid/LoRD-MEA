"""
======================================================================
WATERMARK_DETECT ---

Detect whether a watermark is contained in a given text.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 31 May 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"]="1"
    pass
from extended_watermark_processor import WatermarkDetector
from transformers import AutoTokenizer

import sys
sys.path.append("/home/zi/alignmentExtraction")
sys.path.append("/home/zi/alignmentExtraction/watermark")
from data2text_process import infer_d2t, eval_d2ttt
import json
import numpy as np
from collections import OrderedDict
from pprint import pprint

def wrmk_dtct(output_text,
              tokenizer,
              device):
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25, # should match original setting
        seeding_scheme="selfhash", # should match original setting
        device=device, # must match the original rng device type
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True)

    score_dict = watermark_detector.detect(output_text)
    # or any other text of interest to analyze

    print(f"Score: {score_dict}")
    return score_dict

def eval_varying_train_num():
    taskls = [
        "e2e_nlg",
        "allenai/common_gen",
        # "cs-en",
        # "de-en",
        ]
    mls = [
        "vanilla",
        "LoRD-VI",
        # "pretrained",
        # "gpt-3.5-turbo-1106",
        # "kd",
        ]
    train_times = [
        "1",
        # "2",
        # "3",
        # "4",
        # "5",
        ]
    train_nums = [
        # "8",
        # "16",
        "64",
        # "1000",
        # "128",
        # "256",
        # "512",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer=AutoTokenizer.from_pretrained(base_model_name1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    dir_p = "./watermark_res/"
    res_dict = {}
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)

    res_dict_averaged={}

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                temp_scorels=[]
                for itime in train_times:

                    if task=="e2e_nlg" or task=="allenai/common_gen":
                        prefix = "./watermark/d2t_ckpts/D2TTT"
                        if m=="vanilla":
                            ckpt = (
                                prefix
                                + f"{task}@wrmk{train_num}{itime}{m}___finally/"
                            )
                        elif m =="pretrained":
                            ckpt = f"./text2sql_ckpts/d2t---{task}@wrmk{train_num}{itime}{m}_res.json"
                        elif m=="gpt-3.5-turbo-1106":
                            ckpt=m
                        else:
                            ckpt = prefix + \
                                f"{task}@wrmk{train_num}{itime}{m}___period512/"
                        res_pth = ckpt+f"___{task}@wrmk_d2t_infer_res.json"
                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        if not os.path.exists(dir_p+res_pth):
                            if m=="pretrained":
                                res_ls = infer_d2t(None,
                                                task,
                                                dir_p+res_pth,
                                                test_set_take_num=500,
                                                mnt=64,
                                                base_model_name=base_model_name1,
                                                )
                            else:
                                res_ls = infer_d2t(ckpt,
                                                task,
                                                dir_p+res_pth,
                                                test_set_take_num=500,
                                                mnt=64,
                                                base_model_name=base_model_name1,
                                                )
                        else:
                            # from collections import OrderedDict
                            with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        scores = eval_d2ttt(res_ls)
                    elif task=="cs-en" or task=="de-en":
                        prefix = "./watermark/d2t_ckpts/D2TTT"
                        if m=="vanilla":
                            ckpt = (
                                prefix
                                + f"{task}@wrmk{train_num}{itime}{m}___finally/"
                            )
                        elif m =="pretrained":
                            ckpt = f"./text2sql_ckpts/d2t---{task}@wrmk{train_num}{itime}{m}_res.json"
                        elif m=="gpt-3.5-turbo-1106":
                            ckpt=m
                        else:
                            ckpt = prefix + \
                                f"{task}@wrmk{train_num}{itime}{m}___period512/"
                        res_pth = ckpt+f"___{task}@wrmk_d2t_infer_res.json"
                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        if not os.path.exists(dir_p+res_pth):
                            if m=="pretrained":
                                res_ls = infer_wmt(None,
                                                task,
                                                dir_p+res_pth,
                                                test_set_take_num=500,
                                                mnt=64,
                                                base_model_name=base_model_name1,
                                                )
                            else:
                                res_ls = infer_wmt(ckpt,
                                                task,
                                                dir_p+res_pth,
                                                test_set_take_num=500,
                                                mnt=64,
                                                base_model_name=base_model_name1,
                                                )
                        else:
                            # from collections import OrderedDict
                            with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        scores = eval_wmt(res_ls)
                    else:
                        print("NOT FOUNDDDDDDDDDDDDDDDDDDDDDDD.")

                    another_dict=wrmk_dtct(
                        res_ls,
                        tokenizer,
                        device="cuda:0",
                        )
                    print("=======================================")
                    print("Detect-related Metric:")
                    print(another_dict)
                    print("Detect-related Metric.")
                    print("=======================================")

                    for keyy in another_dict:
                        if keyy =="z_score_at_T":
                            continue
                        scores[keyy]=another_dict[keyy]
                    
                    print(task, ckpt)
                    print(scores)
                    res_dict[task+"-----"+res_pth] = scores
                    score_ls=[
                        scores["bleu"]["1"],
                        scores["bleu"]["2"],
                        scores["bleu"]["3"],
                        scores["bleu"]["4"],
                        scores["bertscore"]["p"],
                        scores["bertscore"]["r"],
                        scores["bertscore"]["f1"],
                        scores["rouge-l"]["p"],
                        scores["rouge-l"]["r"],
                        scores["rouge-l"]["f1"],
                        ]
                    temp_scorels.append(score_ls)

                # obtain the mean value
                # obtain the std value
                temp_scorels=np.array(temp_scorels)
                meanvaluels=np.mean(temp_scorels,axis=0).tolist()
                stdvaluels=np.std(temp_scorels,axis=0,ddof=1).tolist()
                res_dict_averaged[task+"--"+res_pth]=\
                    {"mean": meanvaluels,
                     "std": stdvaluels}

    with open(dir_p+"Overall__d2t_varytrain_num_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    with open(
        dir_p + "OverallScoresAveraged.json",
            "w", encoding="utf8"
    ) as f:
        json.dump(res_dict_averaged, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)
    pprint("-------------------------")
    pprint(res_dict_averaged)
    return res_dict



if __name__=="__main__":
    eval_varying_train_num()
