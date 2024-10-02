"""
======================================================================
COMMON_TASK_PROCESS --- 

This file is used to test the general ability of models, during different
tasks.

Evaluate based on SIQA.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created: 25 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from tqdm import tqdm
import json

from datasets import load_dataset
from gen_pipeline_open import InferObj
from collections import OrderedDict
from pprint import pprint

label2AnswerMap = {
    "1": "A",
    "2": "B",
    "3": "C",
}


def eval_siQA(resls):
    "evaluate SiQA dataset"

    hit_num = 0.
    for predict, label, content_label in resls:
        transferred_label = label2AnswerMap[label]
        if transferred_label in predict or\
           content_label in predict:
            hit_num += 1
    res_dict = {"acc": hit_num/len(resls)}
    return res_dict


def load_siQA(save_pth,
              name="social_i_qa",
              modelname="google/gemma-2b",
              ):
    dataset = load_dataset(name, split="validation[:100]")

    model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=2047)
    gen_pipeline = model.text_gen

    res_ls = []
    for item in tqdm(dataset):
        context = item["context"]
        q = item["question"]
        ans_a = item["answerA"]
        ans_b = item["answerB"]
        ans_c = item["answerC"]
        label = item["label"]

        inps = f"Question: {context} {q} Answer A: {ans_a}. Answer B: {ans_b}. Answer C: {ans_c}. Your selection is "
        res = gen_pipeline(inps, max_new_tokens=16,
                           )[0]["generated_text"]
        res = res.split(inps)[1]
        print(f"inps: {inps}")
        print(f"res: {res}")

        choice = label2AnswerMap[label]
        ans = item[f"answer{choice}"]

        res_ls.append((res, label, ans))

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


def eval_trackingProcessStablity():
    methodls = ["Complex-lord", "vanilla"]
    stepls = [32*(i) for i in range(1, 10)]
    dir_p = "./CiQA_infers_tracking_process_stable/"
    taskls = ["cs-en"]
    res_dict = {}

    for task in taskls:
        res_dict[task] = {}
        for m in methodls:
            for step in stepls:
                if not os.path.exists(dir_p):
                    os.makedirs(dir_p)
                prefix = "./tracking_process_stablecs-en/"
                if m == "Complex-lord" or m == "black--Complex-lord":
                    ckpt = prefix+f"{m}_256{task}_step___{step}"
                else:
                    ckpt = prefix+f"{m}_256{task}_step___{step}"
                res_pth = ckpt+f"___{task}_wmt_infer_res.json"
                res_pth = res_pth.replace("/", "__").replace(".", "")
                if not os.path.exists(dir_p+res_pth):
                    res_ls = load_siQA(dir_p+res_pth,
                                       modelname=ckpt,
                                       )
                else:
                    # from collections import OrderedDict
                    with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                        res_ls = json.load(
                            f, object_pairs_hook=OrderedDict)

                scores = eval_siQA(res_ls)
                res_dict[task][task+"-----"+ckpt] = scores
    with open(dir_p+"SiQA_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


def experiment1():
    ckptls = [
        "google/gemma-2b",
        "./GLUE_ckpts/colablack--Complex-lord256100___period2/",
        "./GLUE_ckpts/colavanilla256100___finally/",
        "./GLUE_ckpts/colakd256100___finally/",
    ]

    dir_p = "./CiQA_infers_tracking_process_stable/"
    taskls = ["CiQA"]
    res_dict = {}

    for task in taskls:
        res_dict[task] = {}
        for ckpt in ckptls:
            if not os.path.exists(dir_p):
                os.makedirs(dir_p)
            res_pth = ckpt+f"___{task}_wmt_infer_res.json"
            res_pth = res_pth.replace("/", "__").replace(".", "")
            if not os.path.exists(dir_p+res_pth):
                res_ls = load_siQA(dir_p+res_pth,
                                   modelname=ckpt,
                                   )
            else:
                # from collections import OrderedDict
                with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                    res_ls = json.load(
                        f, object_pairs_hook=OrderedDict)

            scores = eval_siQA(res_ls)
            res_dict[task][task+"-----"+ckpt] = scores
    with open(dir_p+"SiQA_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


# running entry
if __name__ == "__main__":
    experiment1()
    # main()
    # eval_trackingProcessStablity()
    print("EVERYTHING DONE.")
