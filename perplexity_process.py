"""
======================================================================
PERPLEXITY_PROCESS ---

Evaluate the models' perplexity

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
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
from collections import OrderedDict
from pprint import pprint

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
import torch


def inferInDataset(
    # name="social_i_qa",
    name="wikitext",
        subsetname="wikitext-103-raw-v1",
        modelname="google/gemma-2b",
):
    device = "auto"
    dataset = load_dataset(name, subsetname,
                           split="train[:100]")
    tokenizer = AutoTokenizer.from_pretrained(modelname,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        modelname,
        device_map=device,
        trust_remote_code=True,
        offload_folder="offload",
    )

    res_ls = []
    for item in tqdm(dataset):
        text = item["text"]
        inps = text
        # print(inps)
        inps = tokenizer([inps],
                         return_tensors="pt",
                         max_length=512,).input_ids
        inps = inps.to("cuda")

        neg_log_llh = model(inps, labels=inps).loss
        res_ls.append(neg_log_llh.item())

    ppl = torch.exp(sum(res_ls)/len(res_ls))

    return ppl


def experiment_ppl1():
    ckptls = [
        "google/gemma-2b",
        "./GLUE_ckpts/colablack--Complex-lord256100___period2/",
        "./GLUE_ckpts/colavanilla256100___finally/",
        "./GLUE_ckpts/colakd256100___finally/",
    ]

    dir_p = "./wikitext_infers_tracking_process_stable/"
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
                res = inferInDataset(
                    modelname=ckpt,
                )
            res_dict[task][task+"-----"+ckpt] = res
    with open(dir_p+"ppl_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


# running entry
if __name__ == "__main__":
    experiment_ppl1()
    print("EVERYTHING DONE.")
