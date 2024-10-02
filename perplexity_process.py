"""
======================================================================
PERPLEXITY_PROCESS ---

Evaluate the models' perplexity

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created: 25 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from tqdm import tqdm
import json

from datasets import load_dataset
from collections import OrderedDict
from pprint import pprint
from math import exp

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
    tokenizer = AutoTokenizer.from_pretrained(modelname,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        modelname,
        device_map=device,
        trust_remote_code=True,
        offload_folder="offload",
    )
    dataset = load_dataset(name, subsetname,
                           split="test[:500]")
    encodings = tokenizer("\n\n".join(dataset["text"]),
                          return_tensors="pt")
    # msl = 512
    msl = 128
    sl = encodings.input_ids.size(1)

    nlls = []
    for begin in tqdm(range(0, sl, msl)):
        if begin+msl >= sl:
            break
        inps = encodings.input_ids[:, begin:begin+msl].to("cuda")

        with torch.no_grad():
            outputs = model(inps, labels=inps).loss
            nlls.append(outputs)

    ppl = torch.exp(torch.stack(nlls).mean())

    return float(ppl)


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
