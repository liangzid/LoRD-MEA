"""
======================================================================
SAFETY_PROCESS ---

Safety related experiments.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 27 June 2024
======================================================================
"""

# ------------------------ Code --------------------------------------
import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

import torch
from datasets import load_dataset
from openai import OpenAI as oa
import json
from collections import OrderedDict
import os
from math import exp
import random

import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from training_data_collecting_openai import chatWithOpenAI_APIs
from training_data_collecting_openai import chatWithOpenAI__LogLogits

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process
# from wmt_process import eval_wmt as eval_sum

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch
from pprint import pprint
import numpy as np

def load_safe_datals(tokenizer,
                    task_name="allenai/prosocial-dialog",
                    train_num=100,
                    model_name="gpt-3.5-turbo-1106",
                    topk=5,
                    max_length=1024,
                    openai_tmp_save_pth="./STEALED_PKLS/wmt_data_saveto_"):

    lm_tokenizer = tokenizer

    V = lm_tokenizer.vocab_size
    tasks_we_used = [
        "allenai/prosocial-dialog",
    ]
    assert task_name in tasks_we_used
    dataset_name = task_name
    inp_ls = []
    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")
        for item in trainset_text:
            inp = item["context"]
            resp = item["response"]
            inp_ls.append(inp)

    assert inp_ls != []

    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    openai_tmp_save_pth += f"safetask_{task_name}-trainNUM_{train_num}.pkl"

    return commonly_used_openai_post_process(
        openai_tmp_save_pth,
        inp_ls,
        pp,
        model_name,
        topk,
        max_length,
        p_idxls,
        V,
        lm_tokenizer,
    )


def infer_safe(modelname, task_name, res_pth,
              test_set_take_num=100,
              mnt=32,
              base_model_name=None,
              ):

    save_pth = res_pth

    tasks_we_used = [
        "allenai/prosocial-dialog",
    ]
    assert task_name in tasks_we_used

    task_seqlen_map = {
        "allenai/prosocial-dialog":256,
    }

    inp_ls = []

    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(task_name,
                                     split=f"test")\
            .shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)
        for item in trainset_text:
            inp = item["context"]
            resp = item["response"]
            inp_ls.append((inp,resp))
    assert inp_ls != []

    res_ls = []
    if modelname=="gpt-3.5-turbo-1106":
        from training_data_collecting_openai import chatWithOpenAI_APIs
        res_ls=[]
        for d in tqdm(inp_ls):
            inps,summary = d
            res=chatWithOpenAI_APIs(modelname, "", inps)
            print(f"Generated Text: {res}")
            res_ls.append((res, summary))
    elif base_model_name is None:
        model = InferObj(
            model_name=modelname, device="auto",
            max_length=task_seqlen_map[task_name],
        )
        gen_pipeline = model.text_gen
        res_ls = []
        for d in tqdm(inp_ls):
            inps, summary = d
            final_inps = f"User: {inps} Assistant: "
            res = gen_pipeline(
                final_inps,
                max_new_tokens=mnt,
            )[
                0
            ]["generated_text"]
            res = res.split(final_inps)[1]
            print(f"Text Generated:>>> {res}")
            res_ls.append((res, summary))
    else:
        print("USING PEFT: BASE MODEL + LORA")
        # load model based on our idea
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if modelname is not None:
            model = PeftModel.from_pretrained(model, modelname)
        tokenizer = AutoTokenizer\
            .from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        res_ls = []
        for d in tqdm(inp_ls):
            inps, summary = d
            final_inps = f"User: {inps} Assistant: "
            inps_idx=tokenizer.encode(final_inps,max_length=2048,
                                      padding="longest",
                                      return_tensors="pt")
            print(inps_idx)
            inps_idx=inps_idx.to("cuda")
            res = model.generate(inps_idx,
                                 max_new_tokens=mnt,)
            print(res)
            res=tokenizer.decode(res[0])
            if final_inps in res:
                res = res.split(final_inps)[1]
            else:
                res = res
            print(f"Text Generated:>>> {res}")
            res_ls.append((res, summary))

    model = None
    gen_pipeline = None
    tokenizer = None
    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


def eval_safe(res_ls):
    from nlg_metric import overall_metrics
    hyps, refs = zip(*res_ls)
    ## note that here we need to add a evaluation comparison.
    # TODO!!!
    res_dict1=overall_metrics(hyps, refs)
    return res_dict1

def eval_varying_train_num():
    taskls = [
        "deepmind/code_contests",
        ]
    mls = [
        "vanilla",
        # "LoRD-VI",
        # "pretrained",
        # "gpt-3.5-turbo-1106",
        # "kd",
        ]
    # mls = ["vanilla", "kd", "google/gemma-2b", "Complex-lord",]
    train_times = [
        "1",
        # "2",
        # "3",
        # "4",
        # "5",
        ]
    train_nums = [
        "2",
        # "16",
        # "64",
        # "128",
        # "256",
        # "512",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./code_0627_dataset_res/"
    res_dict = {}
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)

    res_dict_averaged={}

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                temp_scorels=[]
                for itime in train_times:
                    prefix = "./codee_ckpts/CODEEE"
                    if m=="vanilla":
                        ckpt = (
                            prefix
                            + f"{task}{train_num}{itime}{m}___finally/"
                        )
                    elif m =="pretrained":
                        ckpt = f"./codee_ckpts/code---{task}{train_num}{itime}{m}_res.json"
                    elif m=="gpt-3.5-turbo-1106":
                        ckpt=m
                    else:
                        ckpt = prefix + \
                            f"{task}{train_num}{itime}{m}___period512/"
                    res_pth = ckpt+f"___{task}_code_infer_res.json"
                    res_pth = res_pth.replace("/", "__").replace(".", "")

                    if not os.path.exists(dir_p+res_pth):
                        if m=="pretrained":
                            res_ls = infer_code(None,
                                            task,
                                            dir_p+res_pth,
                                            test_set_take_num=500,
                                            mnt=256,
                                            base_model_name=base_model_name1,
                                            )
                        else:
                            res_ls = infer_code(ckpt,
                                            task,
                                            dir_p+res_pth,
                                            test_set_take_num=500,
                                            mnt=256,
                                            base_model_name=base_model_name1,
                                            )
                    else:
                        # from collections import OrderedDict
                        with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                            res_ls = json.load(
                                f, object_pairs_hook=OrderedDict)

                    scores = eval_sum(res_ls)
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

    with open(dir_p+"Overall__code_varytrain_num_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    with open(
        dir_p + "OverallScoresAveraged.json",
            "w", encoding="utf8"
    ) as f:
        json.dump(res_dict_averaged, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)
    print("------------------------------------------")
    pprint(res_dict_averaged)
    return res_dict


# running entry
if __name__ == "__main__":
    eval_varying_train_num()
    print("EVERYTHING DONE.")