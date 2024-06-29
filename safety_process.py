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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

def load_safety_datals(tokenizer,
                    task_name="allenai/prosocial-dialog",
                    train_num=100,
                    model_name="gpt-3.5-turbo-1106",
                    topk=5,
                    max_length=1024,
                       openai_tmp_save_pth="./STEALED_PKLS/wmt_data_saveto_",
                       tokenizer_name=None,):

    lm_tokenizer = tokenizer
    pp=""

    V = lm_tokenizer.vocab_size
    tasks_we_used = [
        "PKU-Alignment/PKU-SafeRLHF",
        "thu-coai/diasafety",
        "allenai/prosocial-dialog",
    ]
    assert task_name in tasks_we_used
    dataset_name = task_name
    inp_ls = []
    if task_name == tasks_we_used[2]:
        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")
        for item in trainset_text:
            inp = item["context"]
            resp = item["response"]
            inp_ls.append(inp)
    elif task_name == tasks_we_used[0]:
        trainset_text = load_dataset(dataset_name,
                                     # split=f"train[:{train_num}]",
                                     split=f"train",
                                     ).shuffle(20240307)
        for item in trainset_text:
            inp = item["prompt"]
            safe_label = item["is_response_1_safe"]
            print(f"Safe Label: {safe_label}")
            if safe_label==False:
                inp_ls.append(inp)
            if len(inp_ls)>=train_num:
                break
    elif task_name == tasks_we_used[1]:
        trainset_text = load_dataset(dataset_name,
                                     # split=f"train[:{train_num}]",
                                     split=f"train",
                                     ).shuffle(20240307)
        for item in trainset_text:
            inp = item["context"]
            inp_ls.append(inp)
            if len(inp_ls)>=train_num:
                break

    assert inp_ls != []

    p_idxls = []
    prompts=[f"User: {x} Assistant: " for x in inp_ls]
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


def infer_safety(modelname, task_name, res_pth,
              test_set_take_num=100,
              mnt=16,
              base_model_name=None,
              ):
    save_pth = res_pth

    tasks_we_used = [
        "PKU-Alignment/PKU-SafeRLHF",
        "thu-coai/diasafety",
        # "allenai/prosocial-dialog",
    ]

    assert task_name in tasks_we_used

    if task_name == tasks_we_used[0]:
        dataset = load_dataset(task_name,
                            split=f"test").shuffle(20240307)\
            .to_iterable_dataset()
            # .take(test_set_take_num)
        # sets = dataset
        sets = []
        for item in dataset:
            if item["is_response_1_safe"]==False:
                sets.append(item)
            if len(sets) >= test_set_take_num:
                break
    elif task_name == tasks_we_used[1]:
        dataset = load_dataset(task_name,
                            split=f"test").shuffle(20240307)\
            .to_iterable_dataset()
            # .take(test_set_take_num)
        # sets = dataset
        sets = []
        for item in dataset:
            if item["label"]=="Unsafe":
                sets.append(item)
            if len(sets) >= test_set_take_num:
                break

    if modelname=="gpt-3.5-turbo-1106":
        from training_data_collecting_openai import chatWithOpenAI_APIs
        res_ls=[]
        for d in tqdm(sets):
            if task_name == tasks_we_used[0]:
                d = d["context"]
            elif task_name == tasks_we_used[1]:
                inp = d["context"]
                out = d["response"]
                category = d["category"]
                label = d["label"]
                d = inp
            res=chatWithOpenAI_APIs(modelname, "", query)
            print(f"Generated Text: {res}")
            res_ls.append((res,""))
    elif base_model_name is None:
        model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=2047,
                     open_16_mode=True,)
        gen_pipeline = model.text_gen

        res_ls = []
        pp = task_prompt_map[task_name]
        for d in tqdm(sets,total=test_set_take_num):
            if task_name == tasks_we_used[0]:
                d = d["context"]
            elif task_name == tasks_we_used[1]:
                inp = d["context"]
                out = d["response"]
                category = d["category"]
                label = d["label"]
                d = inp
            final_inps = \
                " User: "+d+" Assistant: "
            res = gen_pipeline(final_inps,
                            do_sample=True,
                            max_new_tokens=mnt,
                            )[0]["generated_text"]

            print("++++++++++++++++++DEBUG INFO+++++++++++++++++++++++")
            print(f">>>Res with Inpus: {res}")
            res = res.split(final_inps)[1]
            print(f">>>Res without Inpus: {res}")
            res_ls.append((res, ""))
            # break
    else:
        print("USING PEFT: BASE MODEL + LORA")
        # load model based on our idea
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            # trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, modelname)
        tokenizer = AutoTokenizer\
            .from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        res_ls = []
        pp = ""
        input_idxls=[]
        for d in tqdm(sets,total=test_set_take_num):
            if task_name == tasks_we_used[0]:
                d = d["context"]
            elif task_name == tasks_we_used[1]:
                inp = d["context"]
                out = d["response"]
                category = d["category"]
                label = d["label"]
                d = inp
            final_inps = \
                " User: "+d+" Assistant: "
            inps_idx=tokenizer.encode(final_inps,max_length=128,
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
            res_ls.append((res, ""))
        
    model = None
    gen_pipeline = None
    tokenizer = None

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls

def eval_varying_train_num():

    print("LoAdInG Evaluation model...")
    from safety.perspectiveAPI import PerspectiveAPI
    eval_model=PerspectiveAPI()

    taskls = [
        # "PKU-Alignment/PKU-SafeRLHF",
        "thu-coai/diasafety",
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
        # "2",
        # "4",
        # "8",
        # "16",
        # "32",
        "64",
        # "128",
        # "256",
        # "512",
        # "1024",
        # "2048",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./safety_0627_dataset_res/"
    res_dict = {}
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)

    res_dict_averaged={}

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                temp_scorels=[]
                for itime in train_times:
                    prefix = "./safety_ckpts/SAFETY"
                    if m=="vanilla":
                        ckpt = (
                            prefix
                            + f"{task}{train_num}{itime}{m}___finally/"
                        )
                    elif m =="pretrained":
                        ckpt = f"./safety_ckpts/code---{task}{train_num}{itime}{m}_res.json"
                    elif m=="gpt-3.5-turbo-1106":
                        ckpt=m
                    else:
                        ckpt = prefix + \
                            f"{task}{train_num}{itime}{m}___period512/"
                    res_pth = ckpt+f"___{task}_safety_infer_res.json"
                    res_pth = res_pth.replace("/", "__").replace(".", "")

                    if not os.path.exists(dir_p+res_pth):
                        if m=="pretrained":
                            res_ls = infer_safety(None,
                                            task,
                                            dir_p+res_pth,
                                            test_set_take_num=500,
                                            mnt=256,
                                            base_model_name=base_model_name1,
                                            )
                        else:
                            res_ls = infer_safety(ckpt,
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

                    query_ls,resls=zip(*res_ls)
                    score_mean_dict,score_dict_ls =\
                        eval_model.eval_ls(query_ls)
                    print(task, ckpt)
                    print(score_mean_dict)
                    res_dict[task+"-----"+res_pth] = score_mean_dict
                    scores=score_mean_dict
                    score_ls=[
                        scores["toxicity"],
                        scores["severe_toxicity"],
                        scores["insult"],
                        scores["profanity"],
                        scores["threat"],
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

    with open(dir_p+"Overall__safety_varytrain_num_inference_scores.json",
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
