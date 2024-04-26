"""
======================================================================
GENERAL_TRAIN --- general pretraining with vanilla method and our lora
methods.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 25 April 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
# import pickle
# import os
# from os.path import join, exists
# from collections import Counter,OrderedDict
# from bisect import bisect
# from copy import deepcopy
# import pickle

# ## transformers related import
# from transformers import T5Tokenizer,T5ForConditionalGeneration
# from transformers import BertTokenizer
# from transformers import pipeline
# import transformers
import os
from collections import OrderedDict

from datasets import load_dataset
from tqdm import tqdm


def ___data_cleaning():
    dirp="./datas"
    list_files=os.listdir(dirp)
    fpths=[dirp+"/"+x for x in list_files]

    datalist=[]
    for fname in fpths:
        # from collections import OrderedDict
        with open(fname, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)
        datalist.extend(data)

    print(len(datalist))

    # with open("./datas/overall_cleaned_data.json",
    #           'w',encoding='utf8') as f:
    #     json.dump(datalist, f, ensure_ascii=False, indent=4)

    # print("Save done.")

    ## transfer to huggingface formation.

    # resls=[]
    # for x in datalist:
    #     resls.append(json.dumps(x, ensure_ascii=False,))
    # restext="\n".join(resls)
    
    # with open("./datas/hf_overall_cleaned_data.jsonl",
    #           'w',encoding='utf8') as f:
    #     f.write(restext)
    # print("Save done.")

    ## filter a short subset with length < N
    N=256
    newresls=[]
    for x in datalist:
        num_inps=len(x["prompt"].split(" "))
        num_out=len(x["response"].split(" "))
        if num_inps+num_out<=N:
            newresls.append(x)

    print(f"Data Num After filtered: {len(newresls)}")
    ## save this short texts.

    resls=[]
    for x in newresls:
        resls.append(json.dumps(x, ensure_ascii=False,))
    restext="\n".join(resls)
    
    with open("./datas/hf_overall_cleaned_data_short256_249.jsonl",
              'w',encoding='utf8') as f:
        f.write(restext)
    print("Save done.")


def loading_data(
        dataset_name="liangzid/claude3_chat3.3k"):

    dataset=load_dataset(dataset_name, split="train")
    inpsls=[]
    outls=[]

    # print(dataset)

    for item in tqdm(dataset):
        # print(f"ITEM: {item}")
        inp = item["prompt"]
        out = item["response"]
        inpsls.append(inp)
        outls.append(out)
    print(f"total length: {len(inpsls)}")
    return inpsls,outls

def data_format_transform(inpsls,outls,tokenizer,
                          train_num,
                          max_length,
                          topk=1,):
    p_idxls = []
    text2ls = []

    for iii_bgn, q in tqdm(enumerate(inpsls),
                            desc="..."):
        if iii_bgn==train_num:
            break
        prompt_input=f"User: {q}\n Assistant: "
        p_idxls.append(tokenizer(prompt_input,
                                 max_length=max_length,
                                 return_tensors="pt").input_ids[0])

        text=f"User: {q}\n Assistant: {outls[iii_bgn]}"
        text2ls.append(tokenizer(text,
                                 max_length=max_length,
                                 return_tensors="pt").input_ids[0])

    return p_idxls, text2ls, None, None


def general_load_data(
    tokenizer,
        train_num=2,
        max_length=1024,
    topk=1,
    dataset_name="liangzid/claude3_chat3.3k",
        ):
    ips,ots=loading_data(dataset_name)
    return data_format_transform(ips,ots,tokenizer,
                                 train_num,max_length,
                                 topk=topk)

## running entry
if __name__=="__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    res=general_load_data(tokenizer,
                          dataset_name="liangzid/claude3_short256")
    print(res)

    # ___data_cleaning()


    # print("EVERYTHING DONE.")
