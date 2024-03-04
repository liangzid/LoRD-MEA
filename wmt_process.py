"""
======================================================================
WMT_PROCESS ---

WMT dataset process scripts.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  4 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import torch
from datasets import load_dataset
from openai import OpenAI as oa
# import time
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

def load_wmt_datals(tokenizer,
                    task_name,
                    train_num=100,
                    model_name="gpt-3.5-turbo-1106",
                    topk=5,
                    max_length=1024,
                    openai_tmp_save_pth="./wmt_data_saveto_"):

    lm_tokenizer=tokenizer
    tasks_we_used=[
        "cs-en",
        "du-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
        ]

    task_prompt_map={
        "cs-en": "Translate the sentence from Czech to English Please.",
        "du-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
        }
    
    assert task_name in tasks_we_used
    
    V = lm_tokenizer.vocab_size
    dataset_name = "wmt16"
    trainset_text = load_dataset(dataset_name, task_name,
                                 split=f"train[:{train_num}]")
    inp_ls = []

    from_lang,to_lange=task_name.split("-")
    for text in trainset_text["translation"]:
        from_text=text[from_lang]
        to_text=text[to_lange]
        inp_ls.append(from_text)

    pp=task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    openai_tmp_save_pth += f"WMTtask_{task_name}-trainNUM_{train_num}.pkl"

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
    
def commonly_used_openai_post_process(
        openai_tmp_save_pth,
        inp_ls,
        pp,
        model_name,
        topk,
        max_length,
        p_idxls,
        V,
        lm_tokenizer,
        ):

    if not os.path.exists(openai_tmp_save_pth):
        print("RUNNING ChatGPT Stealing...")
        text2ls = []
        idx2_dist_ls = []
        probsls = []
        iii_bgn = 0
        for q in tqdm(inp_ls, desc="ChatGPT Inference:"):
            qd = [{"role": "system", "content": "Instruction: "+pp},
                  {"role": "user", "content": q}]
            res = chatWithOpenAI__LogLogits(
                model_name,
                qd,
                topk,
            )
            resp, logprb = res
            bgn_idx = lm_tokenizer([resp],
                                   # padding="longest",
                                   truncation=True,
                                   max_length=max_length,
                                   return_tensors="pt"
                                   ).input_ids[0][0]

            idx2 = p_idxls[iii_bgn].tolist()
            logits_distr = torch.nn.functional.one_hot(
                p_idxls[iii_bgn][1:],
                num_classes=V,
            ).float()

            idx2_dist = [[x,] for x in idx2]
            for i in range(len(idx2_dist)):
                for _ in range(topk-1):
                    idxxx = random.randint(0, V-1)
                    idx2_dist[i].append(idxxx)
            idx2_dist = idx2_dist[1:]

            # print(logits_distr.shape)
            # logits_distr=logits_distr[torch.tensor(idx2_dist,
            #                                        dtype=torch.long)]
            logits_distr = torch.gather(logits_distr, 1,
                                        torch.tensor(idx2_dist,
                                                     dtype=torch.long))

            logits_distr = [logits_distr[i]
                            for i in range(len(logits_distr))]

            for i, topkdict in enumerate(logprb):
                selected_token = topkdict.token
                subtokens = lm_tokenizer.tokenize(selected_token)
                sub_idxes = lm_tokenizer.convert_tokens_to_ids(subtokens)
                idx2.extend(sub_idxes)

                topk_tokens = [x.token for x in topkdict.top_logprobs]
                topk_subtokenss = [lm_tokenizer.tokenize(x)
                                   for x in topk_tokens]
                topk_subidxes = [lm_tokenizer.convert_tokens_to_ids(x)
                                 for x in topk_subtokenss]
                topk_logits = [x.logprob
                               for x in topkdict.top_logprobs]
                topk_logits = [exp(x) for x in topk_logits]

                # idx2_dist.extend(topk_subidxes)
                # logits_distr.extend(topk_logits)

                for j in range(len(subtokens)):
                    # dist = torch.tensor(topk_logits)
                    idx2_tmp_token_dist = torch.zeros(topk,
                                                      dtype=torch.long)
                    dist = torch.tensor(topk_logits)
                    # print("dist: ",dist)
                    for k, subidx in enumerate(topk_subidxes):
                        if len(subidx) <= j:
                            idx2_tmp_token_dist[k] = subidx[0]
                        else:
                            idx2_tmp_token_dist[k] = subidx[j]

                    logits_distr.append(dist)
                    idx2_dist.append(idx2_tmp_token_dist)
            # print("logits_distr: ",logits_distr)
            # print("idx2: ",idx2)
            # print("idx2_dist: ",idx2_dist)

            # print(len(idx2), len(logits_distr))
            assert len(idx2) == len(logits_distr)+1
            probsls.append(logits_distr)
            text2ls.append(idx2)
            idx2_dist_ls.append(idx2_dist)

        with open(openai_tmp_save_pth,
                  'wb') as f:
            pickle.dump([text2ls, probsls, idx2_dist_ls],
                        f,)
    else:
        print("Directly Loading...")
        # from collections import OrderedDict
        with open(openai_tmp_save_pth, 'rb') as f:
            data = pickle.load(f,)
        text2ls = data[0]
        probsls = data[1]
        idx2_dist_ls = data[2]

    return p_idxls, text2ls, probsls, idx2_dist_ls





























## running entry
if __name__=="__main__":
    # main()
    print("EVERYTHING DONE.")


