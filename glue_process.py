"""
======================================================================
GLUE_PROCESS ---

Process GLUE dataset.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  1 March 2024
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

from training_data_collecting_openai import chatWithOpenAI_APIs
from training_data_collecting_openai import chatWithOpenAI__LogLogits


def load_glue_datals(lm_tokenizer,
                     task_name,
                     train_num=1,
                      model_name="gpt-3.5-turbo-1106",
                      topk=5,
                      max_length=1024,
                      openai_tmp_save_pth="./glue_openai_probs_res1__",
                      ):
    ## some preliminary knowledge of GLUE
    tasks_we_used = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",]

    task_label_map = {
        "cola": {"1": "acceptable", "0": "unacceptable"},
        "mnli": {"1": "neutral", "0": "entailment", "2": "contradiction"},
        "mrpc": {"1": "equivalent", "2": "not_equivalent"},
        "qnli": {"1": "not_entailment", "0": "entailment"},
        "qqp": {"1": "duplicate", "0": "not_duplicate"},
        "rte": {"1": "not_entailment", "0": "entailment"},
        "sst2": {"1": "positive", "0": "negative"},
        "wnli": {"0": "not_entailment", "1": "entailment"},
    }
    task_key_map = {
        "mrpc": ["sentence1", "sentence2"],
        "qnli": ["question", "sentence"],
        "qqp": ["question1", "question2"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],

    }
    task_prompt_map={
        "cola": "In your role as a grammar check tool, assess the following sentence and classify it as 'acceptable' if it is grammatically correct or 'unacceptable' if it is incorrect.",
        "mnli": "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction'.",
        "mrpc": "As a semantic comparison expert, evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'.",
        "qnli": "As a language expert, assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'.",
        "qqp": "In your role as a question comparison tool, assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'.",
        "rte": "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment'.",
        "sst2": "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'.",
        "wnli": "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment'.",
        }
    
    single_input_tasks = ["cola", "sst2",]
    double_input_tasks = ["mrpc", "qnli", "qqp", "rte", "wnli",]
    
    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "glue"
    trainset_text = load_dataset(dataset_name,task_name,
                                 split=f"train[:{train_num}]")

    inp_ls=[]
    ## collecting the input prompts
    if task_name in single_input_tasks:
        for d in trainset_text:
            inps = d["sentence"]
            inp_ls.append(inps)
            # break

    elif task_name == "mnli":
        for d in sets:
            inps = d["premise"]+" <SEP> "+d["hypothesis"]
            # label = d["label"]
            inp_ls.append(inps)
    elif task_name in double_input_tasks:
        for d in sets:
            inps = d[task_key_map[task_name][0]]+" <SEP> " +\
                d[task_key_map[task_name][1]]
            # label = d["label"]
            # label = task_label_map[task_name][str(label)]
            inp_ls.append(inps)
    else:
        logging.error(f"task name: {task_name} not found.")
    

    pp=task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: "\
               for x in inp_ls]
    p_idxls = lm_tokenizer(prompts,
                           padding="longest",
                           truncation=True,
                           max_length=max_length,
                           return_tensors="pt"
                           ).input_ids

    openai_tmp_save_pth+=f"task_{task_name}-trainNUM_{train_num}.pkl"

    if not os.path.exists(openai_tmp_save_pth):
        print("RUNNING ChatGPT Stealing...")
        text2ls = []
        idx2_dist_ls=[]
        probsls = []
        iii_bgn=0
        for q in tqdm(inp_ls, desc="ChatGPT Inference:"):
            qd=[{"role":"system","content":"Instruction: "+pp},
                {"role":"user","content":q}]
            res = chatWithOpenAI__LogLogits(
                model_name,
                qd,
                topk,
            )
            resp, logprb=res
            bgn_idx = lm_tokenizer([resp],
                        padding="longest",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                        ).input_ids[0][0]
            

            idx2=p_idxls[iii_bgn].tolist()
            logits_distr = torch.nn.functional.one_hot(
                p_idxls[iii_bgn][1:],
                num_classes=V,
                ).float() 

            idx2_dist=[[x,] for x in idx2]
            for i in range(len(idx2_dist)):
                for _ in range(topk-1):
                    idxxx=random.randint(0, V-1)
                    idx2_dist[i].append(idxxx)
            idx2_dist=idx2_dist[1:]

            # print(logits_distr.shape)
            # logits_distr=logits_distr[torch.tensor(idx2_dist,
            #                                        dtype=torch.long)]
            logits_distr=torch.gather(logits_distr, 1,
                                      torch.tensor(idx2_dist,
                                                   dtype=torch.long))
                    
            logits_distr=[logits_distr[i]\
                          for i in range(len(logits_distr))]

            for i, topkdict in enumerate(logprb):
                selected_token = topkdict.token
                subtokens = lm_tokenizer.tokenize(selected_token)
                sub_idxes=lm_tokenizer.convert_tokens_to_ids(subtokens)
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
                    dist = torch.zeros(topk)
                    idx2_tmp_token_dist=torch.zeros(topk,
                                                    dtype=torch.long)
                    dist = torch.tensor(topk_logits)
                    for k, subidx in enumerate(topk_subidxes):
                        if len(subidx)<=j:
                            idx2_tmp_token_dist[k]=subidx[0]
                        else:
                            idx2_tmp_token_dist[k]=subidx[j]
                    
                    logits_distr.append(dist)
                    idx2_dist.append(idx2_tmp_token_dist)
                    

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


# def eval_model():



