"""
======================================================================
GLUE_PROCESS ---

Process GLUE dataset.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  1 March 2024
======================================================================
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
from pprint import pprint

import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from training_data_collecting_openai import chatWithOpenAI_APIs
from training_data_collecting_openai import chatWithOpenAI__LogLogits

from gen_pipeline_open import InferObj

task_prompt_map = {
    "cola": "Assess the following sentence and classify it as 'acceptable' or 'unacceptable'.",
    # "mnli": "Assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction'.",
    "mrpc": "Evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'.",
    "qnli": "Assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'.",
    "qqp": "Assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'.",
    "rte": "Assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment'.",
    "sst2": "Determine whether the following text is 'positive' or 'negative'.",
    "wnli": "Assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment'.",
}


def load_glue_nonlabel(lm_tokenizer,
                       task_name,
                       train_num=1,
                       max_length=1024,
                       ):
    # some preliminary knowledge of GLUE
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

    single_input_tasks = ["cola", "sst2",]
    double_input_tasks = ["mrpc", "qnli", "qqp", "rte", "wnli",]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "glue"
    trainset_text = load_dataset(dataset_name, task_name,
                                 split=f"train")\
        .shuffle(seed=20240306).to_iterable_dataset().take(train_num)

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
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
        print(f"task name: {task_name} not found.")

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    return p_idxls, None, None, None


def load_glue_datals(lm_tokenizer,
                     task_name,
                     train_num=1,
                     model_name="gpt-3.5-turbo-1106",
                     topk=5,
                     max_length=1024,
                     openai_tmp_save_pth="./STEALED_PKLS/glue_openai_probs_res1__",
                     ):
    # some preliminary knowledge of GLUE
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

    single_input_tasks = ["cola", "sst2",]
    double_input_tasks = ["mrpc", "qnli", "qqp", "rte", "wnli",]

    assert task_name in tasks_we_used

    V = lm_tokenizer.vocab_size
    dataset_name = "glue"
    trainset_text = load_dataset(dataset_name, task_name,
                                 split=f"train")\
        .shuffle(seed=20240306).to_iterable_dataset().take(train_num)

    sets = trainset_text
    inp_ls = []
    # collecting the input prompts
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
        print(f"task name: {task_name} not found.")

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])
    # p_idxls = lm_tokenizer(prompts,
    #                        # padding="longest",
    #                        # truncation=True,
    #                        # max_length=max_length,
    #                        return_tensors="pt"
    #                        ).input_ids

    openai_tmp_save_pth += f"task_{task_name}-trainNUM_{train_num}.pkl"

    if not os.path.exists(openai_tmp_save_pth):
        print("RUNNING ChatGPT Stealing...")
        text2ls = []
        idx2_dist_ls = []
        probsls = []
        for iii_bgn, q in tqdm(enumerate(inp_ls),
                               desc="ChatGPT Inference:"):
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
            # logits_distr[logits_distr<1e-9]=1e-10
            logits_distr = torch.log(logits_distr)

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
                # topk_logits = [exp(x) for x in topk_logits]

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


def infer_glue(modelname, task_name, res_pth,
               test_set_take_num=1000,
               ):
    """Infer the hf's pretraining models in GLUE"""
    save_pth = res_pth

    model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=2047)
    gen_pipeline = model.text_gen

    prompt = task_prompt_map[task_name]

    # models to be evaluted
    model_ls = [
        "lmsys/vicuna-7b-v1.5-16k",
        "microsoft/phi-1_5",
        "NousResearch/Llama-2-7b-chat-hf",
        "Qwen/Qwen-7B-Chat",
        "01-ai/Yi-6B",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "openchat/openchat_3.5"]

    # tasks for evaluation
    tasks_name = ["valid_parentheses", "bool_logic",
                  "un_multi", "squad_v2",
                  "sst2", "wnli", "rte",
                  "mnli", "cola", "qqp",
                  "qnli", "mrpc",]

    glue_ds = ["ax", "cola", "mnli",
               "mnli_matched",
               "mnli_mismatched", "mrpc",
               "qnli", "qqp", "rte", "sst2",
               "stsb", "wnli",]

    tasks_we_used = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",]

    task_label_map = {
        "cola": {"1": "acceptable", "0": "unacceptable"},
        # "mnli": {"1": "neutral", "0": "entailment", "2": "contradiction"},
        "mrpc": {"1": "equivalent", "0": "not_equivalent"},
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
    single_input_tasks = ["cola", "sst2",]
    double_input_tasks = ["mrpc", "qnli", "qqp", "rte", "wnli",]

    assert task_name in tasks_we_used
    dataset = load_dataset("glue", task_name)
    res_ls = []
    if task_name in single_input_tasks:
        if len(dataset["validation"]) > test_set_take_num:
            sets = dataset["validation"].shuffle(20240307)\
                .to_iterable_dataset().take(test_set_take_num)
            # sets = dataset["train"].shuffle(20240306)\
            # .to_iterable_dataset().take(test_set_take_num)
        else:
            sets = dataset["validation"]
        # sets=sets.shuffle(seed=20240307)
        iii = 0
        for d in tqdm(sets):
            iii += 1
            if iii == 1 or iii == test_set_take_num:
                print(d)
            inps = d["sentence"]
            label = d["label"]
            label = task_label_map[task_name][str(label)]
            final_inps = "Instruction: " + prompt +\
                " User: "+inps+" Assistant: "
            print("Inps: ", final_inps)
            res = gen_pipeline(final_inps,
                               max_new_tokens=16,)[0]["generated_text"]
            res = res.split(final_inps)[1]
            res_ls.append((res, label))
            print("Generations: ", res)
            # break

    elif task_name == "mnli":
        if len(dataset["validation_matched"]) > test_set_take_num:
            sets = dataset["validation_matched"]\
                .to_iterable_dataset().take(test_set_take_num)
        else:
            sets = dataset["validation_matched"]
        for d in tqdm(sets):
            inps = d["premise"]+"SEP"+d["hypothesis"]
            label = d["label"]
            label = task_label_map[task_name][str(label)]

            final_inps = "Instruction: " + prompt +\
                " User: "+inps+" Assistant: "
            res = gen_pipeline(final_inps,
                               max_new_tokens=16,)[0]["generated_text"]
            res = res.split(final_inps)[1]
            res_ls.append((res, label))
    elif task_name in double_input_tasks:
        if len(dataset["validation"]) > test_set_take_num:
            sets = dataset["validation"]\
                .to_iterable_dataset().take(test_set_take_num)
        else:
            sets = dataset["validation"]
        for d in tqdm(sets):
            inps = d[task_key_map[task_name][0]]+"SEP" +\
                d[task_key_map[task_name][1]]
            label = d["label"]
            print(f"task name: {task_name} Label: {label}")
            label = task_label_map[task_name][str(label)]
            final_inps = "Instruction: " + prompt +\
                " User: "+inps+" Assistant: "
            res = gen_pipeline(final_inps,
                               max_new_tokens=16,)[0]["generated_text"]
            res = res.split(final_inps)[1]
            res_ls.append((res, label))
            # break
    else:
        print(f"task name: {task_name} not found.")
    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


# normal import
task_label_map = {
    "cola": {"1": "acceptable", "0": "unacceptable"},
    # "mnli": {"1": "neutral", "0": "entailment", "2": "contradiction"},
    "mrpc": {"1": "equivalent", "0": "not_equivalent"},
    "qnli": {"1": "not_entailment", "0": "entailment"},
    "qqp": {"1": "duplicate", "0": "not_duplicate"},
    "rte": {"1": "not_entailment", "0": "entailment"},
    "sst2": {"1": "positive", "0": "negative"},
    "wnli": {"0": "not_entailment", "1": "entailment"},
}


def eval_glue(task, res):

    predict_ls = []
    label_ls = []
    submap = task_label_map[task]
    sm_r = {v: k for k, v in submap.items()}
    text_dict = list(sm_r.keys())

    for res_sent, lbl in res:
        # print(res_sent)
        res_sent = res_sent.lower()
        # label_ls.append(float(sm_r[lbl]))
        if "not" in text_dict[0] or "not" in text_dict[1]:
            if "not" in text_dict[0]:
                if "not" in res_sent:
                    vv = float(sm_r[text_dict[0]])
                else:
                    vv = float(sm_r[text_dict[1]])
            else:
                if "not" in res_sent:
                    vv = float(sm_r[text_dict[1]])
                else:
                    vv = float(sm_r[text_dict[0]])
        else:
            if text_dict[0] in res_sent and text_dict[1] not in res_sent:
                vv = float(sm_r[text_dict[0]])
            else:
                vv = float(sm_r[text_dict[1]])
        predict_ls.append(vv)
        label_ls.append(float(sm_r[lbl]))

    metric_ls = [accuracy_score, precision_score, recall_score, f1_score]
    scores = []
    for m in metric_ls:
        scores.append(m(label_ls, predict_ls))
    return scores


def evaluation_datas():
    test_task_ckpt_ls = [

        # original
        ["cola", "google/gemma-2b"],

        # ablation study
        # ["cola","./POD_SAVE_CKPTs/kd_3Epochkd_256cola___finally"],
        ["cola", "./POD_SAVE_CKPTs/cola_3Epochvanilla_256cola___finally/"],
        ["cola", "./POD_SAVE_CKPTs/cola_3Epochkd_256cola___finally/"],

        # ## method comparison
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/Complex-lord_256cola___period0"],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/Complex-lord_256cola___period1"],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/Complex-lord_256cola___period2"],



        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/lord_256cola___period0"],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/lord_256cola___period1"],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/lord_256cola___period2"],


        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/reinforce-lord_256cola___period0"],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/reinforce-lord_256cola___period1"],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period0306cola/reinforce-lord_256cola___period2"],


        # varying periods
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period0"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period1"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period2"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period3"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period4"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period5"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period6"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period7"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period8"
         ],
        ["cola",
         "./POD_SAVE_CKPTs/vary_period_complex/Complex-lord256cola1100___period9"
         ],
    ]

    res_dict = {}
    dir_p = "./glue_res/"
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    for task_ckpt in test_task_ckpt_ls:
        task, ckpt = task_ckpt
        res_pth = ckpt+f"___{task}_glue_infer_res.json"
        res_pth = res_pth.replace("/", "__").replace(".", "")
        if not os.path.exists(dir_p+res_pth):
            res_ls = infer_glue(ckpt, task, dir_p+res_pth)
        else:
            # from collections import OrderedDict
            with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                res_ls = json.load(f, object_pairs_hook=OrderedDict)

        scores = eval_glue(task, res_ls)
        print(task, ckpt)
        print(scores)
        res_dict[task+"-----"+ckpt] = scores
    with open(dir_p+"glue_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


def glue_big_evals():
    # methodls = ["Complex-lord", "vanilla", "kd", "black--Complex-lord"]

    methodls = ["vanilla", "kd",]
    train_times = [str(x+1) for x in range(5)]
    dir_p = "./GLUE_infers/"
    res_dict = {}
    for task in task_prompt_map.keys():
        res_dict[task] = {}
        for m in methodls:
            for itime in train_times:
                if not os.path.exists(dir_p):
                    os.makedirs(dir_p)
                prefix = "./GLUE_ckpts/"
                if m == "Complex-lord" or m == "black--Complex-lord":
                    ckpt = prefix+f"{task}{m}256100__{itime}___period2"
                else:
                    ckpt = prefix+f"{task}{m}256100__{itime}___finally"
                res_pth = ckpt+f"___{task}_glue_infer_res.json"
                res_pth = res_pth.replace("/", "__").replace(".", "")
                if not os.path.exists(dir_p+res_pth):
                    res_ls = infer_glue(ckpt, task, dir_p+res_pth,
                                        test_set_take_num=100)
                else:
                    # from collections import OrderedDict
                    with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                        res_ls = json.load(
                            f, object_pairs_hook=OrderedDict)

                scores = eval_glue(task, res_ls)
                print(task, ckpt)
                print(scores)
                res_dict[task][task+"-----"+ckpt] = scores
    with open(dir_p+"glue_inference_scores5times.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


if __name__ == "__main__":
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # res = load_glue_datals(tokenizer,
    #                        "cola",
    #                        train_num=3,
    #                        model_name="gpt-3.5-turbo-1106",
    #                        topk=5,
    #                        max_length=1024,
    #                        openai_tmp_save_pth="./temp.pkl.deletethis.txt",
    #                        )
    # prompt=res[0]
    # text2=res[1]
    # print("-------------------------")
    # print(tokenizer.decode(prompt[0]))
    # print(tokenizer.decode(text2[0]))
    # print("-------------------------")
    # print(tokenizer.decode(prompt[1]))
    # print(tokenizer.decode(text2[1]))
    # print("-------------------------")
    # print(tokenizer.decode(prompt[2]))
    # print(tokenizer.decode(text2[2]))
    # print("-------------------------")

    # evaluation_datas()
    glue_big_evals()
