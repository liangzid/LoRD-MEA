"""
======================================================================
WMT_PROCESS ---

WMT dataset process scripts.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  4 March 2024
======================================================================
"""

import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,2,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from gen_pipeline_open import InferObj
from training_data_collecting_openai import chatWithOpenAI__LogLogits
from training_data_collecting_openai import chatWithOpenAI_APIs
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from tqdm import tqdm
import pickle
from pprint import pprint
import random
from math import exp
from collections import OrderedDict
import json
from openai import OpenAI as oa
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import numpy as np
import math
from sequence_utils import left_pad
# ------------------------ Code --------------------------------------
# import time


def load_wmt_nonlabel(tokenizer,
                      task_name,
                      train_num=100,
                      max_length=1024,
                      ):

    lm_tokenizer = tokenizer
    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
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
    trainset_text = load_dataset(dataset_name, task_name,
                                 split=f"train")\
        .shuffle(20240306)\
        .to_iterable_dataset()\
        .take(train_num)
    # print(trainset_text[0])
    # print("------------------------")

    inp_ls = []

    from_lang, to_lange = task_name.split("-")
    for text in trainset_text:
        text = text["translation"]
        from_text = text[from_lang]
        to_text = text[to_lange]
        inp_ls.append(from_text)

    pp = task_prompt_map[task_name]
    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    return p_idxls, None, None, None


def load_wmt_datals(tokenizer,
                    task_name,
                    train_num=100,
                    model_name="gpt-3.5-turbo-1106",
                    topk=5,
                    max_length=1024,
                    openai_tmp_save_pth="./STEALED_PKLS/wmt_data_saveto_"):

    lm_tokenizer = tokenizer
    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
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
    trainset_text = load_dataset(dataset_name, task_name,
                                 split=f"train")\
        .shuffle(20240306)\
        .to_iterable_dataset()\
        .take(train_num)
    # print(trainset_text[0])
    # print("------------------------")

    inp_ls = []

    from_lang, to_lange = task_name.split("-")
    for text in trainset_text:
        text = text["translation"]
        from_text = text[from_lang]
        to_text = text[to_lange]
        inp_ls.append(from_text)

    pp = task_prompt_map[task_name]
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

def load_wmt_hyprid_gathering(tokenizer,
                         max_length=1024,
                         train_num=64,
                         hyprid_ls=[
                             "cs-en",
                             "de-en",
                             "fi-en",
                             ],):
    
    lm_tokenizer = tokenizer
    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
    }

    plsss=[]
    text2lsss=[]
    probslsss=[]
    idx2_dist_lsss=[]

    for task_name in hyprid_ls:
        assert task_name in tasks_we_used
        V = lm_tokenizer.vocab_size

        dataset_name = "wmt16"
        trainset_text = load_dataset(dataset_name, task_name,
                                    split=f"train[:{train_num}]")
        trainset_text = load_dataset(dataset_name, task_name,
                                    split=f"train")\
            .shuffle(20240306)\
            .to_iterable_dataset()\
            .take(train_num)

        inp_ls = []
        from_lang, to_lange = task_name.split("-")
        for text in trainset_text:
            text = text["translation"]
            from_text = text[from_lang]
            to_text = text[to_lange]
            inp_ls.append(from_text)

        pp = task_prompt_map[task_name]
        prompts = [f"Instruction: {pp} User: {x} Assistant: "
                for x in inp_ls]
        p_idxls = []
        for p in prompts:
            p_idxls.append(lm_tokenizer(p,
                return_tensors="pt").input_ids[0])
        openai_tmp_save_pth = f"./STEALED_PKLS/wmt_data_saveto_WMTtask_{task_name}-trainNUM_{train_num}.pkl"
        with open(openai_tmp_save_pth, 'rb') as f:
            data = pickle.load(f,)
        text2ls = data[0]
        probsls = data[1]
        idx2_dist_ls = data[2]

        plsss.extend(p_idxls)
        text2lsss.extend(text2ls)
        probslsss.extend(probsls)
        idx2_dist_lsss.extend(idx2_dist_ls)

    import random
    i_ls=list(range(len(plsss)))
    random.seed(20240306)
    random.shuffle(i_ls)

    npls=[plsss[x] for x in i_ls]
    ntls=[text2lsss[x] for x in i_ls]
    nprls=[probslsss[x] for x in i_ls]
    nils=[idx2_dist_lsss[x] for x in i_ls]
        
    overall_save_p="./STEALED_PKLS/wmt_hyprid.pkl"
    with open(overall_save_p,
              'wb') as f:
        pickle.dump(
            [
                ntls,nprls,nils,
            ],
            f,)
    print("Hyprid Save DONE.")
    return npls,ntls,nprls,nils


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
            # iii_bgn+=1



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


def infer_wmt(modelname, task_name, res_pth,
              test_set_take_num=100,
              mnt=16,
              base_model_name=None,
              ):
    save_pth = res_pth


    task_prompt_map = {
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
    }

    prompt = task_prompt_map[task_name]

    tasks_we_used = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    assert task_name in tasks_we_used
    dataset = load_dataset("wmt16",
                           task_name,
                           split=f"test").shuffle(20240307)\
        .to_iterable_dataset()\
        .take(test_set_take_num)
    # print("DATASET 0: ",dataset[0])
    # print("DATASET 1: ",dataset[1])
    sets = dataset
    from_lang, to_lange = task_name.split("-")


    if modelname=="gpt-3.5-turbo-1106":
        from training_data_collecting_openai import chatWithOpenAI_APIs
        res_ls=[]
        pp = task_prompt_map[task_name]
        for d in tqdm(sets):
            d=d["translation"]
            inps=d[from_lang]
            label=d[to_lange]
            res=chatWithOpenAI_APIs(modelname, pp, inps)
            print(f"Generated Text: {res}")
            res_ls.append((res, label))
    elif base_model_name is None:
        model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=2047,
                     open_16_mode=True,)
        gen_pipeline = model.text_gen

        res_ls = []
        pp = task_prompt_map[task_name]
        for d in tqdm(sets,total=test_set_take_num):
            d = d["translation"]
            inps = d[from_lang]
            label = d[to_lange]
            final_inps = "Instruction: " + pp +\
                " User: "+inps+" Assistant: "
            res = gen_pipeline(final_inps,
                            do_sample=True,
                            max_new_tokens=mnt,
                            )[0]["generated_text"]

            print("++++++++++++++++++DEBUG INFO+++++++++++++++++++++++")
            print(f">>>Res with Inpus: {res}")
            res = res.split(final_inps)[1]
            print(f">>>Res without Inpus: {res}")
            print(f">>>Labels: {label}")
            res_ls.append((res, label))
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
        pp = task_prompt_map[task_name]
        input_idxls=[]
        for d in tqdm(sets,total=test_set_take_num):
            d = d["translation"]
            inps = d[from_lang]
            label = d[to_lange]
            final_inps = "Instruction: " + pp +\
                " User: "+inps+" Assistant: "
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
            res_ls.append((res, label))

        #     # print("------------------")
        #     # print(inps_idx)
        #     # print(tokenizer.bos_token_id)
        #     input_idxls.append(inps_idx[0])
        # infer_batch_size=24
        # num_chunks=math.floor(len(input_idxls)/infer_batch_size)
        # for i_chunked in tqdm(range(num_chunks+1)):
        #     if i_chunked==num_chunks and num_chunks*infer_batch_size!=len(input_idxls):
        #         INPS=input_idxls[i_chunked*infer_batch_size:]
        #     else:
        #         INPS=input_idxls[i_chunked*infer_batch_size:\
        #                         (i_chunked+1)*infer_batch_size]

        #     INPS=left_pad(INPS, tokenizer.bos_token_id)
        #     INPS=INPS.to("cuda")
        #     res = model.generate(INPS,
        #                          # do_sample=True,
        #                          max_new_tokens=mnt,)
        #     for res_per in res:
        #         res_per=tokenizer.decode(res_per)
        #         if "Assistant: " in res_per:
        #             res_per = res_per.split("Assistant: ")[1]
        #         elif "Assistant:" in res_per:
        #             res_per = res_per.split("Assistant:")[1]
        #         else:
        #             res_per = res_per

        #         if "<|eot_id|>" in res_per:
        #             res_per = res_per.split("<|eot_id|>")[0]
        #         elif "<|end_of_text|>" in res_per:
        #             res_per = res_per.split("<|end_of_text|>")[0]
        #         print(f"Text Generated:>>> {res_per}")
        #     res_ls.append((res_per, label))
        #     # break
        
    model = None
    gen_pipeline = None
    tokenizer = None

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


def eval_wmt(res_ls):
    """
    1. BERTscore
    3. BLEU-4
    4. ROUGE
    """
    from nlg_metric import overall_metrics
    hyps, refs = zip(*res_ls)
    return overall_metrics(hyps, refs)


def evaluation_datas():
    ckpt_ls = [
        # ["cs-en", "google/gemma-2b",],
        # ["cs-en", "./wmt_ckpt/vanilla256cs-en100___finally/",],
        # ["cs-en", "./wmt_ckpt/kd256cs-en100___finally/",],
        # ["cs-en", "./wmt_ckpt/Complex-lord256cs-en100___finally/",],
        # ["cs-en", "./wmt_ckpt/Complex-lord256cs-en100___period0/",],
        # ["cs-en", "./wmt_ckpt/Complex-lord256cs-en100___period1/",],
        # ["cs-en", "./POD_SAVE_CKPTs/vary_period0306cs-en/kd_256cs-en_newkd___finally/",],
        # ["cs-en", "./POD_SAVE_CKPTs/vary_period0306cs-en/kd_256cs-en_30epochs___finally/",],

        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1615256cs-en64__long_stage_style_ckpt___period2/",],
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1615256cs-en64__long_stage_style_ckpt___period5/",],
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1615256cs-en64__long_stage_style_ckpt___period8/",],
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1615256cs-en64__long_stage_style_ckpt___period11/",],
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1615256cs-en64__long_stage_style_ckpt___period14/",],
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1615256cs-en64__long_stage_style_ckpt___period14/"],

        # 0.85
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1003256cs-en64__long_stage_style_ckpt___period2/"],

        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II336256cs-en64__hyper-para-search_ckpt___period5"],
        # ["cs-en", "./lord-IV_ckpt/cs-en/LoRD-IV1003256cs-en64__long_stage_style_ckpt___period2"],

        # evaluate the experimental results of LoRD-II.
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II816256cs-en64__hyper-para-search_ckpt___period14/"],

        # ["cs-en",
        #  "./lord-lord-IV_ckpt/cs-en/"
        #  ]


        # ?
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II43256cs-en4__long_stage_style_ckpt___period2/"],
        # ["cs-en", "./lordii_ckpt/cs-en/LoRD-II1003256cs-en64__long_stage_style_ckpt___period2/"],

        # ["cs-en", "./lord-IV_ckpt/cs-en/LoRD-IV1003256cs-en64__long_stage_style_ckpt___period2/"],
        #################################
        # ["cs-en",
        #  "gpt-3.5-turbo-1106",
        #  ],

        # ["cs-en",
        #  "meta-llama/Meta-Llama-3-8B-Instruct",
        #  ],

        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period1/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period2/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period3/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period4/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period5/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period6/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period7/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period8/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period9/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period10/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period11/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period12/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___period13/",],
        # ["cs-en","./wmt16_ckpts/WMTTT------TEMP___finally/",],

        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period1/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period2/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period3/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period4/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period5/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period6/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period7/",],
        # ["cs-en","./wmt16_ckpts/WMTTToldcs-en------TEMP___period8/",],

        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en------TEMP___period512/",],

        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en5121LoRD-VI___period512/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en5121LoRD-VI___period512/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en5121LoRD-VI___period512/",],

        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en2561LoRD-VI___period256/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en2561LoRD-VI___period256/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en2561LoRD-VI___period256/",],
        
        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en1281LoRD-VI___period128/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en1281LoRD-VI___period128/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en1281LoRD-VI___period128/",],

        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en641LoRD-VI___period64/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en641LoRD-VI___period64/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en641LoRD-VI___period64/",],


        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en5121vanilla___finally/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en5121vanilla___finally/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en5121vanilla___finally/",],

        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en2561vanilla___finally/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en2561vanilla___finally/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en2561vanilla___finally/",],
        
        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en1281vanilla___finally/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en1281vanilla___finally/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en1281vanilla___finally/",],

        # ["cs-en","./wmt16_ckpts/WMTTTnewcs-en641vanilla___finally/",],
        # ["de-en","./wmt16_ckpts/WMTTTnewde-en641vanilla___finally/",],
        # ["fi-en","./wmt16_ckpts/WMTTTnewfi-en641vanilla___finally/",],

        ["cs-en","./wmt16_ckpts/WMTTT0519cs-en161LoRD-VI___period256/"],
        # ["de-en","./wmt16_ckpts/WMTTT0519cs-en321LoRD-VI___period512/"],
        # ["fi-en","./wmt16_ckpts/WMTTT0519cs-en321LoRD-VI___period512/"],
        ["cs-en","./wmt16_ckpts/WMTTT0519cs-en161vanilla___finally/"],
        # ["de-en","./wmt16_ckpts/WMTTT0519cs-en321vanilla___finally/"],
        # ["fi-en","./wmt16_ckpts/WMTTT0519cs-en321vanilla___finally/"],

    ]
    base_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
    res_dict = {}
    dir_p = "./wmt16_res/"
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    for task_ckpt in ckpt_ls:
        task, ckpt = task_ckpt
        res_pth = ckpt+f"___{task}_glue_infer_res"
        res_pth = res_pth.replace("/", "__").replace(".", "")
        res_pth += ".json"
        if not os.path.exists(dir_p+res_pth):
            res_ls = infer_wmt(ckpt, task, dir_p+res_pth,
                               # test_set_take_num=100,
                               test_set_take_num=500,
                               mnt=64,
                               base_model_name=base_model_name)
        else:
            # from collections import OrderedDict
            with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                res_ls = json.load(f, object_pairs_hook=OrderedDict)

        scores = eval_wmt(res_ls)
        print(task, ckpt)
        print(scores)
        res_dict[task+"-----"+ckpt] = scores
    with open(dir_p+"wmt_inference_scores_overall.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


def eval_all():
    # methodls = ["Complex-lord", "vanilla", "kd", "black--Complex-lord"]
    methodls = ["vanilla", "kd",]
    train_times = [str(x+1) for x in range(5)]
    taskls = ["cs-en", "de-en", "fi-en", "ro-en", "ru-en", "tr-en"]
    dir_p = "./WMT16_infers/"
    res_dict = {}

    for task in taskls:
        res_dict[task] = {}
        for m in methodls:
            for itime in train_times:
                if not os.path.exists(dir_p):
                    os.makedirs(dir_p)
                prefix = "./wmt2b_ckpts/"
                if m == "Complex-lord" or m == "black--Complex-lord":
                    ckpt = prefix+f"{task}{m}256100__{itime}___period2"
                else:
                    ckpt = prefix+f"{task}{m}256100__{itime}___finally"
                res_pth = ckpt+f"___{task}_wmt_infer_res.json"
                res_pth = res_pth.replace("/", "__").replace(".", "")
                if not os.path.exists(dir_p+res_pth):
                    res_ls = infer_wmt(ckpt, task, dir_p+res_pth,
                                       test_set_take_num=100,
                                       mnt=64)
                else:
                    # from collections import OrderedDict
                    with open(dir_p+res_pth, 'r', encoding='utf8') as f:
                        res_ls = json.load(
                            f, object_pairs_hook=OrderedDict)

                scores = eval_wmt(res_ls)
                # print(task, ckpt)
                # print(scores)
                res_dict[task][task+"-----"+ckpt] = scores
    with open(dir_p+"glue_inference_scores.json",
              'w', encoding='utf8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


def eval_varying_train_num():
    taskls = [
        "cs-en",
        "de-en",
        "fi-en",
        ]
    mls = [
        "vanilla",
        "LoRD-VI",
        # "kd",
        ]
    # mls = ["vanilla", "kd", "google/gemma-2b", "Complex-lord",]
    train_times = [
        "1",
        "2",
        "3",
        "4",
        "5",
        ]
    train_nums = [
        "16",
        # "64",
        # "128",
        # "256",
        # "512",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./wmt_0519_dataset_res/"
    res_dict = {}
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)

    res_dict_averaged={}

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                temp_scorels=[]
                for itime in train_times:
                    prefix = "./wmt16_ckpts/WMTTT0519"
                    if m=="vanilla":
                        ckpt = (
                            prefix
                            + f"{task}{train_num}{itime}{m}___finally/"
                        )
                    else:
                        ckpt = prefix + \
                            f"{task}{train_num}{itime}{m}___period256/"
                    res_pth = ckpt+f"___{task}_wmt_infer_res.json"
                    res_pth = res_pth.replace("/", "__").replace(".", "")

                    if not os.path.exists(dir_p+res_pth):
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

    with open(dir_p+"Overall__wmt_varytrain_num_inference_scores.json",
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

def eval_tau1_res():
    taskls = [
        "cs-en",
        # "de-en",
        # "fi-en",
    ]
    mls = [
        "LoRD-VI",
        # "vanilla",
        ]
    train_times = [
        "1",
        # "2",
        # "3",
        # "4",
        # "5",
    ]
    train_nums = [
        "64",
        # "128",
        # "256",
        # "512",
        ]
    tau1ls= [
        "0.4",
        "0.5",
        "0.6",
        # "0.70",
        # "0.75",
        # "0.80",
        # "0.85",
        # "0.90",
        # "0.95",
        # "1.0",
        ]
    # tau2="1.0"
    tau2ls=[
        # "0.80",
        # "0.85",
        # "0.90",
        # "0.95",
        "1.0",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./wmt_0516_tau1_res/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    res_dict_averaged={}

    for tau1 in tau1ls:
        for tau2 in tau2ls:
            for task in taskls:
                for train_num in train_nums:
                    for m in mls:
                        temp_scorels=[]
                        for itime in train_times:
                            prefix = f"./qa_ckpts/WMTTT-TAU1{tau1}TAU2{tau2}"
                            ckpt = (
                                prefix
                                + f"{task}{train_num}{itime}{m}___period512/"
                            )
                            res_pth = ckpt + f"___{task}_qa_infer_res.json"
                            res_pth = res_pth.replace("/", "__").replace(".", "")

                            if not os.path.exists(dir_p+res_pth):
                                res_ls = infer_wmt(ckpt, task, dir_p+res_pth,
                                                test_set_take_num=500,
                                                mnt=64,
                                                base_model_name=base_model_name1,
                                                )
                            else:
                                print(
                                    f"{dir_p+res_pth} file already exists. directly loading...")
                                # from collections import OrderedDict
                                with open(dir_p + res_pth, "r", encoding="utf8") as f:
                                    res_ls = json.load(
                                        f, object_pairs_hook=OrderedDict)

                            scores = eval_wmt(res_ls)
                            res_dict[task + "-----" + res_pth] = scores

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

    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    with open(
        dir_p + "OverallScoresAveraged.json",
            "w", encoding="utf8"
    ) as f:
        json.dump(res_dict_averaged, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)
    pprint(res_dict_averaged)


# running entry
if __name__ == "__main__":
    # main()
    # evaluation_datas()
    # eval_all()
    eval_varying_train_num()
    # eval_tau1_res()
    print("EVERYTHING DONE.")
