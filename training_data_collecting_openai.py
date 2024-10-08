"""
======================================================================
TRAINING_DATA_COLLECTING_OPENAI ---

Collecting Training Data from OpenAI's APIs

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 23 February 2024
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

client = oa()


def chatWithOpenAI_APIs(modelname="gpt-3.5-turbo-1106",
                        prompt="",
                        utter="",
                        ):

    res = client.chat.completions.create(
        model=modelname,
        # prompt=f"Instruction: {prompt}. User: {utter}. System: ",
        messages=[
            {"role": "system", "content": "Instruction: " + prompt},
            {"role": "user", "content": utter}
        ]
    )
    # time.sleep(1)
    return res.choices[0].message.content


def chatWithOpenAI__LogLogits(modelname="gpt-3.5-turbo-1106",
                              messages=[],
                              num_top_logprobs=5):
    # empty_prefix=[{"role":"system","content":""},]
    # empty_prefix.extend(messages)

    # print("messages: ",messages)
    # print(type(messages[0]))

    res = client.chat.completions.create(
        model=modelname,
        messages=messages,
        logprobs=True,
        top_logprobs=num_top_logprobs,
    )
    # print("Inference Results: ",res)
    generated_text = res.choices[0].message.content
    logprobs = res.choices[0].logprobs.content

    # print("-----------------------")

    return generated_text, logprobs

def obtain_beginning_sents():
    dataset_name = "Anthropic/hh-rlhf"
    dataset = load_dataset(dataset_name, split="train")["chosen"]

    split_word = " Assistant: "
    extracted_beggin_sents = []
    for d in dataset:
        if split_word not in d:
            continue
        else:
            utterance = d.split(split_word)[0]
            extracted_beggin_sents.append(utterance)

    if not os.path.exists("./intermediate_data"):
        os.makedirs("./intermediate_data")
    with open("./intermediate_data/extracted_biggin_sentence.json",
              'w', encoding='utf8') as f:
        json.dump(extracted_beggin_sents,
                  f, ensure_ascii=False, indent=4)

    return extracted_beggin_sents


def free_sampled_utterance_by_Prompts(
        model="gpt-3.5-turbo-1106",
        num=10,
):
    prompt = "Suppose you are a user. Please generate one utterance (a question) to begin the dialogue."

    utter_ls = []
    for _ in range(num):
        res = chatWithOpenAI_APIs(model, prompt, prompt)
        utter_ls.append(res)

    if not os.path.exists("./intermediate_data"):
        os.makedirs("./intermediate_data")
    with open(f"./intermediate_data/{model}_{num}_free_gen.json",
              'w', encoding='utf8') as f:
        json.dump(utter_ls,
                  f, ensure_ascii=False, indent=4)
    return utter_ls


def chatting_to_generate_positive_dialogues(
    utter,
    modelname="gpt-3.5-turbo-1106",
        L=10):
    dialogue = [
        {"role": "user", "content": utter}
    ]
    for _ in range(L):
        res = client.chat.completions.create(
            model=modelname,
            # prompt=f"Instruction: {prompt}. User: {utter}. System: ",
            messages=dialogue)
        resp = res.choices[0].message.content
        role = "assistant"
        if dialogue[-1]["role"] == "user":
            dialogue.append({
                "role": role,
                "content": resp,
            })
    return dialogue


def generate_training_data(
        chatting_method="anthropic",
):
    if chatting_method == "anthropic":
        utterls = obtain_beginning_sents()
    elif chatting_method == "free":
        utterls = free_sampled_utterance_by_Prompts(num=15)

    # then generate the training dataset.
    dialogue_ls = []
    for utter in utterls:
        dialogue = chatting_to_generate_positive_dialogues(utter)
        dialogue_ls.append(dialogue)

    if not os.path.exists("./intermediate_data"):
        os.makedirs("./intermediate_data")
    with open(f"./intermediate_data/{chatting_method}Positive_set.json",
              'w', encoding='utf8') as f:
        json.dump(dialogue_ls,
                  f, ensure_ascii=False, indent=4)
    return dialogue_ls


def load_raw_train_datals(lm_tokenizer, max_length=1024):
    dataset_name = "Anthropic/hh-rlhf"
    trainset_text = load_dataset(dataset_name, split="train")
    trainset_text = trainset_text["chosen"][:100]

    data = lm_tokenizer(trainset_text,
                        padding="longest",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                        ).input_ids

    return data


def load_steal_datals(lm_tokenizer,
                      model_name="gpt-3.5-turbo-1106",
                      topk=5,
                      max_length=1024,
                      openai_tmp_save_pth="./ultrachat2k_openai_probs_res1.pickle",
                      ):

    V = lm_tokenizer.vocab_size
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    trainset_text = load_dataset(dataset_name, split="train_sft[:1]")

    prompts = trainset_text["prompt"]
    prompts = [f"###User: {x} ###Assistant: " for x in prompts]
    p_idxls = lm_tokenizer(prompts,
                           padding="longest",
                           truncation=True,
                           max_length=max_length,
                           return_tensors="pt"
                           ).input_ids

    messages = trainset_text["messages"]
    # messages=[x[1] for x in messages]

    if not os.path.exists(openai_tmp_save_pth):
        print("RUNNING ChatGPT Stealing...")
        text2ls = []
        idx2_dist_ls=[]
        probsls = []
        iii_bgn=0
        for q in tqdm(messages, desc="ChatGPT Inference:"):
            qd=[{"role":"user",
                "content":q[0]["content"]}]
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
                # topk_logits = [exp(x) for x in topk_logits]

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


def most_vanilla_anthropicModel():
    dataset_name = "Anthropic/hh-rlhf"
    train_set = load_dataset(dataset_name, split="train")
    test_set = load_dataset(dataset_name, split="test")

    return train_set, test_set


def main():
    pass


# running entry
if __name__ == "__main__":
    # main()
    # res=chatWithOpenAI_APIs(prompt="1",utter="1")
    # ms=[{'content': "These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+). What theme version am I using?\nOn your Collections pages & Featured Collections sections, you can easily show the secondary image of a product on hover by enabling one of the theme's built-in settings!\nYour Collection pages & Featured Collections sections will now display the secondary product image just by hovering over that product image thumbnail.\nDoes this feature apply to all sections of the theme or just specific ones as listed in the text material?", 'role': 'user'}]
    # res=chatWithOpenAI__LogLogits(messages=ms)
    # print(res)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    raw_train_datals = load_steal_datals(tokenizer,
                                         max_length=1024)

    print("EVERYTHING DONE.")
