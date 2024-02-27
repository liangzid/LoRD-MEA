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

def chatWithOpenAI__LogLogits(modelname,
                              messages,
                              num_top_logprobs=5):
    completions=client.chat.completions.create(
        model=modelname,
        messages=messages,
        logprobs=True,
        top_logprobs=num_top_logprobs,
        )
    generated_text=completions.choices[0]["message"]["content"]
    logprobs=completions.choices[0].logprobs
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
    trainset_text= load_dataset(dataset_name, split="train")
    trainset_text =trainset_text["chosen"][:100]

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
                      openai_tmp_save_pth="./ultrachat2k_openai_probs_res100.json",
                      ):

    V=lm_tokenizer.vocab_size
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    trainset_text= load_dataset(dataset_name, split="train_gen[:10]")

    prompts =trainset_text["prompt"]
    prompts=[f"###User: {x} ###Assistant: " for x in prompts]
    p_idxls = lm_tokenizer(prompts,
                        padding="longest",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                        ).input_ids

    messages=trainset_text["messages"]
    # messages=[x[1] for x in messages]

    if not os.path.exists(openai_tmp_save_pth):
        query_mess=[x[0] for x in messages]
        text2ls=[]
        probsls=[]
        for q in query_mess:
            resp,logprb=chatWithOpenAI__LogLogits(
                model_name,
                messages=[q],
                num_top_logprobs=topk,
                )
            idx2=lm_tokenizer([resp],
                            padding="longest",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
            ).input_ids
            text2ls.append(idx2[0])
            tokenls=lm_tokenizer.tokenize(resp)
            logits_distr=[]
            subtoken_idx=0
            for i,topkdict in logprb:
                selected_token=topkdict["token"]
                subtokens=lm_tokenizer.tokenize(selected_token)
                topk_tokens=[x["token"] for x in topkdict["top_logprobs"]]
                topk_subtokenss=[lm_tokenizer.tokenize(x)\
                                for x in topk_tokens]
                topk_subidxes=[lm_tokenizer.convert_tokens_to_ids(x)\
                            for x in topk_subtokenss]
                topk_logits=[x["logprob"]\
                             for x in topkdict["top_logprobs"]]
                topk_logits=[exp(x) for x in topk_logits]
                res=1-sum(topk_logits)
                banality_value=res/(V-len(subtokens))
                for i in range(len(subtokens)):
                    dist=torch.ones(V)*banality_value
                    for j,subidx in enumerate(topk_subidxes):
                        dist[subidx[i]]=topk_logits[j]/len(subtokens)
                    logits_distr.append(dist)
            assert len(idx2[0]) == len(logits_distr)
            probsls.append(logits_distr)

        with open(openai_tmp_save_pth,
                'w',encoding='utf8') as f:
            json.dump([text2ls,probsls],
                    f,ensure_ascii=False,indent=4)
    else:
        # from collections import OrderedDict
        with open(openai_tmp_save_pth, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)
        text2ls=data[0]
        probsls=data[1]

    return p_idxls, text2ls, probsls


def most_vanilla_anthropicModel():
    dataset_name = "Anthropic/hh-rlhf"
    train_set = load_dataset(dataset_name, split="train")
    test_set = load_dataset(dataset_name, split="test")

    return train_set, test_set


def main():
    pass


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
