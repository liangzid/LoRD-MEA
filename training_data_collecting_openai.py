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
import os
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
    trainset_text =trainset_text["chosen"][:10]

    data=lm_tokenizer(trainset_text,
                      padding="longest",
                      truncation=True,
                      max_length=max_length,
                      return_tensors="pt"
                      ).input_ids

    return data

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
