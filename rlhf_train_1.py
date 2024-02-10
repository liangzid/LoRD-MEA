"""
======================================================================
RLHF_TRAIN_1 ---

Initial test of the RLHF model training.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 10 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
import torch

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import SFTTrainer, RewardTrainer, RewardConfig
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch


def sft_train():
    # get dataset
    dataset = load_dataset("imdb", split="train")

    # get trainer
    trainer = SFTTrainer(
        "facebook/opt-350m",
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
    )

    # train
    trainer.train()

# Reward trainer


def rewardModel_train(
    dataset,
        model_name="microsoft/phi-1_5",):

    rwd_conf=RewardConfig(output_dir="temp_output",
                          num_train_epochs=1,
                          learning_rate=1e-5,
                          max_steps=1000)
    rwd_conf.max_length=512

    # load model and dataset - dataset needs to be in a specific format
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1,
        )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # load trainer
    trainer = RewardTrainer(
        args=rwd_conf,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()


def rlhf_train(rw_pth="gpt2"):
    # get models
    model = AutoModelForCausalLMWithValueHead\
        .from_pretrained(rw_pth)
    model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(rw_pth)
    tokenizer.pad_token = tokenizer.eos_token

    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=1,
    )

    # encode a query
    query_txt = "This morning I went to the "
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

    # get model response
    response_tensor = respond_to_batch(model, query_tensor)

    # create a ppo trainer
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

    # define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    reward = [torch.tensor(1.0)]

    for i in range(10):
        # train model for one step with ppo
        train_stats = ppo_trainer.step(
            [query_tensor[0]],
            [response_tensor[0]],
            reward)  # RLHF Trainer


def main():
    # print("test rlhf train.")
    # rlhf_train()

    print("test of reward model training.")
    rw_dataset = load_dataset("Anthropic/hh-rlhf",
                              split="train").to_iterable_dataset()\
                              .take(100)
    rewardModel_train(rw_dataset, model_name="microsoft/phi-1_5")


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
