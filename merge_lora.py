"""
======================================================================
MERGE_LORA ---

This file is to merge LoRA with pretrained models.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created:  6 June 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

import torch
from datasets import load_dataset
import json
from collections import OrderedDict
from math import exp
import random

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from tqdm import tqdm
import pickle
from pprint import pprint
from collections import OrderedDict
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import numpy as np


def main():
    pretrained_p="meta-llama/Meta-Llama-3-8B-Instruct"
    # lora_p="./general_train/ckpts/boring_test/NewTemperatureLoRD-VIINewLoss___period1000/"
    lora_p="./general_train/ckpts/boring_test/NewTemperatureNewTau8BvanillaNewLoss___finally/"
    save_p="./general_train/ckpts/MERGED/llama38b-vanilla-Claude3short256/"
    upload_p="liangzid/llama38b-LoRD-Claude3short256"

    mergelora(pretrained_p,lora_p,save_p,)
    # uploadmodel(save_p, upload_p,)

def mergelora(pretrained_p,
              lora_p,
              save_p):
    pretrained_model=AutoModelForCausalLM.from_pretrained(
        pretrained_p,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        )
    model=PeftModel.from_pretrained(
        pretrained_model,
        lora_p,
        )
    tokenizer=AutoTokenizer.from_pretrained(pretrained_p)

    model = model.merge_and_unload()
    model.save_pretrained(save_p)
    tokenizer.save_pretrained(save_p)
    print("TOKENIZER AND THE MODEL SAVED DONE.")

# def uploadmodel(model_p,upload_p):

#     print("First valid the model loading...")

#     from transformers import AutoConfig, AutoModel, AutoTokenizer
#     config = AutoConfig.from_pretrained(
#         model_p, revision=revision)
#     model = AutoModel.from_pretrained(
#         model_p, revision=revision)
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_p, revision=revision)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


