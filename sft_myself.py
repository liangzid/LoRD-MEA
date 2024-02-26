"""
======================================================================
SFT_MYSELF ---

self-implementated SFT.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 26 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

import torch
import json
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel

from training_data_collecting_openai import load_raw_train_datals

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:3", type=str,
                        required=False)
    parser.add_argument('--epoch', default=2, type=int,
                        required=False)
    parser.add_argument('--acc_step', default=4, type=int,
                        required=False)
    parser.add_argument('--log_step', default=100, type=int,
                        required=False)
    parser.add_argument('--save_step', default=10000, type=int,
                        required=False)


    parser.add_argument('--LR', default=3e-4, type=float,
                        required=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        required=False)
    parser.add_argument('--task', default="vanilla", type=str,
                        required=False,)
    parser.add_argument("--max_length", default=1024,
                        type=int, required=False)

    parser.add_argument('--from_path', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--save_path',
                        default='model_training_results',
                        type=str, required=True,)

    return parser.parse_args()


def main():

    args=setup_train_args()
    
    lm=AutoModelForCausalLM.from_pretrained(
        args.from_path,
        device_map="auto",
        )

    lm_tokenizer=AutoTokenizer.from_pretrained(args.from_path)
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token=lm_tokenizer.eos_token

    raw_train_datals=load_raw_train_datals(lm_tokenizer, args.max_length)
    print("Data LOADING done.")
    trainset=TensorDataset(raw_train_datals)
    loader=DataLoader(trainset,
                      batch_size=args.batch_size,
                      shuffle=True)
    
    tensorboard_name=f"BLABLABLA"
    tb_writer=SummaryWriter(log_dir=args.save_path+"___log_writer")

    train(
        lm,
        lm_tokenizer,
        loader,
        args.epoch,
        args.device,
        tb_writer,
        tensorboard_name,
        args.save_path,
        args.LR,
        args.acc_step,
        args.log_step,
        args.save_step,
        )

    print("EVERYTHING in the TRAINING now DONE.")

def train(lm,
                     lm_tokenizer,
                     loader, epoch, device,
                     tb_writer,
                     tensorboard_name,
                     save_path,
                     v_save_path,
                     LR=3e-5,
                     acc_step=1,
                     log_step=100,
                     save_step=1000,
                     ):
    overall_loss=0.
    overall_step=0

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="loader"):
            overall_step+=1

            # print(item)
            inps_idxs=item[0]
            bs, sqlen=inps_idxs.shape

            inps_idxs=inps_idxs.to(device) # bs, sql
            loss=lm(inps_idxs,labels=inps_idxs).loss

            overall_loss += loss 

            if overall_step % log_step ==0:
                print(" LOSS: {}".format(
                    overall_loss,
                    ))
                tb_writer.add_scalar("train-loss", overall_loss.item(),
                                     overall_step)
                
            if overall_loss % save_step==0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___"+overall_step)
                lm.save_pretrained(save_path+"___"+overall_step)

            if overall_step % acc_step == 0:
                opt1.zero_grad()
                
                overall_loss.backward()
                opt1.step()
                overall_loss=0.

    print(" -->Finally Saving.")
    lm_tokenizer.save_pretrained(save_path+"___STEPfinally")
    lm.save_pretrained(save_path+"___STEPfinally")

    print("ONE PERIOD TRAINING DONE!")

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


