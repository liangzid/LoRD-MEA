"""
======================================================================
POD_TRAIN ---

Supervised POD Training Loss

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 27 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
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
from training_data_collecting_openai import load_steal_datals


def train_one_period(lm,
                     lm_tokenizer,
                     loader, epoch, device,
                     tb_writer,
                     tensorboard_name,
                     save_path,
                     LR=3e-5,
                     acc_step=1,
                     log_step=100,
                     save_step=1000,
                     beta=0.7,
                     ):
    overall_loss = 0.
    overall_step = 0

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="loader"):
            overall_step += 1
            loss_constractive = 0.
            loss_logits = 0.

            # print(item)
            idxs1, idxs2, old_logits1, vic_logits2 = item
            bs, sqlen = idxs1.shape

            idxs1 = idxs1.to(device)  # bs, sql
            idxs2 = idxs2.to(device)  # bs, sql
            old_logits1 = old_logits1.to(device)  # bs, sql, Vocab
            old_logits1 = torch.softmax(old_logits1, dim=-1)
            old_logits1 = old_logits1[torch.arange(bs).unsqueeze(1),
                                      torch.arange(sqlen-1).unsqueeze(0),
                                      idxs1[:, 1:]]
            vic_logits2 = vic_logits2.to(device)
            vic_logits2 = torch.softmax(vic_logits2, dim=-1)
            vic_logits2_cons = vic_logits2[torch.arange(bs).unsqueeze(1),
                                           torch.arange(sqlen-1).unsqueeze(0),
                                           idxs2[:, 1:]]

            logits1 = lm(idxs1).logits[:, :-1, :]
            logits1 = torch.softmax(logits1, dim=-1)
            logits1 = logits1[torch.arange(bs).unsqueeze(1),
                              torch.arange(sqlen-1).unsqueeze(0),
                              idxs1[:, 1:]]

            logits2 = lm(idxs2).logits[:, :-1, :]
            logits2 = torch.softmax(logits2, dim=-1)
            logits2_cons = logits2[torch.arange(bs).unsqueeze(1),
                                   torch.arange(sqlen-1).unsqueeze(0),
                                   idxs2[:, 1:]]

            loss_constractive = beta * (torch.log(logits2_cons)
                                        - torch.log(vic_logits2_cons)
                                        )\
                - torch.log(logits1)\
                + torch.log(old_logits1)
            loss_constractive = - torch.sum(loss_constractive)

            loss_logits = beta *\
                torch.sum(logits2*torch.log(logits2/vic_logits2))

            overall_loss += loss_constractive + loss_logits

            if overall_step % log_step == 0:
                print(" LOSS: {}\tReward: {}\tKL-D: {}".format(
                    overall_loss,
                    loss_constractive,
                    loss_logits
                ))
                tb_writer.add_scalar("loss", overall_loss.item(),
                                     overall_step)
                tb_writer.add_scalar("rewardloss",
                                     loss_constractive.item(),
                                     overall_step)
                tb_writer.add_scalar("KLloss", loss_logits.item(),
                                     overall_step)

            if overall_loss % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___"+overall_step)
                lm.save_pretrained(save_path+"___"+overall_step)

            if overall_step % acc_step == 0:
                opt1.zero_grad()

                overall_loss.backward()
                opt1.step()
                overall_loss = 0.

    print(" -->Finally Saving.")
    lm_tokenizer.save_pretrained(save_path+"___STEPfinally")
    lm.save_pretrained(save_path+"___STEPfinally")

    print("ONE PERIOD TRAINING DONE!")
    return lm


def train_pod(lm,
              lm_tokenizer,
              args, raw_train_datals):
    print(">>>> DATA PREPERATION")
    # STEP 1: DATA Preperation.
    ITER_num = args.period_num
    tb_writer = SummaryWriter(log_dir=args.save_path+"___log_writer")
    for iter_idx in range(ITER_num):
        tensorboard_name = f"Period {iter_idx}"
        idxs1_ls = []
        old_logits1_ls = []
        # collect data.
        with torch.no_grad():
            for prompt, idxs2, logits2 in tqdm(raw_train_datals,
                                               desc="Data Collecting..."):
                idxs2 = idxs2.to(args.device).unsqueeze(0)
                bs, sql = idxs2.shape
                prompt = prompt.to(args.device).unsqueeze(0)
                # Generate New Tokens
                idxs1 = lm.generate(prompt,
                                    do_sample=True,
                                    max_length=args.max_length,
                                    temperature=args.temperature)
                old_logits = lm(idxs1[:, :-1]).logits

                idxs1_ls.append(idxs1.squeeze(0).to("cpu"))
                old_logits1_ls.append(old_logits.squeeze(0).to("cpu"))

        pls, idx2ls, logits2ls = zip(*raw_train_datals)
        idx2ls = torch.stack(idx2ls)
        logits2ls = torch.stack(logits2ls)
        idxs1_ls = torch.stack(idxs1_ls)
        old_logits1_ls = torch.stack(old_logits1_ls)

        trainset = TensorDataset(
            idxs1_ls,
            idx2ls,
            old_logits1_ls,
            logits2ls,
        )

        loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            )
        print(">>>> Period {}".format(iter_idx))
        # STEP 2: Train the Model in a period
        lm = train_one_period(lm,
                              lm_tokenizer,
                              loader,
                              args.epoch, args.device,
                              tb_writer,
                              tensorboard_name,
                              args.save_path,
                              args.LR,
                              args.acc_step, args.log_step,
                              args.save_step,
                              args.epsilon,
                              args.beta,
                              )

        print(" -->NOW save the ckpt in each period.")
        print(f"in period {iter_idx}.")
        lm_tokenizer.save_pretrained(args.save_path+"___period"+str(iter_idx))
        lm.save_pretrained(args.save_path+"___period"+str(iter_idx))

    print(" -->ALL TRAINING DONE.")
    lm_tokenizer.save_pretrained(args.save_path+"___finally")
    lm.save_pretrained(args.save_path+"___finally")
    print(" -->Save DONE.")


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:3", type=str,
                        required=False)
    parser.add_argument('--epoch', default=2, type=int,
                        required=False)
    parser.add_argument('--period_num', default=3, type=int,
                        required=False)
    parser.add_argument('--acc_step', default=4, type=int,
                        required=False)
    parser.add_argument('--log_step', default=100, type=int,
                        required=False)
    parser.add_argument('--save_step', default=10000, type=int,
                        required=False)

    parser.add_argument('--LR', default=3e-4, type=float,
                        required=False)
    parser.add_argument('--beta', default=0.7, type=float,
                        required=False)
    parser.add_argument('--temperature', default=0.8, type=float,
                        required=False)

    parser.add_argument('--batch_size', default=1, type=int,
                        required=False)
    parser.add_argument('--task', default="pod", type=str,
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

    args = setup_train_args()

    lm = AutoModelForCausalLM.from_pretrained(
        args.from_path,
        device_map="auto",
    )

    lm_tokenizer = AutoTokenizer.from_pretrained(args.from_path)
    tokenizer = AutoTokenizer.from_pretrained(args.from_path)

    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    raw_train_datals = load_steal_datals(tokenizer,
                                         max_length=args.max_length)
    print("Data LOADING done.")

    train_pod(
        lm,
        lm_tokenizer,
        args,
        raw_train_datals)

    print("EVERYTHING in the TRAINING now DONE.")


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
