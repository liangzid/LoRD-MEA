"""
======================================================================
RLHF_TRAIN ---

Training Language Models with Self-designed Training Loss Function.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 11 February 2024
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


def clip(tnsr, epsilon=0.5):
    # print(f"TNSR: {tnsr}")
    one_tensor = torch.ones_like(tnsr)
    # print(f"ONE-TNSR: {one_tensor}")
    one_tensor = one_tensor.to(tnsr.device)
    tnsr = torch.min(tnsr, one_tensor*(1+epsilon)
                     )
    # print(f"ONE-TNSR-MIN: {tnsr}")
    # print(f"ONE-TNSR: {one_tensor}")
    tnsr = torch.max(tnsr, one_tensor*(1-epsilon)
                     )
    # print(f"ONE-TNSR-MAX: {tnsr}")
    # print(f"ONE-TNSR: {one_tensor}")
    return tnsr


def log_clip(tnsr, epsilon=0.2):
    one_tensor = torch.ones_like(tnsr)
    one_tensor = one_tensor.to(tnsr.device)
    tnsr = torch.min(tnsr, torch.log(one_tensor*(1+epsilon))
                     )
    tnsr = torch.max(tnsr, torch.log(one_tensor*(1-epsilon))
                     )
    return tnsr


def train_one_period(lm, vmodel,
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
                     lambda1=0.7,
                     lambda2=0.7,
                     epsilon=0.2
                     ):
    overall_loss = 0.
    overall_step = 0

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    opt2 = torch.optim.AdamW(vmodel.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="loader"):
            overall_step += 1
            loss_clip = 0.
            loss_vfunc = 0.
            loss_entropy = 0.

            # print(item)
            inps_idxs, reward, old_logits, A, V = item
            bs, sqlen = inps_idxs.shape

            inps_idxs = inps_idxs.to(device)  # bs, sql
            reward = reward.to(device)  # bs, sql
            old_logits = old_logits.to(device)  # bs, sql, Vocab
            old_logits = torch.softmax(old_logits, dim=-1)
            old_logits = old_logits[torch.arange(bs).unsqueeze(1),
                                    torch.arange(sqlen-1).unsqueeze(0),
                                    inps_idxs[:, 1:]]
            A = A.to(device)  # bs, sql
            V = V.to(device)  # bs, sql

            logits = lm(inps_idxs).logits[:, :-1, :]
            logits = torch.softmax(logits, dim=-1)
            logits = logits[torch.arange(bs).unsqueeze(1),
                            torch.arange(sqlen-1).unsqueeze(0),
                            inps_idxs[:, 1:]]

            convince_gen = logits/old_logits
            # print(convince_gen.shape)
            # print(A.shape)

            loss_clip = torch.sum(torch.min(
                convince_gen*A,
                clip(convince_gen, epsilon)*A))

            values = vmodel(inps_idxs).logits.expand(bs, sqlen-1)
            loss_vfunc = torch.sum((values-V)**2)

            entropy = torch.sum(Categorical(logits).entropy())
            loss_entropy = entropy

            overall_loss += -1*loss_clip + lambda1*loss_vfunc\
                + lambda2*loss_entropy

            if overall_step % log_step == 0:
                print(" LOSS: {}\tCLIP: {}\tV: {}\tEntropy: {}".format(
                    overall_loss, loss_clip, loss_vfunc, loss_entropy
                ))
                tb_writer.add_scalar("loss", overall_loss.item(),
                                     overall_step)
                tb_writer.add_scalar("cliploss", loss_clip.item(),
                                     overall_step)
                tb_writer.add_scalar("vfuncloss", loss_vfunc.item(),
                                     overall_step)
                tb_writer.add_scalar("entropyloss", loss_entropy.item(),
                                     overall_step)

            if overall_loss % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___"+overall_step)
                lm.save_pretrained(save_path+"___"+overall_step)
                lm_tokenizer.save_pretrained(v_save_path+"___"+overall_step)
                vmodel.save_pretrained(v_save_path+"___"+overall_step)

            if overall_step % acc_step == 0:
                opt1.zero_grad()
                opt2.zero_grad()

                overall_loss.backward()
                opt1.step()
                opt2.step()
                overall_loss = 0.

    print(" -->Finally Saving.")
    lm_tokenizer.save_pretrained(save_path+"___STEPfinally")
    lm.save_pretrained(save_path+"___STEPfinally")
    lm_tokenizer.save_pretrained(v_save_path+"___STEPfinally")
    vmodel.save_pretrained(v_save_path+"___STEPfinally")

    print("ONE PERIOD TRAINING DONE!")
    return lm, vmodel


def ___V_target_compute(reward, lambdaa=0.95):
    """
    shape of `reward`: bs, sql
    """
    bs, sql = reward.shape
    window_size = 150
    V = torch.zeros((bs, sql)).to(reward.device)

    for i in range(sql):
        for j in range(i, min(i+window_size, sql)):
            V[:, i] += reward[:, j]*(lambdaa**(j-1))

    return V


def train_pod(lm, vmodel, rewardmodel,
              lm_tokenizer,
              args, raw_train_datals):
    print(">>>> DATA PREPERATION")
    # STEP 1: DATA Preperation.
    rewardls = None
    ITER_num = args.period_num
    tb_writer = SummaryWriter(log_dir=args.save_path+"___log_writer")
    for iter_idx in range(ITER_num):
        tensorboard_name = f"Period {iter_idx}"
        old_logitsls = []
        Als = []
        Vls = []
        # collect data.
        with torch.no_grad():
            if rewardls is None:
                rewardls = []
                for inps_idxs in tqdm(raw_train_datals,
                                      desc="Data Collecting..."):
                    inps_idxs = inps_idxs.to(args.device).unsqueeze(0)
                    bs, sql = inps_idxs.shape
                    reward = rewardmodel(inps_idxs[:, :-1])\
                        .logits.expand(bs, sql-1)
                    rewardls.append(reward.squeeze(0).to("cpu"))

                    old_logits = lm(inps_idxs[:, :-1])\
                        .logits
                    V = ___V_target_compute(reward, lambdaa=args.lambdaa)
                    A = V-vmodel(inps_idxs[:, :-1]).logits.expand(bs, sql-1)
                    old_logitsls.append(old_logits.squeeze(0).to("cpu"))
                    Als.append(A.squeeze(0).to("cpu"))
                    Vls.append(V.squeeze(0).to("cpu"))
                rewardls = torch.stack(rewardls)
            else:
                rewardmodel = None
                for i, inps_idxs in tqdm(
                        enumerate(raw_train_datals),
                        desc="Data Collecting..."):
                    inps_idxs = inps_idxs.to(args.device).unsqueeze(0)
                    bs, sql = inps_idxs.shape
                    reward = rewardls[i].to(args.device).unsqueeze(0)

                    old_logits = lm(inps_idxs[:, :-1])\
                        .logits
                    V = ___V_target_compute(reward, lambdaa=args.lambdaa)
                    A = V-vmodel(inps_idxs[:, :-1]).logits.expand(bs, sql-1)
                    old_logitsls.append(old_logits.squeeze(0).to("cpu"))
                    Als.append(A.squeeze(0).to("cpu"))
                    Vls.append(V.squeeze(0).to("cpu"))
        overall_ls = zip(raw_train_datals, rewardls, old_logitsls,
                         Als, Vls)

        old_logitsls = torch.stack(old_logitsls)
        Als = torch.stack(Als)
        Vls = torch.stack(Vls)
        trainset = TensorDataset(
            raw_train_datals,
            rewardls,
            old_logitsls,
            Als, Vls)

        loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            )
        print(">>>> Period {}".format(iter_idx))
        # STEP 2: Train the Model in a period
        lm, vmodel = train_one_period(lm, vmodel,
                                      lm_tokenizer,
                                      loader,
                                      args.epoch, args.device,
                                      tb_writer,
                                      tensorboard_name,
                                      args.save_path,
                                      args.v_save_path,
                                      args.LR,
                                      args.acc_step, args.log_step,
                                      args.save_step,
                                      args.epsilon,
                                      )

        print(" -->NOW save the ckpt in each period.")
        print(f"in period {iter_idx}.")
        lm_tokenizer.save_pretrained(args.save_path+"___period"+str(iter_idx))
        lm.save_pretrained(args.save_path+"___period"+str(iter_idx))
        lm_tokenizer.save_pretrained(
            args.v_save_path+"___period"+str(iter_idx))
        vmodel.save_pretrained(args.v_save_path+"___period"+str(iter_idx))

    print(" -->ALL TRAINING DONE.")
    lm_tokenizer.save_pretrained(args.save_path+"___finally")
    lm.save_pretrained(args.save_path+"___finally")
    lm_tokenizer.save_pretrained(args.v_save_path+"___finally")
    vmodel.save_pretrained(args.v_save_path+"___finally")
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
    parser.add_argument('--lambdaa', default=0.95, type=float,
                        required=False)
    parser.add_argument('--lambda1', default=0.95, type=float,
                        required=False)
    parser.add_argument('--lambda2', default=0.95, type=float,
                        required=False)
    parser.add_argument('--epsilon', default=0.2, type=float,
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
    parser.add_argument('--v_save_path',
                        default='value_model_training_results',
                        type=str, required=True,)
    parser.add_argument('--v_from_path',
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
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    rewardmodel = AutoModelForSequenceClassification.from_pretrained(
        args.v_from_path,
        device_map="auto",
        num_labels=1,
    )

    vmodel = AutoModelForSequenceClassification.from_pretrained(
        args.v_from_path,
        device_map="auto",
        num_labels=1,
    )

    raw_train_datals = load_raw_train_datals(lm_tokenizer, args.max_length)
    print("Data LOADING done.")

    train_pod(
        lm, vmodel, rewardmodel,
        lm_tokenizer,
        args, raw_train_datals)

    print("EVERYTHING in the TRAINING now DONE.")


if __name__ == "__main__":
    main()
