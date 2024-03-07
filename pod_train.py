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
from glue_process import load_glue_datals

from sequence_utils import my_padding, my_padding_logits
from sequence_utils import my_padding_token_dist
from sequence_utils import my_padding_logit

import torch.nn.functional as F

from rlhf_train import clip, log_clip

def train_one_period(args, lm,
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
                     epsln=1e-6,
                     method_type="1",
                     ):
    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id
    sigmoid=torch.nn.Sigmoid()
    kl_loss = torch.nn.KLDivLoss(reduction="none")

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="loader"):
            overall_step += 1
            loss_constractive = 0.
            loss_logits = 0.

            # print(item)
            idxs1, idxs2, mask1, mask2, old_logits1, old_logits2, vic_logits2, idxs2_dist = item
            bs, sqlen1 = idxs1.shape
            bs, sqlen2 = idxs2.shape
            # print(sqlen1, sqlen2)
            sqlen = min(sqlen1, sqlen2)

            idxs1 = idxs1.to(device)  # bs, sql
            idxs2 = idxs2.to(device)  # bs, sql
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            # already normalized by log_softmax
            old_logits1 = old_logits1.to(device)  # bs, sql,
            old_logits2 = old_logits2.to(device)  # bs, sql,
            vic_logits2 = vic_logits2.to(device)  # bs, sql, 5
            
            idxs2_dist = idxs2_dist.to(device)

            print("idx1text: ", lm_tokenizer.decode(idxs1[0]))
            print("idx2text: ", lm_tokenizer.decode(idxs2[0]))
            # print("old_logits1: ",old_logits1)
            # print("vic_logits2: ",vic_logits2)
            # print("idxs2_dist: ",idxs2_dist)

            logits1 = lm(idxs1).logits[:, :-1, :]
            logits1 = F.log_softmax(logits1, dim=-1)
            logits1 = logits1[torch.arange(bs).unsqueeze(1),
                              torch.arange(sqlen-1).unsqueeze(0),
                              idxs1[:, 1:sqlen]]

            logits2 = lm(idxs2).logits[:, :-1, :]
            logits2 = F.log_softmax(logits2, dim=-1)
            logits2_cons = logits2[torch.arange(bs).unsqueeze(1),
                                   torch.arange(sqlen-1).unsqueeze(0),
                                   idxs2[:, 1:sqlen]]
            logits2_dist = torch.gather(logits2, 2, idxs2_dist)

            # with torch.no_grad():
            #     logits2 = lm(idxs2).logits[:, :-1, :]
            #     logits2 = F.log_softmax(logits2, dim=-1)
            #     logits2_cons1 = logits2[torch.arange(bs).unsqueeze(1),
            #                        torch.arange(sqlen-1).unsqueeze(0),
            #                        idxs2[:, 1:sqlen]]

            if method_type=="2":
                loss_vic=logits2_cons-old_logits2
                loss_vic=log_clip(loss_vic)
                loss_vic = torch.sum(
                (loss_vic,
                    )*mask2[:, :-1])

                loss_reward=torch.sum(logits2_cons*mask2[:,:-1])\
                    -torch.sum(logits1*mask1[:,:-1])
            elif method_type=="1":
                loss_vic=logits2_cons-old_logits2
                loss_vic=log_clip(loss_vic)
                loss_vic = torch.sum(
                (loss_vic)*mask2[:, :-1])

                loss_reward=torch.sum(
                    (logits2_cons- vic_logits2[:,:,0])*mask2[:,:-1])\
                    -torch.sum(log_clip(logits1-old_logits1)\
                               *mask1[:,:-1])

            loss_constractive=-loss_vic-loss_reward

            vic_logits2=torch.softmax(torch.exp(vic_logits2)\
                                      /args.temperature,
                                      dim=-1)
            logits2_dist=F.log_softmax(torch.exp(logits2_dist)\
                                   /args.temperature,
                                      dim=-1)
            mask2l=mask2[:,:-1].unsqueeze(-1).expand(-1, -1, 5)
            loss_logits=(mask2l*kl_loss(logits2_dist,
                                        vic_logits2)).mean()*beta
            # loss_logits = beta *\
            #     torch.sum(mask2[:, :-1]
            #               .unsqueeze(-1)
            #               .expand(-1, -1, logits2_dist.shape[2])
            #               * logits2_dist*torch.log(
            #         logits2_dist/(vic_logits2+epsln)
            #         + epsln))/torch.sum(mask2)

            if args.use_entropy=="1":
                loss_entropy=torch.sum(torch.exp(logits2)*\
                                        logits2*mask2[:,:-1])
            else:
                loss_entropy=0.

            if args.use_kld!="1":
                loss_logits = 0.

            overall_loss += loss_constractive + loss_logits \
                + loss_entropy

            if overall_step % log_step == 0:
                print(" LOSS: {}\tGain: {}\tReward: {}\tL: {}\tKL-D: {}\tEntropy: {}".format(
                    overall_loss,
                    loss_vic,
                    loss_reward,
                    loss_constractive,
                    loss_logits,
                    loss_entropy,
                ))
                tb_writer.add_scalar("loss", overall_loss.item(),
                                     overall_step)
                if args.use_vic_logits=="1":
                    tb_writer.add_scalar("Gain",
                                     loss_vic.item(),
                                     overall_step)
                if args.use_old_logits=="1":
                    tb_writer.add_scalar("reward",
                                     loss_reward.item(),
                                     overall_step)
                if args.use_kld=="1":
                    tb_writer.add_scalar("KLloss", loss_logits.item(),
                                     overall_step)
                if args.use_entropy=="1":
                    tb_writer.add_scalar("Entropy", loss_entropy.item(),
                                     overall_step)

            if overall_step % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___"+str(overall_step))
                lm.save_pretrained(save_path+"___"+str(overall_step))

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
              args, raw_train_datals,
              max_new_tokens=-1,
              ):
    print(">>>> DATA PREPERATION")
    # STEP 1: DATA Preperation.
    ITER_num = args.period_num
    tb_writer = SummaryWriter(log_dir=args.save_path+"___log_writer")
    p_ls, idx2ls, logits2ls, idx2_dist = raw_train_datals
    # print(lm_tokenizer.decode(p_ls[0]),lm_tokenizer.decode(p_ls[1]),
    #                    lm_tokenizer.decode(p_ls[2]))
    # print(lm_tokenizer.decode(idx2ls[0]),lm_tokenizer.decode(idx2ls[1]),
    #                    lm_tokenizer.decode(idx2ls[2]))
    for iter_idx in range(ITER_num):
        tensorboard_name = f"Period {iter_idx}"
        idxs1_ls = []
        old_logits1_ls = []
        old_logits2_ls = []
        # collect data.
        with torch.no_grad():
            for i,prompt in tqdm(enumerate(p_ls),
                               desc="Data Collecting..."):
                prompt = prompt.to(args.device).unsqueeze(0)
                # Generate New Tokens
                if max_new_tokens > 0:
                    idxs1 = lm.generate(prompt,
                                        do_sample=True,
                                        max_length=args.max_length,
                                        # early_stopping=True,
                                        max_new_tokens=max_new_tokens,
                                        # temperature=args.temperature,
                                        )
                else:
                    idxs1 = lm.generate(prompt,
                                        do_sample=True,
                                        max_length=args.max_length,
                                        # early_stopping=True,
                                        # temperature=args.temperature,
                                        )

                bs, sqqql = idxs1.shape
                # print(idxs1)
                print(lm_tokenizer.decode(idxs1[0]))
                old_logits = lm(idxs1[:, :-1]).logits
                old_logits = F.log_softmax(old_logits, dim=-1)
                old_logits = old_logits[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sqqql-1).unsqueeze(0),
                    idxs1[:, 1:sqqql]
                ]

                idxs2=torch.tensor(idx2ls[i],dtype=torch.long)\
                           .to(args.device).unsqueeze(0)
                old_logits2 = lm(idxs2[:,:-1]).logits
                old_logits2 = F.log_softmax(old_logits2, dim=-1)
                bs, sql2 = idxs2.shape
                old_logits2 = old_logits2[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sql2-1).unsqueeze(0),
                    idxs2[:, 1:sql2]
                ]


                idxs1_ls.append(idxs1.squeeze(0).to("cpu"))
                old_logits1_ls.append(old_logits.squeeze(0).to("cpu"))
                old_logits2_ls.append(old_logits2.squeeze(0).to("cpu"))

        # do truncations and paddings.
        max_token_num_1 = min(args.max_length,
                              max([len(x) for x in idxs1_ls]))
        max_token_num_2 = min(args.max_length,
                              max([len(x) for x in idx2ls]))
        max_token_num = max(max_token_num_1, max_token_num_2)
        print("max_token_num", max_token_num)
        pad_idx = lm_tokenizer.pad_token_id

        idx2ls, mask2 = my_padding(idx2ls, p_ls, max_token_num, pad_idx)
        idxs1_ls, mask1 = my_padding(idxs1_ls, p_ls, max_token_num, pad_idx)

        old_logits1_ls = my_padding_logit(old_logits1_ls,
                                          max_token_num-1, pad_idx)
        old_logits2_ls = my_padding_logit(old_logits2_ls,
                                          max_token_num-1, pad_idx)
        newlogits2ls = []
        for per_data in logits2ls:
            sl = len(per_data)
            v = len(per_data[0])
            tmp_ts = torch.ones((sl, v), dtype=torch.float)
            for jjjj, per_token_logit in enumerate(per_data):
                tmp_ts[jjjj] = torch.tensor(per_token_logit,)
            newlogits2ls.append(tmp_ts)

        logits2ls = my_padding_logits(newlogits2ls,
                                      max_token_num-1, pad_idx)
        idxs2_dist = my_padding_token_dist(idx2_dist,
                                           max_token_num-1, pad_idx)

        print(old_logits1_ls.shape)
        trainset = TensorDataset(
            idxs1_ls,
            idx2ls,
            mask1,
            mask2,
            old_logits1_ls,
            old_logits2_ls,
            logits2ls,
            idxs2_dist,
        )

        loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            )
        print(">>>> Period {}".format(iter_idx))
        # STEP 2: Train the Model in a period
        if args.task=="lord":
            lm = train_one_period(args, lm,
                                lm_tokenizer,
                                loader,
                                args.epoch, args.device,
                                tb_writer,
                                tensorboard_name,
                                args.save_path,
                                args.LR,
                                args.acc_step, args.log_step,
                                args.save_step,
                                args.beta,
                                )
        elif args.task=="Complex-lord":
            from lord_complex_train import complex_train_one_period as ct
            lm = ct(args, lm,
                    lm_tokenizer,
                    loader,
                    args.epoch, args.device,
                    tb_writer,
                    tensorboard_name,
                    args.save_path,
                    args.LR,
                    args.acc_step, args.log_step,
                    args.save_step,
                    args.beta,
                    )
        elif args.task=="reinforce-lord":
            from lord_reinforce_train import reinforce_train_one_period
            lm = reinforce_train_one_period(args, lm,
                    lm_tokenizer,
                    loader,
                    args.epoch, args.device,
                    tb_writer,
                    tensorboard_name,
                    args.save_path,
                    args.LR,
                    args.acc_step, args.log_step,
                    args.save_step,
                    args.beta,
                    )
        else:
            print("ERROR: CANNOT FIND THE TRAIN LOSS OF THE TASK.")

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
    parser.add_argument('--train_num', default=100, type=int,
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
    parser.add_argument('--task', default="lord", type=str,
                        required=False,)
    parser.add_argument('--use_old_logits', default="1", type=str,
                        required=False,)
    parser.add_argument('--use_vic_logits', default="1", type=str,
                        required=False,)
    parser.add_argument('--use_kld', default="1", type=str,
                        required=False,)
    parser.add_argument('--use_entropy', default="1", type=str,
                        required=False,)
    parser.add_argument('--dataset_task', default="glue", type=str,
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

    # if lm_tokenizer.pad_token is None:
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    tasks_glue = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",]

    tasks_wmt16= [
        "cs-en",
        "du-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
        ]

    tasks_supported=tasks_glue.copy()
    tasks_supported += ["sum",]
    tasks_supported.extend(tasks_wmt16)

    print("---------------------------------------------------------")
    print(f"TASKS NOW Supported: {tasks_supported}")

    if args.dataset_task in tasks_supported:
        
        if args.dataset_task in tasks_glue:
            print(f"RUN GLUE task: {args.dataset_task}")
            raw_train_datals = load_glue_datals(tokenizer,
                                            task_name=args.dataset_task,
                                            train_num=args.train_num,
                                            max_length=args.max_length)
        elif args.dataset_task in tasks_wmt16:
            print(f"RUN wmt task: {args.dataset_task}")
            from wmt_process import load_wmt_datals
            raw_train_datals = load_wmt_datals(
                tokenizer,
                task_name=args.dataset_task,
                train_num=args.train_num,
                max_length=args.max_length
                )
        elif args.dataset_task== "sum":
            print(f"RUN summarization task: {args.dataset_task}")
            from sum_process import load_sum_datals
            raw_train_datals = load_sum_datals(
                tokenizer,
                task_name=args.dataset_task,
                train_num=args.train_num,
                max_length=args.max_length
                )
        else:
            print("EEERRROOORRR: EEERRROOORRR.")
        print("Data LOADING done.")
        
        if "lord" in args.task:
            print("TRAIN WITH LORD!!!")
            train_pod(
                lm,
                lm_tokenizer,
                args,
                raw_train_datals,
                max_new_tokens=16,
            )
        elif args.task in ["kd", "vanilla"]:
            print("TRAIN WITH KD~~~")
            p_ls, idx2ls, logits2ls, idx2_dist = raw_train_datals
            max_token_num = min(args.max_length,
                              max([len(x) for x in idx2ls]))
            pad_idx = lm_tokenizer.pad_token_id
            idx2ls, mask2 = my_padding(idx2ls, p_ls,
                                       max_token_num, pad_idx)
            # print("--DEBUG the MASK:-------")
            # print(idx2ls[0])
            # print(lm_tokenizer.decode(idx2ls[0]))
            # print(lm_tokenizer.convert_ids_to_tokens(idx2ls[0]))
            # print(mask2[0])
            # print("--DEBUG the MASK:-------")
            # print(idx2ls[1])
            # print(lm_tokenizer.decode(idx2ls[1]))
            # print(lm_tokenizer.convert_ids_to_tokens(idx2ls[1]))
            # print(mask2[1])
            # print("--DEBUG the MASK:-------")
            # print(idx2ls[2])
            # print(lm_tokenizer.decode(idx2ls[2]))
            # print(lm_tokenizer.convert_ids_to_tokens(idx2ls[2]))
            # print(mask2[2])
            newlogits2ls = []
            for per_data in logits2ls:
                sl = len(per_data)
                v = len(per_data[0])
                tmp_ts = torch.ones((sl, v), dtype=torch.float)
                for jjjj, per_token_logit in enumerate(per_data):
                    tmp_ts[jjjj] = torch.tensor(per_token_logit,)
                newlogits2ls.append(tmp_ts)
            logits2ls = my_padding_logits(newlogits2ls,
                                        max_token_num-1, pad_idx)
            idxs2_dist = my_padding_token_dist(idx2_dist,
                                            max_token_num-1, pad_idx)
            
            trainset = TensorDataset(
                idx2ls,
                mask2,
                logits2ls,
                idxs2_dist,
            )

            loader=DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )

            tb_writer = SummaryWriter(log_dir=args.save_path+\
                                      "___log_writer")
            tensorboard_name="nothing"

            if args.task=="kd":
                from supervised_distillation import train_distill
                train_distill(
                    lm,
                    lm_tokenizer,
                    loader,
                    args.epoch,args.device,
                    tb_writer,
                    tensorboard_name,
                    args.save_path,
                    args.lr,
                    args.acc_step, args.log_step,
                    args.save_step,
                    args.temperature,
                )

            elif args.task=="vanilla":
                from supervised_training import train_supervised
                train_supervised(
                    lm,
                    lm_tokenizer,
                    loader,
                    args.epoch,args.device,
                    tb_writer,
                    tensorboard_name,
                    args.save_path,
                    args.lr,
                    args.acc_step, args.log_step,
                    args.save_step,
                    args.temperature,
                )
    else:
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
