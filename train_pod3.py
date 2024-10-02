"""
======================================================================
TRAIN_POD3 ---

Beyond `train_pod2.py`, we add unlabeled samples to simulate the attack
procedure.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created:  1 April 2024
======================================================================
"""

# ------------------------ Code --------------------------------------

import torch
# import json
from torch.utils.tensorboard import SummaryWriter
# from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import math
import time
# import argparse
# from transformers import AutoModelForCausalLM
# from transformers import AutoModelForSequenceClassification
# from transformers import AutoModelForTokenClassification
# from transformers import AutoTokenizer, AutoConfig, AutoModel

# from training_data_collecting_openai import load_raw_train_datals
# from training_data_collecting_openai import load_steal_datals
# from glue_process import load_glue_datals

from sequence_utils import my_padding, my_padding_logits
from sequence_utils import my_padding_token_dist
from sequence_utils import my_padding_logit

import torch.nn.functional as F

# from rlhf_train import clip, log_clip
import random

from rlhf_train import clip, log_clip

from train_pod2 import one_period

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print = logger.info


def random_take(num, ls, seed,):
    random.seed(seed)
    random.shuffle(ls)
    if num >= len(ls):
        return ls
    return ls[:num]


def train(lm, lm_tokenizer, args,
          raw_train_datals,
          nonlabel_trainls,
          max_new_tokens=16):
    sub_stage_num = args.sub_stage_num

    steps = sub_stage_num*args.sub_set_num *\
        args.period_num
    print(f"OVERALL STEPS: {steps}.")

    p_i_11_ls = None
    p_i_12_ls = None
    p_m_11_ls = None
    p_m_12_ls = None
    p_logits_11_ls = None
    p_logits_12_ls = None

    p_i2ls = None
    pmask2s = None
    p_logits2ls = None
    p_vic_logits2ls = None

    for ssn in range(sub_stage_num):

        lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
            p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
            p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls\
            = train_pod(lm, lm_tokenizer,
                        args,
                        raw_train_datals,
                        nonlabel_trainls,
                        max_new_tokens,
                        p_i_11_ls, p_i_12_ls, p_m_11_ls,
                        p_m_12_ls, p_logits_11_ls, p_logits_12_ls,
                        p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls
                        )
        if (ssn+1) % 3 == 0:
            print(f" -->NOW save the ckpt in stage {ssn}.")
            lm_tokenizer.save_pretrained(args.save_path +
                                         "___period"+str(ssn))
            lm.save_pretrained(args.save_path +
                               "___period"+str(ssn))


def train_pod(lm,
              lm_tokenizer,
              args,
              raw_train_datals,
              nonlabel_trainls,
              max_new_tokens,
              p_i_11_ls, p_i_12_ls, p_m_11_ls,
              p_m_12_ls, p_logits_11_ls, p_logits_12_ls,
              p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls
              ):
    print(">>>> DATA PREPERATION")
    tau1 = args.tau1
    tau2 = args.tau2
    print(f" Tau1: {tau1}\t Tau2: {tau2}.")
    # STEP 1: DATA Preperation.
    ITER_num = args.period_num
    tb_writer = SummaryWriter(log_dir=args.save_path+"___log_writer")
    op_ls, oidx2ls, ologits2ls, oidx2_dist = raw_train_datals
    ul_p_ls, _, _, _ = raw_train_datals

    subset_num = args.sub_set_num

    # 1. in every period, random take a subset.
    seed = time.time()
    p_ls = random_take(subset_num, op_ls, seed,)
    idx2ls = random_take(subset_num, oidx2ls, seed)
    vic_logits2ls = random_take(subset_num, ologits2ls, seed)
    idx2_dist = random_take(subset_num, oidx2_dist, seed)

    ulpls = random_take(subset_num, ul_p_ls, seed)

    need_pre_store = 0
    if p_i_11_ls is None:
        p_i_11_ls = []
        p_i_12_ls = []
        p_logits_11_ls = []
        p_logits_12_ls = []
        p_m_11_ls = []
        p_m_12_ls = []

        need_pre_store = 1
        period_break = 0

        p_i2ls = []
        pmask2s = []
        p_logits2ls = []
        p_vic_logits2ls = []
    else:
        period_break = 1

    for iter_idx in range(ITER_num):
        tensorboard_name = f"Period {iter_idx}"

        idxs11_ls = []
        idxs12_ls = []
        old_logits11_ls = []
        old_logits12_ls = []

        old_logits2_ls = []

        # 2. generate
        with torch.no_grad():
            for i, prompt in tqdm(enumerate(p_ls),
                                  desc="Data Collecting..."):
                prompt = prompt.to(args.device).unsqueeze(0)
                # Generate New Tokens
                idxs12 = lm.generate(prompt,
                                     do_sample=True,
                                     max_length=args.max_length,
                                     max_new_tokens=max_new_tokens,
                                     # temperature=args.temperature,
                                     )

                idxs11 = lm.generate(prompt,
                                     do_sample=True,
                                     max_length=args.max_length,
                                     max_new_tokens=max_new_tokens,
                                     # temperature=args.temperature,
                                     )

                bs, sqqql = idxs11.shape
                # print(idxs1)
                print(f"idxs11 {lm_tokenizer.decode(idxs11[0])}")
                print(f"idxs12 {lm_tokenizer.decode(idxs12[0])}")

                old_logits11 = lm(idxs11[:, :-1]).logits
                old_logits11 = F.log_softmax(old_logits11, dim=-1)
                old_logits11 = old_logits11[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sqqql-1).unsqueeze(0),
                    idxs11[:, 1:sqqql]
                ]

                bs, sqqql2 = idxs12.shape
                old_logits12 = lm(idxs12[:, :-1]).logits
                old_logits12 = F.log_softmax(old_logits12, dim=-1)
                old_logits12 = old_logits12[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sqqql2-1).unsqueeze(0),
                    idxs12[:, 1:sqqql2]
                ]

                idxs2 = torch.tensor(idx2ls[i], dtype=torch.long)\
                    .to(args.device).unsqueeze(0)
                print(f"idxs2 {lm_tokenizer.decode(idxs2[0])}")
                old_logits2 = lm(idxs2[:, :-1]).logits
                old_logits2 = F.log_softmax(old_logits2, dim=-1)
                bs, sql2 = idxs2.shape
                old_logits2 = old_logits2[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sql2-1).unsqueeze(0),
                    idxs2[:, 1:sql2]
                ]

                idxs11_ls.append(idxs11.squeeze(0).to("cpu"))
                idxs12_ls.append(idxs12.squeeze(0).to("cpu"))
                old_logits11_ls.append(old_logits11
                                       .squeeze(0).to("cpu"))
                old_logits12_ls.append(old_logits12
                                       .squeeze(0).to("cpu"))
                old_logits2_ls.append(old_logits2.squeeze(0).to("cpu"))

            for i, prompt in tqdm(enumerate(ulpls),
                                  "collecting unlabeled data",
                                  ):
                # SKIP the first stage of training.
                if need_pre_store == 1:
                    break
                prompt = prompt.to(args.device).unsqueeze(0)

                idxs12 = lm.generate(prompt,
                                     do_sample=True,
                                     max_length=args.max_length,
                                     max_new_tokens=max_new_tokens,
                                     # temperature=args.temperature,
                                     )

                idxs11 = lm.generate(prompt,
                                     do_sample=True,
                                     max_length=args.max_length,
                                     max_new_tokens=max_new_tokens,
                                     # temperature=args.temperature,
                                     )

                bs, sqqql = idxs11.shape
                # print(idxs1)
                print(f"UL-idxs11 {lm_tokenizer.decode(idxs11[0])}")
                print(f"UL-idxs12 {lm_tokenizer.decode(idxs12[0])}")

                old_logits11 = lm(idxs11[:, :-1]).logits
                old_logits11 = F.log_softmax(old_logits11, dim=-1)
                old_logits11 = old_logits11[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sqqql-1).unsqueeze(0),
                    idxs11[:, 1:sqqql]
                ]

                bs, sqqql2 = idxs12.shape
                old_logits12 = lm(idxs12[:, :-1]).logits
                old_logits12 = F.log_softmax(old_logits12, dim=-1)
                old_logits12 = old_logits12[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sqqql2-1).unsqueeze(0),
                    idxs12[:, 1:sqqql2]
                ]

                idxs11_ls.append(idxs11.squeeze(0).to("cpu"))
                idxs12_ls.append(idxs12.squeeze(0).to("cpu"))
                old_logits11_ls.append(old_logits11
                                       .squeeze(0).to("cpu"))
                old_logits12_ls.append(old_logits12
                                       .squeeze(0).to("cpu"))
                old_logits2_ls.append(old_logits2.squeeze(0).to("cpu"))

        # do truncations and paddings.
        # max_token_num_11 = min(args.max_length,
        #                        max([len(x) for x in p_i_11_ls]))
        # max_token_num_12 = min(args.max_length,
        #                        max([len(x) for x in p_i_12_ls]))
        # max_token_num_2 = min(args.max_length,
        #                       max([len(x) for x in p_i2ls]))

        cmax_token_num_2 = min(args.max_length,
                               max([len(x) for x in idx2ls]))
        cmax_token_num_11 = min(args.max_length,
                                max([len(x) for x in idxs11_ls]))
        cmax_token_num_12 = min(args.max_length,
                                max([len(x) for x in idxs12_ls]))

        max_token_num = max(cmax_token_num_2, cmax_token_num_11)
        max_token_num = max(max_token_num, cmax_token_num_12)

        print(f"max_token_num: {max_token_num}")
        pad_idx = lm_tokenizer.pad_token_id

        templs=[]
        templs.extend(p_ls)
        if need_pre_store==0:
            templs.extend(ulpls)

        idx2ls, mask2 = my_padding(idx2ls, p_ls,
                                   max_token_num, pad_idx)
        idxs11_ls, mask11 = my_padding(idxs11_ls,
                                       templs, max_token_num, pad_idx)
        idxs12_ls, mask12 = my_padding(idxs12_ls,
                                       templs, max_token_num, pad_idx)

        old_logits11_ls = my_padding_logit(old_logits11_ls,
                                           max_token_num-1, pad_idx)
        old_logits12_ls = my_padding_logit(old_logits12_ls,
                                           max_token_num-1, pad_idx)
        old_logits2_ls = my_padding_logit(old_logits2_ls,
                                          max_token_num-1, pad_idx)

        # p_i_11_ls, p_m_11_ls = my_padding(p_i_11_ls,
        #                                   p_ls,
        #                                   max_token_num, pad_idx)
        # p_i_12_ls, p_m_12_ls = my_padding(p_i_12_ls,
        #                                   p_ls,
        #                                   max_token_num, pad_idx)
        # p_logits_11_ls = my_padding_logit(p_logits_11_ls,
        #                                   max_token_num-1,
        #                                   pad_idx)
        # p_logits_12_ls = my_padding_logit(p_logits_12_ls,
        #                                   max_token_num-1,
        #                                   pad_idx)

        newvic_logits2ls = []
        for per_data in vic_logits2ls:
            sl = len(per_data)
            v = len(per_data[0])
            tmp_ts = torch.ones((sl, v), dtype=torch.float)
            for jjjj, per_token_logit in enumerate(per_data):
                tmp_ts[jjjj] = torch.tensor(per_token_logit,)
            newvic_logits2ls.append(tmp_ts)

        vic_logits2ls = my_padding_logits(newvic_logits2ls,
                                          max_token_num-1, pad_idx)
        idxs2_dist = my_padding_token_dist(idx2_dist,
                                           max_token_num-1, pad_idx)
        # print(f"{old_logits11_ls.shape}")

        # ---------------------------------------------------------
        # now fix the logic of constructive two samples
        if need_pre_store == 1:

            need_pre_store = 0
            # at first stage, we have no previous stage, so we use
            # the ground truth in current stage.
            p_i_11_ls = idx2ls  # absolute positive label
            p_i_12_ls = idxs12_ls

            p_logits_11_ls = old_logits2_ls
            p_logits_12_ls = old_logits12_ls

            p_m_11_ls = mask2
            p_m_12_ls = mask12

            p_i2ls = idx2ls
            pmask2s = mask2
            p_logits2ls = old_logits2_ls
            p_vic_logits2ls = vic_logits2ls

            # Dataset what we seen is about the last stage,
            # not current stage.
            # If it is the first stage, then we use the victim's label
            # to guide the training, for a better bootstrapping.
            trainset = TensorDataset(
                p_i_11_ls,
                p_i_12_ls,
                p_i2ls,
                p_m_11_ls,
                p_m_12_ls,
                pmask2s,
                p_logits_11_ls,
                p_logits_12_ls,
                p_logits2ls,
                p_vic_logits2ls,
            )

            # reuse the tensors in current stage as the training dataset
            # for next stage.
            p_i_11_ls = idxs11_ls
            p_i_12_ls = idxs12_ls
            p_i2ls = idx2ls
            p_m_11_ls = mask11
            p_m_12_ls = mask12
            pmask2s = mask2
            p_logits_11_ls = old_logits11_ls
            p_logits_12_ls = old_logits12_ls
            p_logits2ls = old_logits2_ls
            p_vic_logits2ls = vic_logits2ls

            if period_break == 1:
                print("\n\n NOW BREAK SINCE ENOUGH TRAINING\n\n")
                return lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
                    p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
                    p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls

            loader = DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                )
            print(">>>> Period {}".format(iter_idx))
            lm = one_period(args, lm,
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
                            is_black_box=1,
                            method="LoRD-II-no_vic",
                            )

        elif need_pre_store == 0:
            for i, prompt in enumerate(p_ls):
                pidx11 = p_i_11_ls[i].unsqueeze(0).to(args.device)
                pidx12 = p_i_12_ls[i].unsqueeze(0).to(args.device)
                llh1 = lm(pidx11,
                          # p_m_11_ls[i].unsqueeze(0).to(args.device),
                          labels=pidx11).loss
                llh2 = lm(pidx12,
                          # p_m_12_ls[i].unsqueeze(0).to(args.device),
                          labels=pidx12).loss
                sl = pidx12.shape[1]
                m11_num = float(torch.sum(p_m_11_ls[i, :-1]))
                m12_num = float(torch.sum(p_m_12_ls[i, :-1]))
                print(f"num_m11: {m11_num}\t num_m12: {m12_num}")
                print(f"p_m_12_ls[i]: {p_m_12_ls[i]}")
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(f"likelihood, 1: {llh1/sl}, 2: {llh2/sl}")
                if llh2 < llh1:
                    print("SWAP.")
                    p_i_11_ls[i] = pidx12.squeeze(0).to("cpu")
                    p_i_12_ls[i] = pidx11.squeeze(0).to("cpu")

                    temppp = p_logits_11_ls[i]
                    p_logits_11_ls[i] = p_logits_12_ls[i]
                    p_logits_12_ls[i] = temppp

                    temppp = p_m_11_ls[i]
                    p_m_11_ls[i] = p_m_12_ls[i]
                    p_m_12_ls[i] = temppp
                if min(llh1/m11_num, llh2/m12_num) > -math.log(tau1):
                    print("BUT still use the VIC's labels.")
                    # print(f"shape of 11: {p_i_11_ls.shape}")
                    # print(f"shape of 2: {idx2ls.shape}")
                    # print(f"shape of 12: {p_i_12_ls.shape}")

                    p_i_11_ls[i] = p_i2ls[i]
                    p_m_11_ls[i] = pmask2s[i]
                    p_logits_11_ls[i] = p_logits2ls[i]

                if min(llh1/m11_num, llh2/m12_num) > -math.log(tau2):
                    period_break = 0



            # Dataset what we seen is about the last stage,
            # not current stage.
            # If it is the first stage, then we use the victim's label
            # to guide the training, for a better bootstrapping.
            trainset = TensorDataset(
                p_i_11_ls,
                p_i_12_ls,
                p_i_11_ls,
                p_m_11_ls,
                p_m_12_ls,
                p_m_11_ls,
                p_logits_11_ls,
                p_logits_12_ls,
                p_logits_11_ls,
                p_logits_11_ls,
            )

            # reuse the tensors in current stage as the training dataset
            # for next stage.
            p_i_11_ls = idxs11_ls
            p_i_12_ls = idxs12_ls
            p_i2ls = idx2ls
            p_m_11_ls = mask11
            p_m_12_ls = mask12
            pmask2s = mask2
            p_logits_11_ls = old_logits11_ls
            p_logits_12_ls = old_logits12_ls
            p_logits2ls = old_logits2_ls
            p_vic_logits2ls = vic_logits2ls

            if period_break == 1:
                print("\n\n NOW BREAK SINCE ENOUGH TRAINING\n\n")
                return lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
                    p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
                    p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls

            loader = DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                )
            print(">>>> Period {}".format(iter_idx))
            lm = one_period(args, lm,
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
                            is_black_box=1,
                            method="LoRD-II-no_vic",
                            )

        need_pre_store = 0


    # lm_tokenizer.save_pretrained(args.save_path+"___finally")
    # lm.save_pretrained(args.save_path+"___finally")
    return lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
        p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
        p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
