"""
======================================================================
LORD_COMPLEX_TRAIN ---

The comprehensive and overall training function. In `pod_train.py` it
is a simplified version of LoRD loss function.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  5 March 2024
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


def complex_train_one_period(args, lm,
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
                             is_black_box=0,
                             method="complex",
                             ):
    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id
    kl_loss = torch.nn.KLDivLoss(reduction="none")
    sigmoid = torch.nn.Sigmoid()

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
            # already normalized by softmax
            old_logits1 = old_logits1.to(device)  # bs, sql,
            old_logits2 = old_logits2.to(device)  # bs, sql,

            vic_logits2 = vic_logits2.to(device)  # bs, sql, 5
            idxs2_dist = idxs2_dist.to(device)

            print("idx1text: ", lm_tokenizer.decode(idxs1[0]))
            print("idx2text: ", lm_tokenizer.decode(idxs2[0]))

            logits1 = lm(idxs1).logits[:, :-1, :]
            logits1 = F.log_softmax(logits1, dim=-1)
            logits1 = logits1[torch.arange(bs).unsqueeze(1),
                              torch.arange(sqlen-1).unsqueeze(0),
                              idxs1[:, 1:sqlen]]

            logits2 = lm(idxs2).logits[:, :-1, :]
            logits2 = torch.log_softmax(logits2, dim=-1)
            logits2_cons = logits2[torch.arange(bs).unsqueeze(1),
                                   torch.arange(sqlen-1).unsqueeze(0),
                                   idxs2[:, 1:sqlen]]

            logits2_dist = torch.gather(logits2, 2, idxs2_dist)

            if method == "complex":
                if is_black_box == 0:
                    loss_constractive_good = -torch.sum(
                        (logits2_cons*2 -
                         vic_logits2[:, :, 0]-old_logits2)*mask2[:, :-1])
                else:
                    loss_constractive_good = -torch.sum(
                        (logits2_cons*2 -
                         -old_logits2)*mask2[:, :-1])

                zelta_logits1 = log_clip(old_logits1-logits1)
                loss_constractive_past = -torch.sum(
                    (zelta_logits1) * mask1[:, :-1])

                if args.use_old_logits != "1":
                    loss_constractive_past = 0.
                if args.use_vic_logits != "1":
                    loss_constractive_good = 0.

                loss_constractive = loss_constractive_good \
                    + loss_constractive_past

                overall_loss = loss_constractive

            elif method == "VeryComplex":

                term1 = -torch.exp(old_logits1)*(
                    log_clip(old_logits1-logits1))

                if is_black_box == 0:
                    term3 = torch.exp(vic_logits2[:, :, 0])\
                        * (
                        (vic_logits2[:, :, 0]-logits2_cons))\
                        + (old_logits2 - logits2_cons)
                else:
                    term3 = - logits2_cons*2

                loss_constractive_past = torch.sum(
                    term1*mask1[:, :-1])
                loss_constractive_good = torch.sum(
                    term3*mask2[:, :-1])

                loss_constractive = loss_constractive_good +\
                    loss_constractive_past

                if args.use_old_logits != "1":
                    loss_constractive_past = 0.
                if args.use_vic_logits != "1":
                    loss_constractive_good = 0.

                overall_loss = loss_constractive

            elif method == "nologComplex":

                mask = torch.logical_or(mask1, mask2).long()
                # print(mask1)
                # print(mask2)
                # print(mask)
                # print("_____________")
                term1 = log_clip(-old_logits1+logits1)
                term2 = (old_logits2-logits2_cons)

                if is_black_box == 0:
                    term3 = \
                        (vic_logits2[:, :, 0]-logits2_cons)
                else:
                    term3 = - logits2_cons

                loss_1 = term2 + term3
                # loss_2 = torch.exp(term1)
                loss_2 = term1

                loss = sigmoid(loss_1)+loss_2

                if torch.sum(mask[:, :-1]) >= 1:
                    loss = torch.sum(loss*mask[:, :-1])
                    # / torch.sum(mask[:, :-1])
                else:
                    loss = 0.
                if loss == torch.tensor(float("nan")):
                    print("++++++++++++++++++++++")
                    print(f"term1: {term1}")
                    print(f"term2: {term3}")
                    print(f"loss1: {loss_1}")
                    print(f"loss2: {loss_2}")
                    print(f"loss: {loss}")
                    print(f"mask: {mask[:,:-1]}")
                    print("++++++++DEBUG DONE.++++++++")

                loss_constractive = loss

                loss_constractive_past = 0.
                loss_constractive_good = 0.
                loss_logits = 0.

                overall_loss += loss_constractive + loss_logits

            elif method == "ComplexV3":

                mask = torch.logical_or(mask1, mask2).long()

                term1 = (-old_logits1+logits1)
                term2 = log_clip(old_logits2-logits2_cons)

                if is_black_box == 0:
                    term3 = \
                        (vic_logits2[:, :, 0]-logits2_cons)
                else:
                    term3 = - logits2_cons

                loss_1 = term1 + term3
                loss_2 = torch.exp(term2)

                loss = sigmoid(loss_1)*loss_2

                if torch.sum(mask[:, :-1]) >= 1:
                    loss = torch.sum(loss*mask[:, :-1])
                    # / torch.sum(mask[:, :-1])
                else:
                    loss = 0.
                if loss == torch.tensor(float("nan")):
                    print("++++++++++++++++++++++")
                    print(f"term1: {term1}")
                    print(f"term2: {term3}")
                    print(f"loss1: {loss_1}")
                    print(f"loss2: {loss_2}")
                    print(f"loss: {loss}")
                    print(f"mask: {mask[:,:-1]}")
                    print("++++++++DEBUG DONE.++++++++")

                loss_constractive = loss

                loss_constractive_past = 0.
                loss_constractive_good = 0.
                loss_logits = 0.

                overall_loss += loss_constractive + loss_logits

            if overall_step % log_step == 0:
                print(" LOSS: {}\tGoodRewardLoss: {}\tToPassRewardLoss: {}\tKL-D: {}".format(
                    overall_loss,
                    loss_constractive_good,
                    loss_constractive_past,
                    loss_logits
                ))
                tb_writer.add_scalar("loss", overall_loss,
                                     overall_step)
                if args.use_vic_logits == "1":
                    tb_writer.add_scalar("rewardloss_good",
                                         loss_constractive_good,
                                         overall_step)
                if args.use_old_logits == "1":
                    tb_writer.add_scalar("rewardloss_past",
                                         loss_constractive_past,
                                         overall_step)
                if args.use_kld == "1":
                    tb_writer.add_scalar("KLloss", loss_logits,
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
    # lm_tokenizer.save_pretrained(save_path+"___STEPfinally")
    # lm.save_pretrained(save_path+"___STEPfinally")

    print("ONE PERIOD TRAINING DONE!")
    return lm
