"""
======================================================================
SUPERVISED_DISTILLATION ---

Model Extraction with supervised distillation LOSS.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  2 March 2024
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


def train_distill(lm,
                     lm_tokenizer,
                     loader, epoch, device,
                     tb_writer,
                     tensorboard_name,
                     save_path,
                     LR=3e-5,
                     acc_step=1,
                     log_step=100,
                     save_step=1000,
                     temperature=1.0,
                  epsln=1e-6,
                     ):
    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="loader"):
            overall_step += 1

            # print(item)
            idxs2, mask2, vic_logits2, idxs2_dist = item
            bs, sqlen = idxs2.shape

            idxs2 = idxs2.to(device)  # bs, sql
            mask2 = mask2.to(device)
            # already normalized by softmax
            vic_logits2 = vic_logits2.to(device)  # bs, sql, 5

            idxs2_dist = idxs2_dist.to(device)

            print("idx2text: ", lm_tokenizer.decode(idxs2[0]))

            logits2 = lm(idxs2).logits[:, :-1, :]
            logits2 = torch.softmax(logits2, dim=-1)

            logits2_dist = torch.gather(logits2, 2, idxs2_dist)

            logits2_dist/=temperature
            vic_logits2/=temperature
            logits2 = torch.softmax(logits2_dist, dim=-1)
            vic_logits2 = torch.softmax(vic_logits2, dim=-1)

            loss_logits = torch.mean(mask2[:, :-1]
                          .unsqueeze(-1)
                          .expand(-1, -1, logits2_dist.shape[2])
                          * logits2_dist*torch.log(
                    logits2_dist/(vic_logits2+epsln)
                    + epsln))

            overall_loss += loss_logits

            if overall_step % log_step == 0:
                print(" LOSS: {}".format(
                    overall_loss,
                ))
                tb_writer.add_scalar("loss", overall_loss.item(),
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
    lm_tokenizer.save_pretrained(save_path+"___finally")
    lm.save_pretrained(save_path+"___finally")

    print("ONE PERIOD TRAINING DONE!")
    return lm
