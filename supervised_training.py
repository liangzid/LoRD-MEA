"""
======================================================================
SUPERVISED_TRAINING ---

The most straightforward version: directly training with supervised
learning.


Convergence Problem of Supervised Training:
1. Not about the mask.
2. Not about the KL divergence.
3. Any Others?

Even input the training data the models cannot generate the responses
which are making sense. Quite strange.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created:  7 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import torch
import torch.nn.functional as F
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


def train_supervised(lm,
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
    print("VVVAAANNNIIILLLAAA---TRAIN!!!")
    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id
    ce = torch.nn.CrossEntropyLoss()

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    for e in tqdm(range(epoch), desc="epoch"):
        for item in tqdm(loader, desc="loader"):
            overall_step += 1

            # print(item)
            idxs2, mask2, _, _ = item
            bs, sqlen = idxs2.shape

            idxs2 = idxs2.to(device)  # bs, sql
            mask2 = mask2.to(device)

            print("Input Index: ", idxs2)
            print("Input Index Text: ", lm_tokenizer.decode(idxs2[0]))

            # logits2 = lm(idxs2).logits[:, :-1, :]
            # logits_hard = ce(logits2, idxs2[:,1:])

            logits_hard = lm(idxs2,
                             # attention_mask=mask2,
                             labels=idxs2,
                             ).loss

            # print(logits_hard)

            overall_loss += logits_hard

            if overall_step % log_step == 0:
                print(" LOSS: {}".format(
                    overall_loss,
                    # logits_hard,
                    # loss_logits,
                ))

                tb_writer.add_scalar("loss", overall_loss.item(),
                                     overall_step)

            if overall_step % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___" +
                                             str(overall_step))
                lm.save_pretrained(save_path+"___" +
                                   str(overall_step))

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
