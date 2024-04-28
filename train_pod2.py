"""
======================================================================
TRAIN_POD2 ---

New stealing mechamism.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 27 March 2024
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
from sequence_utils import left_pad
from sequence_utils import random_shut 

import torch.nn.functional as F

# from rlhf_train import clip, log_clip
import random

from rlhf_train import clip, log_clip

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
          raw_train_datals, max_new_tokens=16):
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

    pp11ls=None
    pp12ls=None

    preset_subset_num=args.sub_set_num

    for ssn in range(sub_stage_num):
        if ssn<3:
            args.sub_set_num=16
        else:
            args.sub_set_num=preset_subset_num

        lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
            p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
            p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls, \
            pp11ls, pp12ls = train_pod(
                lm, lm_tokenizer,
                        args, raw_train_datals,
                        max_new_tokens,
                        p_i_11_ls, p_i_12_ls, p_m_11_ls,
                        p_m_12_ls, p_logits_11_ls, p_logits_12_ls,
                        p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls,
                        pp11ls, pp12ls,
                        )
        if (ssn+1) % 3 == 0:
            print(f" -->NOW save the ckpt in stage {ssn}.")
            lm_tokenizer.save_pretrained(args.save_path +
                                         "___period"+str(ssn))
            lm.save_pretrained(args.save_path +
                               "___period"+str(ssn))


def train_pod(lm,
              lm_tokenizer,
              args, raw_train_datals,
              max_new_tokens,
              p_i_11_ls, p_i_12_ls, p_m_11_ls,
              p_m_12_ls, p_logits_11_ls, p_logits_12_ls,
              p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls,
              pp11ls, pp12ls,
              ):
    # print(">>>> DATA PREPERATION")
    tau1 = args.tau1
    tau2 = args.tau2
    print(f" Tau1: {tau1}\t Tau2: {tau2}.")
    print(f"MAX NEW TOKENS: {max_new_tokens}.")
    # STEP 1: DATA Preperation.
    ITER_num = args.period_num
    tb_writer = SummaryWriter(log_dir=args.save_path+"___log_writer")
    op_ls, oidx2ls, ologits2ls, oidx2_dist = raw_train_datals

    subset_num = args.sub_set_num

    # 1. in every period, random take a subset.
    seed = time.time()
    p_ls = random_take(subset_num, op_ls, seed,)
    idx2ls = random_take(subset_num, oidx2ls, seed)
    if ologits2ls is not None:
        vic_logits2ls = random_take(subset_num, ologits2ls, seed)
        idx2_dist = random_take(subset_num, oidx2_dist, seed)
    else:
        vic_logits2ls = [None for _ in range(subset_num)]
        idx2_dist = [None for _ in range(subset_num)]

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
            ## =======================================================
            ## New version of the code: chunked generation. 
            ## =======================================================

            if args.with_early_shut==1:
                print("EXECUTE ERALY SHUT...")
                p_ls=random_shut(p_ls)

            chunked_size=args.infer_batch_size
            num_chunks=math.floor(len(p_ls)/chunked_size)
            # 1. first divided the model into chunks.
            for i_chunked in range(num_chunks+1):
                if i_chunked == num_chunks:
                    if i_chunked*chunked_size!=len(p_ls):
                        prompt=p_ls[i_chunked*chunked_size:]
                        ## left padding
                        print(f"BOS TOKEN ID: {lm_tokenizer.bos_token_id}")
                        prompt=left_pad(prompt,lm_tokenizer.bos_token_id)
                        prompt=prompt.to(args.device)
                    else:
                        break
                else:
                    print(f"BOS TOKEN ID: {lm_tokenizer.bos_token_id}")
                    prompt=p_ls[i_chunked*chunked_size:\
                                (i_chunked+1)*chunked_size]
                    prompt=left_pad(prompt,lm_tokenizer.bos_token_id)
                    prompt=prompt.to(args.device)

                gen_idx=lm.generate(
                    prompt,
                    do_sample=True,
                    max_length=args.max_length,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=2,
                    temperature=1.5,
                    top_p=0.98,
                    use_cache=True,
                    )

                # gen_idx=lm.generate(
                #     prompt,
                #     do_sample=False,
                #     num_beams=2,
                #     num_beam_groups=2,
                #     diversity_penalty=0.3,
                #     num_return_sequences=2,
                #     max_length=args.max_length,
                #     max_new_tokens=max_new_tokens,
                #     use_cache=True,
                #     )


                # 2. extract idx12 and idx11 from gen_idx
                idxs11=gen_idx[0::2,:]
                idxs12=gen_idx[1::2,:]

                # print(gen_idx.shape)
                # print(idxs11.shape)
                # print(idxs12.shape)

                # shape of gen idx: (2*chunked_size, msl)
                bs, sqqql = idxs11.shape
                # print(idxs1)
                # print(f"idxs11 {lm_tokenizer.decode(idxs11[0])}")
                # print(f"idxs12 {lm_tokenizer.decode(idxs12[0])}")

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


                idxs11_ls.extend([x for x in idxs11.to("cpu")])
                idxs12_ls.extend([x for x in idxs12.to("cpu")])
                old_logits11_ls.extend([x for x in old_logits11
                                       .to("cpu")])
                old_logits12_ls.extend([x for x in old_logits12
                                       .to("cpu")])
            # print(idxs11_ls)
            # print(idxs12_ls)
            # assert len(idxs11_ls)==len(idxs12_ls)
            # assert len(idxs11_ls)==len(p_ls)
                
            for i in range(len(p_ls)):
                idxs2 = torch.tensor(idx2ls[i], dtype=torch.long)\
                    .to(args.device).unsqueeze(0)
                # print(f"idxs2 {lm_tokenizer.decode(idxs2[0])}")
                old_logits2 = lm(idxs2[:, :-1]).logits
                old_logits2 = F.log_softmax(old_logits2, dim=-1)
                bs, sql2 = idxs2.shape
                old_logits2 = old_logits2[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sql2-1).unsqueeze(0),
                    idxs2[:, 1:sql2]
                ]
                old_logits2_ls.append(old_logits2.squeeze(0).to("cpu"))

            ## =======================================================
            ## OLD CODE: NOT FAST ENOUGH MAYBE.
            ## =======================================================
            # for i, prompt in tqdm(enumerate(p_ls),
            #                       desc="Data Collecting..."):
            #     prompt = prompt.to(args.device).unsqueeze(0)
            #     # Generate New Tokens
            #     idxs12 = lm.generate(prompt,
            #                          do_sample=True,
            #                          max_length=args.max_length,
            #                          max_new_tokens=max_new_tokens,
            #                          # temperature=args.temperature,
            #                          )

            #     idxs11 = lm.generate(prompt,
            #                          do_sample=True,
            #                          max_length=args.max_length,
            #                          max_new_tokens=max_new_tokens,
            #                          # temperature=args.temperature,
            #                          )

            #     bs, sqqql = idxs11.shape
            #     # print(idxs1)
            #     print(f"idxs11 {lm_tokenizer.decode(idxs11[0])}")
            #     print(f"idxs12 {lm_tokenizer.decode(idxs12[0])}")

            #     old_logits11 = lm(idxs11[:, :-1]).logits
            #     old_logits11 = F.log_softmax(old_logits11, dim=-1)
            #     old_logits11 = old_logits11[
            #         torch.arange(1).unsqueeze(1),
            #         torch.arange(sqqql-1).unsqueeze(0),
            #         idxs11[:, 1:sqqql]
            #     ]

            #     bs, sqqql2 = idxs12.shape
            #     old_logits12 = lm(idxs12[:, :-1]).logits
            #     old_logits12 = F.log_softmax(old_logits12, dim=-1)
            #     old_logits12 = old_logits12[
            #         torch.arange(1).unsqueeze(1),
            #         torch.arange(sqqql2-1).unsqueeze(0),
            #         idxs12[:, 1:sqqql2]
            #     ]

            #     idxs2 = torch.tensor(idx2ls[i], dtype=torch.long)\
            #         .to(args.device).unsqueeze(0)
            #     print(f"idxs2 {lm_tokenizer.decode(idxs2[0])}")
            #     old_logits2 = lm(idxs2[:, :-1]).logits
            #     old_logits2 = F.log_softmax(old_logits2, dim=-1)
            #     bs, sql2 = idxs2.shape
            #     old_logits2 = old_logits2[
            #         torch.arange(1).unsqueeze(1),
            #         torch.arange(sql2-1).unsqueeze(0),
            #         idxs2[:, 1:sql2]
            #     ]

            #     idxs11_ls.append(idxs11.squeeze(0).to("cpu"))
            #     idxs12_ls.append(idxs12.squeeze(0).to("cpu"))
            #     old_logits11_ls.append(old_logits11
            #                            .squeeze(0).to("cpu"))
            #     old_logits12_ls.append(old_logits12
            #                            .squeeze(0).to("cpu"))
            #     old_logits2_ls.append(old_logits2.squeeze(0).to("cpu"))

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

        idx2ls, mask2 = my_padding(idx2ls, p_ls,
                                   max_token_num, pad_idx)
        idxs11_ls, mask11 = my_padding(idxs11_ls,
                                       p_ls, max_token_num, pad_idx)
        idxs12_ls, mask12 = my_padding(idxs12_ls,
                                       p_ls, max_token_num, pad_idx)
        if args.with_early_shut:
            mask11=torch.ones_like(mask11)
            mask12=torch.ones_like(mask12)

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

        if vic_logits2ls[0] is not None:
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

            pp11ls=[]
            pp12ls=[]
            for i, prompt in enumerate(p_ls):
                p11 = float(torch.sum(torch.exp(p_logits_11_ls[i])
                                      * p_m_11_ls[i, :-1])
                            / torch.sum(p_m_11_ls[i, :-1]))
                p12 = float(torch.sum(torch.exp(p_logits_12_ls[i])
                                      * p_m_12_ls[i, :-1])
                            / torch.sum(p_m_12_ls[i, :-1]))
                pp11ls.append(p11)
                pp12ls.append(p12)

        elif need_pre_store == 0:
            for i, prompt in enumerate(p_ls):
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                # print(f"shape of logits 11: {p_logits_11_ls[i].shape}")
                # print(f"value of logits 11: {torch.sum(torch.exp(p_logits_11_ls[i])*p_m_11_ls[i,:-1])/torch.sum(p_m_11_ls[i,:-1])}")
                # print(f"value of logits 12: {torch.sum(torch.exp(p_logits_12_ls[i])*p_m_11_ls[i,:-1])/torch.sum(p_m_12_ls[i,:-1])}")

                pidx11 = p_i_11_ls[i].unsqueeze(0).to(args.device)
                pidx12 = p_i_12_ls[i].unsqueeze(0).to(args.device)

                p11 = float(torch.sum(torch.exp(p_logits_11_ls[i])
                                      * p_m_11_ls[i, :-1])
                            / torch.sum(p_m_11_ls[i, :-1]))
                p12 = float(torch.sum(torch.exp(p_logits_12_ls[i])
                                      * p_m_12_ls[i, :-1])
                            / torch.sum(p_m_12_ls[i, :-1]))

                p11=p11-pp11ls[i]
                p12=p11-pp12ls[i]

                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(f"confidence, 1: {p11}, 2: {p12}")
                if p12 > p11:
                    print("SWAP.")
                    p_i_11_ls[i] = pidx12.squeeze(0).to("cpu")
                    p_i_12_ls[i] = pidx11.squeeze(0).to("cpu")

                    temppp = p_logits_11_ls[i]
                    p_logits_11_ls[i] = p_logits_12_ls[i]
                    p_logits_12_ls[i] = temppp

                    temppp = p_m_11_ls[i]
                    p_m_11_ls[i] = p_m_12_ls[i]
                    p_m_12_ls[i] = temppp
                if max(p11, p12) < tau1:
                    print("BUT still use the VIC's labels.")
                    # print(f"shape of 11: {p_i_11_ls.shape}")
                    # print(f"shape of 2: {idx2ls.shape}")
                    # print(f"shape of 12: {p_i_12_ls.shape}")

                    p_i_11_ls[i] = p_i2ls[i]
                    p_m_11_ls[i] = pmask2s[i]
                    p_logits_11_ls[i] = p_logits2ls[i]

                if min(p11, p12) < tau2:
                   period_break = 0

            ## current the current idx11's probablity on current model
            pp11ls=[]
            pp12ls=[]
            for i, prompt in enumerate(p_ls):
                p11 = float(torch.sum(torch.exp(old_logits11_ls[i])
                                      * p_m_11_ls[i, :-1])
                            / torch.sum(p_m_11_ls[i, :-1]))
                p12 = float(torch.sum(torch.exp(old_logits12_ls[i])
                                      * p_m_12_ls[i, :-1])
                            / torch.sum(p_m_12_ls[i, :-1]))
                pp11ls.append(p11)
                pp12ls.append(p12)

        need_pre_store = 0

        # Dataset what we seen is about the last stage,
        # not current stage.
        # If it is the first stage, then we use the victim's label
        # to guide the training, for a better bootstrapping.
        if vic_logits2ls[0] is not None:
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
        else:
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
                p_logits2ls,
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
                p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls,\
                pp11ls, pp12ls

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
                        is_black_box=0,
                        method=args.task,
                        )

    # lm_tokenizer.save_pretrained(args.save_path+"___finally")
    # lm.save_pretrained(args.save_path+"___finally")
    return lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
        p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
        p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls, \
        pp11ls, pp12ls


def one_period(args, lm,
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
               method="LORD-II",
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

            # print(item)
            idxs11, idxs12, idxs2, mask11, mask12,\
                mask2, old_logits11, old_logits12,\
                old_logits2, vic_logits2 = item

            bs, sqlen1 = idxs11.shape
            sqlen = sqlen1

            idxs11 = idxs11.to(device)  # bs, sql
            idxs12 = idxs12.to(device)  # bs, sql
            idxs2 = idxs2.to(device)  # bs, sql
            mask11 = mask11.to(device)
            mask12 = mask12.to(device)
            mask2 = mask2.to(device)

            mask11 = mask11 == 0
            mask12 = mask12 == 0

            # already normalized by softmax
            old_logits11 = old_logits11.to(device)  # bs, sql,
            old_logits12 = old_logits12.to(device)  # bs, sql,
            old_logits2 = old_logits2.to(device)  # bs, sql,

            if args.is_black_box==0:
                vic_logits2 = vic_logits2.to(device)  # bs, sql, 5

            # print("===========================================")
            # print(f">>>idx11text:  {lm_tokenizer.decode(idxs11[0])}")
            # print(f">>>idx12text:  {lm_tokenizer.decode(idxs12[0])}")
            # print(f">>>idx2text:  {lm_tokenizer.decode(idxs2[0])}")

            # loss1 = lm(idxs11,
            #            labels=labels11).loss
            # loss2 = lm(idxs12,
            #            labels=labels12).loss
            # # loss = loss1-loss2
            # loss = loss1

            # hard to say: our method.
            logits11 = lm(idxs11).logits[:, :-1, :]
            logits11 = F.log_softmax(logits11, dim=-1)
            logits11 = logits11[torch.arange(bs).unsqueeze(1),
                                torch.arange(sqlen-1).unsqueeze(0),
                                idxs11[:, 1:sqlen]]

            logits12 = lm(idxs12).logits[:, :-1, :]
            logits12 = F.log_softmax(logits12, dim=-1)
            logits12 = logits12[torch.arange(bs).unsqueeze(1),
                                torch.arange(sqlen-1).unsqueeze(0),
                                idxs12[:, 1:sqlen]]

            logits2 = lm(idxs2).logits[:, :-1, :]
            logits2 = torch.log_softmax(logits2, dim=-1)
            logits2_cons = logits2[torch.arange(bs).unsqueeze(1),
                                   torch.arange(sqlen-1).unsqueeze(0),
                                   idxs2[:, 1:sqlen]]

            # mask = torch.logical_or(mask11, mask12).long()

            term1 = torch.sum(log_clip(-old_logits12+logits12)
                              * mask12[:, :-1])
            term2 = torch.sum(log_clip(old_logits11-logits11)
                              * mask11[:, :-1])

            if args.is_black_box == 0:
                term3 = \
                    (vic_logits2[:, :, 0]+old_logits2-2*logits2_cons)
            else:
                term3 = old_logits2 - logits2_cons

            term3 = torch.sum(term3 * mask2[:, :-1])

            if method == "LoRD-II":
                loss_1 = term1 + term2
                loss_2 = term3
            elif method == "LoRD-II-no_vic":
                loss_1 = term2
                loss_2 = term1

            loss = loss_1 + loss_2

            # if loss == torch.tensor(float("nan")):
            #     print("++++++++++++++++++++++")
            #     print(f"term1: {term1}")
            #     print(f"term2: {term3}")
            #     print(f"loss1: {loss_1}")
            #     print(f"loss2: {loss_2}")
            #     print(f"loss: {loss}")
            #     print(f"mask: {mask[:,:-1]}")
            #     print("++++++++DEBUG DONE.++++++++")

            overall_loss += loss

            overall_loss = loss
            if overall_step % log_step == 0:
                print(" LOSS: {}".format(
                    overall_loss,
                ))
                print(" Neg Loss: {}".format(
                    term1
                ))
                print(" Pos Loss: {}".format(
                    term2
                ))
                print(" Standard Loss: {}".format(
                    term3
                ))
                tb_writer.add_scalar("loss", overall_loss,
                                     overall_step)
                tb_writer.add_scalar("term1", term1,
                                     overall_step)
                tb_writer.add_scalar("term2", term2,
                                     overall_step)
                tb_writer.add_scalar("term3", term3,
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


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
