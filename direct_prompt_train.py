"""
======================================================================
DIRECT_PROMPT_TRAIN ---

Code for a direct prompt.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  3 December 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import time
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
from pprint import pprint as ppp
from transformers import AutoModelForCausalLM
from transformers import AutoModelWithLMHead
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

from openai import OpenAI as oa

client = oa()


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("--epoch", default=2, type=int, required=False)
    parser.add_argument("--period_num", default=3, type=int, required=False)
    parser.add_argument("--sub_stage_num", default=10, type=int, required=False)
    parser.add_argument("--sub_set_num", default=16, type=int, required=False)
    parser.add_argument("--train_num", default=100, type=int, required=False)
    parser.add_argument("--acc_step", default=4, type=int, required=False)
    parser.add_argument("--log_step", default=1, type=int, required=False)
    parser.add_argument("--save_step", default=64, type=int, required=False)

    parser.add_argument("--use_pure_blackbox", default=0, type=int, required=False)
    parser.add_argument("--extra_nonlabel_data", default=0, type=int, required=False)
    parser.add_argument("--nonlabel_data_num", default=32, type=int, required=False)

    parser.add_argument("--with_early_shut", default=0, type=int, required=False)
    parser.add_argument("--use_opensource", default=0, type=int, required=False)

    parser.add_argument("--lambda1", default=0.5, type=float, required=False)
    parser.add_argument("--tau1", default=0.99, type=float, required=False)
    parser.add_argument("--tau_delta", default=0.999, type=float, required=False)
    parser.add_argument("--tau2", default=0.998, type=float, required=False)
    parser.add_argument("--LR", default=3e-4, type=float, required=False)
    parser.add_argument("--beta", default=0.7, type=float, required=False)
    parser.add_argument("--temperature", default=0.8, type=float, required=False)
    parser.add_argument("--T", default=1.0, type=float, required=False)

    parser.add_argument("--use_lora", default=0, type=int, required=False)
    parser.add_argument("--rank", default=64, type=int, required=False)
    parser.add_argument("--lora_alpha", default=128, type=int, required=False)

    parser.add_argument("--batch_size", default=1, type=int, required=False)
    parser.add_argument("--infer_batch_size", default=32, type=int, required=False)
    parser.add_argument(
        "--task",
        default="lord",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_old_logits",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_vic_logits",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_kld",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_entropy",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--is_black_box",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--dataset_task",
        default="glue",
        type=str,
        required=False,
    )
    parser.add_argument("--max_length", default=1024, type=int, required=False)
    parser.add_argument("--max_new_tokens", default=16, type=int, required=False)

    parser.add_argument(
        "--victim_path",
        default="gpt-3.5-turbo",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--from_path",
        default="bert-tiny",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        default="model_training_results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--temp_save_path",
        default="model_training_results",
        type=str,
        required=False,
    )

    return parser.parse_args()


def main():
    args = setup_train_args()

    print("----------------------------------------------------------")
    ppp(args)
    print("----------------------------------------------------------")

    if "t5" in args.from_path:
        lm = AutoModelWithLMHead.from_pretrained(
            args.from_path,
            device_map="auto",
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            args.from_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    lm_tokenizer = AutoTokenizer.from_pretrained(
        args.from_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.from_path,
        trust_remote_code=True,
    )

    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    tasks_glue = [
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "wnli",
    ]

    tasks_wmt16 = [
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    tasks_wmt16_wrmk = [
        "cs-en@wrmk",
        "de-en@wrmk",
        "fi-en@wrmk",
        "ro-en@wrmk",
    ]

    tasks_qa = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc",
    ]

    tasks_code = [
        "deepmind/code_contests",
    ]

    tasks_data2text = [
        "e2e_nlg",
        "allenai/common_gen",
    ]

    tasks_data2text_wrmk = [
        "e2e_nlg@wrmk",
        "allenai/common_gen@wrmk",
    ]

    tasks_sum = [
        "UCL-DARK/openai-tldr-filtered",
        "cnn_dailymail",
        "samsum",
    ]

    tasks_text2sql = [
        "wikisql",
        "spider",
    ]

    tasks_safety = [
        "PKU-Alignment/PKU-SafeRLHF",
        "thu-coai/diasafety",
        "Anthropic/hh-rlhf",
    ]

    tasks_general = [
        "liangzid/claude3_chat3.3k",
        "liangzid/claude3_short256",
        "teknium/GPT4-LLM-Cleaned",
        "BAAI/Infinity-Instruct",
    ]

    tasks_supported = tasks_glue.copy()
    tasks_supported.extend(tasks_sum)
    tasks_supported.extend(tasks_wmt16)
    # tasks_supported.extend(tasks_qa)
    # tasks_supported.extend(tasks_data2text)
    # tasks_supported.extend(tasks_text2sql)
    # tasks_supported.extend(tasks_code)
    # tasks_supported.extend(tasks_safety)
    # tasks_supported.extend(tasks_general)
    # tasks_supported.append("wmt_mix")
    # tasks_supported.extend(tasks_data2text_wrmk)
    # tasks_supported.extend(tasks_wmt16_wrmk)

    print("---------------------------------------------------------")
    print(f"TASKS NOW Supported: {tasks_supported}")

    if args.dataset_task in tasks_wmt16:
        print(f"RUN wmt task: {args.dataset_task}")
        from wmt_process import load_wmt_query

        p_idxls, inp_ls, prompts = load_wmt_query(
            tokenizer,
            task_name=args.dataset_task,
            train_num=args.train_num,
            max_length=args.max_length,
            tokenizer_name=args.from_path,
            use_opensource=args.use_opensource,
        )
    else:
        print("Error: cannot find correct dataset.")
        return -1

    ## load the pre-trained model.

    print(f">>/> Num of params: {lm.num_parameters()}")
    if float(lm.num_parameters()) > 6e9:
        print(">>/> The model is larger than 6B. We use LoRA.")
        args.use_lora = 1

    # if use lora, then set new `lm` with the peft library
    if args.use_lora == 1:
        from peft import (
            LoraConfig,
            # PeftConfig,
            # PeftModel,
            get_peft_model,
            # prepare_model_for_kbit_training,
        )

        # apply lora here
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            # target_modules=["embed_tokens", "lm_head",
            #                 "q_proj", "v_proj",],
            target_modules="all-linear",
        )
        model = get_peft_model(lm, lora_config)
        lm = model
        print(f">>/> Type of the model: {type(lm)}")

    sigmoid = torch.nn.Sigmoid()
    print(">>>> DATA PREPERATION")
    # STEP 1: DATA Preperation.
    tb_writer = SummaryWriter(log_dir=args.save_path + "___log_writer")
    max_new_tokens = args.max_new_tokens
    device = args.device
    log_step = args.log_step
    save_step = args.save_step
    epoch = args.epoch
    save_path = args.save_path
    acc_step = args.acc_step
    LR = args.LR

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)

    data_ls=[]
    judge_ls=[]

    overall_loss = 0.0
    overall_step = 0
    for e in tqdm(range(epoch), desc="epoch"):
        for i, pidx in tqdm(enumerate(p_idxls), desc="Direct Prompting"):
            overall_step += 1
            pidx = pidx.to(args.device).unsqueeze(0)
            ## generate two sentences.

            if max_new_tokens > 0:
                idxs11 = lm.generate(
                    pidx,
                    do_sample=True,
                    max_length=args.max_length,
                    early_stopping=True,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                )
                idxs12 = lm.generate(
                    pidx,
                    do_sample=True,
                    max_length=args.max_length,
                    early_stopping=True,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                )
            else:
                idxs11 = lm.generate(
                    pidx,
                    do_sample=True,
                    max_length=args.max_length,
                    # early_stopping=True,
                    # temperature=args.temperature,
                )
                idxs12 = lm.generate(
                    pidx,
                    do_sample=True,
                    max_length=args.max_length,
                    # early_stopping=True,
                    # temperature=args.temperature,
                )
            # print(f"idxs11: {idxs11}")
            text1 = tokenizer.decode(idxs11[0])
            text2 = tokenizer.decode(idxs12[0])

            if "\n" in text1:
                text1 = text1.split("\n")[0]
            if "User" in text1:
                text1 = text1.split("User")[0]
            if "\n" in text2:
                text2 = text2.split("\n")[0]
            if "User" in text2:
                text2 = text2.split("User")[0]

            print("Text before the swap-----------------------------")
            print(f"text1: {text1}")
            print(f"text2: {text2}")
            

            ## propose the direct prompting.

            # INSTRUCTION = f"Regarding a translation task that requires to translating the given `Text` into English. Now I will provide you two versions of the translation marked with `A` and `B`. Your responsibility is to return the *correct letter of the translation* without any other outputs.\n"
            INSTRUCTION = "For a translation task involving the conversion of the given `Text` into English, The user will provide two translation versions labeled `A` and `B`. Your task is to return the *letter corresponding to the correct translation* without including any additional output."

            query = f"Text:{inp_ls[i]}\nA:{text1}\nB:{text2}"

            modelname = "gpt-3.5-turbo-1106"

            res = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": query},
                ],
            )
            # time.sleep(0.5)
            judge = res.choices[0].message.content
            print(f"judge: {judge}.")

            # filter low quality feedback
            if ("A" in judge and "B" in judge) or (
                "A" not in judge and "B" not in judge
            ):
                continue
            else:
                if "A" in judge:
                    judge_ls.append("A")
                    # A is the better one.
                    pass
                else:
                    # Swap two responses to keep the first one is better.
                    judge_ls.append("B")
                    temp = text1
                    text1 = text2
                    text2 = temp

            data_ls.append([text1,text2])
            # ---------------------------------------------------------
            print("DONE for the feedback. Now optimze the local model.")

            overtext1 = prompts[i] + text1
            overtext2 = prompts[i] + text2

            idxss = lm_tokenizer([
                overtext1,
                overtext2,
                ],
                padding="longest",
                return_tensors="pt",).input_ids
            idx11 = idxss[0]
            idx12 = idxss[1]
            idx11 = idxs11.to(device)  # bs, sql
            idx12 = idxs12.to(device)  # bs, sql

            bs, sqlen1 = idx11.shape
            sqlen2=idx12.shape[1]
            sqlen = sqlen1

            logits11 = lm(idx11).logits[:, :-1, :]
            logits11 = F.log_softmax(logits11, dim=-1)
            logits11 = logits11[
                torch.arange(bs).unsqueeze(1),
                torch.arange(sqlen - 1).unsqueeze(0),
                idx11[:, 1:sqlen],
            ]

            logits12 = lm(idx12).logits[:, :-1, :]
            logits12 = F.log_softmax(logits12, dim=-1)
            logits12 = logits12[
                torch.arange(bs).unsqueeze(1),
                torch.arange(sqlen2 - 1).unsqueeze(0),
                idx12[:, 1:sqlen2],
            ]

            if args.task == "LoRD-VI":
                ## cannot use LoRD because there is no victim model's feedback.
                pass
            elif args.task == "simPO":
                beta = 2.5
                gamma = 0.55 * beta
                loss = beta * torch.mean(logits11) - beta * torch.mean(logits12)
                loss = loss - gamma
                loss = -torch.log(sigmoid(loss))

            # overall_loss += loss
            overall_loss = loss
            if overall_step % log_step == 0:
                print(
                    " LOSS: {}".format(
                        overall_loss,
                    )
                )
                tb_writer.add_scalar("loss", overall_loss, overall_step)

            if overall_step % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path + "___" + str(overall_step))
                lm.save_pretrained(save_path + "___" + str(overall_step))

            if overall_step % acc_step == 0:
                opt1.zero_grad()

                overall_loss.backward()
                opt1.step()
                overall_loss = 0.0

    print(" -->Finally Saving.")
    lm_tokenizer.save_pretrained(save_path + "___STEPfinally")
    lm.save_pretrained(save_path + "___STEPfinally")

    with open(f"./save_path_temperature{args.temperature}_query256_internaldata_directprompt.json",
              'w',encoding='utf8') as f:
        json.dump({
            "datals":data_ls,
            "judgels":judge_ls,
            },f,ensure_ascii=False,indent=4)
    print("Save done.")
        

    return lm


if __name__ == "__main__":
    main()
