"""
======================================================================
LLAMA3_WATERMARK_GEN ---

Generate watermarks from the victim model.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 31 May 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"]="1"
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import torch.nn.functional as F
from tqdm import tqdm

torch.arrange=torch.arange

import sys
sys.path.append("/home/zi/alignmentExtraction/watermark")
from extended_watermark_processor import WatermarkLogitsProcessor
from transformers import LogitsProcessorList
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset


def wrmk_gen(tokenizer, model, input_text,):
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=2.0,
        seeding_scheme="selfhash")
    #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
    # Note:
    # You can turn off self-hashing by setting the seeding scheme
    # to `minhash`.
    tokenized_input = tokenizer(input_text,
                                return_tensors='pt').to(model.device)
    idxx=tokenized_input.input_ids

    # note that if the model is on cuda, then the input is on cuda
    # and thus the watermarking rng is cuda-based.
    # This is a different generator than the cpu-based rng in pytorch!

    output_tokens = model.generate(
        idxx,
        logits_processor=LogitsProcessorList(
            [watermark_processor]))

    # if decoder only model, then we need to isolate the
    # newly generated tokens as only those are watermarked,
    # the input/prompt is not
    output_tokens = output_tokens[:,
                                  tokenized_input["input_ids"]
                                  .shape[-1]:]
    output_text = tokenizer.batch_decode(output_tokens,
                                         skip_special_tokens=True)[0]
    return output_text


def wrmk_gen_list(modelname, task_name, res_pth,
              test_set_take_num=100,
              mnt=32,
              base_model_name=None,):

    save_pth = res_pth

    tasks_we_used = [
        "e2e_nlg",
        "allenai/common_gen",
        "cs-en",
        "de-en",
        "fi-en",
        "ro-en",
        "ru-en",
        "tr-en",
    ]

    assert task_name in tasks_we_used

    task_seqlen_map = {
        "e2e_nlg": 512,
        "allenai/common_gen": 256,
    }

    pls = {
        "e2e_nlg": "Please translate the information to a sentence with natural language.",
        "allenai/common_gen": "Please generate a sentence based on the words provided by User.",
        "cs-en": "Translate the sentence from Czech to English Please.",
        "de-en": "Translate the sentence from Dutch to English Please.",
        "fi-en": "Translate the sentence from Finnish to English Please.",
        "ro-en": "Translate the sentence from Romanian to English Please.",
        "ru-en": "Translate the sentence from Russian to English Please.",
        "tr-en": "Translate the sentence from Turkish to English Please.",
    }
    pp = pls[task_name]

    inp_ls = []

    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(task_name,
                                     split=f"test")\
            .shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)

        for item in trainset_text:
            question = item["meaning_representation"]
            label = item["human_reference"]
            inp_ls.append((f"Information: {question}.", label))

    elif task_name == tasks_we_used[1]:

        trainset_text = load_dataset(task_name,
                                     split=f"validation")\
            .shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)

        for item in trainset_text:
            question = item["concepts"]
            question = ", ".join(question)
            label = item["target"]
            inp_ls.append((f"Words: {question}.", label))
    else:
        dataset = load_dataset("wmt16",
                            task_name,
                            split=f"test").shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)
        # print("DATASET 0: ",dataset[0])
        # print("DATASET 1: ",dataset[1])
        sets = dataset
        from_lang, to_lang = task_name.split("-")

        for item in trainset_text:
            inp_ls.append((item[from_lang],
                           item[to_lang]))

    assert inp_ls != []
    if True:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
        )

        tokenizer = AutoTokenizer\
            .from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        res_ls = []
        for d in tqdm(inp_ls):
            inps, summary = d
            final_inps = "Instruction: " + pp + " User: " + inps + " Assistant: "
            output_text=wrmk_gen(tokenizer,model,
                                 final_inps,)
            res_ls.append((output_text, summary))
    return res_ls
    

def wrmk_gen2(model, tokenizer, input_idx,):
    """
    input_idx: shape[1, MSL_INPUT]
    generated_idx: shape[1, MSL]
    generated_logits: shape[1, MSL]
    """
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=2.0,
        seeding_scheme="selfhash")
    #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
    # Note:
    # You can turn off self-hashing by setting the seeding scheme
    # to `minhash`.
    # note that if the model is on cuda, then the input is on cuda
    # and thus the watermarking rng is cuda-based.
    # This is a different generator than the cpu-based rng in pytorch!


    input_idx=input_idx.unsqueeze(0).to(model.device)
    output_tokens = model.generate(
        input_idx,
        logits_processor=LogitsProcessorList(
            [watermark_processor]))
    _,sql=output_tokens.shape

    input_idx=None

    # logits=model(output_tokens).logits[:,:-1]
    # logits = F.log_softmax(logits, dim=-1)
    # logits = logits[torch.arrange(1).unsqueeze(1),
                    # torch.arrange(sql-1).unsqueeze(0),
                    # output_tokens[:,1:sql]]
    # return output_tokens,logits

    return output_tokens

def commonly_used_wrmk_post_process(
        save_pth,
        inp_ls,
        pp,
        model_name,
        topk,
        max_length,
        p_idxls,
        V,
        lm_tokenizer,
        victim="meta-llama/Meta-Llama-3-70B-Instruct",
        ):

    if not os.path.exists(save_pth):
        model=AutoModelForCausalLM.from_pretrained(
            victim,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            )
        tokenizer=lm_tokenizer

        print(f"RUNNING {victim} Stealing...")
        text2ls = []
        idx2_dist_ls = []
        probsls = []

        for i_for, inp in tqdm(enumerate(inp_ls),
                               desc="Steal Progress: "):
            generated_tokens=wrmk_gen2(
                model,
                tokenizer,
                p_idxls[i_for],
                )
            # generated_tokens,logits=wrmk_gen2(
                # model,
                # tokenizer,
                # p_idxls[i_for],
                # )
            text2ls.append(generated_tokens.squeeze(0).to("cpu"))
            probsls.append(generated_tokens.squeeze(0).to("cpu"))
            idx2_dist_ls.append(generated_tokens
                                .squeeze(0).to("cpu"))
            generated_tokens=None
            logits=None

        with open(save_pth,
                  'wb') as f:
            pickle.dump([text2ls, probsls, idx2_dist_ls],
                        f,)
    else:
        print("Directly Loading...")
        # from collections import OrderedDict
        with open(save_pth, 'rb') as f:
            data = pickle.load(f,)
        text2ls = data[0]
        probsls = data[1]
        idx2_dist_ls = data[2]

    return p_idxls, text2ls, probsls, idx2_dist_ls


def main():
    pass


## running entry
if __name__=="__main__":
    print("EVERYTHING DONE.")


