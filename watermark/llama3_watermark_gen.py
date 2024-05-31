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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import torch.nn.functional as F

import sys
sys.path.append("/home/zi/alignmentExtraction/watermark")
from extended_watermark_processor import WatermarkLogitsProcessor
from transformers import LogitsProcessorList


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

    # note that if the model is on cuda, then the input is on cuda
    # and thus the watermarking rng is cuda-based.
    # This is a different generator than the cpu-based rng in pytorch!

    output_tokens = model.generate(
        **tokenized_input,
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


    input_idx=input_idx.unsqueeze(0).to("cuda")
    output_tokens = model.generate(
        input_idx,
        logits_processor=LogitsProcessorList(
            [watermark_processor]))
    _,sql=output_tokens.shape

    logits=model(output_tokens).logits[:,:-1]
    logits = F.log_softmax(logits, dim=-1)
    logits = logits[torch.arrange(1).unsqueeze(1),
                    torch.arrange(sql-1).unsqueeze(0),
                    output_tokens[:,1:sql]]

    return output_tokens,logits

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

    model=AutoModelForCausalLM.from_pretrained(
        victim,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        )
    tokenizer=lm_tokenizer

    if not os.path.exists(save_pth):
        print(f"RUNNING {victim} Stealing...")
        text2ls = []
        idx2_dist_ls = []
        probsls = []

        for i_for, inp in enumerate(inp_ls):
            generated_tokens,logits=wrmk_gen2(
                model,
                tokenizer,
                p_idxls[i_for],
                )
            text2ls.append(generated_tokens.squeeze(0).to("cpu"))
            probsls.append(logits.squeeze(0).to("cpu"))
            idx2_dist_ls.append(generated_tokens
                                .squeeze(0).to("cpu"))

        with open(openai_tmp_save_pth,
                  'wb') as f:
            pickle.dump([text2ls, probsls, idx2_dist_ls],
                        f,)
    else:
        print("Directly Loading...")
        # from collections import OrderedDict
        with open(openai_tmp_save_pth, 'rb') as f:
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


