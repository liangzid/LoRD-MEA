"""
======================================================================
GEN_PIPELINE_OPEN --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  2 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
import logging
from datasets import load_dataset


class InferObj:
    def __init__(self, model_name="gpt2",
                 meta_prompt_pth="./instructions/meta-1.txt",
                 prompt_dataset="liangzid/prompts",
                 split="train",
                 device="auto",
                 max_length=2047,
                 max_new_tokens=-1,
                 open_16_mode=False,
                 load_in_8_bit=False,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       trust_remote_code=True,)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model_name = model_name

        # Model
        if open_16_mode:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # quantization_config=quant_config,
                device_map=device,
                # load_in_8bit=True,
                trust_remote_code=True,
                # offload_folder="offload",
                torch_dtype=torch.float16,
            )
        elif load_in_8_bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # quantization_config=quant_config,
                device_map=device,
                load_in_8bit=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # quantization_config=quant_config,
                device_map=device,
                # load_in_8bit=True,
                trust_remote_code=True,
                offload_folder="offload",
            )

        self.text_gen = pipeline(task="text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 max_length=max_length,
                                 )

        # self.temp_prompts = load_dataset(prompt_dataset)[split].to_list()
        # self.prompts = []
        # for xx in self.temp_prompts:
        #     self.prompts.append(xx["text"])
        # logging.info("Prompt file loading done.")

        self.meta_instruct = ""
        # with open(meta_prompt_pth, 'r', encoding="utf8") as f:
        #     self.meta_instruct = f.read()
        # logging.info("Meta prompt file loading done.")

        # self.update_prompt()
        self.eos = "### User"
        self.p = ""
        self.prompt = ""

    def update_prompt(self, bigger_than=0, smaller_than=1e5):
        newone = self.prompts[0]
        is_find = 0
        assert smaller_than > bigger_than
        random.shuffle(self.prompts)
        for x in self.prompts:
            if len(x.split(" ")) < smaller_than \
               and len(x.split(" ")) > bigger_than:
                newone = x
                is_find = 1
                break
        if is_find == 0:
            logging.info("WARNING: PROMPT NOT FOUND")
        self.prompt = newone

        # # Concentrate them
        # if "<PROMPT>" in self.meta_instruct:
        #     self.p = self.meta_instruct.replace("<PROMPT>",
        #                                         self.prompt,
        #                                         )
        # else:
        #     self.p = self.prompt

        self.p = self.prompt

        logging.info(f"updated prompt: {self.p}")

    def vanilla_prompt_based_attacking(self, query, is_sample=False,
                                       num_beams=1, num_beam_groups=1, dp=0.0,
                                       k=50, p=1.0, t=1.0,
                                       repetition_penalty=2.3,
                                       no_repeat_ngram_size=3,
                                       ):
        if "<QUERY>" in self.p:
            query = self.p.replace("<QUERY>", query)
        else:
            if self.model_name == "microsoft/phi-1_5":
                query = "Instruction: "+self.p+" Alice: "+query+" Bob: "
            else:
                query = "Instruction: "+self.p +\
                    " User: "+query+" Assistant: "

        logging.info(f"Query:{query}")
        output = self.text_gen(f"{query}",
                               do_sample=is_sample,
                               num_beams=num_beams,
                               num_beam_groups=num_beam_groups,
                               diversity_penalty=dp,
                               top_k=k,
                               top_p=p,
                               temperature=t,
                               # repetition_penalty=repetition_penalty,
                               # no_repeat_ngram_size=no_repeat_ngram_size,
                               # sequence_length=4096,
                               )
        logging.info(output)
        resps = []
        for x in output:
            t = x["generated_text"]
            # t = extract_onlyGen(query, t, eos=self.eos)
            resps.append(t)
        logging.info(resps)
        return resps


