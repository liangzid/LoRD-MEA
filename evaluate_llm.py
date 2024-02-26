"""
======================================================================
EVALUATE_LLM ---

Evaluating LLM with some famous datasets.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 26 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
from transformers import AutoModel,AutoTokenizer,AutoConfig, AutoModelForCausalLM

from accelerate import load_checkpoint_and_dispatch

def main():
    ## submit the RL checkpoint
    # pth="POD_SAVE_CKPTs/TheFirstTimeAttempts/policy-___finally"
    pth="./POD_SAVE_CKPTs/TheFirstTimeAttempts/policy-___period4"
    config=AutoConfig.from_pretrained(pth)
    # model=AutoModel.from_config(config)
    # model=load_checkpoint_and_dispatch(model,pth,device_map="auto")

    model=AutoModelForCausalLM.from_pretrained(
        pth,
        device_map="cpu",
        ignore_mismatched_sizes=True,
        )

    tokenizer=AutoTokenizer.from_pretrained(pth)
    # model.push_to_hub(pth.replace("/","_").replace(".",""), config=config)
    tokenizer.push_to_hub(pth.replace("/","_").replace(".",""), config=config)

def save_test2():
    # model=AutoModel.from_pretrained("gpt2",device_map="auto")
    # model.save_pretrained("./current_res_test_delete_this")
    model=AutoModel.from_pretrained("./current_res_test_delete_this",
                                    device_map="auto")
    
    import os
    res=os.listdir("./lm-evaluation-harness/lm_eval/tasks/")
    print(res)

## running entry
if __name__=="__main__":
    # main()
    save_test2()
    print("EVERYTHING DONE.")


