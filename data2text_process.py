"""
======================================================================
DATA2TEXT_PROCESS ---

Data2text datasets process.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 26 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import os
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    os.environ["TORCH_USE_CUDA_DSA"]="1"
from datasets import load_dataset
import json
from collections import OrderedDict
from pprint import pprint

from tqdm import tqdm

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process
from wmt_process import eval_wmt as eval_d2ttt

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch

def load_data2text_datals(tokenizer,
                          task_name="e2e_nlg",
                          train_num=100,
                          model_name="gpt-3.5-turbo-1106",
                          topk=5,
                          max_length=512,
                          openai_tmp_save_pth="./STEALED_PKLS/wmt_data_saveto_"):

    lm_tokenizer = tokenizer

    V = lm_tokenizer.vocab_size
    tasks_we_used = [
        "e2e_nlg",
        "allenai/common_gen",
    ]
    assert task_name in tasks_we_used
    dataset_name = task_name
    inp_ls = []
    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:
            question = item["meaning_representation"]
            label = item["human_reference"]
            inp_ls.append(f"Information: {question}.")

    elif task_name == tasks_we_used[1]:

        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:
            question = item["concepts"]
            question = ", ".join(question)
            label = item["target"]
            inp_ls.append(f"Words: {question}.")
    assert inp_ls != []

    pls = {
        "e2e_nlg": "Please translate the information to a sentence with natural language.",
        "allenai/common_gen": "Please generate a sentence based on the words provided by User.",
    }

    pp = pls[task_name]

    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    openai_tmp_save_pth += f"Data2Ttask_{task_name}-trainNUM_{train_num}.pkl"

    return commonly_used_openai_post_process(
        openai_tmp_save_pth,
        inp_ls,
        pp,
        model_name,
        topk,
        max_length,
        p_idxls,
        V,
        lm_tokenizer,
    )


def infer_d2t(modelname, task_name, res_pth,
              test_set_take_num=100,
              mnt=32,
              base_model_name=None,
              ):

    save_pth = res_pth

    tasks_we_used = [
        "e2e_nlg",
        "allenai/common_gen",
    ]

    assert task_name in tasks_we_used

    task_seqlen_map = {
        "e2e_nlg": 512,
        "allenai/common_gen": 256,
    }

    model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=task_seqlen_map[task_name])

    gen_pipeline = model.text_gen

    pls = {
        "e2e_nlg": "Please translate the information to a sentence with natural language.",
        "allenai/common_gen": "Please generate a sentence based on the words provided by User.",
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

    assert inp_ls != []

    if modelname=="gpt-3.5-turbo-1106":
        from training_data_collecting_openai import chatWithOpenAI_APIs
        res_ls=[]
        for d in tqdm(inp_ls):
            inps,summary = d
            res=chatWithOpenAI_APIs(modelname, pp, inps)
            print(f"Generated Text: {res}")
            res_ls.append((res, summary))
    elif base_model_name is None:
        model = InferObj(
            model_name=modelname, device="auto",
            max_length=task_seqlen_map[task_name],
        )
        gen_pipeline = model.text_gen
        res_ls = []
        for d in tqdm(inp_ls):
            inps, summary = d
            final_inps = "Instruction: " + pp + " User: " + inps + " Assistant: "
            res = gen_pipeline(
                final_inps,
                max_new_tokens=mnt,
            )[
                0
            ]["generated_text"]
            res = res.split(final_inps)[1]
            print(f"Text Generated:>>> {res}")
            res_ls.append((res, summary))
    else:
        print("USING PEFT: BASE MODEL + LORA")
        # load model based on our idea
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, modelname)
        tokenizer = AutoTokenizer\
            .from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        res_ls = []
        for d in tqdm(inp_ls):
            inps, summary = d
            final_inps = "Instruction: " + pp + " User: " + inps + " Assistant: "
            inps_idx=tokenizer.encode(final_inps,max_length=128,
                                      padding="longest",
                                      return_tensors="pt")
            print(inps_idx)
            inps_idx=inps_idx.to("cuda")
            res = model.generate(inps_idx,
                                 max_new_tokens=mnt,)
            print(res)
            res=tokenizer.decode(res[0])
            if final_inps in res:
                res = res.split(final_inps)[1]
            else:
                res = res
            print(f"Text Generated:>>> {res}")
            res_ls.append((res, summary))

    model = None
    gen_pipeline = None
    tokenizer = None

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls



def eval_d2t_res():
    ckpt_ls=[
        ["e2e_nlg","./d2t_ckpts/D2TTTe2e_nlg641vanilla___finally",],
        # ["allenai/common_gen",
        #  "./d2t_ckpts/D2TTTe2e_nlg641vanilla___finally",],

        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    res_dict = {}
    dir_p = "./d2t_dataset_res/"
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    for task_ckpt in ckpt_ls:
        task, ckpt = task_ckpt
        if ckpt==base_model_name1:
            base_model_name=None
        else:
            base_model_name=base_model_name1
        res_pth = ckpt + f"___{task}_d2t_infer_res"
        res_pth = res_pth.replace("/", "__").replace(".", "")
        res_pth += ".json"
        if not os.path.exists(dir_p + res_pth):
            res_ls = infer_d2t(
                ckpt,
                task,
                dir_p + res_pth,
                test_set_take_num=500,
                mnt=128,
                base_model_name=base_model_name
            )
        else:
            # from collections import OrderedDict
            with open(dir_p + res_pth, "r", encoding="utf8") as f:
                res_ls = json.load(f, object_pairs_hook=OrderedDict)

        print(res_ls)
        scores = eval_d2ttt(task, res_ls)
        print(task, ckpt)
        print(scores)
        res_dict[task + "-----" + ckpt] = scores
    with open(dir_p + "temp_boring_res_delete_thisfile_itisuseless.json", "w", encoding="utf8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)

# running entry
if __name__ == "__main__":
    # main()
    eval_d2t_res()
    print("EVERYTHING DONE.")
