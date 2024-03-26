"""
======================================================================
QA_PROCESS ---

QA datasets's process.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 26 March 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

from datasets import load_dataset
import json

import random
from tqdm import tqdm

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process


def load_qa_datals(tokenizer,
                   task_name="piqa",
                   train_num=100,
                   model_name="gpt-3.5-turbo-1106",
                   topk=5,
                   max_length=256,
                   openai_tmp_save_pth="./wmt_data_saveto_"):

    lm_tokenizer = tokenizer

    V = lm_tokenizer.vocab_size
    tasks_we_used = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc"
    ]
    assert task_name in tasks_we_used
    dataset_name = task_name
    inp_ls = []
    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:
            question = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            label = str(item["label"])
            text = f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            inp_ls.append(text)

    elif task_name == tasks_we_used[1]:

        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:

            question = item["question"]
            assert len(item["mc1_targets"]["choices"]) >= 2
            sol1 = item["mc1_targets"]["choices"][0]
            sol2 = item["mc1_targets"]["choices"][1]

            if random.random() > 0.5:
                # then flip
                temp = sol1
                sol1 = sol2
                sol2 = temp
            label = str(item["label"])
            text = f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            inp_ls.append(text)

    elif task_name == tasks_we_used[2]:

        trainset_text = load_dataset(dataset_name,
                                     split=f"train[:{train_num}]")

        for item in trainset_text:

            question = item["question"]+"\n\n"
            choices_text = ""
            for idx in range(len(item["choices"]["label"])):
                choices_text += f"Selection {item['choices']['label'][idx]}"
                choices_text += " "+item["choices"]["text"][idx]+"\n\n"

            label = item["answerKey"]
            text = f"Question: {question}{choices_text}"

            inp_ls.append(text)

    assert inp_ls != []

    pp = "Please select the correct answer for the Question of Users."

    prompts = [f"Instruction: {pp} User: {x} Assistant: "
               for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    openai_tmp_save_pth += f"QAtask_{task_name}-trainNUM_{train_num}.pkl"

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


def infer_qa(modelname, task_name, res_pth,
             test_set_take_num=100,
             mnt=32
             ):

    save_pth = res_pth

    tasks_we_used = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc",
    ]

    assert task_name in tasks_we_used

    task_seqlen_map = {
        "piqa": 256,
        "truthful_qa": 256,
        "allenai/ai2_arc": 256,
    }

    model = InferObj(model_name=modelname,
                     device="auto",
                     max_length=task_seqlen_map[task_name])

    gen_pipeline = model.text_gen

    pp = "Please select the correct answer for the Question of Users."

    inp_ls = []

    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(task_name,
                                     split=f"test")\
            .shuffle(20240307)\
            .to_iterable_dataset()\
            .take(test_set_take_num)

        for item in trainset_text:
            question = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            label = str(item["label"])
            text = f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            inp_ls.append((text, label))

    elif task_name == tasks_we_used[1]:

        trainset_text = load_dataset(task_name, "multi_choice",
                                     split=f"validation")\

        for item in trainset_text:

            question = item["question"]
            assert len(item["mc1_targets"]["choices"]) >= 2
            sol1 = item["mc1_targets"]["choices"][0]
            sol2 = item["mc1_targets"]["choices"][1]

            if random.random() > 0.5:
                # then flip
                temp = sol1
                sol1 = sol2
                sol2 = temp
            label = str(item["label"])
            text = f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            inp_ls.append((text, label))

    elif task_name == tasks_we_used[2]:

        trainset_text = load_dataset(task_name,
                                     split=f"test")\

        for item in trainset_text:

            question = item["question"]+"\n\n"
            choices_text = ""
            for idx in range(len(item["choices"]["label"])):
                choices_text += f"Selection {item['choices']['label'][idx]}"
                choices_text += " "+item["choices"]["text"][idx]+"\n\n"

            label = item["answerKey"]
            text = f"Question: {question}{choices_text}"

            inp_ls.append((text, label))

    assert inp_ls != []

    res_ls = []
    for d in tqdm(inp_ls):
        inps, summary = d
        final_inps = "Instruction: " + pp +\
            " User: "+inps+" Assistant: "
        res = gen_pipeline(final_inps,
                           max_new_tokens=mnt,)[0]["generated_text"]
        res = res.split(final_inps)[1]
        res_ls.append((res, summary))

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
