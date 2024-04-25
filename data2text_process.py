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


from datasets import load_dataset
import json

from tqdm import tqdm

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process


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
              mnt=32
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
