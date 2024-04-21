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
import os

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from datasets import load_dataset
import json

import random
from tqdm import tqdm

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process

import os
from collections import OrderedDict
from pprint import pprint

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def load_qa_datals(
    tokenizer,
    task_name="piqa",
    train_num=100,
    model_name="gpt-3.5-turbo-1106",
    topk=5,
    max_length=256,
    openai_tmp_save_pth="./wmt_data_saveto_",
):
    lm_tokenizer = tokenizer

    V = lm_tokenizer.vocab_size
    tasks_we_used = ["piqa", "truthful_qa", "allenai/ai2_arc"]
    assert task_name in tasks_we_used
    dataset_name = task_name
    inp_ls = []
    if task_name == tasks_we_used[0]:
        trainset_text = load_dataset(
            dataset_name, split=f"train[:{train_num}]")

        for item in trainset_text:
            question = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            label = str(item["label"])
            text = (
                f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            )
            inp_ls.append(text)

    elif task_name == tasks_we_used[1]:
        trainset_text = load_dataset(
            dataset_name, "multiple_choice", split=f"validation[:{train_num}]"
        )

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
            label = str(0)
            text = (
                f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            )
            inp_ls.append(text)

    elif task_name == tasks_we_used[2]:
        trainset_text = load_dataset(
            dataset_name, "ARC-Challenge", split=f"train[:{train_num}]"
        )

        for item in trainset_text:
            question = item["question"] + "\n\n"
            choices_text = ""
            for idx in range(len(item["choices"]["label"])):
                choices_text += f"Selection {item['choices']['label'][idx]}"
                choices_text += " " + item["choices"]["text"][idx] + "\n\n"

            label = item["answerKey"]
            text = f"Question: {question}{choices_text}"

            inp_ls.append(text)

    assert inp_ls != []

    pp = "Please select the correct answer for the Question of Users."

    prompts = [f"Instruction: {pp} User: {x} Assistant: " for x in inp_ls]
    p_idxls = []
    for p in prompts:
        p_idxls.append(lm_tokenizer(p, return_tensors="pt").input_ids[0])

    task_name1 = task_name.replace("/", "_")
    openai_tmp_save_pth += f"QAtask_{task_name1}-trainNUM_{train_num}.pkl"

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


def infer_qa(modelname, task_name, res_pth, test_set_take_num=100,
             mnt=32,
             base_model_name=None,
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

    print(task_name)

    model = InferObj(
        model_name=modelname, device="auto",
        max_length=task_seqlen_map[task_name],
        base_model_name=base_model_name,
    )

    gen_pipeline = model.text_gen

    pp = "Please select the correct answer for the Question of Users."

    inp_ls = []

    if task_name == tasks_we_used[0]:
        trainset_text = (
            load_dataset(task_name, split=f"validation")
            .shuffle(20240307)
            .to_iterable_dataset()
            .take(test_set_take_num)
        )

        for item in trainset_text:
            question = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            label = str(item["label"])
            text = (
                f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            )
            inp_ls.append((text, label))

    elif task_name == tasks_we_used[1]:
        trainset_text = load_dataset(
            task_name, "multiple_choice", split=f"validation")
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
            label = str(0)
            text = (
                f"Question: {question}\n\n Selection 1: {sol1} \n\n Selection 2:{sol2}"
            )
            inp_ls.append((text, label))

    elif task_name == tasks_we_used[2]:
        trainset_text = load_dataset(
            task_name, "ARC-Challenge", split=f"validation")
        for item in trainset_text:
            question = item["question"] + "\n\n"
            choices_text = ""
            for idx in range(len(item["choices"]["label"])):
                choices_text += f"Selection {item['choices']['label'][idx]}"
                choices_text += " " + item["choices"]["text"][idx] + "\n\n"

            label = item["answerKey"]
            text = f"Question: {question}{choices_text}"

            inp_ls.append((text, label))

    assert inp_ls != []

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

    model = None
    gen_pipeline = None
    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


def eval_qa_res():
    ckpt_ls = (
        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaComplex-lord112164256___period0/",
        ],

        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaLoRD-II112164256___period2/",
        ],
        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaLoRD-II112164256___period5/",
        ],
        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaLoRD-II112164256___period8/",
        ],

        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaLoRD-IV112164256___period2/",
        ],

        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaLoRD-IV112164256___period5/",
        ],

        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___41piqaLoRD-IV112164256___period8/",
        ],

        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___161piqaLoRD-IV112164256___period8/",
        ],
        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___161piqaLoRD-IV112164256___period5/",
        ],
        [
            "piqa",
            "./vArY_TrAiN_nUm_ckpts/varyTrainNum___321piqaLoRD-IV112164256___period8/",
        ],
        [
            "piqa",
            "google/gemma-2b",
        ],
    )

    res_dict = {}
    dir_p = "./qa_dataset_res/"
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    for task_ckpt in ckpt_ls:
        task, ckpt = task_ckpt
        res_pth = ckpt + f"___{task}_qa_infer_res"
        res_pth = res_pth.replace("/", "__").replace(".", "")
        res_pth += ".json"
        if not os.path.exists(dir_p + res_pth):
            res_ls = infer_qa(
                ckpt,
                task,
                dir_p + res_pth,
                test_set_take_num=1000,
                # test_set_take_num=50,
                mnt=64,
            )
        else:
            # from collections import OrderedDict
            with open(dir_p + res_pth, "r", encoding="utf8") as f:
                res_ls = json.load(f, object_pairs_hook=OrderedDict)

        print(res_ls)
        scores = eval_qaacc(task, res_ls)
        print(task, ckpt)
        print(scores)
        res_dict[task + "-----" + ckpt] = scores
    with open(dir_p + "temp_boring_res_delete_thisfile_itisuseless.json", "w", encoding="utf8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
    print("OVERALL Save DONE.")
    pprint(res_dict)


def eval_varytrainum_res():
    taskls = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc",
    ]
    # mls = ["vanilla", "kd", "google/gemma-2b"]
    mls = ["LoRD-IV"]
    # mls = ["google/gemma-2b",]
    train_times = [
        "1",
        "2",
        "3",
    ]
    train_nums = ["4", "8", "16", "32", "64", "100", "256", "512"]
    train_nums = ["512",]

    dir_p = "./vary_train_num_qa_infers/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                for itime in train_times:
                    prefix = "./vArY_TrAiN_nUm_ckpts/"
                    if m == "google/gemma-2b":
                        ckpt = m
                    elif m == "Complex-lord":
                        ckpt = (
                            prefix
                            + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___period2"
                        )
                    elif m == "LoRD-IV":
                        ckpt = (
                            prefix
                            + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___period5"
                        )
                    else:
                        ckpt = (
                            prefix
                            + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___finally"
                        )

                    if m == "google/gemma-2b":
                        res_pth = ckpt + f"__{itime}_{task}_qa_infer_res.json"
                    else:
                        res_pth = ckpt + f"___{task}_qa_infer_res.json"

                    res_pth = res_pth.replace("/", "__").replace(".", "")

                    if not os.path.exists(dir_p+res_pth):
                        res_ls = infer_qa(ckpt, task, dir_p+res_pth,
                                          test_set_take_num=1000,
                                          mnt=64)
                    else:
                        print(
                            f"{dir_p+res_pth} file already exists. directly loading...")
                        # from collections import OrderedDict
                        with open(dir_p + res_pth, "r", encoding="utf8") as f:
                            res_ls = json.load(
                                f, object_pairs_hook=OrderedDict)

                    scores = eval_qaacc(task, res_ls)
                    res_dict[task + "-----" + res_pth] = scores
    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)


def eval_varytrainum_231_ours():
    taskls = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc",
    ]
    mls = ["LoRD-II", "LoRD-IV"]
    # mls = ["google/gemma-2b",]
    train_times = [
        "1",
        "2",
        "3",
    ]
    train_nums = ["4", "8", "16", "32", "64", "100", "256", "512"]
    period_nums = ["2", "5", "8"]

    dir_p = "./vary_train_num_qa_infers/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                for itime in train_times:
                    for periodnum in period_nums:
                        prefix = "./vArY_TrAiN_nUm_ckpts/"
                        if m == "google/gemma-2b":
                            ckpt = m
                        elif m == "Complex-lord":
                            ckpt = (
                                prefix
                                + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___period2"
                            )
                        elif "LoRD" in m:
                            ckpt = (
                                prefix
                                + f"varyTrainNum___{train_num}{itime}{task}{m}112164256___period{periodnum}"
                            )
                        else:
                            ckpt = (
                                prefix
                                + f"varyTrainNum___{train_num}{itime}{task}{m}332164256___finally"
                            )

                        if m == "google/gemma-2b":
                            res_pth = ckpt + \
                                f"__{itime}_{task}_qa_infer_res.json"
                        else:
                            res_pth = ckpt + f"___{task}_qa_infer_res.json"

                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        if not os.path.exists(dir_p + res_pth):
                            res_ls = infer_qa(
                                ckpt, task, dir_p + res_pth,
                                test_set_take_num=1000,
                                mnt=64,
                            )
                        else:
                            # from collections import OrderedDict
                            with open(dir_p + res_pth, "r", encoding="utf8") as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        scores = eval_qaacc(task, res_ls)
                        res_dict[task + "-----" + res_pth] = scores
    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)


def eval_qaacc(task, res):
    task_label_map = {
        "piqa": {
            "0": "Selection 1",
            "1": "Selection 2",
        },
        "truthful_qa": {
            "1": "Selection 1",
            "0": "Selection 2",
        },
        "allenai/ai2_arc": {
            "0": "Selection 1",
            "1": "Selection 2",
            "2": "Selection 3",
            "3": "Selection 4",
        },
    }
    extra_ai2 = {
        "0": "Selection A",
        "1": "Selection B",
        "2": "Selection C",
        "3": "Selection D",
    }

    textlabel_to_reallabel_map = {
        "piqa": {
            "0": "0",
            "1": "1",
        },
        "truthful_qa": {
            "0": "0",
            "1": "0",
        },
        "allenai/ai2_arc": {
            "A": "0",
            "B": "1",
            "C": "2",
            "D": "3",
            "E": "4",
            "F": "5",
            "1": "0",
            "2": "1",
            "3": "2",
            "4": "3",
        },
    }

    predict_ls = []
    label_ls = []
    submap = task_label_map[task]
    sm_r = {v: k for k, v in submap.items()}
    for res_sent, lbl in res:
        # print(res_sent)
        res_sent = res_sent.lower()
        if task == "allenai/ai2_arc":
            # label_ls.append(float(sm_r[lbl]))
            if "1" in res_sent or "Selection A" in res_sent:
                vv = sm_r["Selection 1"]
            elif "2" in res_sent or "Selection B" in res_sent:
                vv = sm_r["Selection 2"]
            elif "3" in res_sent or "Selection C" in res_sent:
                vv = sm_r["Selection 3"]
            elif "4" in res_sent or "Selection D" in res_sent:
                vv = sm_r["Selection 4"]
            else:
                vv = sm_r["Selection 4"]
        else:
            if "2" in res_sent:
                vv = sm_r["Selection 2"]
            else:
                vv = sm_r["Selection 1"]

        predict_ls.append(float(vv))
        if task != "truthful_qa":
            # print("task: ", task)
            label = textlabel_to_reallabel_map[task][lbl]
        else:
            label = "0"
        label_ls.append(float(label))

    metric_ls = [precision_score, recall_score, f1_score]
    scores = [accuracy_score(label_ls, predict_ls)]
    for m in metric_ls:
        scores.append(m(label_ls, predict_ls, average="macro"))
    return scores


# running entry
if __name__ == "__main__":
    # main()
    eval_qa_res()
    # eval_varytrainum_res()
    # eval_varytrainum_231_ours()
    print("EVERYTHING DONE.")
