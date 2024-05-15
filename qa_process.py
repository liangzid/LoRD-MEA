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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

from datasets import load_dataset
import json

import random
from tqdm import tqdm
import numpy as np

from gen_pipeline_open import InferObj
from wmt_process import commonly_used_openai_post_process

import os
from collections import OrderedDict
from pprint import pprint

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch


def load_qa_datals(
    tokenizer,
    task_name="piqa",
    train_num=100,
    model_name="gpt-3.5-turbo-1106",
    topk=5,
    max_length=256,
    openai_tmp_save_pth="./STEALED_PKLS/wmt_data_saveto_",
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
            dataset_name, "multiple_choice",
            split=f"validation[:{train_num}]"
        )

        for item in trainset_text:
            question = item["question"]
            assert len(item["mc1_targets"]["choices"]) >= 2
            sol1 = item["mc1_targets"]["choices"][0]
            sol2 = item["mc1_targets"]["choices"][1]
            label = str(0)

            if random.random() > 0.5:
                # then flip
                temp = sol1
                sol1 = sol2
                sol2 = temp
                label = str(1)
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
    print(f"save inference file {save_pth} done.")

    return res_ls


def eval_qa_res():
    ckpt_ls = (
        # [
        #     "piqa",
        #     "gpt-3.5-turbo-1106",
        # ],
        # [
        #     "truthful_qa",
        #     "gpt-3.5-turbo-1106",
        # ],
        # [
            # "allenai/ai2_arc",
            # "gpt-3.5-turbo-1106",
        # ],
    #     [
    #         "piqa",
    #         "google/gemma-7b",
    #     ],
        # ["piqa",
         # "./LoRA-LoRD-ckptsvaryTrainNum___321piqaComplex-lord332164256___period2"
         # ],
        # [
        #     "piqa",
        #     "meta-llama/Meta-Llama-3-8B-Instruct",
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA2561piqaLoRD-VI212132256___period5/",
        # ],


        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA0.202561piqaLoRD-VI212132256___period11/",
        # ],

        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA0.202561piqaLoRD-VI212132256___period5/",
        # ],


        # ## 
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA2561piqavanilla332132256___finally/"
        #     ],

        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period1/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period2/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period3/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period4/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period5/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period6/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period7/"
        # ],
        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period8/"
        # ],

        # [
        #     "piqa",
        #     "./qa_ckpts/QAAA---TEMP--res___period256/"
        # ],

        # ["piqa","./qa_ckpts/QAAAnewpiqa5121LoRD-VI___period512/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa5121LoRD-VI___period512/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc5121LoRD-VI___period512/",],

        # ["piqa","./qa_ckpts/QAAAnewpiqa2561LoRD-VI___period256/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa2561LoRD-VI___period256/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc2561LoRD-VI___period256/",],
        
        # ["piqa","./qa_ckpts/QAAAnewpiqa1281LoRD-VI___period128/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa1281LoRD-VI___period128/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc1281LoRD-VI___period128/",],

        # ["piqa","./qa_ckpts/QAAAnewpiqa641LoRD-VI___period64/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa641LoRD-VI___period64/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc641LoRD-VI___period64/",],

        # ["piqa","./qa_ckpts/QAAAnewpiqa5121vanilla___finally/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa5121vanilla___finally/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc5121vanilla___finally/",],

        # ["piqa","./qa_ckpts/QAAAnewpiqa2561vanilla___finally/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa2561vanilla___finally/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc2561vanilla___finally/",],
        
        # ["piqa","./qa_ckpts/QAAAnewpiqa1281vanilla___finally/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa1281vanilla___finally/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc1281vanilla___finally/",],

        # ["piqa","./qa_ckpts/QAAAnewpiqa641vanilla___finally/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa641vanilla___finally/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc641vanilla___finally/",],




        # ["piqa","./qa_ckpts/QAAAnewpiqa5121LoRD-VI___period512/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa5121LoRD-VI___period512/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc5121LoRD-VI___period512/",],
        # ["piqa","./qa_ckpts/QAAAnewpiqa2561LoRD-VI___period512/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa2561LoRD-VI___period512/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc2561LoRD-VI___period512/",],
        # ["piqa","./qa_ckpts/QAAAnewpiqa1281LoRD-VI___period512/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa1281LoRD-VI___period512/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc1281LoRD-VI___period512/",],
        # ["piqa","./qa_ckpts/QAAAnewpiqa641LoRD-VI___period512/",],
        # ["truthful_qa","./qa_ckpts/QAAAnewtruthful_qa641LoRD-VI___period512/",],
        # ["allenai/ai2_arc","./qa_ckpts/QAAAnewallenai/ai2_arc641LoRD-VI___period512/",],



        ["piqa","./qa_ckpts/QAAAnewpiqa641LoRD-VI___period64/",],
        ["piqa","./qa_ckpts/QAAAnewpiqa641LoRD-VI___period128/",],
        ["piqa","./qa_ckpts/QAAAnewpiqa641LoRD-VI___period256/",],
        ["piqa","./qa_ckpts/QAAAnewpiqa641LoRD-VI___period512/",],

    )

    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    res_dict = {}
    dir_p = "./qa_dataset_res/"
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    for task_ckpt in ckpt_ls:
        task, ckpt = task_ckpt
        if ckpt==base_model_name1:
            base_model_name=None
        else:
            base_model_name=base_model_name1
        res_pth = ckpt + f"___{task}_qa_infer_res"
        res_pth = res_pth.replace("/", "__").replace(".", "")
        res_pth += ".json"
        if not os.path.exists(dir_p + res_pth):
            res_ls = infer_qa(
                ckpt,
                task,
                dir_p + res_pth,
                test_set_take_num=500,
                # test_set_take_num=50,
                # mnt=64,
                mnt=8,
                base_model_name=base_model_name
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

def eval_tau1_res():
    taskls = [
        "piqa",
        # "truthful_qa",
        # "allenai/ai2_arc",
    ]
    mls = [
        "LoRD-VI",
        # "vanilla",
        ]
    train_times = [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    train_nums = [
        "64",
        # "128",
        # "256",
        # "512",
        ]
    tau1ls= [
        "0.70",
        "0.75",
        "0.80",
        "0.85",
        "0.90",
        "0.95",
        "1.0",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./qa_0514_tau1_res/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    res_dict_averaged={}

    for tau1 in tau1ls:
        for task in taskls:
            for train_num in train_nums:
                for m in mls:
                    temp_scorels=[]
                    for itime in train_times:
                        prefix = f"./qa_ckpts/QAAA-TAU1{tau1}"
                        ckpt = (
                            prefix
                            + f"{task}{train_num}{itime}{m}___period{train_num}/"
                        )
                        res_pth = ckpt + f"___{task}_qa_infer_res.json"
                        res_pth = res_pth.replace("/", "__").replace(".", "")

                        if not os.path.exists(dir_p+res_pth):
                            res_ls = infer_qa(ckpt, task, dir_p+res_pth,
                                            test_set_take_num=500,
                                            mnt=8,
                                            base_model_name=base_model_name1,
                                            )
                        else:
                            print(
                                f"{dir_p+res_pth} file already exists. directly loading...")
                            # from collections import OrderedDict
                            with open(dir_p + res_pth, "r", encoding="utf8") as f:
                                res_ls = json.load(
                                    f, object_pairs_hook=OrderedDict)

                        scores = eval_qaacc(task, res_ls)
                        res_dict[task + "-----" + res_pth] = scores
                        temp_scorels.append(scores)

                    # obtain the mean value
                    # obtain the std value
                    temp_scorels=np.array(temp_scorels)
                    meanvaluels=np.mean(temp_scorels,axis=0).tolist()
                    stdvaluels=np.std(temp_scorels,axis=0,ddof=1).tolist()
                    res_dict_averaged[task+"--"+res_pth]=\
                        {"mean": meanvaluels,
                        "std": stdvaluels}

    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    with open(
        dir_p + "OverallScoresAveraged.json",
            "w", encoding="utf8"
    ) as f:
        json.dump(res_dict_averaged, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)
    pprint(res_dict_averaged)


def eval_tau2_res():
    taskls = [
        "piqa",
        # "truthful_qa",
        # "allenai/ai2_arc",
    ]
    mls = [
        "LoRD-VI",
        # "vanilla",
        ]
    train_times = [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    train_nums = [
        "64",
        # "128",
        # "256",
        # "512",
        ]
    tau1ls= [
        "0.80",
        ]
    tau2ls= [
        "0.80",
        "0.85",
        "0.90",
        "0.95",
        "1.0",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./qa_0515_tau2_res/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    res_dict_averaged={}

    for tau1 in tau1ls:
        for tau2 in tau2ls:
            for task in taskls:
                for train_num in train_nums:
                    for m in mls:
                        temp_scorels=[]
                        for itime in train_times:
                            prefix = f"./qa_ckpts/QAAA-TAU1{tau1}TAU2{tau2}"
                            ckpt = (
                                prefix
                                + f"{task}{train_num}{itime}{m}___period512/"
                            )
                            res_pth = ckpt + f"___{task}_qa_infer_res.json"
                            res_pth = res_pth.replace("/", "__").replace(".", "")

                            if not os.path.exists(dir_p+res_pth):
                                res_ls = infer_qa(ckpt, task, dir_p+res_pth,
                                                test_set_take_num=500,
                                                mnt=8,
                                                base_model_name=base_model_name1,
                                                )
                            else:
                                print(
                                    f"{dir_p+res_pth} file already exists. directly loading...")
                                # from collections import OrderedDict
                                with open(dir_p + res_pth, "r", encoding="utf8") as f:
                                    res_ls = json.load(
                                        f, object_pairs_hook=OrderedDict)

                            scores = eval_qaacc(task, res_ls)
                            res_dict[task + "-----" + res_pth] = scores
                            temp_scorels.append(scores)

                        # obtain the mean value
                        # obtain the std value
                        temp_scorels=np.array(temp_scorels)
                        meanvaluels=np.mean(temp_scorels,axis=0).tolist()
                        stdvaluels=np.std(temp_scorels,axis=0,ddof=1).tolist()
                        res_dict_averaged[task+"--"+res_pth]=\
                            {"mean": meanvaluels,
                            "std": stdvaluels}

    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    with open(
        dir_p + "OverallScoresAveraged.json",
            "w", encoding="utf8"
    ) as f:
        json.dump(res_dict_averaged, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)
    pprint(res_dict_averaged)


def eval_varytrainum_res():
    taskls = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc",
    ]
    mls = [
        "LoRD-VI",
        "vanilla",
        ]
    train_times = [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    train_nums = [
        "64",
        "128",
        "256",
        "512",
        ]
    base_model_name1="meta-llama/Meta-Llama-3-8B-Instruct"

    dir_p = "./qa_0513_dataset_res/"
    res_dict = {}

    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    # ===============================================================

    res_dict_averaged={}

    for task in taskls:
        for train_num in train_nums:
            for m in mls:
                temp_scorels=[]
                for itime in train_times:
                    prefix = "./qa_ckpts/QAAAnew"
                    if m=="vanilla":
                        ckpt = (
                            prefix
                            + f"{task}{train_num}{itime}{m}___finally/"
                        )
                    else:
                        ckpt = (
                            prefix
                            + f"{task}{train_num}{itime}{m}___period{train_num}/"
                        )
                    res_pth = ckpt + f"___{task}_qa_infer_res.json"
                    res_pth = res_pth.replace("/", "__").replace(".", "")

                    if not os.path.exists(dir_p+res_pth):
                        res_ls = infer_qa(ckpt, task, dir_p+res_pth,
                                          test_set_take_num=500,
                                          mnt=8,
                                          base_model_name=base_model_name1,
                                          )
                    else:
                        print(
                            f"{dir_p+res_pth} file already exists. directly loading...")
                        # from collections import OrderedDict
                        with open(dir_p + res_pth, "r", encoding="utf8") as f:
                            res_ls = json.load(
                                f, object_pairs_hook=OrderedDict)

                    scores = eval_qaacc(task, res_ls)
                    res_dict[task + "-----" + res_pth] = scores
                    temp_scorels.append(scores)

                # obtain the mean value
                # obtain the std value
                temp_scorels=np.array(temp_scorels)
                meanvaluels=np.mean(temp_scorels,axis=0).tolist()
                stdvaluels=np.std(temp_scorels,axis=0,ddof=1).tolist()
                res_dict_averaged[task+"--"+res_pth]=\
                    {"mean": meanvaluels,
                     "std": stdvaluels}

    with open(
        dir_p + "Overall__qa_varytrain_num_inference_scores.json", "w", encoding="utf8"
    ) as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    with open(
        dir_p + "OverallScoresAveraged.json",
            "w", encoding="utf8"
    ) as f:
        json.dump(res_dict_averaged, f, ensure_ascii=False, indent=4)

    print("OVERALL Save DONE.")
    pprint(res_dict)
    pprint(res_dict_averaged)


def eval_varytrainum_231_ours():
    taskls = [
        "piqa",
        "truthful_qa",
        "allenai/ai2_arc",
    ]
    # taskls = [
    #     "piqa",
    # ]
    # taskls = [
        # "truthful_qa",
    # ]
    taskls = [
        "allenai/ai2_arc",
    ]
    mls = ["LoRD-II"]
    # mls = ["LoRD-IV"]

    # mls = ["google/gemma-2b",]
    train_times = [
        "1",
        "2",
        "3",
    ]
    train_nums = ["4", "8", "16", "32", "64", "100", "256", "512"]
    period_nums = ["8"]

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



def eval_all_samles_in_dir(dirp="./vary_train_num_qa_infers"):
    taskls = [
        "piqa",
        "truthful_qa",
        # "allenai/ai2_arc",
        "allenai__ai2_arc",
    ]

    import os
    fls=os.listdir(dirp)

    res_dict={}

    for fpth in fls:
        fpth=dirp+"/"+fpth
        find_task=None
        for task in taskls:
            if task in fpth:
                find_task=task
                break
        if find_task is None:
            continue
        # if "piqa" not in fpth:
        #     continue
        if "period2" in fpth or "period5" in fpth:
            continue
        # from collections import OrderedDict
        with open(fpth, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)

        if find_task=="allenai__ai2_arc":
            score= eval_qaacc("allenai/ai2_arc", data)
        else:
            score= eval_qaacc(find_task, data)
        res_dict[fpth]=score

    pprint(res_dict)
    with open(dirp+"/"+"res_dict_allfiles.json", 'w',encoding='utf8') as f:
        json.dump(res_dict,f,ensure_ascii=False,indent=4)
    print("Save done.")


def generate_atable(fpth="./vary_train_num_qa_infers/res_dict_allfiles.json",
                    task="allenai__ai2_arc"):
    text="|train-num|repeated-time|method|acc|f1|precision|recall|\n"
    big_ls=[]

    # from collections import OrderedDict
    with open(fpth, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)

    for fname, scorelist in data.items():
        if task not in fname:
            continue
        tt=fname.split(task)[0]
        if "varyTrainNum" not in tt:
            # text+=f"|{train_num}|{repeated_time}|{method}|{acc}|{f1}|{pre}|{rec}|\n"
            continue
        tt=tt.split("varyTrainNum___")[1]
        repeated_time=tt[-1]
        train_num=tt[:-1]
        ttt=fname.split(task)[1]
        if "332164256" in ttt:
            method=ttt.split("332164256")[0]
        else:
            method=ttt.split("112164256")[0]
        acc=round(float(scorelist[0]), 3)
        f1=round(float(scorelist[3]), 3)
        pre=round(float(scorelist[1]), 3)
        rec=round(float(scorelist[2]), 3)

        big_ls.append((train_num,
                       repeated_time,
                       method,
                       acc,f1,pre,rec))
    sorted_ls=sorted(big_ls, key=lambda x: (-float(x[0]),-x[3]))
    for x in sorted_ls:
        text+=f"|{x[0]}|{x[1]}|{x[2]}|{x[3]}|{x[4]}|{x[5]}|{x[6]}|\n"

    print(text)

# running entry
if __name__ == "__main__":
    # main()
    # eval_qa_res()
    # eval_varytrainum_res()
    # eval_varytrainum_231_ours()
    # eval_all_samles_in_dir()
    # generate_atable()
    # eval_tau1_res()
    eval_tau2_res()
    print("EVERYTHING DONE.")
