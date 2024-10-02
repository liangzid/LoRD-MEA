"""
======================================================================
PY2_0_1_TEST_LOADING_LORA ---

Debug the code of loading LoRA Checkpoints.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created: 17 April 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
# import json
# from typing import List, Tuple, Dict
# import random
# from pprint import pprint as ppp

# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer


import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,0"

from qa_process import eval_qaacc, infer_qa


def main():
    # prefix = "./LoRA-LoRD-ckpts/"
    ckpt = "LoRA-LoRD-ckptsvaryTrainNum___41allenai/ai2_arcvanilla122164256___finally/"

    res_pth = ckpt + "test_qa_infer_res.json"

    res_pth = res_pth.replace("/", "__").replace(".", "")

    base_model = "google/gemma-7b"

    res_ls = infer_qa(
        ckpt,
        "allenai/ai2_arc",
        res_pth,
        test_set_take_num=10,
        mnt=64,
        base_model_name=base_model,
    )

    scores = eval_qaacc("allenai/ai2_arc", res_ls)
    print(scores)


if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
