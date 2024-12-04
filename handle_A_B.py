"""
======================================================================
HANDLE_A_B --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  4 December 2024
======================================================================
"""

from collections import OrderedDict
import json


def handle():
    # path = "./save_path_temperature0.8_query256_internaldata_directprompt.json"
    path='./save_path_temperature1.3_query256_internaldata_directprompt.json'

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    judgels = data["judgels"]

    num_A = 0.0

    for x in judgels:
        if x == "A":
            num_A += 1

    print("A's num:", num_A)
    print("total num:", len(judgels))
    print("Rate: ", num_A / len(judgels))


if __name__ == "__main__":
    handle()
