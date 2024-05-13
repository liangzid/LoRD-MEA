"""
======================================================================
ARRANGE_TABLE_DATA ---

Reformat the data of tables.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 13 May 2024
======================================================================
"""
# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
from collections import OrderedDict


def process_wmt_data(fname):
    # from collections import OrderedDict
    with open(fname, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)

    key_ls=[]
    example_dict=data[list(data.keys())[0]]
    for key in example_dict:
        for key2 in example_dict[key]:
            key_ls.append((key,key2))
    print(key_ls)
    flatten_dict={}
    for key in data:
        adict=data[key]
        als=[]
        for k1,k2 in key_ls:
            als.append(round(adict[k1][k2],3))
        flatten_dict[key]=als

    org_table_text=""
    for fname, value_ls in flatten_dict.items():
        org_table_text+=f"|{fname}|"
        value_ls=[str(x) for x in value_ls]
        str_ls="|".join(value_ls)
        org_table_text+=str_ls+"| \n"
    print(f"ORG TABLE TEXT:\n {org_table_text}")
    return org_table_text




if __name__=="__main__":
    process_wmt_data("./wmt16_res/wmt_inference_scores_overall.json")
