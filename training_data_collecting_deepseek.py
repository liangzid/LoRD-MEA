"""
======================================================================
TRAINING_DATA_COLLECTING_DEEPSEEK --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created:  2 April 2025
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from openai import OpenAI
import json
import os

apikey=os.environ["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key=apikey,
    base_url="https://api.deepseek.com",
)

def onetimequery(
    system_prompt,
    user_prompt,
):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    adict = response.choices[0].message.content
    print(adict)
    return adict

# deepseek-reasoner
def chatWithDS__LogLogits(modelname="deepseek-chat",
                              messages=[],
                              num_top_logprobs=5):

    res = client.chat.completions.create(
        model=modelname,
        messages=messages,
        logprobs=True,
        top_logprobs=num_top_logprobs,
    )
    # print("Inference Results: ",res)
    generated_text = res.choices[0].message.content
    logprobs = res.choices[0].logprobs.content

    # print("-----------------------")

    return generated_text, logprobs





