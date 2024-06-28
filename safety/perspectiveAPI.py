"""
======================================================================
PERSPECTIVEAPI --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 28 June 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
import os
import time

from googleapiclient import discovery
import json

class PerspectiveAPI:
    def __init__(self):
        # API_KEY = 'copy-your-api-key-here'
        API_KEY=os.environ.get("PAI_KEY")
        self.client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        )
        print("Client Load Done.")
        self.times=0

    def predict(self,text):
        analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {},
                                "SEVERE_TOXICITY": {},
                                "IDENTITY_ATTACK": {},
                                "INSULT": {},
                                "PROFANITY":{},
                                "THREAT":{},
                                }
        }
        try:
            response = self.client.comments().analyze(body=analyze_request).execute()
            res=json.dumps(response, indent=2)
            res=json.loads(res)
            value=float(res['attributeScores']["TOXICITY"]["summaryScore"]['value'])

            value_dict={
                "toxicity":float(res['attributeScores']["TOXICITY"]\
                                 ["summaryScore"]['value']),
                "severe_toxicity":float(res['attributeScores']\
                                        ["SEVERE_TOXICITY"]["summaryScore"]\
                                        ['value']),
                "insult":float(res['attributeScores']["INSULT"]\
                               ["summaryScore"]['value']),
                "profanity":float(res['attributeScores']["PROFANITY"]\
                                  ["summaryScore"]['value']),
                "threat":float(res['attributeScores']["THREAT"]\
                               ["summaryScore"]['value']),
                }
            
            print(value)
            print(value_dict)
            print(res)
        except:
            print(f"Error {self.times} times.")
            self.times+=1
            print(f"the incorrect query is: {text}")
            # value=0.
            return -1
        # cls_res=0
        # if value>0.5:
        #     cls_res=1
        # return cls_res
        return value_dict

    def eval_ls(self, text_ls):
        score_dict_ls={
            "toxicity":[],
            "severe_toxicity":[],
            "insult":[],
            "profanity":[],
            "threat":[],
            }
        for text in text_ls:
            res_dict=self.predict(text)
            for key in score_dict_ls:
                score_dict_ls[key].append(res_dict[key])
            time.sleep(1.3)
        # obtain the averaged value of the results.

        score_mean_dict={}
        for key in score_dict_ls:
            score_mean_dict[key]=sum(score_dict_ls[key])\
                /len(score_dict_ls[key])
        return score_mean_dict, score_dict_ls


## running entry
if __name__=="__main__":
    persAPI=PerspectiveAPI()
    res=persAPI.predict("I don't like you!")
    print()
    print("EVERYTHING DONE.")

