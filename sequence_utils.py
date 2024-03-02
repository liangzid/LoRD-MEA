"""
======================================================================
SEQUENCE_UTILS ---

Utils for the sequence processing.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 28 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import torch
from typing import List


def my_padding(ts_ls:List[torch.tensor], msl, pad_idx,):
    num=len(ts_ls)
    target_tensor=(torch.ones((num, msl), dtype=torch.long)*pad_idx).to("cpu")
    mask_tensor=torch.zeros((num, msl)).to("cpu")
    for i, ts in enumerate(ts_ls):
        end_idx=min(msl, len(ts))
        target_tensor[i, :end_idx]=torch.tensor(ts[:end_idx],
                                                dtype=torch.long)
        mask_tensor[i, :end_idx]=torch.ones_like(target_tensor[i,
                                                               :end_idx])
    return target_tensor, mask_tensor

def my_padding_token_dist(ts_ls:List[torch.tensor], msl, pad_idx,):
    num=len(ts_ls)
    candidate_num=len(ts_ls[0][0])
    target_tensor=(torch.ones((num, msl,candidate_num), dtype=torch.long)*pad_idx).to("cpu")
    # mask_tensor=torch.zeros((num, msl, candidate_num)).to("cpu")
    for i, ts in enumerate(ts_ls):
        end_idx=min(msl, len(ts))
        target_tensor[i, :end_idx]=torch.tensor(ts[:end_idx],
                                                dtype=torch.long)
        # mask_tensor[i, :end_idx]=torch.ones_like(ts[:end_idx])
    return target_tensor

def my_padding_logits(ts_lss:List[torch.tensor], msl, pad_idx,):
    num=len(ts_lss)
    V=ts_lss[0].shape[1]
    target_tensor=(torch.ones((num, msl, V),
                             dtype=torch.float)*1/V).to("cpu")
    # mask_tensor=torch.zeros((num, msl, V)).to("cpu")
    for i, ts in enumerate(ts_lss):
        end_idx=min(msl, len(ts))
        # target_tensor[i, :end_idx]=ts[:end_idx]
    return target_tensor

def my_padding_logit(ts_lss:List[torch.tensor], msl, pad_idx,):
    num=len(ts_lss)
    sl=ts_lss[0].shape[0]
    V=25600
    target_tensor=(torch.ones((num, msl),
                              dtype=torch.float)*(1/V)).to("cpu")
    for i, ts in enumerate(ts_lss):
        end_idx=min(msl, len(ts))
        target_tensor[i, :end_idx]=ts[:end_idx]
    return target_tensor






## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


