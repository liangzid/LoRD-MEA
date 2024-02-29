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
    for i, ts in enumerate(ts_ls):
        end_idx=min(msl, len(ts))
        target_tensor[i, :end_idx]=torch.tensor(ts[:end_idx],
                                                dtype=torch.long)
    return target_tensor

def my_padding_logits(ts_lss:List[torch.tensor], msl, pad_idx,):
    num=len(ts_lss)
    V=ts_lss[0].shape[1]
    target_tensor=(torch.ones((num, msl, V),
                             dtype=torch.float)*1/V).to("cpu")
    for i, ts in enumerate(ts_lss):
        end_idx=min(msl, len(ts))
        target_tensor[i, :end_idx]=ts[:end_idx]
    return target_tensor







## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


