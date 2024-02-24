"""
======================================================================
RLHF_TRAIN ---

Training Language Models with Self-designed Training Loss Function.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 11 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import torch
from tqdm import tqdm



def clip(tnsr, epsilon):
    tnsr=torch.min(tnsr, 1+epsilon)
    tnsr=torch.max(tnsr, 1-epsilon)
    return tnsr

def train_one_period(lm, loader, epoch, device):
    loss_clip=0.
    loss_vfunc=0.
    loss_entropy=0.
    for e in tqdm(epoch,desc="epoch"):
        for item in tqdm(loader,,desc="loader"):

            inps_idxs, gen_idx, reward, prob_gen, A, V=item

            inps_idxs=inps_idxs.to(device)
            gen_idx=gen_idx.to(device)
            reward=reward.to(device)
            old_logits=old_logits.to(device)
            A=A.to(device)
            V=V.to(device)

            bs, sqlen=inps_idxs.shape

            logits=lm(inps_idxs)[:, sqlen-1, :]
            new_prob_gen=logits[:,gen_idx]

            convince_gen=new_prob_gen/prob_gen

            loss_clip += torch.min(
                new_prob_gen*A,
                clip(new_prob_gen)*A)

            loss_vfunc=
        
        


        
    


def train():
    pass

def main():
    pass




















if __name__=="__main__":
    main()
