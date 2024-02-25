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
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from training_data_collecting_openai import load_raw_train_datals


def clip(tnsr, epsilon):
    tnsr=torch.min(tnsr, 1+epsilon)
    tnsr=torch.max(tnsr, 1-epsilon)
    return tnsr

def train_one_period(lm, vmodel,
                     lm_tokenizer,
                     loader, epoch, device,
                     tb_writer,
                     save_path,
                     LR=3e-5,
                     acc_step=1,
                     log_step=100,
                     save_step=1000,
                     lambda1=0.7,
                     lambda2=0.7,
                     ):
    overall_loss=0.
    overall_step=0

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)
    opt2 = torch.optim.AdamW(vmodel.parameters(), lr=LR)
    for e in tqdm(epoch,desc="epoch"):
        for item in tqdm(loader,,desc="loader"):
            overall_step+=1
            loss_clip=0.
            loss_vfunc=0.
            loss_entropy=0.

            inps_idxs, reward, old_logits, A, V=item

            inps_idxs=inps_idxs.to(device) # bs, sql
            reward=reward.to(device) # bs, sql
            old_logits=old_logits.to(device) # bs, sql, Vocab
            A=A.to(device) # bs, sql
            V=V.to(device) # bs, sql

            bs, sqlen=inps_idxs.shape

            logits=lm(inps_idxs).logits[:, :-1, :]

            convince_gen=new_prob_gen/prob_gen

            loss_clip = torch.sum(torch.min(
                new_prob_gen*A,
                clip(new_prob_gen)*A))

            values=vmodel(inps_idxs).logits
            loss_vfunc = torch.sum((values-V)**2)

            entropy=torch.sum(Categorical(logits).entropy())
            loss_entropy = (-1)*entropy

            overall_loss += loss_clip + lambda1*loss_vfunc\
                + lambda2*loss_entropy

            if overall_step % acc_step == 0:
                opt1.backward()
                opt2.backward()
                
                overall_loss.backward()
                opt1.step()
                opt2.step()
            if overall_step % log_step ==0:
                print(" LOSS: {}\tCLIP: {}\tV: {}\tEntropy: {}".format(
                    overall_loss, loss_clip, loss_vfunc, loss_entropy
                    ))
            if overall_loss % save_step==0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                tokenizer.save_pretrained(save_path+"___"+overall_step)
                model.save_pretrained(save_path+"___"+overall_step)

    print(" -->Finally Saving.")
    tokenizer.save_pretrained(save_path+"___STEPfinally")
    model.save_pretrained(save_path+"___STEPfinally")

    print("ONE PERIOD TRAINING DONE!")
                
def ___V_target_compute(reward, lambdaa=0.95):
    """
    shape of `reward`: bs, sql
    """
    bs, sql=reward.shape
    window_size=150
    V=torch.zero((bs, sql))

    for i in range(sql):
        for j in range(i, min(i+window_size, sql)):
            V[:,i]+=reward[:,j]*(lambdaa**(j-1))
    
    return V


def train_pod(lm, vmodel, rewardmodel,
              lm_tokenizer,
              args, raw_train_datals):
    ## STEP 1: DATA Preperation.
    rewardls=None
    ITER_num=args.period_num
    tb_writer=SummaryWriter(log_dir=args.save_path+"___log_writer")
    for iter_idx in range(ITER_num):
        old_logitsls=[]
        Als=[]
        Vls=[]
        ## collect data.
        with torch.no_grad():
            if inps_idxs_ls is None:
                for inps_idxs in raw_train_datals:
                    inps_idxs=inps_idxs.to(args.device)
                    reward=rewardmodel(inps_idxs[:,:-1]).logits
                    rewardls.append(reward)

                    old_logits=lm(inps_idxs[:,:-1]).logits
                    V=___V_target_compute(reward, lambdaa=args.lambdaa)
                    A=V-vmodel(inps_idxs[:,:-1]).logits
                    old_logitsls.append(old_logits)
                    Als.append(A)
                    Vls.append(V)
            else:
                rewardmodel=None
                for i, inps_idxs in enumerate(raw_train_datals):
                    inps_idxs=inps_idxs.to(args.device)
                    reward=rewardls[i]

                    old_logits=lm(inps_idxs[:,:-1]).logits
                    V=___V_target_compute(reward, lambdaa=args.lambdaa)
                    A=V-vmodel(inps_idxs[:,:-1]).logits
                    old_logitsls.append(old_logits)
                    Als.append(A)
                    Vls.append(V)
        inpsls=TensorDataset(raw_train_datals)
        rewardls=TensorDataset(rewardls)
        old_logitsls=TensorDataset(old_logitsls)
        Als=TensorDataset(Als)
        Vls=TensorDataset(Vls)
        trainset=torch.utils.data.ConcatDataset([inpsls,rewardls,
                                                 old_logitsls,
                                                 Als,Vls,])
        loader=DataLoader(trainset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          )
        ## STEP 2: Train the Model in a period
        lm,vmodel= train_one_period(lm, vmodel,
                                    lm_tokenizer,
                                    loader,
                                    args.epoch, args.device,
                                    tb_writer, args.save_path, args.LR,
                                    args.acc_step, args.log_step,
                                    args.save_step)
            
        print(" -->NOW save the ckpt in each period.")
        print(f"in period {iter_idx}.")
        tokenizer.save_pretrained(args.save_path+"___period"+iter_idx)
        lm.save_pretrained(args.save_path+"___period"+iter_idx)

    print(" -->ALL TRAINING DONE.")
    tokenizer.save_pretrained(args.save_path+"___finally")
    lm.save_pretrained(args.save_path+"___finally")
    print(" -->Save DONE.")

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:3", type=str,
                        required=False)
    parser.add_argument('--epoch', default=2, type=int,
                        required=False)
    parser.add_argument('--period_num', default=3, type=int,
                        required=False)
    parser.add_argument('--acc_step', default=4, type=int,
                        required=False)
    parser.add_argument('--log_step', default=100, type=int,
                        required=False)
    parser.add_argument('--save_step', default=10000, type=int,
                        required=False)


    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False)
    parser.add_argument('--lambdaa', default=0.95, type=float,
                        required=False)
    parser.add_argument('--lambdaa', default=0.95, type=float,
                        required=False)
    parser.add_argument('--max_length', default=1024, type=int,
                        required=False)

    parser.add_argument('--batch_size', default=1, type=int,
                        required=False)
    parser.add_argument('--task', default="pod", type=str,
                        required=False,)
    parser.add_argument("--max_seq_length", default=1024,
                        type=int, required=False)

    parser.add_argument('--from_path', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--save_path',
                        default='model_training_results',
                        type=str, required=True,)
    parser.add_argument('--v_save_path',
                        default='value_model_training_results',
                        type=str, required=True,)
    parser.add_argument('--v_from_path',
                        default='model_training_results',
                        type=str, required=True,)

    return parser.parse_args()

def main():

    args=setup_train_args()
    
    lm=AutoModelForCausalLM.from_pretrained(
        args.from_path,
        device_map="auto",
        )
    lm_tokenizer=AutoTokenizer.from_pretrained(args.from_path)

    rewardmodel=AutoModelForSequenceClassification.from_pretrained(
        args.v_from_path,
        device_map="auto",
        )

    vmodel=AutoModelForSequenceClassification.from_pretrained(
        args.v_from_path,
        device_map="auto",
        )
    # vtokenizer=AutoTokenizer.from_pretrained(args.v_from_path)

    raw_train_datals=load_raw_train_datals(lm_tokenizer, args.max_length)

    train_pod(
        lm, vmodel, rewardmodel,
        lm_tokenizer,
        args, raw_train_datals)

    print("EVERYTHING in the TRAINING now DONE.")


if __name__=="__main__":
    main()
