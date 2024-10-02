"""
======================================================================
DOMAIN_SPECIFIC_VICTIM_PROCESS --- 

A Temporal script.

    Author: Anonymous authors
    Copyright © 2024, Anonymous, all rights reserved.
    Created: 27 May 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

from text2sql_process import eval_varying_train_num as eval1
from data2text_process import eval_varying_train_num as eval2
from sum_process import eval_varying_train_num as eval3
import time

def main():
    eval1()
    print("Now sleep 30 minutes.")
    time.sleep(30*60)
    eval2()
    print("Now sleep 30 minutes.")
    time.sleep(30*60)
    eval3()
    print("Now DONE.")

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


