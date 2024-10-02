"""
======================================================================
BLEU4 ---

From https://github.com/zhyack/BLEU4Python/blob/master/bleu.py
All rights belong to the original author

    Author: Anonymous authors
    Copyright © 2023, Anonymous, all rights reserved.
    Created: 17 November 2023
======================================================================
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp


import math
import copy

# From https://github.com/zhyack/BLEU4Python/blob/master/bleu.py
# All rights belong to the original author


def bleu_count(hypothesis, references, max_n=4):
    ret_len_hyp = 0
    ret_len_ref = 0
    ret_clip_count = [0]*max_n
    ret_count = [0]*max_n
    for m in range(len(hypothesis)):
        hyp, ref = hypothesis[m], references[m]
        x = hyp.split()
        y = [r.split() for r in ref]
        x_len = len(x)
        y_len = [len(s) for s in y]
        n_ref = len(ref)

        closest_diff = 9999
        closest_length = 9999
        ref_ngram = dict()

        for i in range(n_ref):
            diff = abs(y_len[i]-x_len)
            if diff < closest_diff:
                closest_diff = diff
                closest_length = y_len[i]
            elif diff == closest_diff and y_len[i] < closest_length:
                closest_length = y_len[i]

            for n in range(max_n):
                sent_ngram = dict()
                for st in range(0, y_len[i]-n):
                    ngram = "%d" % (n+1)
                    for k in range(n+1):
                        j = st+k
                        ngram += " %s" % (y[i][j])
                    if ngram not in sent_ngram:
                        sent_ngram[ngram] = 0
                    sent_ngram[ngram] += 1
                for ngram in sent_ngram.keys():
                    if ngram not in ref_ngram or ref_ngram[ngram] < sent_ngram[ngram]:
                        ref_ngram[ngram] = sent_ngram[ngram]

        ret_len_hyp += x_len
        ret_len_ref += closest_length

        for n in range(max_n):
            hyp_ngram = dict()
            for st in range(0, x_len-n):
                ngram = "%d" % (n+1)
                for k in range(n+1):
                    j = st+k
                    ngram += " %s" % (x[j])
                if ngram not in hyp_ngram:
                    hyp_ngram[ngram] = 0
                hyp_ngram[ngram] += 1
            for ngram in hyp_ngram.keys():
                if ngram in ref_ngram:
                    ret_clip_count[n] += min(ref_ngram[ngram],
                                             hyp_ngram[ngram])
                ret_count[n] += hyp_ngram[ngram]

    return ret_clip_count, ret_count, ret_len_hyp, ret_len_ref

# From https://github.com/zhyack/BLEU4Python/blob/master/bleu.py
# All rights belong to the original author


def corpus_bleu(hypothesis, references, max_n=4):
    assert (len(hypothesis) == len(references))
    clip_count, count, total_len_hyp, total_len_ref = bleu_count(
        hypothesis, references, max_n=max_n)
    brevity_penalty = 1.0
    bleu_scores = []
    bleu = 0
    for n in range(max_n):
        if count[n] > 0:
            bleu_scores.append(clip_count[n]/count[n])
        else:
            bleu_scores.append(0)
    if total_len_hyp < total_len_ref:
        if total_len_hyp == 0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = math.exp(1 - total_len_ref/total_len_hyp)

    def my_log(x):
        if x == 0:
            return -9999999999.0
        elif x < 0:
            raise Exception("Value Error")
        return math.log(x)
    log_bleu = 0.0
    for n in range(max_n):
        log_bleu += my_log(bleu_scores[n])
    bleu = brevity_penalty*math.exp(log_bleu / float(max_n))
    return [bleu]+bleu_scores, [brevity_penalty, total_len_hyp/total_len_ref, total_len_hyp, total_len_ref]


# running entry
if __name__ == "__main__":
    print("EVERYTHING DONE.")
