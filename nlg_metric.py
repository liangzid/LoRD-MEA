"""
======================================================================
NLG_METRIC ---

Some commonly used NLG metrics for generaiton evaluation.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created:  5 March 2024
======================================================================
"""
import torch


def bleu_1to4(hyp_ls, ref_ls):
    p_ls = ref_ls
    gen_p_ls = hyp_ls
    p_ls = [[p] for p in p_ls]
    from bleu4 import corpus_bleu
    res_ls = []

    for n in range(1, 5):
        res = corpus_bleu(gen_p_ls, p_ls, max_n=n)
        res_ls.append(res[0][0])
    return res_ls


def BERTscore(hyps, refs):
    """pip install bert_score"""
    import bert_score as bs
    gens = hyps
    ps = refs
    p, r, f1 = bs.score(gens, ps, lang="en", verbose=True)

    # then average this score into the same one.
    p = torch.mean(p)
    r = torch.mean(r)
    f1 = torch.mean(f1)
    return p, r, f1


def get_rouge_L(hyps, refs):
    """ pip install rouge"""
    from rouge import Rouge
    rouge = Rouge()

    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores["rouge-l"]


def overall_metrics(hyps, refs):
    newhyps = []
    for h in hyps:
        if h == "":
            newhyps.append(" ")
        else:
            newhyps.append(h)
    hyps = newhyps
    refs = list(refs)

    res_dict = {}
    bleures = bleu_1to4(hyps, refs)
    res_dict["bleu"] = {}
    res_dict["bertscore"] = {}
    res_dict["rouge-l"] = {}
    res_dict["bleu"] = {}
    res_dict["bleu"]["1"] = bleures[0]
    res_dict["bleu"]["2"] = bleures[1]
    res_dict["bleu"]["3"] = bleures[2]
    res_dict["bleu"]["4"] = bleures[3]

    bertscore = BERTscore(hyps, refs)
    res_dict["bertscore"]["p"] = float(bertscore[0])
    res_dict["bertscore"]["r"] = float(bertscore[1])
    res_dict["bertscore"]["f1"] = float(bertscore[2])

    rouges = get_rouge_L(hyps, refs)
    res_dict["rouge-l"]["p"] = rouges["p"]
    res_dict["rouge-l"]["r"] = rouges["r"]
    res_dict["rouge-l"]["f1"] = rouges["f"]

    return res_dict


def main():
    gens = ["please do this! Can you do this? yes, I can!",
            "What day is it today?",
            "can you understand me?",
            "A C match B C."]

    ps = ["please do not do this! You cannot do that!",
          "What date is it today?",
          "It is difficult to follow you.",
          "A C match B C.",]
    print(bleu_1to4(gens, ps))
    print(BERTscore(gens, ps))
    print(get_rouge_L(gens, ps))


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
