#!/bin/bash
######################################################################
#EAI_HARNESS_EVALUATE ---

# Evaluate LLMs with E-AI's LM HARNESS

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: 26 February 2024
######################################################################

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align/bin/python3
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export all_t=("siqa" "xnli" "xstorycloze" "hendrycks_ethics" "gpqa"\
		      "mc_taco" "glue" "csatqa" "sciq" "xcopa" "wmt2016"\
		      "race" "ceval" "nq_open" "cmmlu" "truthfulqa"\
		      "lambada_cloze" "crows_pairs" "prost" "paws-x"\
		      "swag" "translation" "bbh" "scrolls" "code_x_glue"\
		      "mathqa" "kobest" "pile" "benchmarks" "gsm8k"\
		      "mgsm" "belebele" "anli" "polemo2" "arc" "okapi"\
		      "wsc273" "asdiv" "logiqa2" "winogrande" "webqs"\
		      "toxigen" "triviaqa" "pubmedqa" "storycloze"\
		      "lambada" "xwinograd" "super_glue" "haerae"\
		      "piqa" "lambada_multilingual" "medmcqa" "blimp"\
		      "squadv2" "wikitext" "babi" "minerva_math" "coqa"\
		      "fld" "qa4mre"  "drop"  "realtoxicityprompts"\
		      "openbookqa" "mutual" "medqa" "hellaswag"\
		      "bigbench" "model_written_evals" "qasper"\
		      "unscramble" "ifeval" "mmlu" "logiqa" "headqa"\
		      "arithmetic" "kmmlu")

# export model_path="./POD_SAVE_CKPTs/TheFirstTimeAttempts/policy-___period4"
export model_path="google/gemma-2b"
# export model_path="./SFT_SAVE_CKPTs/TheFirstTimeAttempts/policy-___STEPfinally"
# export tasks="mmlu"
# export tasks="toxigen"
export tasks="ethics_cm,ethics_deontology,ethics_justice,ethics_utilitarianism,ethics_virtue,toxigen"
export output_path="${model_path}___inference_results"
export device="cuda:1"

$python -m lm_eval \
	--model hf \
	--model_args \
	pretrained=$model_path \
	--tasks=$tasks \
	--limit 10 \
	--num_fewshot 1 \
	--write_out \
	--device $device \
	--batch_size auto \
	--output_path $output_path


echo "RUNNING eai_harness_evaluate.sh DONE."
# eai_harness_evaluate.sh ends here
