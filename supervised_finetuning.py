"""
======================================================================
SUPERVISED_FINETUNING ---

SFT code, which is quite simpler to the original version.

Reference: https://huggingface.co/google/gemma-7b/blob/main/examples/example_sft_qlora.py

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 23 February 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
from dataclasses import dataclass, field
from typing import Optional
from pprint import pprint
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
hf_token=os.environ["HF_TOKEN"]

import torch

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        default="google/gemma-7b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="stingning/ultrachat",
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={
                           "help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={
                                "help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=1000, metadata={
                            "help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=500, metadata={
                               "help": "Log every X updates steps."})
    output_dir: str = field(
        default="./sft__results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."},
    )


def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_id = script_args.model_name

    quantization_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    if not script_args.use_flash_attention_2:
        attn_implementation = "sdpa"
    else:
        attn_implementation = "flash_attention_2"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
        attn_implementation=attn_implementation,
        token=hf_token,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=script_args.lora_r,
        target_modules=["q_proj", "o_proj", "k_proj",
                        "v_proj", "gate_proj", "up_proj",
                        "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout
    )

    print("=====LORA ARGS=====")
    pprint(lora_config)
    print("=====Quantization ARGS=====")
    pprint(quantization_config)
    print("=====SCRIPT ARGS=====")
    pprint(script_args)

    dataset_name = script_args.dataset_name
    train_dataset = load_dataset(script_args.dataset_name,
                                 split="train[:1%]")

    # TODO: make that configurable
    YOUR_HF_USERNAME = "liangzid"
    output_dir = f"{YOUR_HF_USERNAME}/gemma-qlora-{dataset_name}"
    output_dir = script_args.output_dir

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_checkpointing=script_args.gradient_checkpointing,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        peft_config=lora_config,
        packing=script_args.packing,
        dataset_text_field="id",
        tokenizer=tokenizer,
        max_seq_length=script_args.max_seq_length,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(output_dir)


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
