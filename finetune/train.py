#!/usr/bin/env python3
"""
QLoRA fine-tuning of microsoft/phi-2 for IT helpdesk intent classification.

Requirements: transformers, peft, bitsandbytes, datasets, torch
Hardware: Minimum 24GB VRAM recommended (runs on 48GB A6000/A100).

Usage:
    python finetune/train.py
    # Checkpoint saved to finetune/checkpoints/phi2-qlora/
"""
import json
from pathlib import Path
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

BASE_MODEL = "microsoft/phi-2"
TRAIN_FILE = Path(__file__).parent / "dataset_train.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "phi2-qlora"

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def format_example(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- QLoRA config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config for phi-2 ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    raw = load_jsonl(TRAIN_FILE)
    formatted = [format_example(e) for e in raw]
    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    dataset = Dataset.from_dict(tokenized)

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nModel saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
