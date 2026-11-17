#!/usr/bin/env python3
"""
Evaluate base microsoft/phi-2 vs QLoRA-adapted phi-2 on the held-out eval set.
Computes ROUGE-L and keyword-match accuracy.

Usage:
    python finetune/evaluate.py
    # Requires finetune/checkpoints/phi2-qlora/ to exist.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate as hf_evaluate

BASE_MODEL = "microsoft/phi-2"
ADAPTED_DIR = Path(__file__).parent / "checkpoints" / "phi2-qlora"
EVAL_FILE = Path(__file__).parent / "dataset_eval.jsonl"

CATEGORY_KEYWORDS = [
    "VPN", "Remote Access", "Password", "Account Lockout",
    "Hardware", "Software", "Crash", "Network", "Outage",
    "Printer", "Email", "Security", "Phishing",
]

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def format_prompt(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n"
    )

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def keyword_accuracy(predictions: list[str], references: list[str]) -> float:
    correct = 0
    for pred, ref in zip(predictions, references):
        for kw in CATEGORY_KEYWORDS:
            if kw.lower() in ref.lower() and kw.lower() in pred.lower():
                correct += 1
                break
    return correct / len(predictions) if predictions else 0.0

def evaluate_model(model, tokenizer, examples: list[dict]) -> dict:
    rouge = hf_evaluate.load("rouge")
    predictions, references = [], []
    for ex in examples:
        prompt = format_prompt(ex)
        pred = generate(model, tokenizer, prompt)
        predictions.append(pred)
        references.append(ex["output"])
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    return {
        "rouge_l": round(rouge_scores["rougeL"], 4),
        "keyword_accuracy": round(keyword_accuracy(predictions, references), 4),
        "n_examples": len(examples),
    }

def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    examples = load_jsonl(EVAL_FILE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    print("Evaluating base model ...")
    base_results = evaluate_model(base_model, tokenizer, examples)
    print(f"Base model: {base_results}")
    del base_model
    torch.cuda.empty_cache()

    if ADAPTED_DIR.exists():
        print("Loading adapted model ...")
        adapted_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        adapted_model = PeftModel.from_pretrained(adapted_model, str(ADAPTED_DIR))
        print("Evaluating adapted model ...")
        adapted_results = evaluate_model(adapted_model, tokenizer, examples)
        print(f"Adapted model: {adapted_results}")
    else:
        print(f"Checkpoint not found at {ADAPTED_DIR}. Run train.py first.")
        adapted_results = None

    print("\n--- RESULTS SUMMARY ---")
    print(f"Base phi-2:    ROUGE-L={base_results['rouge_l']:.4f}  KeywordAcc={base_results['keyword_accuracy']:.4f}")
    if adapted_results:
        print(f"Adapted phi-2: ROUGE-L={adapted_results['rouge_l']:.4f}  KeywordAcc={adapted_results['keyword_accuracy']:.4f}")
    else:
        print("Adapted phi-2: UNEXECUTED — checkpoint not found")

if __name__ == "__main__":
    main()
