import json
import pytest
from pathlib import Path

TRAIN_FILE = Path(__file__).parent.parent / "finetune" / "dataset_train.jsonl"
EVAL_FILE = Path(__file__).parent.parent / "finetune" / "dataset_eval.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    # utf-8-sig handles files with or without a UTF-8 BOM (common on Windows)
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_train_file_exists():
    assert TRAIN_FILE.exists(), f"Training file not found: {TRAIN_FILE}"


def test_eval_file_exists():
    assert EVAL_FILE.exists(), f"Eval file not found: {EVAL_FILE}"


def test_train_set_has_minimum_examples():
    examples = load_jsonl(TRAIN_FILE)
    assert len(examples) >= 80, f"Expected >= 80 training examples, got {len(examples)}"


def test_eval_set_has_minimum_examples():
    examples = load_jsonl(EVAL_FILE)
    assert len(examples) >= 20, f"Expected >= 20 eval examples, got {len(examples)}"


def test_train_examples_have_required_keys():
    examples = load_jsonl(TRAIN_FILE)
    for i, ex in enumerate(examples):
        assert "instruction" in ex, f"Example {i} missing 'instruction'"
        assert "input" in ex, f"Example {i} missing 'input'"
        assert "output" in ex, f"Example {i} missing 'output'"


def test_eval_examples_have_required_keys():
    examples = load_jsonl(EVAL_FILE)
    for i, ex in enumerate(examples):
        assert "instruction" in ex, f"Example {i} missing 'instruction'"
        assert "input" in ex, f"Example {i} missing 'input'"
        assert "output" in ex, f"Example {i} missing 'output'"


def test_no_empty_fields_in_train():
    examples = load_jsonl(TRAIN_FILE)
    for i, ex in enumerate(examples):
        assert ex["instruction"].strip(), f"Example {i}: empty instruction"
        assert ex["input"].strip(), f"Example {i}: empty input"
        assert ex["output"].strip(), f"Example {i}: empty output"


def test_no_empty_fields_in_eval():
    examples = load_jsonl(EVAL_FILE)
    for i, ex in enumerate(examples):
        assert ex["instruction"].strip(), f"Example {i}: empty instruction"
        assert ex["input"].strip(), f"Example {i}: empty input"
        assert ex["output"].strip(), f"Example {i}: empty output"


def test_train_covers_vpn_category():
    examples = load_jsonl(TRAIN_FILE)
    vpn_examples = [
        e for e in examples
        if "vpn" in e["output"].lower() or "remote access" in e["output"].lower()
    ]
    assert len(vpn_examples) >= 5, "Expected at least 5 VPN examples in training set"


def test_eval_covers_multiple_categories():
    examples = load_jsonl(EVAL_FILE)
    categories_found = set()
    keywords = {
        "VPN": "vpn",
        "Password": "password",
        "Hardware": "hardware",
        "Network": "network",
        "Printer": "printer",
        "Email": "email",
    }
    for ex in examples:
        for cat, kw in keywords.items():
            if kw in ex["output"].lower():
                categories_found.add(cat)
    assert len(categories_found) >= 4, (
        f"Expected >= 4 categories in eval set, found: {categories_found}"
    )
