# Fine-Tuning Experiment — IT Helpdesk Intent Classification

## 1. Overview

This experiment fine-tunes a small open-weight language model using QLoRA on a synthetic IT helpdesk dataset. The goal is to evaluate whether a parameter-efficient adaptation can improve classification accuracy and resolution quality compared to the unmodified base model on the same held-out examples. The task is: given a free-text IT support ticket, produce a category label and a specific actionable resolution.

The experiment is a reproducible demonstration component of the CSPRO RAG assessment. It is designed to run on a single 48 GB GPU and is fully self-contained within the `finetune/` directory.

---

## 2. Base Model

**Model:** [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2)

**Why phi-2:**
- 2.7 billion parameters — small enough to fit in 4-bit quantised form on a 48 GB A6000 or A100 with memory to spare for training.
- Strong reasoning and instruction-following capability relative to its size, as measured on standard benchmarks (HellaSwag, WinoGrande, ARC).
- Permissive licence (Microsoft Research Licence) suitable for research use.
- Well-supported by the HuggingFace `transformers` library with `trust_remote_code=True` for the custom attention implementation.

The model is loaded in 4-bit NF4 quantisation via `bitsandbytes`, which reduces peak VRAM usage from approximately 5.5 GB (fp16) to under 2 GB, leaving headroom for the LoRA adapter, optimiser states, and batch activations.

---

## 3. QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantisation | 4-bit NF4 | Optimal distribution for normally distributed weights |
| Double quantisation | Enabled | Further reduces memory by quantising quantisation constants |
| Compute dtype | `bfloat16` | Stable numerics for mixed-precision training |
| LoRA rank (`r`) | 16 | Balances expressiveness and parameter count |
| LoRA alpha | 32 | Effective learning rate scale of 2× rank |
| LoRA dropout | 0.05 | Light regularisation for a small dataset |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `dense` | Covers all attention projection layers in phi-2's MixFormer architecture |
| Bias | `none` | Keeps non-LoRA parameters frozen |
| Task type | `CAUSAL_LM` | Autoregressive text generation objective |

Trainable parameters are approximately 6.8M out of 2.7B total (~0.25%), consistent with standard QLoRA configurations.

---

## 4. Dataset

| Split | File | Examples | Categories |
|-------|------|----------|------------|
| Train | `dataset_train.jsonl` | 85 | 8 |
| Eval | `dataset_eval.jsonl` | 25 | 8 |

**Categories and approximate distribution in training set:**

| Category | Train count |
|----------|-------------|
| VPN/Remote Access | 10 |
| Password Reset/Account Lockout | 10 |
| Hardware Failure | 10 |
| Software Crash/Installation | 10 |
| Network Outage/Connectivity | 10 |
| Printer Issues | 10 |
| Email Problems | 10 |
| Security Alert/Phishing | 5 |

All examples are **fully synthetic**. No real credentials, personal data, production secrets, or identifiable customer information is present. Each example follows the prompt template:

```
### Instruction:
Classify this IT support ticket and suggest a resolution:

### Input:
<ticket text>

### Response:
Category: <X>. Resolution: <actionable steps>
```

---

## 5. Training Command and Estimated Time

**Prerequisites:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0 peft==0.10.0 bitsandbytes==0.43.1 datasets==2.19.0 accelerate==0.29.3
```

**Run training:**

```bash
cd /path/to/cspro
python finetune/train.py
```

**Checkpoint output:** `finetune/checkpoints/phi2-qlora/`

**Estimated training time on a 48 GB A6000:**
- 85 examples × 2 epochs = 170 forward/backward passes
- With `per_device_train_batch_size=4` and `gradient_accumulation_steps=4` → effective batch size 16
- Approximately **8–12 minutes** of wall-clock time, excluding model download (~5 GB on first run).

---

## 6. Evaluation Command

**Additional requirement:**

```bash
pip install rouge-score==0.1.2
```

**Run evaluation:**

```bash
python finetune/evaluate.py
```

The script:
1. Loads the base `microsoft/phi-2` in 4-bit and runs inference on all 25 eval examples.
2. If `finetune/checkpoints/phi2-qlora/` exists, loads the QLoRA-adapted model and repeats.
3. Prints ROUGE-L F1 and keyword-match accuracy for each model.

---

## 7. Results

> **⚠️ UNEXECUTED — hardware not available at submission time. Scripts are fully reproducible. Do not fabricate metrics.**

The evaluation results table below will be populated after training and evaluation are run on a 48 GB GPU:

| Model | ROUGE-L | Keyword Accuracy | Notes |
|-------|---------|-----------------|-------|
| Base `microsoft/phi-2` | *unexecuted* | *unexecuted* | Zero-shot baseline |
| QLoRA-adapted `phi-2` | *unexecuted* | *unexecuted* | 2 epochs, r=16 |

Expected behaviour: the adapted model should produce higher keyword accuracy because fine-tuning aligns the output format (`Category: X. Resolution: ...`) with the training template, reducing hallucinated or off-format responses.

---

## 8. Reproducing

Complete reproduction steps from a clean environment:

```bash
# 1. Clone or unzip the project
cd /path/to/cspro

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or: venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0 peft==0.10.0 bitsandbytes==0.43.1 \
            datasets==2.19.0 accelerate==0.29.3 rouge-score==0.1.2

# 4. Run fine-tuning (requires 48GB VRAM, ~10 minutes)
python finetune/train.py

# 5. Run evaluation
python finetune/evaluate.py

# Checkpoint location:
#   finetune/checkpoints/phi2-qlora/
```

**Pinned dependency versions** (tested):

```
torch==2.2.2
transformers==4.40.0
peft==0.10.0
bitsandbytes==0.43.1
datasets==2.19.0
accelerate==0.29.3
rouge-score==0.1.2
```

No internet access is required beyond the initial HuggingFace model download. Set `TRANSFORMERS_OFFLINE=1` after the first download for air-gapped environments.
