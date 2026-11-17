# CSPRO RAG Assessment — Multi-Tenant IT Support Service

## Overview

This project is a multi-tenant Retrieval-Augmented Generation (RAG) service for IT support, built as the technical assessment for CSPRO's Senior AI Developer / Open Source LLM Engineer position.

The service allows multiple tenants (customers) to ingest their own IT support documents—tickets, logs, and knowledge-base articles—and query them through a natural-language interface backed by an open-weight LLM running locally via Ollama.

**Core security scenario.** When a RAG system uses a global vector store with a low similarity threshold, a query from Tenant A can surface documents belonging to Tenant B if their embeddings are geometrically close. This is a cross-tenant data leakage vector. This project fixes it with collection-level tenant partitioning in ChromaDB: every tenant gets its own isolated collection, so a query issued with Tenant A's API key physically cannot reach Tenant B's collection regardless of similarity scores. Isolation is enforced in application code, not in prompts.

---

## Architecture

### RAG Service

```
User / Client
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                          │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │   Auth Layer     │    │       Security Layer         │  │
│  │  (API Key →      │───▶│  (Prompt-injection detection │  │
│  │   Tenant ID)     │    │   + XML source tagging)      │  │
│  └──────────────────┘    └──────────────────────────────┘  │
│                                    │                        │
│                                    ▼                        │
│                        ┌───────────────────────┐           │
│                        │      RAG Pipeline     │           │
│                        └───────────────────────┘           │
│                          /         |          \             │
│            ┌────────────▼──┐  ┌───▼───────┐  ┌──▼──────┐  │
│            │  Embeddings   │  │ ChromaDB  │  │  Ollama  │  │
│            │ all-MiniLM-   │  │ Per-tenant│  │ command- │  │
│            │  L6-v2        │  │collections│  │  r:35b   │  │
│            └───────────────┘  └───────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Fine-tuning Path (separate from the service)

```
phi-2 base model
      │
      ▼
┌─────────────────┐
│  QLoRA (PEFT)   │  ◀── synthetic train dataset (data/finetune/train.jsonl)
│  train.py       │
└─────────────────┘
      │
      ▼
  LoRA adapter
  checkpoint
      │
      ▼
┌─────────────────┐
│  evaluate.py    │  ◀── held-out dataset (data/finetune/eval.jsonl)
└─────────────────┘
      │
      ▼
  adapted phi-2
  (baseline vs. adapted results)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and accessible
- `command-r:35b` model already pulled on your Ollama server
- (Optional, for fine-tuning) CUDA GPU with ≥ 24 GB VRAM

```bash
# 1. Clone the repository
git clone <repo>
cd cspro

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env to set OLLAMA_BASE_URL if your Ollama server is remote

# 4. Start Ollama and pull the model (skip if already running)
ollama serve
ollama pull command-r:35b

# 5. Verify Ollama connectivity
python scripts/run_ollama_check.py

# 6. Ingest synthetic data
python scripts/ingest.py

# 7. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

---

## API Reference

### GET `/health`

Returns service status.

**Headers:** none required

**Response `200`:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

```bash
curl http://localhost:8000/health
```

---

### POST `/ingest`

Ingests one or more documents into a tenant's isolated ChromaDB collection.

**Headers:**

| Header | Value |
|--------|-------|
| `X-API-Key` | Tenant API key |
| `Content-Type` | `application/json` |

**Request body:**
```json
{
  "tenant_id": "tenant_alpha",
  "documents": [
    {
      "doc_id": "TEST-001",
      "doc_type": "ticket",
      "title": "Test document",
      "content": "Full text of the document.",
      "metadata": {}
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tenant_id` | string | Must match the tenant bound to the API key |
| `documents` | array | One or more document objects |
| `doc_id` | string | Unique document identifier |
| `doc_type` | string | `ticket`, `log`, or `kb` |
| `title` | string | Document title |
| `content` | string | Full document text to embed |
| `metadata` | object | Arbitrary key-value pairs stored alongside the embedding |

**Response `200`:**
```json
{
  "ingested": 1,
  "tenant_id": "tenant_alpha"
}
```

**Error codes:**

| Code | Reason |
|------|--------|
| `401` | Missing or invalid API key |
| `403` | API key does not belong to the requested `tenant_id` |
| `422` | Validation error in request body |

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: alpha-key-00000000000000000000000000000001" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "tenant_alpha", "documents": [{"doc_id": "TEST-001", "doc_type": "ticket", "title": "Test", "content": "Test content", "metadata": {}}]}'
```

---

### POST `/ask`

Submits a natural-language question and returns an LLM-generated answer grounded in the tenant's documents, with source citations.

**Headers:**

| Header | Value |
|--------|-------|
| `X-API-Key` | Tenant API key |
| `Content-Type` | `application/json` |

**Request body:**
```json
{
  "question": "How do I fix VPN error 691?",
  "tenant_id": "tenant_alpha",
  "top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | — | Natural-language question |
| `tenant_id` | string | — | Must match the API key's tenant |
| `top_k` | integer | `5` | Number of documents to retrieve |

**Response `200`:**
```json
{
  "answer": "VPN error 691 indicates an authentication failure...",
  "sources": [
    {
      "doc_id": "KB-0042",
      "title": "VPN Troubleshooting Guide",
      "excerpt": "Error 691 means the username or password was rejected...",
      "score": 0.87
    }
  ],
  "abstained": false,
  "conflict_detected": false
}
```

**Error codes:**

| Code | Reason |
|------|--------|
| `400` | Prompt injection detected in question |
| `401` | Missing or invalid API key |
| `403` | API key does not belong to the requested `tenant_id` |
| `422` | Validation error in request body |

```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: alpha-key-00000000000000000000000000000001" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I fix VPN error 691?", "tenant_id": "tenant_alpha", "top_k": 5}'
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Security and prompt-injection tests
pytest tests/test_security.py -v

# Fine-tuning dataset integrity tests
pytest tests/test_finetune.py -v
```

Test categories:

| File | Covers |
|------|--------|
| `tests/test_api.py` | Health, ingest, ask — happy paths and validation errors |
| `tests/test_retrieval.py` | Retrieval accuracy, citation presence, abstention on weak evidence |
| `tests/test_security.py` | Cross-tenant isolation, prompt-injection attempts, forged-instruction documents |
| `tests/test_finetune.py` | Dataset schema, no overlap between train and eval splits, no fabricated metrics |

---

## Fine-tuning

The fine-tuning experiment uses **phi-2** (Microsoft, 2.7 B parameters) as the base model with QLoRA (4-bit quantisation via `bitsandbytes`) to adapt it toward IT support question-answering.

### Hardware requirements

- GPU with ≥ 24 GB VRAM recommended (tested target: A100/H100 40–80 GB)
- 16 GB system RAM minimum
- ~10 GB free disk for checkpoints

### Training

```bash
python finetune/train.py \
  --base_model microsoft/phi-2 \
  --train_data data/finetune/train.jsonl \
  --output_dir checkpoints/phi2-itsupport-qlora \
  --epochs 3 \
  --batch_size 4 \
  --lora_r 16 \
  --lora_alpha 32
```

Checkpoints are saved to `checkpoints/phi2-itsupport-qlora/` after each epoch. The final adapter is at `checkpoints/phi2-itsupport-qlora/final/`.

### Evaluation

```bash
python finetune/evaluate.py \
  --base_model microsoft/phi-2 \
  --adapter checkpoints/phi2-itsupport-qlora/final \
  --eval_data data/finetune/eval.jsonl \
  --output results/eval_results.json
```

Results are written to `results/eval_results.json` as ROUGE-L and exact-match scores, with baseline (phi-2 without adapter) and adapted columns.

> **Note:** Training and evaluation results in this repository are **unexecuted**. The implementation is fully reproducible; results have not been run due to hardware scheduling constraints at submission time. No metrics have been fabricated.

---

## Security Model

The service uses a two-layer defence.

### Layer 1 — Tenant isolation

**ChromaDB collection partitioning:** Each tenant's documents are stored in a dedicated ChromaDB collection named `tenant_{tenant_id}`. Query calls never fan out across collections; the collection name is derived from the API-key-to-tenant mapping resolved in application code before any vector operation runs.

**API key enforcement in code:** The `X-API-Key` header is verified against a key-to-tenant registry on every request. If the key is absent, invalid, or does not match the `tenant_id` in the request body, the request is rejected with `401` or `403` before the RAG pipeline runs.

**Why isolation is in code, not prompts:** A prompt instruction such as "only return documents for the requesting tenant" is not a security control. An adversary can override it with a carefully crafted query or a poisoned document. Application-code enforcement means the LLM never sees another tenant's documents to begin with; there is nothing to leak regardless of what the model does.

### Layer 2 — Input and content integrity

**Prompt-injection detection:** Incoming questions are scanned for patterns that attempt to override system instructions (e.g., "ignore previous instructions", role-switching commands, base64-encoded payloads). Requests that match are rejected with `400` before reaching the LLM.

**XML source tagging:** Retrieved documents are injected into the prompt inside XML tags (`<source id="...">...</source>`) with an explicit system instruction that all content inside those tags is untrusted external data. This structural separation makes it harder for a poisoned document to blur the boundary between retrieved evidence and system instructions.

---

## Known Limitations

1. **API keys are hardcoded.** The key-to-tenant registry lives in `app/auth.py` as a Python dict. A production deployment would store keys in a secrets manager (AWS Secrets Manager, HashiCorp Vault) and rotate them on demand.

2. **Conflict detection is heuristic.** The service flags contradictory evidence by checking whether the top-k retrieved documents contain semantically opposing statements. This is based on keyword and negation heuristics, not a dedicated entailment model, so it will miss subtle contradictions.

3. **Ollama latency.** `command-r:35b` running on a single GPU (even 48 GB) can take 10–30 seconds per response under moderate load. There is no streaming or timeout handling beyond Ollama's own defaults.

4. **No authentication beyond API keys.** The service has no user accounts, session tokens, MFA, or rate limiting. API keys provide tenant-level access but not per-user audit trails.

5. **ChromaDB HNSW approximate nearest neighbour.** ChromaDB's default index is HNSW, which trades recall for speed. At very high collection sizes or with unusual embedding distributions, the top-k results may miss the closest true neighbours. Exact-search mode is available but slower and not enabled by default.

6. **Single-node, no persistence guarantees.** ChromaDB is run in-process with local disk persistence. There is no replication, backup, or crash-recovery mechanism. A server restart after a failed write could leave a collection in an inconsistent state.

7. **Synthetic data only.** The retrieval quality and conflict-detection logic have been validated only against the eight synthetic documents provided. Performance on real IT support corpora has not been measured.

---

## AI Assistance Disclosure

Portions of this codebase were scaffolded and generated with AI coding assistance (GitHub Copilot and Claude). This is disclosed in accordance with the assessment's explicit requirement.

All design decisions are the developer's own: the two-layer security architecture (collection partitioning + code-level enforcement rather than prompt-level enforcement), the choice of ChromaDB collection-per-tenant as the isolation primitive, the prompt-injection detection strategy, the XML source-tagging approach for untrusted-data separation, the QLoRA experiment design, and the test strategy covering cross-tenant leakage and injection regression. AI tools were used to accelerate implementation of boilerplate (FastAPI route skeletons, Pydantic schema definitions, pytest fixtures) and to draft documentation; they did not originate the security model or the architectural choices.
