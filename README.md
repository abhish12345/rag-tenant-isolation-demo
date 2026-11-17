# RAG Tenant Isolation Demo — Multi-Tenant IT Support Service

## Overview

A multi-tenant Retrieval-Augmented Generation (RAG) service for IT support, built as the technical assessment for CSPRO's Senior AI Developer / Open Source LLM Engineer position.

Each tenant (customer) ingests their own IT support documents — tickets, logs, and knowledge-base articles — and queries them through a natural-language interface backed by an open-weight LLM running locally via Ollama.

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
│  QLoRA (PEFT)   │  ◀── finetune/dataset_train.jsonl (85 examples)
│  train.py       │
└─────────────────┘
      │
      ▼
  LoRA adapter
  checkpoint
      │
      ▼
┌─────────────────┐
│  evaluate.py    │  ◀── finetune/dataset_eval.jsonl (25 examples)
└─────────────────┘
      │
      ▼
  baseline vs. adapted ROUGE-L + keyword accuracy
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and accessible (local or remote server)
- `command-r:35b` model already pulled on your Ollama server
- (Optional, for fine-tuning) CUDA GPU with ≥ 24 GB VRAM

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/abhish12345/rag-tenant-isolation-demo.git
cd rag-tenant-isolation-demo

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies (wheels only — no compiler needed)
pip install -r requirements.txt --only-binary=:all:
# ChromaDB must be installed separately (C extension, no --only-binary)
pip install chromadb

# 4. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
# Edit .env — set OLLAMA_BASE_URL if your Ollama server is remote

# 5. Verify Ollama connectivity
python scripts/run_ollama_check.py

# 6. Ingest synthetic data
python scripts/ingest.py

# 7. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

---

## API Keys (for testing)

| Tenant | API Key | tenant_id |
|--------|---------|-----------|
| Contoso Retail | `alpha-key-00000000000000000000000000000001` | `tenant_alpha` |
| Nexus Freight  | `beta-key-00000000000000000000000000000001` | `tenant_beta` |

> These are hardcoded synthetic keys for demonstration only. See [Known Limitations](#known-limitations).

---

## API Reference

### GET `/health`

Returns service status.

**Response `200`:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model": "command-r:35b"
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
| `doc_id` | string | Unique document identifier |
| `doc_type` | string | `ticket`, `kb_article`, or `log` |
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
  -d '{
    "tenant_id": "tenant_alpha",
    "documents": [{
      "doc_id": "TEST-001",
      "doc_type": "ticket",
      "title": "Test",
      "content": "Test content",
      "metadata": {}
    }]
  }'
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
| `question` | string | — | Natural-language question (max 1000 characters) |
| `tenant_id` | string | — | Must match the API key's tenant |
| `top_k` | integer | `5` | Number of documents to retrieve |

**Response `200`:**
```json
{
  "answer": "VPN error 691 indicates an authentication failure...",
  "sources": [
    {
      "doc_id": "ALPHA-KB-001",
      "excerpt": "Error 691 means the username or password was rejected...",
      "score": 0.87
    }
  ],
  "confidence": "high",
  "conflict_note": null
}
```

| Field | Description |
|-------|-------------|
| `answer` | LLM-generated answer grounded in retrieved sources |
| `sources` | List of retrieved documents with `doc_id`, `excerpt`, and similarity `score` |
| `confidence` | `high`, `medium`, `low`, or `abstain` (abstain when no relevant evidence found) |
| `conflict_note` | Non-null when retrieved sources contain contradictory evidence |

**Error codes:**

| Code | Reason |
|------|--------|
| `400` | Prompt injection detected in question |
| `401` | Missing or invalid API key |
| `403` | API key does not belong to the requested `tenant_id` |
| `422` | Validation error (e.g. question exceeds 1000 characters) |

```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: alpha-key-00000000000000000000000000000001" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I fix VPN error 691?",
    "tenant_id": "tenant_alpha",
    "top_k": 5
  }'
```

---

## Running Tests

```bash
# Full test suite (29 tests)
pytest tests/ -v

# Security and prompt-injection tests only
pytest tests/test_security.py -v

# Fine-tuning dataset integrity tests only
pytest tests/test_finetune.py -v
```

| File | Covers |
|------|--------|
| `tests/test_api.py` | Health, ingest, ask — happy paths and validation errors |
| `tests/test_retrieval.py` | Retrieval accuracy, citation presence, abstention on empty KB |
| `tests/test_security.py` | Cross-tenant isolation, prompt-injection attempts, adversarial doc regression |
| `tests/test_finetune.py` | Dataset schema, minimum example counts, category coverage |

Ollama calls are mocked in all tests. ChromaDB runs in-memory (no server required). All 29 tests pass on Python 3.14.

---

## Fine-tuning

The fine-tuning experiment uses **microsoft/phi-2** (2.7B parameters) as the base model with QLoRA (4-bit NF4 quantisation via `bitsandbytes`, r=16, alpha=32) to adapt it toward IT helpdesk question-answering.

### Hardware requirements

- GPU with ≥ 24 GB VRAM recommended
- Tested target: A100 / H100 40–80 GB
- ~10 GB free disk for checkpoints

### Dataset

| Split | File | Examples | Categories |
|-------|------|----------|------------|
| Train | `finetune/dataset_train.jsonl` | 85 | VPN, password, hardware, software, network, printer, email, security |
| Eval  | `finetune/dataset_eval.jsonl`  | 25 | Same 8 categories |

Format (instruction-following JSONL):
```jsonl
{"instruction": "Classify this ticket and suggest a resolution:", "input": "User cannot connect to VPN after password reset", "output": "Category: Access/Authentication. Resolution: Check that the new password meets VPN client requirements and re-authenticate."}
```

### Training

```bash
python finetune/train.py
# Checkpoint saved to finetune/checkpoints/phi2-qlora/
```

### Evaluation

```bash
python finetune/evaluate.py
# Prints ROUGE-L and keyword accuracy for base phi-2 vs adapted phi-2
```

> **⚠ Results are UNEXECUTED.** The training and evaluation scripts are fully reproducible but were not run during submission due to hardware scheduling constraints. No metrics have been fabricated. See `finetune/README_finetune.md` for the full configuration and expected result template.

---

## Security Model

The service uses a two-layer defence.

### Layer 1 — Tenant isolation in application code

**ChromaDB collection partitioning:** Each tenant's documents are stored in a dedicated ChromaDB collection named `tenant_{tenant_id}`. Query calls never fan out across collections; the collection name is derived from the API-key-to-tenant mapping resolved in application code before any vector operation runs.

**API key enforcement:** The `X-API-Key` header is verified against a key-to-tenant registry (`app/auth.py`) on every request. If the key is absent, invalid, or does not match the `tenant_id` in the request body, the request is rejected with `401` or `403` before the RAG pipeline runs.

**Why isolation is in code, not prompts:** A prompt instruction such as "only return documents for the requesting tenant" is not a security control. An adversary can override it with a carefully crafted query or a poisoned document. Application-code enforcement means the LLM never sees another tenant's documents to begin with; there is nothing to leak regardless of what the model does.

### Layer 2 — Input and content integrity

**Prompt-injection detection:** Incoming questions are scanned for patterns that attempt to override system instructions (e.g. "ignore previous instructions", role-switching commands, jailbreak phrases). Requests that match are rejected with `400` before reaching the LLM.

**XML source tagging:** Retrieved documents are injected into the prompt inside `<source id="...">...</source>` tags with an explicit system instruction that all content inside those tags is untrusted external data. This structural separation limits the ability of a poisoned document to blur the boundary between retrieved evidence and system instructions.

---

## Known Limitations

1. **API keys are hardcoded.** The key-to-tenant registry lives in `app/auth.py` as a Python dict. A production deployment would use a secrets manager (AWS Secrets Manager, HashiCorp Vault) with key rotation.

2. **Conflict detection is heuristic.** Contradictory evidence is flagged using keyword and negation heuristics, not a dedicated entailment model. Subtle contradictions may be missed.

3. **Ollama latency.** `command-r:35b` on a single 48 GB GPU can take 10–30 seconds per response. There is no streaming or timeout handling beyond Ollama's defaults.

4. **No per-user authentication.** API keys provide tenant-level access but no per-user audit trails, rate limiting, or MFA.

5. **ChromaDB HNSW approximate search.** The default HNSW index trades recall for speed. At very high collection sizes, the top-k results may miss the closest true neighbours.

6. **Single-node, no persistence guarantees.** ChromaDB runs with local disk persistence and no replication or crash-recovery mechanism.

7. **Synthetic data only.** Retrieval quality and conflict detection have been validated only against the 16 synthetic documents provided.

---

## AI Assistance Disclosure

Portions of this codebase were scaffolded with AI coding assistance (Kiro AI / Claude). This is disclosed in accordance with the assessment's explicit requirement.

All design decisions are the developer's own: the two-layer security architecture (collection partitioning + code-level enforcement rather than prompt-level enforcement), the choice of ChromaDB collection-per-tenant as the isolation primitive, the prompt-injection detection strategy, the XML source-tagging approach for untrusted-data separation, the QLoRA experiment design, and the test strategy covering cross-tenant leakage and injection regression. AI tools were used to accelerate implementation of boilerplate (FastAPI route skeletons, Pydantic schema definitions, pytest fixtures) and to draft documentation; they did not originate the security model or the architectural choices.
