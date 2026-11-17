#!/usr/bin/env python3
"""Load synthetic data from data/synthetic/ into ChromaDB.

Usage:
    python scripts/ingest.py                     # persistent ChromaDB
    CHROMA_IN_MEMORY=true python scripts/ingest.py  # ephemeral (CI)
"""
import json
import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.embeddings import get_embedding_model
from app.vectorstore import get_chroma_client, ingest_documents

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

def load_tenant_file(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)

def main():
    in_memory = os.environ.get("CHROMA_IN_MEMORY", "").lower() in ("true", "1", "yes")
    client = get_chroma_client(in_memory=in_memory)
    emb_model = get_embedding_model()

    files = [
        DATA_DIR / "tenant_alpha.json",
        DATA_DIR / "tenant_beta.json",
    ]

    total = 0
    for fpath in files:
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping.")
            continue
        docs = load_tenant_file(fpath)
        if not docs:
            continue
        # Group by tenant_id
        by_tenant: dict[str, list[dict]] = {}
        for doc in docs:
            tid = doc["tenant_id"]
            by_tenant.setdefault(tid, []).append(doc)
        for tenant_id, tenant_docs in by_tenant.items():
            texts = [d["content"] for d in tenant_docs]
            embeddings = emb_model.embed(texts)
            ingest_documents(tenant_id, tenant_docs, embeddings, client)
            print(f"Ingested {len(tenant_docs)} docs for {tenant_id} from {fpath.name}")
            total += len(tenant_docs)

    print(f"\nTotal ingested: {total} documents")

if __name__ == "__main__":
    main()
