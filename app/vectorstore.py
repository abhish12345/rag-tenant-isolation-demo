import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

_client: chromadb.Client | None = None


def get_chroma_client(in_memory: bool = False) -> chromadb.Client:
    global _client
    if _client is None:
        if in_memory:
            _client = chromadb.Client()
        else:
            _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def reset_client():
    """Reset the global client (used in tests)."""
    global _client
    _client = None


def get_tenant_collection(tenant_id: str, client: chromadb.Client | None = None) -> chromadb.Collection:
    c = client or get_chroma_client()
    collection_name = f"tenant_{tenant_id}"
    return c.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_documents(
    tenant_id: str,
    documents: list[dict],
    embeddings: list[list[float]],
    client: chromadb.Client | None = None,
):
    collection = get_tenant_collection(tenant_id, client)
    ids = [d["doc_id"] for d in documents]
    metadatas = [
        {"tenant_id": tenant_id, "doc_type": d["doc_type"], "title": d["title"], **d.get("metadata", {})}
        for d in documents
    ]
    contents = [d["content"] for d in documents]
    collection.upsert(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)


def query_tenant(
    tenant_id: str,
    query_embedding: list[float],
    top_k: int = 5,
    client: chromadb.Client | None = None,
) -> list[dict]:
    collection = get_tenant_collection(tenant_id, client)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"tenant_id": tenant_id},
        include=["documents", "metadatas", "distances"],
    )
    output = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = max(0.0, 1.0 - distance)
            output.append({
                "doc_id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(score, 4),
            })
    return output
