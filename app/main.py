import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Header

from app.auth import require_api_key
from app.config import settings
from app.embeddings import get_embedding_model
from app.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from app.rag import run_rag
from app.security import detect_injection, sanitise_input
from app.vectorstore import get_chroma_client, ingest_documents

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model
    get_embedding_model()
    logger.info("Embedding model loaded.")
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=settings.version, model=settings.ollama_model)


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest,
    tenant_id: str = Depends(require_api_key),
):
    if body.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="tenant_id in body does not match API key")
    emb_model = get_embedding_model()
    docs = [d.model_dump() for d in body.documents]
    texts = [d["content"] for d in docs]
    embeddings = emb_model.embed(texts)
    client = get_chroma_client(in_memory=settings.chroma_in_memory)
    ingest_documents(tenant_id, docs, embeddings, client)
    return IngestResponse(ingested=len(docs), tenant_id=tenant_id)


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    tenant_id: str = Depends(require_api_key),
):
    if body.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="tenant_id in body does not match API key")
    clean_question = sanitise_input(body.question)
    if detect_injection(clean_question):
        raise HTTPException(
            status_code=400,
            detail="Input rejected: potential prompt injection detected",
        )
    result = run_rag(
        question=clean_question,
        tenant_id=tenant_id,
        top_k=body.top_k,
        chroma_client=get_chroma_client(in_memory=settings.chroma_in_memory),
    )
    return AskResponse(**result)
