from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator


class DocType(str, Enum):
    ticket = "ticket"
    kb_article = "kb_article"
    log = "log"


class DocumentIn(BaseModel):
    doc_id: str
    doc_type: DocType
    title: str
    content: str
    metadata: dict = {}


class IngestRequest(BaseModel):
    tenant_id: str
    documents: list[DocumentIn]
    model_config = ConfigDict(str_strip_whitespace=True)


class IngestResponse(BaseModel):
    ingested: int
    tenant_id: str


class AskRequest(BaseModel):
    question: str
    tenant_id: str
    top_k: int = 5

    @field_validator('question')
    @classmethod
    def question_length(cls, v):
        if len(v) > 1000:
            raise ValueError('Question exceeds 1000 character limit')
        return v


class SourceCitation(BaseModel):
    doc_id: str
    excerpt: str
    score: float


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    abstain = "abstain"


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: Confidence
    conflict_note: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
