"""
Tests for retrieval quality, source citations, and abstention behaviour.
"""
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import ALPHA_DOCS, ALPHA_KEY, TEST_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_alpha(client):
    payload = {"tenant_id": "tenant_alpha", "documents": ALPHA_DOCS}
    resp = client.post("/ingest", json=payload, headers={"X-API-Key": ALPHA_KEY})
    assert resp.status_code == 200


def _mock_ollama_citing(doc_id: str):
    """Return a patch whose mock returns an answer that cites *doc_id*."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "response": f"Use GlobalProtect to set up VPN. See [src:{doc_id}] for details."
    }
    return mock_resp


def _mock_ollama_no_citation():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "response": "I cannot find specific information in the provided sources."
    }
    return mock_resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieval_finds_correct_doc(client):
    """After ingesting ALPHA_DOCS, asking a VPN question should return TEST-ALPHA-001."""
    _ingest_alpha(client)

    mock_resp = _mock_ollama_citing("TEST-ALPHA-001")

    with patch("app.rag.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = mock_resp

        payload = {
            "question": "How do I set up VPN for TenantAlpha?",
            "tenant_id": "tenant_alpha",
        }
        resp = client.post("/ask", json=payload, headers={"X-API-Key": ALPHA_KEY})

    assert resp.status_code == 200
    body = resp.json()
    source_ids = [s["doc_id"] for s in body["sources"]]
    assert "TEST-ALPHA-001" in source_ids


def test_sources_have_non_empty_excerpts(client):
    """Every source citation must carry a non-empty excerpt string."""
    _ingest_alpha(client)

    mock_resp = _mock_ollama_citing("TEST-ALPHA-001")

    with patch("app.rag.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = mock_resp

        payload = {
            "question": "How do I set up VPN for TenantAlpha?",
            "tenant_id": "tenant_alpha",
        }
        resp = client.post("/ask", json=payload, headers={"X-API-Key": ALPHA_KEY})

    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert len(sources) > 0, "Expected at least one source citation"
    for src in sources:
        assert isinstance(src["excerpt"], str)
        assert len(src["excerpt"]) > 0, f"Empty excerpt on source {src['doc_id']}"


def test_no_results_returns_abstain(client):
    """With no documents ingested, confidence must be low or abstain."""
    mock_resp = _mock_ollama_no_citation()

    with patch("app.rag.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = mock_resp

        payload = {
            "question": "What is the helpdesk email address?",
            "tenant_id": "tenant_test",
        }
        resp = client.post("/ask", json=payload, headers={"X-API-Key": TEST_KEY})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] in ("low", "abstain")
