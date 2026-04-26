from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["project"] == "rag-ops-platform"
    assert payload["status"] == "v1-local"
    assert payload["indexed_assets"]["documents"] == 3


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["indexed_assets"]["chunks"] >= 3


def test_documents_endpoint_lists_indexed_corpus() -> None:
    response = client.get("/documents")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["documents"]) == 3
    assert payload["documents"][0]["path"].startswith("corpus/")


def test_query_returns_grounded_citations() -> None:
    response = client.post(
        "/query",
        json={"question": "How does the platform reduce hallucinations?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "citations" in payload
    assert payload["citations"][0]["doc_id"] == "retrieval-quality-playbook"
    assert "hallucinations" in payload["answer"].lower()
    assert payload["diagnostics"]["latency_ms"]["total"] >= 0.0
    assert payload["diagnostics"]["ranking"]["retrieved_chunk_count"] == 3
    assert "hallucinations" in payload["retrieval"][0]["overlap_terms"]


def test_evaluation_endpoint_reports_summary_metrics() -> None:
    response = client.get("/evaluation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["cases"] == 3
    assert payload["summary"]["retrieval_hit_rate_at_3"] >= 0.66
    assert payload["summary"]["citation_hit_rate"] >= 0.66
    assert payload["summary"]["latency_ms"]["retrieval_p50"] >= 0.0
    assert payload["summary"]["ranking_diagnostics"]["mean_top_result_margin"] >= 0.0
    assert "ranking_diagnostics" in payload["cases"][0]
