from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Ops Platform" in response.text
    assert "/evaluation" in response.text


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
    assert len(payload["documents"]) == 5
    assert payload["documents"][0]["path"].startswith("corpus/")
    by_type = {document["content_type"] for document in payload["documents"]}
    assert by_type == {"markdown", "html", "pdf"}
    pdf_doc = next(document for document in payload["documents"] if document["content_type"] == "pdf")
    assert pdf_doc["metadata"]["page_count"] == 1


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
    assert payload["diagnostics"]["embedding"]["provider"] == "local_tfidf_svd"
    assert payload["retrieval"][0]["embedding_provider"] == "local_tfidf_svd"
    assert payload["answer_diagnostics"]["faithfulness"]["supported_sentence_ratio"] >= 0.5
    assert payload["answer_diagnostics"]["completeness"]["question_term_coverage_ratio"] > 0.0


def test_evaluation_endpoint_reports_summary_metrics() -> None:
    response = client.get("/evaluation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["cases"] == 3
    assert payload["summary"]["retrieval_hit_rate_at_3"] >= 0.66
    assert payload["summary"]["citation_hit_rate"] >= 0.66
    assert payload["summary"]["answer_diagnostics"]["mean_faithfulness_score"] >= 0.66
    assert payload["summary"]["answer_diagnostics"]["mean_question_term_coverage"] > 0.0
    assert payload["summary"]["latency_ms"]["retrieval_p50"] >= 0.0
    assert payload["summary"]["ranking_diagnostics"]["mean_top_result_margin"] >= 0.0
    assert "ranking_diagnostics" in payload["cases"][0]
    assert "answer_diagnostics" in payload["cases"][0]
