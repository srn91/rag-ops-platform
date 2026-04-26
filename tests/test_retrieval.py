from app.answering import build_grounded_answer
from app.corpus import load_documents
from app.retrieval import HybridIndex, chunk_document, split_sentences


def test_chunk_document_keeps_one_sentence_overlap() -> None:
    document = load_documents()[0]
    chunks = chunk_document(document, target_tokens=20, overlap_sentences=1)

    assert len(chunks) >= 2

    first_chunk_last_sentence = split_sentences(chunks[0].text)[-1]
    second_chunk_first_sentence = split_sentences(chunks[1].text)[0]
    assert first_chunk_last_sentence == second_chunk_first_sentence


def test_hybrid_index_returns_empty_results_for_blank_query() -> None:
    index = HybridIndex(load_documents())

    assert index.search("   ", top_k=3) == []


def test_load_documents_supports_markdown_html_and_pdf_sources() -> None:
    documents = load_documents()

    assert {document.content_type for document in documents} == {"markdown", "html", "pdf"}
    html_doc = next(document for document in documents if document.content_type == "html")
    pdf_doc = next(document for document in documents if document.content_type == "pdf")
    assert html_doc.metadata["meta_description"].startswith("Operational runbook")
    assert pdf_doc.metadata["page_count"] == 1
    assert "Compliance Answering Checklist" in pdf_doc.text


def test_hybrid_index_can_retrieve_html_ingested_content() -> None:
    index = HybridIndex(load_documents())
    results = index.search("What should teams inspect before changing prompts?", top_k=3)

    assert results
    assert results[0].chunk.doc_id == "support-copilot-runbook"
    assert results[0].embedding_provider == "local_tfidf_svd"


def test_hybrid_index_can_retrieve_pdf_ingested_content() -> None:
    index = HybridIndex(load_documents())
    results = index.search("What should be recorded for compliance-sensitive answers?", top_k=3)

    assert results
    assert results[0].chunk.doc_id == "compliance-answering-checklist"
    assert results[0].chunk.content_type == "pdf"


def test_grounded_answer_falls_back_cleanly_when_no_results_exist() -> None:
    payload = build_grounded_answer("What is the answer?", [])

    assert payload["citations"] == []
    assert payload["retrieval"] == []
    assert "No grounded answer" in payload["answer"]
    assert payload["answer_diagnostics"]["faithfulness"]["supported_sentence_ratio"] == 0.0
