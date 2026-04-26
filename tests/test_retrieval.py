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


def test_grounded_answer_falls_back_cleanly_when_no_results_exist() -> None:
    payload = build_grounded_answer("What is the answer?", [])

    assert payload["citations"] == []
    assert payload["retrieval"] == []
    assert "No grounded answer" in payload["answer"]
