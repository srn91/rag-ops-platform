from __future__ import annotations

from app.models import SearchResult
from app.retrieval import split_sentences, tokenize


def build_grounded_answer(question: str, results: list[SearchResult]) -> dict[str, object]:
    if not results:
        return {
            "question": question,
            "answer": "No grounded answer could be produced because the query did not match the indexed sample corpus.",
            "citations": [],
            "retrieval": [],
        }

    question_tokens = set(tokenize(question))
    sentence_candidates: list[tuple[float, str, SearchResult]] = []
    for result in results[:3]:
        for sentence in split_sentences(result.chunk.text):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(question_tokens & sentence_tokens)
            if overlap == 0:
                continue
            score = overlap + result.rerank_score
            sentence_candidates.append((score, sentence, result))

    sentence_candidates.sort(key=lambda item: item[0], reverse=True)

    selected_sentences: list[str] = []
    citation_map: dict[str, dict[str, object]] = {}
    for _, sentence, result in sentence_candidates:
        cleaned = sentence.strip()
        if cleaned in selected_sentences:
            continue
        selected_sentences.append(cleaned)
        citation_map.setdefault(
            result.chunk.chunk_id,
            {
                "doc_id": result.chunk.doc_id,
                "title": result.chunk.title,
                "chunk_id": result.chunk.chunk_id,
                "path": result.chunk.path,
                "snippet": cleaned,
                "score": result.rerank_score,
            },
        )
        if len(selected_sentences) == 2:
            break

    if not selected_sentences:
        selected_sentences.append(results[0].chunk.text[:220].strip())
        citation_map.setdefault(
            results[0].chunk.chunk_id,
            {
                "doc_id": results[0].chunk.doc_id,
                "title": results[0].chunk.title,
                "chunk_id": results[0].chunk.chunk_id,
                "path": results[0].chunk.path,
                "snippet": results[0].chunk.text[:220].strip(),
                "score": results[0].rerank_score,
            },
        )

    answer = " ".join(selected_sentences)
    citations = list(citation_map.values())
    retrieval = [
        {
            "doc_id": result.chunk.doc_id,
            "title": result.chunk.title,
            "chunk_id": result.chunk.chunk_id,
            "path": result.chunk.path,
            "sparse_score": result.sparse_score,
            "dense_score": result.dense_score,
            "combined_score": result.combined_score,
            "rerank_score": result.rerank_score,
            "overlap_terms": result.overlap_terms,
            "overlap_term_count": len(result.overlap_terms),
        }
        for result in results
    ]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieval": retrieval,
    }
