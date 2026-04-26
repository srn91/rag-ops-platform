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
            "answer_diagnostics": {
                "faithfulness": {
                    "supported_sentence_ratio": 0.0,
                    "supported_sentences": 0,
                    "unsupported_sentences": [],
                },
                "completeness": {
                    "question_term_coverage_ratio": 0.0,
                    "covered_question_terms": [],
                    "missing_question_terms": [],
                },
            },
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
                "content_type": result.chunk.content_type,
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
                "content_type": results[0].chunk.content_type,
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
            "content_type": result.chunk.content_type,
            "sparse_score": result.sparse_score,
            "dense_score": result.dense_score,
            "combined_score": result.combined_score,
            "rerank_score": result.rerank_score,
            "overlap_terms": result.overlap_terms,
            "overlap_term_count": len(result.overlap_terms),
            "embedding_provider": result.embedding_provider,
        }
        for result in results
    ]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieval": retrieval,
        "answer_diagnostics": _answer_diagnostics(question, answer, results, citations),
    }


def _answer_diagnostics(
    question: str,
    answer: str,
    results: list[SearchResult],
    citations: list[dict[str, object]],
) -> dict[str, object]:
    answer_sentences = [sentence.strip() for sentence in split_sentences(answer) if sentence.strip()]
    citation_context = " ".join(str(citation["snippet"]) for citation in citations)
    retrieval_context = " ".join(result.chunk.text for result in results[:3])
    evidence_tokens = set(tokenize(f"{citation_context} {retrieval_context}"))

    supported_sentences = 0
    unsupported_sentences: list[str] = []
    for sentence in answer_sentences:
        sentence_tokens = set(tokenize(sentence))
        if sentence_tokens and sentence_tokens <= evidence_tokens:
            supported_sentences += 1
        else:
            unsupported_sentences.append(sentence)

    question_terms = sorted(set(tokenize(question)))
    answer_tokens = set(tokenize(answer))
    covered_question_terms = sorted(term for term in question_terms if term in answer_tokens)
    missing_question_terms = sorted(term for term in question_terms if term not in answer_tokens)

    return {
        "faithfulness": {
            "supported_sentence_ratio": round(supported_sentences / len(answer_sentences), 4) if answer_sentences else 0.0,
            "supported_sentences": supported_sentences,
            "unsupported_sentences": unsupported_sentences,
        },
        "completeness": {
            "question_term_coverage_ratio": round(len(covered_question_terms) / len(question_terms), 4)
            if question_terms
            else 0.0,
            "covered_question_terms": covered_question_terms,
            "missing_question_terms": missing_question_terms,
        },
    }
