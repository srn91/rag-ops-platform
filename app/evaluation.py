from __future__ import annotations

import json

from app.config import EVALUATION_CASES_PATH
from app.models import EvaluationCase
from app.retrieval import HybridIndex
from app.answering import build_grounded_answer


def load_evaluation_cases() -> list[EvaluationCase]:
    raw_cases = json.loads(EVALUATION_CASES_PATH.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            question=case["question"],
            expected_doc_id=case["expected_doc_id"],
            rationale=case["rationale"],
        )
        for case in raw_cases
    ]


def run_evaluation(index: HybridIndex) -> dict[str, object]:
    cases = load_evaluation_cases()
    case_results: list[dict[str, object]] = []
    hits = 0
    reciprocal_rank_total = 0.0
    citation_hits = 0

    for case in cases:
        search_results = index.search(case.question, top_k=3)
        answer = build_grounded_answer(case.question, search_results)
        retrieved_doc_ids = [result.chunk.doc_id for result in search_results]
        citations = answer["citations"]

        rank = next(
            (position for position, doc_id in enumerate(retrieved_doc_ids, start=1) if doc_id == case.expected_doc_id),
            None,
        )
        hit = rank is not None
        citation_hit = any(citation["doc_id"] == case.expected_doc_id for citation in citations)

        if hit:
            hits += 1
            reciprocal_rank_total += 1.0 / rank
        if citation_hit:
            citation_hits += 1

        case_results.append(
            {
                "question": case.question,
                "expected_doc_id": case.expected_doc_id,
                "rationale": case.rationale,
                "retrieved_doc_ids": retrieved_doc_ids,
                "top_citation_doc_ids": [citation["doc_id"] for citation in citations],
                "retrieval_hit": hit,
                "citation_hit": citation_hit,
                "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
            }
        )

    total_cases = len(cases)
    return {
        "summary": {
            "cases": total_cases,
            "retrieval_hit_rate_at_3": round(hits / total_cases, 4),
            "citation_hit_rate": round(citation_hits / total_cases, 4),
            "mean_reciprocal_rank": round(reciprocal_rank_total / total_cases, 4),
        },
        "cases": case_results,
    }

