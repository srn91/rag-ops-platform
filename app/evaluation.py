from __future__ import annotations

import json
from statistics import mean
from time import perf_counter

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
    retrieval_latencies_ms: list[float] = []
    answer_latencies_ms: list[float] = []
    total_latencies_ms: list[float] = []
    top_result_margins: list[float] = []

    for case in cases:
        retrieval_started = perf_counter()
        search_results = index.search(case.question, top_k=3)
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000, 3)
        answer_started = perf_counter()
        answer = build_grounded_answer(case.question, search_results)
        answer_ms = round((perf_counter() - answer_started) * 1000, 3)
        total_ms = round(retrieval_ms + answer_ms, 3)
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
        retrieval_latencies_ms.append(retrieval_ms)
        answer_latencies_ms.append(answer_ms)
        total_latencies_ms.append(total_ms)
        top_result_margin = (
            round(search_results[0].rerank_score - search_results[1].rerank_score, 4)
            if len(search_results) > 1
            else round(search_results[0].rerank_score, 4)
            if search_results
            else 0.0
        )
        top_result_margins.append(top_result_margin)

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
                "latency_ms": {
                    "retrieval": retrieval_ms,
                    "answer": answer_ms,
                    "total": total_ms,
                },
                "ranking_diagnostics": {
                    "expected_doc_rank": rank,
                    "top_result_margin": top_result_margin,
                    "top_overlap_terms": search_results[0].overlap_terms if search_results else [],
                    "top_chunk_id": search_results[0].chunk.chunk_id if search_results else None,
                },
            }
        )

    total_cases = len(cases)
    return {
        "summary": {
            "cases": total_cases,
            "retrieval_hit_rate_at_3": round(hits / total_cases, 4),
            "citation_hit_rate": round(citation_hits / total_cases, 4),
            "mean_reciprocal_rank": round(reciprocal_rank_total / total_cases, 4),
            "latency_ms": {
                "retrieval_p50": _percentile(retrieval_latencies_ms, 0.5),
                "retrieval_p95": _percentile(retrieval_latencies_ms, 0.95),
                "answer_p50": _percentile(answer_latencies_ms, 0.5),
                "answer_p95": _percentile(answer_latencies_ms, 0.95),
                "total_p50": _percentile(total_latencies_ms, 0.5),
                "total_p95": _percentile(total_latencies_ms, 0.95),
            },
            "ranking_diagnostics": {
                "mean_top_result_margin": round(mean(top_result_margins), 4) if top_result_margins else 0.0,
                "max_top_result_margin": round(max(top_result_margins), 4) if top_result_margins else 0.0,
            },
        },
        "cases": case_results,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)
