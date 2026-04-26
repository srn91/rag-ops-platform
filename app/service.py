from __future__ import annotations

from time import perf_counter

from app.answering import build_grounded_answer
from app.corpus import load_documents
from app.evaluation import run_evaluation
from app.models import SearchResult
from app.retrieval import HybridIndex


class RAGService:
    def __init__(self) -> None:
        self.documents = load_documents()
        self.index = HybridIndex(self.documents)

    def stats(self) -> dict[str, object]:
        return self.index.stats()

    def list_documents(self) -> list[dict[str, object]]:
        return [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "path": document.path,
                "content_type": document.content_type,
                "metadata": document.metadata,
            }
            for document in self.documents
        ]

    def query(self, question: str, *, top_k: int = 5) -> dict[str, object]:
        retrieval_started = perf_counter()
        results = self.index.search(question, top_k=top_k)
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000, 3)

        answer_started = perf_counter()
        payload = build_grounded_answer(question, results)
        answer_ms = round((perf_counter() - answer_started) * 1000, 3)
        total_ms = round(retrieval_ms + answer_ms, 3)
        payload["diagnostics"] = _query_diagnostics(results, retrieval_ms, answer_ms, total_ms)
        return payload

    def evaluate(self) -> dict[str, object]:
        return run_evaluation(self.index)


def _query_diagnostics(
    results: list[SearchResult],
    retrieval_ms: float,
    answer_ms: float,
    total_ms: float,
) -> dict[str, object]:
    if not results:
        return {
            "latency_ms": {
                "retrieval": retrieval_ms,
                "answer": answer_ms,
                "total": total_ms,
            },
            "ranking": {
                "retrieved_chunk_count": 0,
                "top_result_margin": 0.0,
                "top_overlap_terms": [],
            },
            "embedding": {
                "provider": None,
            },
        }

    top_margin = (
        round(results[0].rerank_score - results[1].rerank_score, 4)
        if len(results) > 1
        else round(results[0].rerank_score, 4)
    )
    return {
        "latency_ms": {
            "retrieval": retrieval_ms,
            "answer": answer_ms,
            "total": total_ms,
        },
        "ranking": {
            "retrieved_chunk_count": len(results),
            "top_result_margin": top_margin,
            "top_overlap_terms": results[0].overlap_terms,
            "top_chunk_id": results[0].chunk.chunk_id,
        },
        "embedding": {
            "provider": results[0].embedding_provider,
        },
    }
