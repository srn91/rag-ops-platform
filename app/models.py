from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    title: str
    path: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    path: str
    text: str
    token_counts: dict[str, int]
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    sparse_score: float
    dense_score: float
    combined_score: float
    rerank_score: float
    overlap_terms: list[str]


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_doc_id: str
    rationale: str
