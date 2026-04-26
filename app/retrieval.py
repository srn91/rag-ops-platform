from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from app.embeddings import EmbeddingProvider, build_embedding_provider
from app.models import DocumentChunk, SearchResult, SourceDocument


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_BREAK_PATTERN = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not normalized:
        return []
    return [sentence.strip() for sentence in SENTENCE_BREAK_PATTERN.split(normalized) if sentence.strip()]


def chunk_document(
    document: SourceDocument,
    *,
    target_tokens: int = 110,
    overlap_sentences: int = 1,
) -> list[DocumentChunk]:
    sentences = split_sentences(document.text)
    if not sentences:
        return []

    chunks: list[DocumentChunk] = []
    chunk_sentences: list[str] = []
    chunk_token_count = 0
    chunk_number = 1

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if not sentence_tokens:
            continue

        if chunk_sentences and chunk_token_count + len(sentence_tokens) > target_tokens:
            text = " ".join(chunk_sentences)
            token_counts = Counter(tokenize(text))
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_number:02d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    path=document.path,
                    content_type=document.content_type,
                    metadata=document.metadata,
                    text=text,
                    token_counts=dict(token_counts),
                )
            )
            chunk_number += 1
            chunk_sentences = chunk_sentences[-overlap_sentences:] if overlap_sentences else []
            chunk_token_count = sum(len(tokenize(existing)) for existing in chunk_sentences)

        chunk_sentences.append(sentence)
        chunk_token_count += len(sentence_tokens)

    if chunk_sentences:
        text = " ".join(chunk_sentences)
        token_counts = Counter(tokenize(text))
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.doc_id}-chunk-{chunk_number:02d}",
                doc_id=document.doc_id,
                title=document.title,
                path=document.path,
                content_type=document.content_type,
                metadata=document.metadata,
                text=text,
                token_counts=dict(token_counts),
            )
        )

    return chunks


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(minimum, maximum):
        return [1.0 if maximum > 0 else 0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


class HybridIndex:
    def __init__(
        self,
        documents: list[SourceDocument],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.documents = documents
        self.chunks = [chunk for document in documents for chunk in chunk_document(document)]
        if not self.chunks:
            raise RuntimeError("hybrid index requires at least one chunk")

        self.embedding_provider = embedding_provider or build_embedding_provider()
        self.doc_frequency: Counter[str] = Counter()
        self.average_chunk_length = 0.0

        total_length = 0
        for chunk in self.chunks:
            total_length += sum(chunk.token_counts.values())
            self.doc_frequency.update(chunk.token_counts.keys())

        self.average_chunk_length = total_length / len(self.chunks)
        self._chunk_embeddings = self.embedding_provider.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

    def stats(self) -> dict[str, object]:
        content_types = Counter(document.content_type for document in self.documents)
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedding_provider": self.embedding_provider.name,
            "content_types": dict(content_types),
        }

    def search(self, question: str, *, top_k: int = 5) -> list[SearchResult]:
        query_counts = Counter(tokenize(question))
        if not query_counts:
            return []

        sparse_scores = [self._bm25(query_counts, chunk) for chunk in self.chunks]
        query_embedding = self.embedding_provider.transform_query(question)
        dense_scores = [
            float(np.dot(query_embedding[0], chunk_embedding))
            for chunk_embedding in self._chunk_embeddings
        ]
        normalized_sparse = _normalize_scores(sparse_scores)
        normalized_dense = _normalize_scores(dense_scores)

        question_tokens = set(query_counts)
        results: list[SearchResult] = []
        for chunk, sparse_score, dense_score, sparse_norm, dense_norm in zip(
            self.chunks,
            sparse_scores,
            dense_scores,
            normalized_sparse,
            normalized_dense,
            strict=True,
        ):
            combined = 0.55 * sparse_norm + 0.45 * dense_norm
            rerank = combined + self._rerank_bonus(question_tokens, chunk)
            overlap_terms = sorted(question_tokens & set(chunk.token_counts))
            results.append(
                SearchResult(
                    chunk=chunk,
                    sparse_score=round(sparse_score, 4),
                    dense_score=round(dense_score, 4),
                    combined_score=round(combined, 4),
                    rerank_score=round(rerank, 4),
                    overlap_terms=overlap_terms,
                    embedding_provider=self.embedding_provider.name,
                )
            )

        return sorted(results, key=lambda result: result.rerank_score, reverse=True)[:top_k]

    def _bm25(
        self,
        query_counts: Counter[str],
        chunk: DocumentChunk,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        score = 0.0
        chunk_length = sum(chunk.token_counts.values())
        total_chunks = len(self.chunks)

        for token in query_counts:
            if token not in chunk.token_counts:
                continue

            term_frequency = chunk.token_counts[token]
            document_frequency = self.doc_frequency[token]
            idf = math.log(
                1
                + ((total_chunks - document_frequency + 0.5) / (document_frequency + 0.5))
            )
            denominator = term_frequency + k1 * (
                1 - b + b * chunk_length / self.average_chunk_length
            )
            score += idf * ((term_frequency * (k1 + 1)) / denominator)

        return score

    def _rerank_bonus(self, question_tokens: set[str], chunk: DocumentChunk) -> float:
        chunk_tokens = set(chunk.token_counts)
        overlap_ratio = len(question_tokens & chunk_tokens) / max(len(question_tokens), 1)
        title_bonus = 0.08 if question_tokens & set(tokenize(chunk.title)) else 0.0
        exact_bonus = 0.12 if all(token in chunk_tokens for token in question_tokens) else 0.0
        type_bonus = 0.03 if chunk.content_type in {"html", "pdf"} and overlap_ratio > 0 else 0.0
        return overlap_ratio * 0.3 + title_bonus + exact_bonus + type_bonus
