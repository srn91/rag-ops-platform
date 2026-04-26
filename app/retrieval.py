from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

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


def _hash_vector(token_counts: Counter[str], dimensions: int = 64) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for token, count in token_counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * float(count)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


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
                    text=text,
                    token_counts=dict(token_counts),
                    vector=_hash_vector(token_counts),
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
                text=text,
                token_counts=dict(token_counts),
                vector=_hash_vector(token_counts),
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
    def __init__(self, documents: list[SourceDocument]) -> None:
        self.documents = documents
        self.chunks = [chunk for document in documents for chunk in chunk_document(document)]
        if not self.chunks:
            raise RuntimeError("hybrid index requires at least one chunk")

        self.doc_frequency: Counter[str] = Counter()
        self.average_chunk_length = 0.0

        total_length = 0
        for chunk in self.chunks:
            total_length += sum(chunk.token_counts.values())
            self.doc_frequency.update(chunk.token_counts.keys())

        self.average_chunk_length = total_length / len(self.chunks)

    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

    def search(self, question: str, *, top_k: int = 5) -> list[SearchResult]:
        query_counts = Counter(tokenize(question))
        if not query_counts:
            return []

        query_vector = _hash_vector(query_counts)
        sparse_scores = [self._bm25(query_counts, chunk) for chunk in self.chunks]
        dense_scores = [_cosine_similarity(query_vector, chunk.vector) for chunk in self.chunks]
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
            combined = 0.6 * sparse_norm + 0.4 * dense_norm
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
                )
            )

        return sorted(results, key=lambda result: result.rerank_score, reverse=True)[:top_k]

    def _bm25(self, query_counts: Counter[str], chunk: DocumentChunk, *, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        chunk_length = sum(chunk.token_counts.values())
        total_chunks = len(self.chunks)

        for token in query_counts:
            if token not in chunk.token_counts:
                continue

            term_frequency = chunk.token_counts[token]
            document_frequency = self.doc_frequency[token]
            idf = math.log(1 + ((total_chunks - document_frequency + 0.5) / (document_frequency + 0.5)))
            denominator = term_frequency + k1 * (1 - b + b * chunk_length / self.average_chunk_length)
            score += idf * ((term_frequency * (k1 + 1)) / denominator)

        return score

    def _rerank_bonus(self, question_tokens: set[str], chunk: DocumentChunk) -> float:
        chunk_tokens = set(chunk.token_counts)
        overlap_ratio = len(question_tokens & chunk_tokens) / max(len(question_tokens), 1)
        title_bonus = 0.08 if question_tokens & set(tokenize(chunk.title)) else 0.0
        exact_bonus = 0.12 if all(token in chunk_tokens for token in question_tokens) else 0.0
        return overlap_ratio * 0.3 + title_bonus + exact_bonus
