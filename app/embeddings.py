from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class EmbeddingProvider(Protocol):
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        ...

    def transform_query(self, text: str) -> np.ndarray:
        ...


@dataclass
class DeterministicLocalEmbedder:
    name: str = "local_tfidf_svd"
    max_features: int = 2048
    ngram_range: tuple[int, int] = (1, 2)
    random_state: int = 7

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        if matrix.shape[1] >= 2:
            components = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if components >= 2:
                self._svd = TruncatedSVD(
                    n_components=components,
                    random_state=self.random_state,
                )
                dense = self._svd.fit_transform(matrix)
                self._fitted = True
                return normalize(dense)
        self._fitted = True
        return normalize(matrix.toarray())

    def transform_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("embedding provider has not been fitted")
        matrix = self.vectorizer.transform([text])
        if self._svd is not None:
            return normalize(self._svd.transform(matrix))
        return normalize(matrix.toarray())


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "sentence_transformer_all_minilm_l6_v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed; use the default local provider "
                "or install the optional dependency to enable this embedder."
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=float)

    def transform_query(self, text: str) -> np.ndarray:
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embedding, dtype=float)


def build_embedding_provider() -> EmbeddingProvider:
    provider = os.getenv("RAG_EMBEDDING_PROVIDER", "local").strip().lower()
    if provider in {"local", "local_tfidf_svd", ""}:
        return DeterministicLocalEmbedder()
    if provider in {"sentence-transformer", "sentence_transformer"}:
        return SentenceTransformerEmbedder()
    raise ValueError(
        "unsupported RAG_EMBEDDING_PROVIDER. Expected one of: "
        "'local', 'local_tfidf_svd', 'sentence-transformer'."
    )
