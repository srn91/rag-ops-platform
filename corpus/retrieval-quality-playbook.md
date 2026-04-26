# Retrieval Quality Playbook

The platform uses hybrid retrieval because pure keyword search misses loosely related passages and single-signal ranking can miss useful chunk overlap. Query terms first score every chunk with a BM25-style sparse ranker, then the same query is projected into a deterministic local dense embedding space built from TF-IDF plus truncated SVD so adjacent lexical structure can still influence recall without introducing an external embedding dependency.

After the two retrieval signals are combined, a lightweight reranker boosts chunks whose titles and token overlap closely match the question. This extra pass is deliberately simple, but it keeps the ranking traceable because every boost can be explained from the actual chunk text.

The platform reduces hallucinations by drafting answers only from reranked chunks and by returning citations alongside every response. When the top results do not provide enough overlap with the user question, the answer generator falls back to quoting the highest-ranked chunk instead of inventing unsupported claims.

This repository keeps the quality strategy transparent on purpose. The objective is not to hide the retrieval logic behind a framework, but to show the engineering tradeoffs between explainability, determinism, and production realism.
