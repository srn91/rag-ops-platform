# RAG Ops Platform - System Brief

## Problem

RAG systems fail when retrieval, grounding, and answer quality are treated as separate concerns. This project keeps ingestion, chunking, retrieval, reranking, citation handling, evaluation, and serving in one inspectable workflow.

## System Design

```mermaid
flowchart LR
    A["Markdown corpus"] --> B["Sentence-aware chunking"]
    B --> C["Sparse + dense retrieval"]
    C --> D["Reranking"]
    D --> E["Grounded answer builder"]
    E --> F["FastAPI response"]
    F --> G["Evaluation report"]
```

## Stack

- Python, FastAPI, pytest
- Hybrid sparse + dense retrieval
- Reranking, citation-backed answers, deterministic evaluation
- Render-hosted read-only demo

## Metrics

- `retrieval_hit_rate@3 = 1.0`
- `citation_hit_rate = 1.0`
- `MRR = 1.0`
- `11` quality checks passing

## Run

```bash
make setup
make test
make serve
```

Live demo: https://rag-ops-platform.onrender.com

## Production Scale Improvements

- Move document and chunk storage to PostgreSQL or a warehouse-backed document index.
- Replace the local dense retrieval path with a managed vector index when corpus size grows.
- Add per-query latency tracing and retrieval-cache metrics.
- Add evaluation suites by document class, query type, and expected citation behavior.
