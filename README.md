# rag-ops-platform

An auditable RAG pipeline for grounded question answering that ingests a small document corpus, builds a transparent hybrid retrieval index, reranks candidate chunks, and returns citation-backed answers with retrieval evaluation metrics.

This repo is aimed at the production risks companies actually care about: reducing hallucinations, surfacing why retrieval failed, and making LLM-backed answers easier to trust in internal knowledge search, support copilots, and compliance-sensitive Q&A systems.

The current V1 runs fully local and does not require external model credentials or hosted vector infrastructure, so the full retrieval path stays reproducible and inspectable.

## Problem

Many RAG demos prove that a model can answer questions, but they do not prove that the retrieval layer is grounded, inspectable, or testable. This project focuses on the infrastructure around the answer: corpus ingestion, chunking, hybrid retrieval, ranking traces, citations, and evaluation hooks that reduce hallucination risk and make retrieval failures debuggable.

## Architecture

The current V1 is intentionally local and deterministic. Instead of hiding the system behind hosted vector services or framework abstractions, the repo shows the moving parts directly:

- Markdown documents are loaded from a versioned sample corpus.
- The ingestion layer creates sentence-aware chunks with one-sentence overlap.
- Each chunk is indexed with a BM25-style sparse representation and a deterministic hashed vector sketch.
- Query-time ranking combines sparse and hashed-vector scores, then applies a simple reranker.
- The answer generator selects the highest-overlap sentences from retrieved chunks and returns citations.
- Query traces surface overlap terms, rerank margins, and latency measurements for operational debugging.
- A golden question set measures retrieval hit rate, citation hit rate, mean reciprocal rank, and latency/ranking diagnostics.

```mermaid
flowchart LR
    A["corpus/*.md"] --> B["Document loader"]
    B --> C["Sentence-aware chunker"]
    C --> D["Hybrid index<br/>BM25-style sparse + hashed vector sketches"]
    D --> E["Hybrid search"]
    E --> F["Lightweight reranker"]
    F --> G["Answer composer"]
    F --> H["Evaluation runner"]
    G --> I["/query"]
    H --> J["/evaluation"]
    D --> K["/documents"]
    D --> L["/health"]
```

## Query Contract

The main request shape is intentionally small so the retrieval path stays easy to test and explain:

```json
{
  "question": "How does the platform reduce hallucinations?",
  "top_k": 3
}
```

The `/query` response returns grounded answer data, citations, a retrieval trace, and query diagnostics. The important fields are `question`, `answer`, `citations`, `retrieval`, and `diagnostics`.

## Repo Layout

```text
rag-ops-platform/
├── app/
│   ├── answering.py
│   ├── cli.py
│   ├── corpus.py
│   ├── evaluation.py
│   ├── main.py
│   ├── models.py
│   ├── retrieval.py
│   └── service.py
├── corpus/
├── eval/
└── tests/
```

## Tradeoffs

This implementation makes three deliberate V1 tradeoffs:

1. The vector side uses deterministic hashed term sketches instead of learned embeddings. That keeps the repo runnable without credentials and makes ranking behavior stable in tests, but it is a lexical proxy rather than semantic embedding retrieval.
2. Answer generation is extractive rather than generative. The current goal is grounded retrieval and citation quality, not free-form model fluency.
3. The corpus is small and local. This repo is proving system shape and evaluation discipline before adding PDF ingestion, remote storage, or hosted vector infrastructure.

## Run Steps

### Local API

```bash
git clone https://github.com/srn91/rag-ops-platform.git
cd rag-ops-platform
python3 -m pip install -r requirements.txt
make run
```

Open the API docs at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/documents`

### Example Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does the platform reduce hallucinations?","top_k":3}'
```

Example response shape:

```json
{
  "question": "How does the platform reduce hallucinations?",
  "answer": "...",
  "citations": [
    {
      "doc_id": "retrieval-quality-playbook",
      "chunk_id": "retrieval-quality-playbook:2"
    }
  ],
  "retrieval": [
    {
      "doc_id": "retrieval-quality-playbook",
      "rerank_score": 0.91,
      "overlap_terms": ["hallucinations", "platform", "reduce"]
    }
  ],
  "diagnostics": {
    "latency_ms": {
      "retrieval": 1.2,
      "answer": 0.4,
      "total": 1.6
    },
    "ranking": {
      "retrieved_chunk_count": 3,
      "top_result_margin": 0.18
    }
  }
}
```

### CLI Evaluation

```bash
make evaluate
```

### Docker Compose

```bash
docker compose up --build
```

The Docker path is also credential-free because retrieval uses deterministic hashed vectors rather than external embedding APIs.

Make sure the Docker daemon is running before you start the stack. On macOS that usually means Docker Desktop is open before you run the compose command.

If host port `8000` is already occupied on your machine, you can override it without editing repo files:

```bash
RAG_PORT=8006 docker compose up --build
```

Then open the containerized API on `http://127.0.0.1:8006`.

Under the hood, `docker compose up --build` now builds a dedicated image with dependencies baked in and runs the API without live-reload flags, so the container path matches the published V1 story rather than a development-only shell command.

## Hosted Deployment

- Live URL: `https://rag-ops-platform.onrender.com`
- Click first: [`/evaluation`](https://rag-ops-platform.onrender.com/evaluation)
- Browser smoke: Render-hosted `/evaluation` loaded in a real browser and returned the live retrieval-quality summary with all three golden cases present.
- Render service config: Python web service on `main`, auto-deploy on commit, region `oregon`, plan `free`, build `pip install -r requirements.txt`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check `/health`.
- Render deploy command: `render deploys create srv-d7n6572pmmbs73cb5i10 --confirm`

## Validation

The repo includes three verification paths:

- `make lint` runs Ruff against the application and tests.
- `make test` exercises the API, retrieval, chunking, grounded-answer, and evaluation paths with pytest.
- `make evaluate` runs the golden question set and reports retrieval hit rate, citation hit rate, and mean reciprocal rank.

Expected local verification flow:

```bash
make verify
```

Current verification snapshot from the latest local run:

- `make lint`: passed
- `make test`: passed (`8 passed`)
- `make evaluate`: passed with `retrieval_hit_rate_at_3=1.0`, `citation_hit_rate=1.0`, `mean_reciprocal_rank=1.0`, plus latency and rerank-margin diagnostics

## Current Capabilities

The current V1 supports:

- corpus ingestion from versioned Markdown files
- sentence-aware chunking with overlap
- hybrid retrieval using sparse and hashed-vector signals
- reranked, citation-backed answers
- per-query latency and ranking diagnostics in both `/query` and `/evaluation`
- document inventory and health endpoints
- retrieval evaluation with golden questions

## What This Proves

This repo proves the retrieval layer, not just the answer text. A hiring reviewer can inspect chunking, hybrid ranking, reranking, citations, and evaluation metrics from the same local codebase, which makes the system easier to trust and harder to hand-wave.

## Next Steps

Realistic follow-up work for the next milestone:

1. add PDF and HTML ingestion with metadata extraction
2. replace the hashed vector sketch with real embedding generation behind a pluggable interface
3. add faithfulness diagnostics that compare answer text against the cited retrieval context
4. support larger corpora with persistent vector and sparse indexes
5. expand evaluation into faithfulness and answer completeness checks
