# Evaluation and Operations

The repository includes a golden question set so retrieval quality can be checked every time the corpus or ranking logic changes. Each evaluation case names the expected source document, which lets the project track hit rate at top three results, mean reciprocal rank, and whether the final answer cites the right document.

Evaluation is exposed through both the FastAPI service and the local CLI. That keeps validation close to the application surface instead of burying it in notebook-only scripts. The goal is to make quality checks part of the normal developer workflow before any future publish or deployment step.

Operationally, the service exposes health, corpus inventory, query, and evaluation endpoints. Those endpoints create a small but realistic control plane for a retrieval system: operators can confirm what is indexed, run a known question, inspect citations, and compare current behavior with the expected answer sources.

This evaluation layer is intentionally deterministic and local-first, which keeps latency tracking, faithfulness checks, and offline replay reproducible without external credentials.
