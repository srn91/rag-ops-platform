# Ingestion Pipeline

The RAG Ops Platform ingests Markdown operating notes, incident retrospectives, and architecture records from a local corpus directory. Each source document is normalized into UTF-8 text before indexing so the retrieval path is deterministic and easy to test during local development.

After loading the raw files, the ingestion layer splits each document into sentence-aware chunks instead of fixed character windows. This keeps section meaning intact while still producing chunks small enough for retrieval, reranking, and citations. Adjacent chunks share one overlapping sentence so operational details are less likely to get cut across chunk boundaries.

Every chunk stores document metadata, token counts, and a lightweight dense representation derived from hashed term frequencies. That gives the system enough structure to evaluate retrieval behavior without depending on a hosted vector database during the first portfolio milestone.

The current ingestion path is intentionally local-first. It is designed to be replaced later with connectors for PDFs, S3, or object-store sync once the public repository has a stable baseline.

