from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.service import RAGService


class QueryRequest(BaseModel):
    question: str = Field(min_length=5, description="Question to answer from the local corpus")
    top_k: int = Field(default=5, ge=1, le=8, description="Maximum number of chunks to retrieve")


service = RAGService()

app = FastAPI(
    title="RAG Ops Platform",
    description="A local, production-style RAG demo with ingestion, hybrid retrieval, reranking, citations, and evaluation.",
    version="0.2.0",
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "project": "rag-ops-platform",
        "status": "v1-local",
        "capabilities": [
            "markdown corpus ingestion",
            "sentence-aware chunking",
            "hybrid sparse+dense retrieval",
            "lightweight reranking",
            "citation-backed answers",
            "retrieval evaluation",
        ],
        "indexed_assets": service.stats(),
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "indexed_assets": service.stats()}


@app.get("/documents")
def documents() -> dict[str, object]:
    return {"documents": service.list_documents()}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, object]:
    return service.query(request.question, top_k=request.top_k)


@app.get("/evaluation")
def evaluation() -> dict[str, object]:
    return service.evaluate()

