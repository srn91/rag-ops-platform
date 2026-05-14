from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.service import RAGService


class QueryRequest(BaseModel):
    question: str = Field(min_length=5, description="Question to answer from the local corpus")
    top_k: int = Field(default=5, ge=1, le=8, description="Maximum number of chunks to retrieve")


service = RAGService()

app = FastAPI(
    title="RAG Ops Platform",
    description="A local, inspectable RAG demo with ingestion, hybrid retrieval, reranking, citations, and evaluation.",
    version="0.2.0",
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    stats = service.stats()
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Ops Platform</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:860px;margin:48px auto;padding:0 24px;line-height:1.5;color:#111}}a{{color:#0645ad}}</style></head>
<body>
<h1>RAG Ops Platform</h1>
<p>Inspectable RAG service with corpus ingestion, chunking, hybrid retrieval, reranking, citations, and evaluation output.</p>
<ul><li>Documents indexed: {stats.get("documents")}</li><li>Chunks indexed: {stats.get("chunks")}</li></ul>
<h2>Open endpoints</h2>
<ul>
<li><a href="/evaluation">Evaluation summary</a></li>
<li><a href="/documents">Indexed documents</a></li>
<li><a href="/docs">API docs</a></li>
</ul>
</body></html>"""


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
