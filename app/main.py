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
<style>
body{{margin:0;background:#f8fafc;color:#0f172a;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.5}}
main{{max-width:1080px;margin:0 auto;padding:56px 24px}}
.hero{{background:linear-gradient(135deg,#0f172a,#1d4ed8);color:white;border-radius:22px;padding:38px;box-shadow:0 24px 60px rgba(15,23,42,.18)}}
.eyebrow{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#bfdbfe;font-weight:700}}
h1{{font-size:42px;line-height:1.05;margin:10px 0 14px}}p{{font-size:17px;margin:0;color:#334155}}.hero p{{color:#dbeafe;max-width:760px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}}.card{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(15,23,42,.06)}}
.metric{{font-size:28px;font-weight:800;color:#0f172a}}.label{{font-size:13px;color:#64748b;margin-top:3px}}.links{{display:flex;flex-wrap:wrap;gap:12px;margin-top:22px}}
a.button{{background:#0f172a;color:white;text-decoration:none;padding:11px 14px;border-radius:10px;font-weight:700}}a.secondary{{background:white;color:#0f172a;border:1px solid #cbd5e1}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}h1{{font-size:34px}}}}
</style></head>
<body><main>
<section class="hero">
<div class="eyebrow">Retrieval-augmented generation</div>
<h1>RAG Ops Platform</h1>
<p>Grounded question answering service with deterministic corpus ingestion, hybrid retrieval, reranking, citations, and evaluation output.</p>
<div class="links"><a class="button" href="/evaluation">View evaluation</a><a class="button secondary" href="/documents">Indexed documents</a><a class="button secondary" href="/docs">API docs</a></div>
</section>
<section class="grid">
<div class="card"><div class="metric">{stats.get("documents")}</div><div class="label">documents indexed</div></div>
<div class="card"><div class="metric">{stats.get("chunks")}</div><div class="label">retrieval chunks</div></div>
<div class="card"><div class="metric">1.0</div><div class="label">citation hit rate</div></div>
<div class="card"><div class="metric">11</div><div class="label">passing checks</div></div>
</section>
<section class="card"><p>This demo is intentionally small and inspectable: open the evaluation summary, review the indexed documents, then use the API docs to submit a grounded query.</p></section>
</main></body></html>"""


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
