from __future__ import annotations

from html import unescape
from pathlib import Path
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import CORPUS_DIR, PROJECT_ROOT
from app.models import SourceDocument

HEADING_PATTERN = re.compile(r"^#\s+(?P<title>.+)$", re.MULTILINE)


def _fallback_title(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()


def _extract_markdown(path: Path) -> tuple[str, str, dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    heading_match = HEADING_PATTERN.search(text)
    title = heading_match.group("title").strip() if heading_match else _fallback_title(path)
    line_count = len([line for line in text.splitlines() if line.strip()])
    return title, text, {"line_count": line_count}


def _extract_html(path: Path) -> tuple[str, str, dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    title = (
        (soup.title.string.strip() if soup.title and soup.title.string else "")
        or (soup.find(["h1", "h2"]).get_text(" ", strip=True) if soup.find(["h1", "h2"]) else "")
        or _fallback_title(path)
    )
    description = ""
    meta_description = soup.find("meta", attrs={"name": "description"})
    if meta_description and meta_description.get("content"):
        description = meta_description["content"].strip()
    text = " ".join(
        part.strip()
        for part in soup.get_text("\n").splitlines()
        if part.strip()
    )
    return title, unescape(text), {
        "html_title": title,
        "meta_description": description,
    }


def _extract_pdf(path: Path) -> tuple[str, str, dict[str, object]]:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    metadata = reader.metadata or {}
    title = (str(metadata.get("/Title", "")).strip() or _fallback_title(path))
    author = str(metadata.get("/Author", "")).strip()
    return title, text, {
        "page_count": len(reader.pages),
        "pdf_title": title,
        "author": author,
    }


def _load_one(path: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    if suffix == ".md":
        title, text, metadata = _extract_markdown(path)
        content_type = "markdown"
    elif suffix in {".html", ".htm"}:
        title, text, metadata = _extract_html(path)
        content_type = "html"
    elif suffix == ".pdf":
        title, text, metadata = _extract_pdf(path)
        content_type = "pdf"
    else:  # pragma: no cover - guarded by glob pattern
        raise ValueError(f"unsupported corpus file type: {path.suffix}")

    if not text.strip():
        raise RuntimeError(f"source document {path} did not yield extractable text")

    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    metadata = {"source_extension": suffix.lstrip("."), **metadata}
    return SourceDocument(
        doc_id=path.stem,
        title=title,
        path=relative_path,
        content_type=content_type,
        metadata=metadata,
        text=text,
    )


def load_documents(corpus_dir: Path | None = None) -> list[SourceDocument]:
    directory = corpus_dir or CORPUS_DIR
    documents = [_load_one(path) for path in sorted(directory.glob("*")) if path.suffix.lower() in {".md", ".html", ".htm", ".pdf"}]
    if not documents:
        raise RuntimeError(f"no source documents found under {directory}")
    return documents
