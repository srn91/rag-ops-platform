from pathlib import Path

from app.config import CORPUS_DIR, PROJECT_ROOT
from app.models import SourceDocument


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return fallback.replace("-", " ").title()


def load_documents(corpus_dir: Path | None = None) -> list[SourceDocument]:
    directory = corpus_dir or CORPUS_DIR
    documents: list[SourceDocument] = []

    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        documents.append(
            SourceDocument(
                doc_id=path.stem,
                title=_extract_title(text, path.stem),
                path=relative_path,
                text=text,
            )
        )

    if not documents:
        raise RuntimeError(f"no source documents found under {directory}")

    return documents

