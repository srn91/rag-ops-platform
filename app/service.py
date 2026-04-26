from app.answering import build_grounded_answer
from app.corpus import load_documents
from app.evaluation import run_evaluation
from app.retrieval import HybridIndex


class RAGService:
    def __init__(self) -> None:
        self.documents = load_documents()
        self.index = HybridIndex(self.documents)

    def stats(self) -> dict[str, int]:
        return self.index.stats()

    def list_documents(self) -> list[dict[str, str]]:
        return [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "path": document.path,
            }
            for document in self.documents
        ]

    def query(self, question: str, *, top_k: int = 5) -> dict[str, object]:
        results = self.index.search(question, top_k=top_k)
        return build_grounded_answer(question, results)

    def evaluate(self) -> dict[str, object]:
        return run_evaluation(self.index)

