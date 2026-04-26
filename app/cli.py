from __future__ import annotations

import argparse
import json

from app.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Local tooling for rag-ops-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="run a grounded query against the sample corpus")
    query_parser.add_argument("question", help="question to ask the RAG service")

    subparsers.add_parser("evaluate", help="run retrieval and citation checks")

    args = parser.parse_args()
    service = RAGService()

    if args.command == "query":
        payload = service.query(args.question)
    else:
        payload = service.evaluate()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

