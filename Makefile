.PHONY: run test lint evaluate verify

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	ruff check app tests

evaluate:
	python3 -m app.cli evaluate

verify: lint test evaluate
