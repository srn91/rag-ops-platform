from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus"
EVALUATION_CASES_PATH = PROJECT_ROOT / "eval" / "golden_questions.json"

