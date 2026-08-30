"""Centralized configuration constants for the RAG QnA pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ── Embedding model ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-mpnet-base-v2"

# ── Chunking defaults ─────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 500        # characters
DEFAULT_CHUNK_OVERLAP = 50      # characters
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1000
CHUNK_SIZE_STEP = 50
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 200
CHUNK_OVERLAP_STEP = 25

# ── Retrieval defaults ────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 10

# ── QA models ─────────────────────────────────────────────────────────────
QA_MODELS = {
    "BERT-Large (Accurate)": "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "DistilBERT (Fast)": "distilbert/distilbert-base-cased-distilled-squad",
}
DEFAULT_QA_MODEL_KEY = "DistilBERT (Fast)"

# ── Evaluation experiment grid ────────────────────────────────────────────
EVAL_CHUNK_SIZES = [300, 500, 700, 1000]
EVAL_TOP_K_VALUES = [1, 3, 5, 7]
