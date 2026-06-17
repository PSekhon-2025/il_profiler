"""Central configuration: paths, models, and pipeline hyperparameters.

Everything tunable lives here so the rest of the codebase stays free of magic
numbers and hardcoded paths. The corpus root is resolved relative to this file
(config.py sits at Research Project/code/il_profiler/il_rag/), so the project
works regardless of the current working directory.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pull TOGETHER_API_KEY from .env if present

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # .../code/il_profiler
CORPUS_ROOT = Path(__file__).resolve().parents[3]           # .../Research Project

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
PROFILES_DIR = DATA_DIR / "profiles"
for _d in (DATA_DIR, CHROMA_DIR, PROFILES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API / models
# ---------------------------------------------------------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
GENERATION_MODEL = "openai/gpt-oss-120b"                     # RAG answers + matching
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"  # 1024-dim
EMBEDDING_DIM = 1024

# ---------------------------------------------------------------------------
# Chunking / retrieval
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1400        # target characters per chunk
CHUNK_OVERLAP = 150      # characters of overlap between consecutive chunks
TOP_K = 5                # chunks retrieved per question
COLLECTION_NAME = "il_corpus"

# ---------------------------------------------------------------------------
# Study design
# ---------------------------------------------------------------------------
ORGS = ["OpenAI", "DeepMind", "Anthropic"]

# The two source types are analysed SEPARATELY by design: "published" captures
# each lab's self-presentation, "thirdparty" captures external perception.
# Comparing the two profiles per lab is part of the paper's argument.
SOURCE_TYPES = ["published", "thirdparty"]

# Published-document corpora: each is one big text file produced by converting
# the lab's PDFs; documents inside are delimited by "FILE:" header blocks.
PUBLISHED_CORPUS = {
    "OpenAI":    CORPUS_ROOT / "OpenAI"    / "OpenAI PDF's"   / "pdf_corpus.txt",
    "DeepMind":  CORPUS_ROOT / "DM"        / "Deepmind PDF's" / "pdf_corpus.txt",
    "Anthropic": CORPUS_ROOT / "Anthropic" / "Anthropic PDF's" / "pdf_corpus.txt",
}

# Third-party article dumps: directories of large concatenated RTF files.
ARTICLE_DIRS = {
    "OpenAI":    CORPUS_ROOT / "OpenAI"    / "OpenAI Articles",
    "DeepMind":  CORPUS_ROOT / "DM"        / "Deepmind Articles",
    "Anthropic": CORPUS_ROOT / "Anthropic" / "Anthropic Articles",
}


def require_api_key() -> str:
    """Return the Together API key or fail with an actionable message."""
    if not TOGETHER_API_KEY:
        raise RuntimeError(
            "TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return TOGETHER_API_KEY
