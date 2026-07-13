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
# Locally the raw corpus lives three levels up (.../Research Project). In a
# container the app sits at /app, which has no parents[3] — indexing it raised
# IndexError at import and crashed the whole app. Fall back to PROJECT_ROOT:
# cloud deployments ship only the prebuilt index, so the corpus paths derived
# from CORPUS_ROOT simply won't exist there (ingestion is disabled anyway).
_here = Path(__file__).resolve()
CORPUS_ROOT = _here.parents[3] if len(_here.parents) > 3 else PROJECT_ROOT

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
# Hallucination / grounding checks — ALL opt-in; a default run never reads
# these except at import, and produces byte-identical output with them present.
# ---------------------------------------------------------------------------
# Feature 1 (grounding pre-check): rows whose retrieval-grounding score (max
# lexical content-token recall of the question against its retrieved chunks,
# in [0, 1]) falls below this are bucketed "retrieval_missed". Tune per corpus.
GROUNDING_LOW_THRESHOLD = 0.2

# Feature 3 (metamorphic label-stability eval).
METAMORPHIC_PARAPHRASES = 3          # meaning-preserving paraphrases per item
# label_stability below this flags an item unstable. 1.0 = any paraphrase that
# flips the predicted logic flags the item; relax if paraphrase noise is high.
METAMORPHIC_STABILITY_THRESHOLD = 1.0
# Paraphrases are sampled at nonzero temperature so the k variants differ;
# answering and matching stay at temperature 0 like the production path.
METAMORPHIC_PARAPHRASE_TEMPERATURE = 0.9

# Lab-name swap: which lab replaces which in the swap variant, and the alias
# spellings to rewrite. The swap is a deterministic regex substitution — no
# LLM — so the swap itself cannot drift the text's meaning.
LAB_SWAP = {
    "OpenAI": "DeepMind",
    "DeepMind": "Anthropic",
    "Anthropic": "OpenAI",
}
LAB_ALIASES = {
    "OpenAI": ["OpenAI", "Open AI"],
    "DeepMind": ["Google DeepMind", "DeepMind", "Deep Mind"],
    "Anthropic": ["Anthropic"],
}

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
