"""Corpus ingestion: parse -> chunk -> embed -> persist to Chroma.

Two corpus shapes are handled, mapping to the study's two source types:

  published  — one pdf_corpus.txt per lab: the lab's PDFs converted to text and
               concatenated, each document introduced by a header block:
                   ==============================
                   FILE: <name>.pdf
                   METHOD: native | PAGES: 13
                   ------------------------------
               We split on those headers so every chunk keeps its source
               document's filename.

  thirdparty — directories of large RTF press-clipping dumps. Each RTF bundles
               many articles; we convert with striprtf and split on common
               article delimiters so chunks don't straddle unrelated articles.

Every chunk is stored with metadata {org, source_type, doc_type, filename} —
the retriever's compound filter on (org, source_type) is what keeps the six
profiles independent.

Ingestion is resumable: chunk ids are deterministic and upserted, so re-running
after an interruption just overwrites what's already there and continues.
"""
import re
from typing import Iterator

import chromadb
from chromadb.config import Settings
from striprtf.striprtf import rtf_to_text
from tqdm import tqdm

from .config import (
    ARTICLE_DIRS,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    PUBLISHED_CORPUS,
)
from .llm import embed

EMBED_BATCH = 32

# Header block introducing each document inside pdf_corpus.txt.
PDF_HEADER_RE = re.compile(
    r"=+\s*\nFILE:\s*(?P<filename>.+?)\s*\n(?:METHOD:[^\n]*\n)?-+\s*\n"
)

# Delimiters between articles inside a concatenated RTF press dump.
ARTICLE_SPLIT_RE = re.compile(
    r"\n(?=(?:End of Document|Title:|Headline:|HEADLINE:|Byline:|"
    r"Publication-Date:|Length:\s*\d+\s*words)\b)",
    re.IGNORECASE,
)

MIN_ARTICLE_CHARS = 200  # skip fragments too short to carry meaning

# --- Third-party boilerplate stripping -------------------------------------
# The RTF press dumps are BuySellSignals / News Bites / Nexis exports whose
# machine-generated boilerplate (one item per line) dominated retrieval before
# we addressed it. Three line patterns are dropped; editorial prose is kept.
# These patterns are specific to THESE exports, not a general HTML/news cleaner.

# 1. Label-prefixed metadata fields (PermID:, Load-Date:, Length: N words, ...).
_BOILERPLATE_LABEL_RE = re.compile(
    r"^\s*(PermID|Website|Address|Load-Date|Length|Byline|Dateline|Section|"
    r"Publication-Date|Publication|Copyright|Language|Distribution|Graphic|"
    r"Company|Ticker|Industry|Subject|Organization|Classification|Geographic|"
    r"Created by|Document type|Source)\b.*$",
    re.IGNORECASE,
)
# 2. Auto-generated company-profile scaffolding (SECTION 2 ..., 1.2 SUMMARY,
#    'Top Management', and bare 'Body' / 'End of Document' delimiters).
_BOILERPLATE_SCAFFOLD_RE = re.compile(
    r"^\s*(SECTION\s+\d+\b.*|\d+\.\d+\s+[A-Z][A-Z ]+|Top Management|Body|"
    r"End of Document|Name\|Designation\|.*)\s*$"
)
# 3. Pipe-delimited table rows, e.g. 'Sam Altman|Chief Executive Officer|'.
_BOILERPLATE_PIPE_RE = re.compile(r"^\s*[^|\n]{1,60}\|[^|\n]*\|\s*$")


def strip_boilerplate(text: str) -> str:
    """Remove Nexis/BuySellSignals export boilerplate, preserving prose.

    Drops metadata-field lines, company-profile scaffolding, and pipe-table
    rows; collapses the blank lines left behind. Applied per article at ingest
    so retrieval slots go to editorial text, not database metadata.
    """
    kept = [
        line for line in text.splitlines()
        if not (_BOILERPLATE_LABEL_RE.match(line)
                or _BOILERPLATE_SCAFFOLD_RE.match(line)
                or _BOILERPLATE_PIPE_RE.match(line))
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

# Length of the normalized text prefix used as a near-duplicate signature.
# The third-party press dumps are ~50% duplicates (syndicated articles + a
# LexisNexis company-profile boilerplate block repeated in every export). If
# those copies are embedded, near-duplicate passages dominate every retrieval
# and starve answers of real evidence — the main cause of spurious abstentions.
# We drop duplicate chunks at ingest, scoped per (org, source_type), so we never
# pay to embed a copy and the index stays clean.
_DEDUP_SIGNATURE_CHARS = 240


def _dedup_signature(text: str) -> str:
    """Normalized prefix used to detect near-duplicate chunks."""
    return re.sub(r"\s+", " ", text).strip().lower()[:_DEDUP_SIGNATURE_CHARS]


def dedup_corpus(items: Iterator[dict]) -> Iterator[dict]:
    """Drop near-duplicate chunks, keeping the first per (org, source_type)."""
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        meta = item["metadata"]
        key = (meta["org"], meta["source_type"], _dedup_signature(item["text"]))
        if key in seen:
            continue
        seen.add(key)
        yield item


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunking that prefers natural break points.

    Within each window we break at the last newline (or failing that, sentence
    end) in the second half of the window, so chunks end on coherent boundaries
    instead of mid-sentence wherever possible.
    """
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            brk = text.rfind("\n", start, end)
            if brk <= start + size // 2:
                brk = text.rfind(". ", start, end)
            if brk > start + size // 2:
                end = brk + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def split_published_corpus(raw: str) -> list[tuple[str, str]]:
    """Split a pdf_corpus.txt into (filename, document_text) pairs."""
    headers = list(PDF_HEADER_RE.finditer(raw))
    if not headers:
        # Defensive fallback: treat the whole file as one unnamed document
        # rather than silently ingesting nothing.
        return [("(whole corpus)", raw.strip())] if raw.strip() else []
    docs = []
    for i, h in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        body = raw[h.end():end].strip()
        if body:
            docs.append((h.group("filename").strip(), body))
    return docs


def _iter_published() -> Iterator[dict]:
    """Yield chunks from every lab's published-document corpus."""
    for org, path in PUBLISHED_CORPUS.items():
        if not path.exists():
            print(f"[warn] missing published corpus for {org}: {path}")
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        for doc_idx, (filename, body) in enumerate(split_published_corpus(raw)):
            for ci, chunk in enumerate(chunk_text(body)):
                yield {
                    "id": f"{org}|published|{doc_idx}|{ci}",
                    "text": chunk,
                    "metadata": {
                        "org": org,
                        "source_type": "published",
                        "doc_type": "pdf",
                        "filename": filename,
                    },
                }


def _iter_thirdparty() -> Iterator[dict]:
    """Yield chunks from every lab's third-party RTF article dumps."""
    for org, dirpath in ARTICLE_DIRS.items():
        if not dirpath.exists():
            print(f"[warn] missing article dir for {org}: {dirpath}")
            continue
        rtf_files = sorted(p for p in dirpath.iterdir() if p.suffix.lower() == ".rtf")
        for rtf_path in rtf_files:
            try:
                plain = rtf_to_text(
                    rtf_path.read_text(encoding="utf-8", errors="ignore"),
                    errors="ignore",
                )
            except Exception as e:  # noqa: BLE001 — one bad file must not kill the run
                print(f"[warn] could not parse {rtf_path.name}: {e}")
                continue
            parts = [p.strip() for p in ARTICLE_SPLIT_RE.split(plain) if p.strip()]
            articles = parts if len(parts) > 1 else [plain.strip()]
            for ai, article in enumerate(articles):
                # Strip export boilerplate AFTER splitting: the split markers
                # (End of Document, Length: N words, ...) are themselves
                # boilerplate, so they must survive long enough to delimit
                # articles, then get removed here. Length check uses the cleaned
                # text so metadata-only fragments are dropped.
                article = strip_boilerplate(article)
                if len(article) < MIN_ARTICLE_CHARS:
                    continue
                for ci, chunk in enumerate(chunk_text(article)):
                    yield {
                        "id": f"{org}|thirdparty|{rtf_path.stem}|{ai}|{ci}",
                        "text": chunk,
                        "metadata": {
                            "org": org,
                            "source_type": "thirdparty",
                            "doc_type": "article",
                            "filename": rtf_path.name,
                        },
                    }


def iter_corpus() -> Iterator[dict]:
    yield from _iter_published()
    yield from _iter_thirdparty()


def _embed_resilient(texts: list[str]) -> list[list[float] | None]:
    """Embed a batch; on oversize (400) errors bisect so only the single
    offending chunk is dropped (returned as None), never the whole batch."""
    try:
        return embed(texts)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        oversize = "400" in str(e) or "too long" in msg or "maximum context" in msg
        if not oversize:
            raise
        if len(texts) == 1:
            print(f"  [drop] chunk rejected by embedder ({len(texts[0])} chars)")
            return [None]
        mid = len(texts) // 2
        return _embed_resilient(texts[:mid]) + _embed_resilient(texts[mid:])


def build_index(fresh: bool = False) -> None:
    """Chunk, embed, and upsert the full corpus into Chroma.

    Args:
        fresh: delete any existing collection first and rebuild from scratch.
    """
    chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    if fresh:
        try:
            chroma.delete_collection(COLLECTION_NAME)
        except Exception:  # collection may not exist yet
            pass
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    batch: list[dict] = []
    indexed = dropped = 0

    def flush() -> None:
        nonlocal indexed, dropped
        if not batch:
            return
        vectors = _embed_resilient([b["text"] for b in batch])
        keep = [(b, v) for b, v in zip(batch, vectors) if v is not None]
        dropped += len(batch) - len(keep)
        if keep:
            collection.upsert(
                ids=[b["id"] for b, _ in keep],
                embeddings=[v for _, v in keep],
                documents=[b["text"] for b, _ in keep],
                metadatas=[b["metadata"] for b, _ in keep],
            )
            indexed += len(keep)
        batch.clear()

    for item in tqdm(dedup_corpus(iter_corpus()), desc="ingest", unit="chunk"):
        batch.append(item)
        if len(batch) >= EMBED_BATCH:
            flush()
    flush()

    print(f"indexed {indexed} chunks into '{COLLECTION_NAME}' ({dropped} dropped)")
    # Per-(org, source_type) counts so scoping problems are visible immediately.
    for org in PUBLISHED_CORPUS:
        for st in ("published", "thirdparty"):
            n = len(collection.get(
                where={"$and": [{"org": org}, {"source_type": st}]},
                include=[],
            )["ids"])
            print(f"  {org:<10} {st:<10} {n} chunks")
