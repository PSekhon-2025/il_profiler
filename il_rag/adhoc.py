"""Ad-hoc analysis: run the questionnaire against documents dropped into the UI.

The six research profiles come from a curated, pre-ingested corpus. This module
is for the other case — someone has a document in hand and wants to know how it
reads against the seven logics, without touching that corpus at all.

Deliberately does NOT use Chroma. Uploaded chunks are embedded in memory and
retrieved by brute-force cosine, for two reasons:

  1. ISOLATION. The six profiles are the study's result; a stray upload must
     never be able to contaminate the index they are computed from. Not writing
     is a stronger guarantee than writing carefully and cleaning up.
  2. It is simply faster here. Exhaustive similarity over a few hundred chunks
     is instant, and needs no collection lifecycle, no cleanup, no namespacing.

Everything downstream of retrieval is the PRODUCTION path, unchanged:
rag_qa.answer_question with injected chunks (the same entry point the
metamorphic probes use), then graded_matcher.match_graded, then
profile_harness.aggregate. So an ad-hoc profile is computed by the same code as
a real one — only the evidence source differs.

The questionnaire is templated with a subject name, so the caller must supply
one: the questions literally ask what "{org}" does, and answering them about an
unnamed entity would change what is being measured.
"""
import io
import re
from dataclasses import dataclass

from .embedding_agreement import _cosine
from .graded_matcher import match_graded
from .ingest import chunk_text
from .llm import embed
from .profile_harness import aggregate
from .questionnaire import build_questionnaire
from .rag_qa import answer_question
from .retriever import Chunk

# What a dropped file may be. PDF needs pypdf; RTF needs striprtf (already a
# core dependency); text formats need nothing.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".rtf"}

# Ad-hoc runs are labelled with this source_type so a saved result can never be
# mistaken for one of the two curated corpora.
ADHOC_SOURCE_TYPE = "uploaded"

EMBED_BATCH = 32


@dataclass
class AdhocDoc:
    """One uploaded document after text extraction."""
    filename: str
    text: str
    n_chars: int
    error: str | None = None


def extract_text(filename: str, data: bytes) -> AdhocDoc:
    """Pull plain text out of one uploaded file, by extension.

    Never raises: an unreadable file becomes a doc carrying an `error`, so one
    bad drop degrades that file rather than the whole batch.
    """
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    try:
        if suffix in TEXT_SUFFIXES:
            text = data.decode("utf-8", errors="replace")
        elif suffix == ".rtf":
            from striprtf.striprtf import rtf_to_text
            text = rtf_to_text(data.decode("utf-8", errors="ignore"),
                               errors="ignore")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                return AdhocDoc(filename, "", 0,
                                error="PDF support needs pypdf (pip install pypdf)")
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            return AdhocDoc(filename, "", 0,
                            error=f"unsupported file type '{suffix or '?'}' "
                                  f"(accepts: {', '.join(sorted(SUPPORTED_SUFFIXES))})")
    except Exception as e:  # noqa: BLE001 — one bad file must not kill the batch
        return AdhocDoc(filename, "", 0, error=f"could not read: {e}")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return AdhocDoc(filename, "", 0,
                        error="no extractable text (a scanned PDF needs OCR)")
    return AdhocDoc(filename, text, len(text))


def build_chunks(docs: list[AdhocDoc], subject: str) -> list[Chunk]:
    """Chunk the readable documents, using the same splitter as ingest."""
    chunks: list[Chunk] = []
    for doc in docs:
        if doc.error or not doc.text:
            continue
        for i, piece in enumerate(chunk_text(doc.text)):
            chunks.append(Chunk(
                id=f"adhoc::{doc.filename}::chunk{i}",
                text=piece, org=subject, source_type=ADHOC_SOURCE_TYPE,
                filename=doc.filename, score=0.0,
            ))
    return chunks


def embed_chunks(chunks: list[Chunk], progress=None) -> list[list[float]]:
    """Embed every chunk once, batched. This is the only API cost before the
    questionnaire itself starts."""
    vectors: list[list[float]] = []
    texts = [c.text for c in chunks]
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embed(texts[i:i + EMBED_BATCH]))
        if progress:
            progress(min(i + EMBED_BATCH, len(texts)), len(texts))
    return vectors


def retrieve(question: str, chunks: list[Chunk], vectors: list[list[float]],
             k: int) -> list[Chunk]:
    """Top-k chunks by cosine similarity — exhaustive, no index.

    Returns copies carrying their similarity in `score`, so the UI can show how
    well-grounded each answer was, exactly as the Chroma path does.
    """
    if not chunks:
        return []
    qv = embed([question])[0]
    scored = sorted(
        ((_cosine(qv, v), c) for v, c in zip(vectors, chunks)),
        key=lambda pair: pair[0], reverse=True,
    )[:k]
    return [Chunk(id=c.id, text=c.text, org=c.org, source_type=c.source_type,
                  filename=c.filename, score=round(float(s), 4))
            for s, c in scored]


def analyze(chunks: list[Chunk], vectors: list[list[float]], subject: str,
            k: int = 5, progress=None) -> dict:
    """Run the full questionnaire against uploaded chunks only.

    Identical to the production path from retrieval onward, so the resulting
    percentages mean the same thing as a corpus profile — computed over a
    document set instead of a corpus.
    """
    questions = build_questionnaire(subject)
    rows = []
    for n, q in enumerate(questions, 1):
        evidence = retrieve(q["question"], chunks, vectors, k)
        rag = answer_question(None, q["question"], org=subject,
                              source_type=ADHOC_SOURCE_TYPE, chunks=evidence)
        verdict = match_graded(question=q["question"], candidate=rag.answer,
                               category=q["category"], variant=q["variant"])
        rows.append({
            "org": subject, "source_type": ADHOC_SOURCE_TYPE,
            "qid": q["qid"], "category": q["category"], "variant": q["variant"],
            "question": q["question"], "answer": rag.answer,
            "retrieved_ids": [c.id for c in evidence],
            "retrieved": [{"id": c.id, "filename": c.filename,
                           "score": c.score, "text": c.text} for c in evidence],
            "abstain": verdict["abstain"], "weights": verdict["weights"],
            "reasoning": verdict["reasoning"],
        })
        if progress:
            progress(n, len(questions))

    profiles = aggregate(rows, [subject], [ADHOC_SOURCE_TYPE])
    return {"subject": subject, "rows": rows,
            "profile": profiles[subject][ADHOC_SOURCE_TYPE]}
