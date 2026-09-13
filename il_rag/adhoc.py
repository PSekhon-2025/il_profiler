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

An analysis can be SAVED (save_run) as a real run snapshot — same folder, same
filenames as a corpus run, so the post-hoc judges read it with no special
casing — but stamped kind="adhoc" in meta.json so the study's own pickers can
filter it out. Two deliberate differences from a corpus snapshot:

  - the rows keep their `retrieved` evidence text. Corpus rows store only
    chunk ids because the ids resolve against Chroma; ad-hoc chunk ids resolve
    against nothing once the upload is gone, so the text has to travel with
    the row or the saved audit trail would be empty.
  - CURRENT is never moved. That pointer is the corpus pipeline's resume
    cursor, and a saved document analysis is not something to resume into.
"""
import csv
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime

from . import runs
from .config import EMBEDDING_MODEL, GENERATION_MODEL
from .embedding_agreement import _cosine
from .graded_matcher import match_graded
from .ingest import chunk_text
from .llm import embed
from .profile_harness import aggregate
from .questionnaire import LOGICS, build_questionnaire
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


# ---------------------------------------------------------------------------
# Persistence: an analysis as a tagged run snapshot
# ---------------------------------------------------------------------------
def save_run(result: dict, *, k: int, label: str | None = None,
             documents: list[str] | None = None) -> str:
    """Persist one analyze() result as a run snapshot. Returns the run id.

    Writes exactly the files a corpus run writes — per_question.jsonl,
    company_profiles.json, profiles_matrix.csv, questionnaire.json, meta.json —
    so every reader in the app and every post-hoc check works on it unchanged.
    The snapshot is stamped kind="adhoc", and CURRENT is deliberately left
    alone (see the module docstring).
    """
    subject = result["subject"]
    rows = result["rows"]
    profile = result["profile"]

    run_id = runs._mint_run_id()
    paths = runs.run_paths(run_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    with open(paths["per_question"], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    profiles = {subject: {ADHOC_SOURCE_TYPE: profile}}
    paths["profiles_json"].write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(paths["profiles_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["org", "source_type", "answered", "abstained", *LOGICS])
        w.writerow([subject, ADHOC_SOURCE_TYPE, profile["answered"],
                    profile["abstained"],
                    *[profile["logic_pct"].get(logic, 0.0) for logic in LOGICS]])

    # The judges (embedding agreement, keyword agreement) read this to recover
    # the references a row was graded against, so it is not optional.
    paths["questionnaire"].write_text(
        json.dumps(runs.snapshot_questionnaire(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    now = datetime.now().isoformat(timespec="seconds")
    runs.write_meta(run_id, {
        "run_id": run_id,
        "kind": runs.KIND_ADHOC,
        "label": label or "",
        "subject": subject,
        "documents": list(documents or []),
        "created_at": now,
        "updated_at": now,
        "orgs": [subject],
        "source_types": [ADHOC_SOURCE_TYPE],
        "k": k,
        "generation_model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        # An ad-hoc run answers the whole questionnaire in one pass or fails;
        # there is no resumption, so a saved one is complete by construction.
        "status": "complete",
        "answered": sum(1 for r in rows if not r.get("abstain")),
        "abstained": sum(1 for r in rows if r.get("abstain")),
        "questions": len(rows),
    })
    return run_id


def load_run(run_id: str) -> dict | None:
    """Rebuild an analyze()-shaped result from a saved ad-hoc snapshot.

    Returns None for a run that is missing, unreadable, or not an ad-hoc run —
    the caller is a picker over ad-hoc runs, so a corpus id reaching here is a
    bug to degrade on, not to render.
    """
    meta = runs.read_meta(run_id)
    if not runs.is_adhoc(meta):
        return None
    paths = runs.run_paths(run_id)
    if not paths["per_question"].exists() or not paths["profiles_json"].exists():
        return None

    rows = []
    with open(paths["per_question"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    profiles = json.loads(paths["profiles_json"].read_text(encoding="utf-8"))
    subject = meta.get("subject") or next(iter(profiles), "")
    profile = (profiles.get(subject) or {}).get(ADHOC_SOURCE_TYPE)
    if profile is None or not rows:
        return None
    return {"subject": subject, "rows": rows, "profile": profile,
            "run_id": run_id, "meta": meta}
