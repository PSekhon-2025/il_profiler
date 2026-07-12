"""Retrieval-grounding pre-check: did retrieval plausibly find relevant text?

A wrong answer has two very different causes: the model hallucinated over good
evidence, or retrieval never surfaced relevant evidence in the first place.
This module separates the two cheaply — no LLM call — so the report can bucket
every question into:

  retrieval_missed  the question's content words barely appear in any retrieved
                    chunk: retrieval likely failed, whatever the model then did
  abstained         retrieval looked plausible but the matcher abstained
                    (the model said the excerpts don't answer the question)
  committed         retrieval looked plausible and the model committed to an
                    answer that was graded into logic weights

The thresholded signal is LEXICAL: the fraction of the question's content
tokens that appear in the best retrieved chunk (ROUGE-1-recall-style set
overlap). The retriever's cosine score is kept as a subscore but not
thresholded — e5 embeddings compress cosine into a narrow high band even for
weak matches, so token overlap is the discriminative, interpretable signal.

Everything here is pure computation over already-retrieved chunks; enabling it
adds zero API cost to a run.
"""
import re

from .config import GROUNDING_LOW_THRESHOLD

BUCKET_RETRIEVAL_MISSED = "retrieval_missed"
BUCKET_ABSTAINED = "abstained"
BUCKET_COMMITTED = "committed"
BUCKETS = (BUCKET_RETRIEVAL_MISSED, BUCKET_ABSTAINED, BUCKET_COMMITTED)

# Minimal English stopword list: enough to keep function words from inflating
# overlap, small enough to need no dependency.
_STOPWORDS = frozenset("""
a an the and or but if of to in on for with by at from as is are was were be
been being it its this that these those do does did done what which who whom
when where how why not no nor so than then there their they them he she his
her hers you your yours we our ours i me my mine will would can could should
shall may might must have has had having about into over under between
against during before after above below up down out off again further once
each other some such only own same very more most any all both few many much
per via also
""".split())


def _content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens minus stopwords and short fragments."""
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOPWORDS and len(t) > 2
    }


def lexical_overlap(question: str, chunk_text: str) -> float:
    """Fraction of the question's content tokens present in the chunk, in [0, 1]."""
    q = _content_tokens(question)
    if not q:
        return 0.0
    return len(q & _content_tokens(chunk_text)) / len(q)


def grounding_scores(question: str, chunks) -> dict:
    """Score how well the retrieved set covers the question.

    Returns {"score": <max lexical overlap>, "cosine_top": <max cosine>}.
    Max (not mean) over chunks: one genuinely relevant chunk is enough to
    ground an answer, so a strong hit shouldn't be diluted by weak siblings.
    """
    if not chunks:
        return {"score": 0.0, "cosine_top": 0.0}
    lex = max(lexical_overlap(question, c.text) for c in chunks)
    cos = max(max(0.0, min(1.0, c.score)) for c in chunks)
    return {"score": round(lex, 4), "cosine_top": round(cos, 4)}


def bucket(*, abstain: bool, grounding_score: float,
           threshold: float = GROUNDING_LOW_THRESHOLD) -> str:
    """Assign a result row to one of the three report buckets.

    retrieval_missed takes precedence over abstained: when retrieval failed,
    abstaining was the RIGHT response and the item's failure belongs to
    retrieval, not to the model's grounding.
    """
    if grounding_score < threshold:
        return BUCKET_RETRIEVAL_MISSED
    if abstain:
        return BUCKET_ABSTAINED
    return BUCKET_COMMITTED


def summarize_buckets(rows: list[dict]) -> dict:
    """Per-bucket stats over result rows that carry a grounding_bucket field.

    There are no gold labels in this pipeline, so instead of accuracy each
    bucket reports its size, its abstention rate, and the mean top-logic
    weight of its committed answers (a proxy for how decisively the matcher
    graded them).
    """
    out = {}
    for b in BUCKETS:
        rs = [r for r in rows if r.get("grounding_bucket") == b]
        if not rs:
            out[b] = {"n": 0, "abstain_rate": None, "mean_top_weight": None}
            continue
        abstained = sum(1 for r in rs if r["abstain"])
        tops = [max(r["weights"].values()) for r in rs if not r["abstain"]]
        out[b] = {
            "n": len(rs),
            "abstain_rate": round(abstained / len(rs), 3),
            "mean_top_weight": round(sum(tops) / len(tops), 3) if tops else None,
        }
    return out
