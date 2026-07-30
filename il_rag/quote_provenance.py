"""Quote provenance: grade HOW a quoted span relates to the sources, and
whether its content survives even when the span itself does not.

Feature 2 (rag_qa._verify_quotes) answers one question with one bit: does this
span occur verbatim in the retrieved evidence? Everything that is not a
verbatim hit collapses into a single ❌, which conflates four different things:

  near-verbatim drift   a copy that picked up a curly quote, an em-dash, an
                        elided "...", a dropped comma
  faithful paraphrase   the model summarized the passage instead of copying it
  non-attributive use   scare quotes, terms of art, hypotheticals — quotation
                        marks that never claimed anything about a source
  fabrication           a span with no support in the evidence at all

Only the last is a hallucination. This module separates them, and then asks a
second question the binary check cannot ask at all:

  PROVENANCE  does this TEXT exist in the sources?      (stages A-C, no LLM)
  VERACITY    is what it ASSERTS supported by them?     (stage D, LLM)

These are independent, so the verdict is a 2x2 rather than a line. Its most
useful cell is `misquote_but_true`: the model manufactured a quotation, but the
proposition underneath it does hold up against the evidence — a citation
integrity failure, not a factual one. Without that cell, finding the truth
inside a fabricated-looking span is manual work.

Design notes:
  - POST-HOC over a saved run, like the metamorphic and embedding-agreement
    checks. Retriever.get_by_ids replays the exact evidence a past run answered
    from, so nothing in the answering path has to change and thresholds can be
    retuned without re-running the expensive profile pass.
  - Feature 2's `quotes_verified` is READ, never rewritten. It keeps its strict
    all-spans-verbatim meaning so old and new runs stay comparable; the graded
    verdicts are reported alongside it.
  - The ladder is cheapest-first and short-circuits: a run whose quotes are all
    verbatim costs ZERO generation calls and zero embedding calls.
  - The model attests, the code audits. The adjudicator's label is normalized
    in code to the allowed set, and the verdict itself is derived by a pure
    function of (tier, support, intent) with no LLM in the derivation.
  - The adjudicator never sees the institutional-logics taxonomy — the same
    separation rag_qa's docstring insists on.
  - Quoted spans in the ANSWER PROSE are audited too, not just the structured
    `quotes` field, so the check works on every saved run and covers the
    non-attributive cases, which is where they live.

Scope caveat, stated here because it bounds every number this module reports:
the evidence is scoped by (org, source_type) and there is no world-knowledge
oracle in this pipeline by design. `unsupported` therefore means "unsupported
by this lab's scoped corpus", NOT "false in the world" — which is exactly why
`not_addressed` is a separate label from `contradicted`. Only `contradicted` is
evidence against a span.

Outputs (in <run>/quote_provenance/):
  spans.jsonl   one record per candidate span: intent, tier, verdict, evidence
  summary.json  tier/intent/verdict histograms + rates, overall and per slice

Full literature basis and references: ARCHITECTURE.md §9.5.
"""
import json
import re
from difflib import SequenceMatcher

from . import runs
from .config import (
    QUOTE_MIN_SPAN_TOKENS,
    QUOTE_NEAR_VERBATIM_THRESHOLD,
    QUOTE_PARAPHRASE_COS_THRESHOLD,
    QUOTE_PARAPHRASE_LEX_THRESHOLD,
)
# Reused rather than reimplemented — the same reason json_utils exists instead
# of three copies of one fence-stripping regex. These are module-internal
# helpers elsewhere; importing them keeps ONE definition of each notion
# (normalization, content tokens, cosine) across all the checks.
from .embedding_agreement import _cosine, _embed_batched, _truncate
from .grounding import _content_tokens, lexical_overlap
from .json_utils import extract_json
from .llm import chat
from .rag_qa import _norm_ws
from .retriever import Retriever

OUT_DIR_NAME = runs.QUOTE_PROVENANCE_DIR_NAME

# ---------------------------------------------------------------------------
# Vocabularies. Every label a record can carry is named here so the summary
# histograms and the UI can enumerate them without hardcoding strings.
# ---------------------------------------------------------------------------
SOURCE_QUOTES_FIELD = "quotes_field"
SOURCE_ANSWER_PROSE = "answer_prose"

INTENT_ATTRIBUTIVE = "attributive"
INTENT_SCARE_QUOTE = "scare_quote"
INTENT_TERM_OF_ART = "term_of_art"
INTENT_HYPOTHETICAL = "hypothetical"
INTENTS = (INTENT_ATTRIBUTIVE, INTENT_SCARE_QUOTE, INTENT_TERM_OF_ART,
           INTENT_HYPOTHETICAL)

TIER_EXACT = "exact"
TIER_NEAR_VERBATIM = "near_verbatim"
TIER_PARAPHRASE = "paraphrase"
TIER_UNSUPPORTED = "unsupported"
TIERS = (TIER_EXACT, TIER_NEAR_VERBATIM, TIER_PARAPHRASE, TIER_UNSUPPORTED)
# Tiers at which the span's TEXT was located in the evidence (provenance holds).
TIERS_FOUND = (TIER_EXACT, TIER_NEAR_VERBATIM)

SUPPORT_SUPPORTED = "supported"
SUPPORT_PARTIAL = "partial"
SUPPORT_CONTRADICTED = "contradicted"
SUPPORT_NOT_ADDRESSED = "not_addressed"
SUPPORT_LABELS = (SUPPORT_SUPPORTED, SUPPORT_PARTIAL, SUPPORT_CONTRADICTED,
                  SUPPORT_NOT_ADDRESSED)
# Labels under which the span's CONTENT stands up (veracity holds).
SUPPORT_POSITIVE = (SUPPORT_SUPPORTED, SUPPORT_PARTIAL)

VERDICT_ATTRIBUTED = "attributed"
VERDICT_PARAPHRASE_GROUNDED = "paraphrase_grounded"
VERDICT_MISQUOTE_BUT_TRUE = "misquote_but_true"
VERDICT_MISATTRIBUTED = "misattributed"
VERDICT_FABRICATED = "fabricated"
VERDICT_NON_ATTRIBUTIVE = "non_attributive"
VERDICTS = (VERDICT_ATTRIBUTED, VERDICT_PARAPHRASE_GROUNDED,
            VERDICT_MISQUOTE_BUT_TRUE, VERDICT_MISATTRIBUTED,
            VERDICT_FABRICATED, VERDICT_NON_ATTRIBUTIVE)


# ---------------------------------------------------------------------------
# Stage A — candidate extraction (no LLM)
# ---------------------------------------------------------------------------
# Straight and curly DOUBLE quotes, plus curly singles. Straight single quotes
# are deliberately excluded: apostrophes ("OpenAI's charter", "don't") make
# them unparseable without a tokenizer, and the false-positive spans they
# produce would swamp the real ones.
_QUOTE_RE = re.compile(r'"([^"\n]{2,400})"|“([^”\n]{2,400})”'
                       r'|‘([^’\n]{2,400})’')

# How much text before the opening quote the intent rules get to look at. Long
# enough to catch "according to the charter, ...", short enough that a cue from
# an unrelated earlier clause cannot leak in.
_CONTEXT_CHARS = 70


def _span_tokens(span: str) -> int:
    """Content-token count of a span (stopwords and 1-2 char fragments dropped)."""
    return len(_content_tokens(span))


def extract_prose_spans(answer: str) -> list[dict]:
    """Quoted spans appearing in the answer's own prose, with their left context.

    The structured `quotes` field is only populated on --quotes runs, and even
    there it holds what the model CHOSE to offer as support. Quotation marks in
    the prose are the unfiltered picture, and the only place the non-attributive
    uses (scare quotes, terms of art) ever appear. Spans shorter than
    QUOTE_MIN_SPAN_TOKENS content tokens are dropped as noise.
    """
    out = []
    for m in _QUOTE_RE.finditer(answer or ""):
        span = next((g for g in m.groups() if g), "").strip()
        if _span_tokens(span) < QUOTE_MIN_SPAN_TOKENS:
            continue
        out.append({
            "quote": span,
            "left_context": answer[max(0, m.start() - _CONTEXT_CHARS):m.start()],
        })
    return out


def extract_candidates(row: dict) -> list[dict]:
    """All quote candidates in one result row, structured entries first.

    A prose span that merely repeats a structured entry is dropped: it is the
    same claim surfaced twice, and counting it twice would double-weight rows
    whose answers quote their own supporting spans back to the reader.
    """
    candidates = []
    seen = set()
    for q in row.get("quotes") or []:
        if not isinstance(q, dict):
            continue
        quote = str(q.get("quote", "")).strip()
        if not quote:
            continue
        candidates.append({
            "quote": quote,
            "source": SOURCE_QUOTES_FIELD,
            "excerpt": q.get("excerpt"),
            # Feature 2's own verdict for this span, carried through unchanged
            # so the graded record can be compared against it row by row.
            "verified": bool(q.get("verified")),
            "left_context": "",
        })
        seen.add(_norm_ws(quote))
    for p in extract_prose_spans(row.get("answer", "")):
        norm = _norm_ws(p["quote"])
        if norm in seen or any(norm in s or s in norm for s in seen):
            continue
        seen.add(norm)
        candidates.append({
            "quote": p["quote"],
            "source": SOURCE_ANSWER_PROSE,
            "excerpt": None,
            "verified": None,
            "left_context": p["left_context"],
        })
    return candidates


# ---------------------------------------------------------------------------
# Stage B — intent triage (no LLM, fully auditable)
# ---------------------------------------------------------------------------
# Rule order encodes which way to fail, and one ordering is load-bearing:
# a COUNTERFACTUAL attribution ("a critic might say ...") contains a reporting
# verb but is not an attribution at all, so it has to be tested BEFORE the
# reporting cue or it would be swallowed by it. Everything else is tested
# after, because a reporting verb otherwise wins: if the sentence says the
# source SAYS this, the span is a claim about the source whatever surrounds it.
_COUNTERFACTUAL_CUE = re.compile(
    r"\b((?:would|might|could|may|will|'d)\s+(?:say|claim|argue|note|state|"
    r"describe|call)\w*|imagine|hypothetical(?:ly)?|suppose|were it to)\b", re.I)
_REPORTING_CUE = re.compile(
    r"\b(says?|said|states?|stated|writes?|wrote|notes?|noted|reads?|"
    r"describ\w+|declares?|explains?|argues?|claims?|asserts?|emphasi\w+|"
    r"according to|per the|quoting|cites?|cited)\b", re.I)
_MENTION_CUE = re.compile(
    r"\b(so-?called|what (?:it|they|the \w+) calls?|referred to as|"
    r"refers to (?:it|them|this) as|dubbed|the term|termed|labell?ed|"
    r"known as|a kind of|a sort of|air quotes)\b", re.I)
# Illustrative framing, tested AFTER the reporting cue: "for example, the
# charter says X" is a real attribution that merely opens with an example
# marker, so this may only fire when no reporting verb did.
_EXAMPLE_CUE = re.compile(
    r"\b(for example|for instance|e\.g\.|such as)\b", re.I)

# A span with fewer content tokens than this and no sentence punctuation reads
# as a label, not as a sentence lifted from a document.
_TERM_OF_ART_MAX_TOKENS = 4


def classify_intent(candidate: dict) -> tuple[str, str, str]:
    """Is this span claiming to reproduce a source, or just using quote marks?

    Returns (intent, rule, confidence). The rule that fired is returned with
    the label so every classification is inspectable rather than a black box.

    Entries from the structured `quotes` field are attributive by construction
    — the prompt asked for spans "copied character-for-character", so offering
    one IS the attribution claim. Prose spans are triaged by deterministic cues
    in the ~70 characters before the opening quote, plus the span's own shape,
    in a fixed precedence: counterfactual → reporting verb → mention → example
    → shape → default.

    An unmatched span defaults to ATTRIBUTIVE at low confidence. That direction
    is deliberate: wrongly auditing a scare quote produces a false alarm the
    reviewer can dismiss, while wrongly excusing a fabricated quotation hides
    exactly what this check exists to find.
    """
    if candidate["source"] == SOURCE_QUOTES_FIELD:
        return INTENT_ATTRIBUTIVE, "declared_in_quotes_field", "high"

    ctx = candidate.get("left_context", "")
    if _COUNTERFACTUAL_CUE.search(ctx):
        return INTENT_HYPOTHETICAL, "counterfactual_frame", "high"
    if _REPORTING_CUE.search(ctx):
        return INTENT_ATTRIBUTIVE, "reporting_verb", "high"
    if _MENTION_CUE.search(ctx):
        return INTENT_SCARE_QUOTE, "mention_not_use", "high"
    if _EXAMPLE_CUE.search(ctx):
        return INTENT_HYPOTHETICAL, "example_frame", "high"

    span = candidate["quote"]
    if _span_tokens(span) < _TERM_OF_ART_MAX_TOKENS and not re.search(r"[.!?;]", span):
        return INTENT_TERM_OF_ART, "short_unpunctuated_span", "high"
    return INTENT_ATTRIBUTIVE, "default_no_cue", "low"


# ---------------------------------------------------------------------------
# Stage C — provenance ladder (no LLM)
# ---------------------------------------------------------------------------
def _strip_punct(norm: str) -> str:
    """Drop everything but alphanumerics and single spaces, post-_norm_ws.

    This is the one normalization feature 2 deliberately refuses: it is what
    makes a smart-quote or em-dash copy artifact stop looking like a rewrite.
    Applied only at the near-verbatim tier, never to the exact-match test.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", norm)).strip()


def _ellipsis_fragments(norm: str) -> list[str]:
    """Split a normalized span on ellipses into its retained fragments."""
    parts = [p.strip() for p in re.split(r"\.{3,}|…", norm)]
    return [p for p in parts if p]


def _fragments_in_order(fragments: list[str], hay: str) -> bool:
    """Do all fragments occur in `hay`, in order, without overlapping?

    An elided quotation ("A ... B") claims that A precedes B in the source, so
    order is part of what is being asserted and has to be part of the check.
    """
    pos = 0
    for frag in fragments:
        i = hay.find(frag, pos)
        if i < 0:
            return False
        pos = i + len(frag)
    return True


def _best_window_ratio(span: str, hay: str) -> tuple[float, int]:
    """Best difflib ratio of `span` against any same-length window of `hay`.

    Returns (ratio, window_start). Sliding at a quarter-span stride guarantees
    any true alignment is covered by some window with at most a quarter of the
    span shifted out; the cheap real_quick_ratio/quick_ratio upper bounds skip
    the quadratic comparison for windows that cannot possibly win.
    """
    n, h = len(span), len(hay)
    if not n or not h:
        return 0.0, 0
    if h <= n:
        return SequenceMatcher(None, span, hay).ratio(), 0
    best, best_at = 0.0, 0
    stride = max(1, n // 4)
    sm = SequenceMatcher()
    sm.set_seq2(span)
    for start in range(0, h - n + 1, stride):
        window = hay[start:start + n]
        sm.set_seq1(window)
        if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
            continue
        r = sm.ratio()
        if r > best:
            best, best_at = r, start
    return best, best_at


def _sentence_windows(text: str, size: int = 2) -> list[str]:
    """Sliding windows of `size` sentences — the unit a paraphrase aligns to.

    A whole 1400-character chunk embedded as one vector drowns a one-sentence
    paraphrase in unrelated text; single sentences are too short to carry the
    context. Two-sentence windows are the compromise.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        return []
    if len(sents) <= size:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]


class _WindowEmbeddings:
    """Lazily embedded sentence windows, cached per chunk id across the run.

    Chunks repeat heavily across rows (the same passage answers several
    questions), and most spans never reach the tier that needs a vector at all.
    Embedding lazily and caching by chunk id keeps a verbatim-clean run at zero
    embedding calls and stops a repeated chunk from being paid for twice.
    """

    def __init__(self) -> None:
        self._by_chunk: dict[str, tuple[list[str], list[list[float]]]] = {}
        self._by_span: dict[str, list[float]] = {}
        self.calls = 0

    def windows(self, chunk) -> tuple[list[str], list[list[float]]]:
        if chunk.id not in self._by_chunk:
            wins = _sentence_windows(chunk.text)
            vecs = _embed_batched([_truncate(w) for w in wins]) if wins else []
            self.calls += len(wins)
            self._by_chunk[chunk.id] = (wins, vecs)
        return self._by_chunk[chunk.id]

    def span(self, text: str) -> list[float]:
        if text not in self._by_span:
            self._by_span[text] = _embed_batched([_truncate(text)])[0]
            self.calls += 1
        return self._by_span[text]


def locate_span(span: str, chunks: list, embeddings: "_WindowEmbeddings | None" = None) -> dict:
    """Place a span on the provenance ladder against the evidence it cites.

    The ladder is cheapest-first and the first tier to clear its bar wins:

        exact          norm(s) ⊑ norm(c)                    for some chunk c
        near_verbatim  strip_punct(norm(s)) ⊑ strip_punct(norm(c)), or an
                       ellipsis-elided span whose fragments occur in order, or
                       max-window difflib ratio ≥ τ_near
        paraphrase     lexical_overlap(s, c) ≥ τ_lex  ∨  cos(s, w) ≥ τ_cos
                       for some 2-sentence window w of some chunk
        unsupported    nothing cleared a bar

    Returns the tier, the score that placed it there, and the aligned source
    text (`best_span`) so a reviewer can read claim against source instead of
    taking the tier on faith. `embeddings=None` disables the cosine route,
    which keeps the whole ladder pure computation.
    """
    norm_span = _norm_ws(span)
    result = {"match_tier": TIER_UNSUPPORTED, "match_score": 0.0,
              "match_rule": None, "best_chunk_id": None, "best_span": None}
    if not norm_span or not chunks:
        return result

    norm_chunks = [(c, _norm_ws(c.text)) for c in chunks]

    # --- exact: bit-identical to feature 2's predicate -------------------
    for c, nc in norm_chunks:
        if norm_span in nc:
            at = nc.find(norm_span)
            return {"match_tier": TIER_EXACT, "match_score": 1.0,
                    "match_rule": "substring", "best_chunk_id": c.id,
                    "best_span": nc[at:at + len(norm_span)]}

    # --- near_verbatim: a copy that drifted, not a rewrite ----------------
    stripped_span = _strip_punct(norm_span)
    fragments = _ellipsis_fragments(norm_span)
    for c, nc in norm_chunks:
        stripped_chunk = _strip_punct(nc)
        if stripped_span and stripped_span in stripped_chunk:
            at = stripped_chunk.find(stripped_span)
            return {"match_tier": TIER_NEAR_VERBATIM, "match_score": 1.0,
                    "match_rule": "punctuation_insensitive",
                    "best_chunk_id": c.id,
                    "best_span": stripped_chunk[at:at + len(stripped_span)]}
        if len(fragments) > 1 and _fragments_in_order(fragments, nc):
            return {"match_tier": TIER_NEAR_VERBATIM, "match_score": 1.0,
                    "match_rule": "ellipsis_fragments_in_order",
                    "best_chunk_id": c.id, "best_span": None}

    best_ratio, best_c, best_text = 0.0, None, None
    for c, nc in norm_chunks:
        ratio, at = _best_window_ratio(norm_span, nc)
        if ratio > best_ratio:
            best_ratio, best_c = ratio, c
            best_text = nc[at:at + len(norm_span)]
    if best_ratio >= QUOTE_NEAR_VERBATIM_THRESHOLD:
        return {"match_tier": TIER_NEAR_VERBATIM, "match_score": round(best_ratio, 4),
                "match_rule": "character_similarity", "best_chunk_id": best_c.id,
                "best_span": best_text}

    # --- paraphrase: different words, same content ------------------------
    # Lexical overlap is the primary signal (grounding.py's argument: e5
    # compresses cosine into a narrow band, so token overlap discriminates
    # better); cosine is a second route into the same tier.
    best_lex, lex_c = 0.0, None
    for c, _nc in norm_chunks:
        ov = lexical_overlap(span, c.text)
        if ov > best_lex:
            best_lex, lex_c = ov, c
    if best_lex >= QUOTE_PARAPHRASE_LEX_THRESHOLD:
        return {"match_tier": TIER_PARAPHRASE, "match_score": round(best_lex, 4),
                "match_rule": "lexical_overlap", "best_chunk_id": lex_c.id,
                "best_span": None}

    if embeddings is not None:
        svec = embeddings.span(span)
        best_cos, cos_c, cos_text = 0.0, None, None
        for c, _nc in norm_chunks:
            wins, vecs = embeddings.windows(c)
            for w, v in zip(wins, vecs):
                sim = _cosine(svec, v)
                if sim > best_cos:
                    best_cos, cos_c, cos_text = sim, c, w
        if best_cos >= QUOTE_PARAPHRASE_COS_THRESHOLD:
            return {"match_tier": TIER_PARAPHRASE, "match_score": round(best_cos, 4),
                    "match_rule": "embedding_cosine", "best_chunk_id": cos_c.id,
                    "best_span": cos_text}
        result["match_score"] = round(max(best_lex, best_cos), 4)
        result["best_chunk_id"] = (cos_c or lex_c or best_c).id if (cos_c or lex_c or best_c) else None
        result["best_span"] = cos_text
        return result

    result["match_score"] = round(best_lex, 4)
    result["best_chunk_id"] = (lex_c or best_c).id if (lex_c or best_c) else None
    return result


# ---------------------------------------------------------------------------
# Stage D — entailment adjudication (LLM, flagged spans only)
# ---------------------------------------------------------------------------
ADJUDICATOR_SYSTEM = (
    "You judge whether a claimed quotation's CONTENT is supported by a source "
    "passage. You are not judging whether the wording matches — only whether "
    "the passage asserts what the quotation asserts. Answer strictly from the "
    "passage; never use outside knowledge."
)

ADJUDICATE_TEMPLATE = """Source text:
{passages}

Claimed quotation:
"{span}"

Does the source text support what this quotation asserts? Judge the CONTENT, not the wording — the quotation may be a paraphrase or may not appear in the source at all.

Return strictly this JSON object, nothing else:
{{"support": "supported|partial|contradicted|not_addressed", "evidence_sentence": "<the sentence from the source text that decides it, copied verbatim, or empty>", "grounded_fragment": "<the part of the quotation the source does support, or empty>", "reason": "<one sentence>"}}

Use the labels precisely:
- "supported": the source text asserts what the quotation asserts.
- "partial": the source supports some of the quotation but not all of it; put the supported part in "grounded_fragment".
- "contradicted": the source text asserts something incompatible with the quotation.
- "not_addressed": the source text neither supports nor contradicts it — it simply does not speak to this. Silence is NOT contradiction."""


def _normalize_support(parsed: dict) -> dict:
    """Coerce an adjudicator reply into the fixed label set.

    The model attests, the code audits: an unrecognized or missing label
    degrades to `not_addressed`, the neutral option, so a malformed reply can
    never manufacture a `supported` or a `contradicted` verdict.
    """
    label = str(parsed.get("support", "")).strip().lower()
    if label not in SUPPORT_LABELS:
        label = SUPPORT_NOT_ADDRESSED
    fragment = str(parsed.get("grounded_fragment", "") or "").strip()
    return {
        "support": label,
        "evidence_sentence": str(parsed.get("evidence_sentence", "") or "").strip(),
        # A fragment only means something when part of the span survived.
        "grounded_fragment": fragment or None,
        "support_reason": str(parsed.get("reason", "") or "").strip(),
    }


def adjudicate(span: str, passages: list[str]) -> dict:
    """Ask whether the evidence supports the span's content. One LLM call.

    Follows the codebase's structured-output convention: strict-JSON prompt,
    extract_json, one retry at a doubled budget on a parse failure, then a safe
    degraded record rather than a crash — one malformed reply costs a span's
    veracity verdict, not the run.
    """
    user = ADJUDICATE_TEMPLATE.format(passages="\n\n".join(passages), span=span)
    messages = [{"role": "system", "content": ADJUDICATOR_SYSTEM},
                {"role": "user", "content": user}]
    raw = chat(messages, temperature=0.0, max_tokens=1024)
    parsed = extract_json(raw)
    if parsed is None:
        raw = chat(messages, temperature=0.0, max_tokens=2048)
        parsed = extract_json(raw)
    if parsed is None:
        return {"support": SUPPORT_NOT_ADDRESSED, "evidence_sentence": "",
                "grounded_fragment": None,
                "support_reason": "adjudicator reply did not parse"}
    return _normalize_support(parsed)


# ---------------------------------------------------------------------------
# Stage D' — provenance x veracity
# ---------------------------------------------------------------------------
def derive_verdict(match_tier: str, support: str | None, intent: str) -> str:
    """Combine the two independent axes into one verdict. Pure function.

                            | content supported | content not supported
        --------------------+-------------------+----------------------
        text in sources     | attributed        | misattributed
        text not in sources | misquote_but_true | fabricated

    The bottom-left cell is the point of the whole module: a span that is not
    in the sources as TEXT can still assert something the sources do support.
    It is subdivided by how far the text drifted — `paraphrase_grounded` (the
    model reworded a real passage) is a much milder failure than
    `misquote_but_true` (the model manufactured a quotation whose content
    happens to hold), and collapsing them would hide that difference.

    `support=None` means no adjudication ran, which for a located span is the
    default path: stage D is skipped when the text was found verbatim, so
    provenance alone decides and the verdict is `attributed`. For a span that
    was NOT located, no adjudication means nothing rescued it — `fabricated`.
    """
    if intent != INTENT_ATTRIBUTIVE:
        return VERDICT_NON_ATTRIBUTIVE
    if match_tier in TIERS_FOUND:
        if support == SUPPORT_CONTRADICTED:
            return VERDICT_MISATTRIBUTED
        return VERDICT_ATTRIBUTED
    if support in SUPPORT_POSITIVE:
        return (VERDICT_PARAPHRASE_GROUNDED if match_tier == TIER_PARAPHRASE
                else VERDICT_MISQUOTE_BUT_TRUE)
    return VERDICT_FABRICATED


# ---------------------------------------------------------------------------
# Stage E — per-row verdicts
# ---------------------------------------------------------------------------
def row_verdicts(spans: list[dict]) -> dict:
    """Fold a row's span records into row-level flags.

    Only ATTRIBUTIVE spans count: a scare quote is not a claim about a source,
    so it can neither ground a row nor fabricate one.

        quotes_grounded    = |A| > 0 ∧ ∀ s ∈ A : verdict(s) ∈ {attributed,
                                                paraphrase_grounded}
        quotes_fabricated  = ∃ s ∈ A : verdict(s) = fabricated
        quotes_misquoted   = ∃ s ∈ A : verdict(s) ∈ {misquote_but_true,
                                                     misattributed}

    `quotes_grounded` carries the same |A| > 0 guard as feature 2's
    `quotes_verified`, and for the same reason: a conjunction over an empty set
    is vacuously true, and a row that cited nothing has grounded nothing.

    Note that fabricated and misquoted are NOT mutually exclusive with each
    other across a multi-span row — a row can both invent one span and
    half-rescue another, and flattening that would lose information.
    """
    attributive = [s for s in spans if s["intent"] == INTENT_ATTRIBUTIVE]
    verdicts = {s["verdict"] for s in attributive}
    return {
        "n_spans": len(spans),
        "n_attributive": len(attributive),
        "quotes_grounded": bool(attributive) and verdicts <= {
            VERDICT_ATTRIBUTED, VERDICT_PARAPHRASE_GROUNDED},
        "quotes_fabricated": VERDICT_FABRICATED in verdicts,
        "quotes_misquoted": bool(
            verdicts & {VERDICT_MISQUOTE_BUT_TRUE, VERDICT_MISATTRIBUTED}),
    }


def analyze_row(row: dict, chunks: list, *, embeddings=None,
                adjudicate_verbatim: bool = False) -> list[dict]:
    """Full stage A-D' pass over one result row. Returns its span records.

    The evidence window handed to the adjudicator widens with the tier, because
    the question being asked changes. A PARAPHRASE is judged against the passage
    it aligned to — "did the model reword THIS faithfully?". An UNSUPPORTED span
    aligned to nothing, so it is judged against the row's ENTIRE retrieved set:
    the text is not in the sources, but the sources may still support what it
    says, and that is the difference between a misquote and a fabrication.
    """
    by_id = {c.id: c for c in chunks}
    records = []
    for cand in extract_candidates(row):
        intent, rule, confidence = classify_intent(cand)
        located = (locate_span(cand["quote"], chunks, embeddings)
                   if intent == INTENT_ATTRIBUTIVE
                   else {"match_tier": None, "match_score": None,
                         "match_rule": None, "best_chunk_id": None,
                         "best_span": None})

        support = {"support": None, "evidence_sentence": "",
                   "grounded_fragment": None, "support_reason": ""}
        tier = located["match_tier"]
        needs_adjudication = intent == INTENT_ATTRIBUTIVE and (
            tier in (TIER_PARAPHRASE, TIER_UNSUPPORTED)
            or (adjudicate_verbatim and tier in TIERS_FOUND))
        if needs_adjudication and chunks:
            if tier == TIER_UNSUPPORTED:
                passages = [c.text for c in chunks]
            else:
                aligned = by_id.get(located["best_chunk_id"])
                passages = [aligned.text] if aligned else [c.text for c in chunks]
            support = adjudicate(cand["quote"], passages)

        records.append({
            "org": row.get("org"), "source_type": row.get("source_type"),
            "qid": row.get("qid"), "category": row.get("category"),
            "quote": cand["quote"], "source": cand["source"],
            "excerpt": cand["excerpt"],
            # feature 2's verdict for this span, unchanged (None for prose spans)
            "verbatim_verified": cand["verified"],
            "intent": intent, "intent_rule": rule,
            "intent_confidence": confidence,
            **located,
            **support,
            "verdict": derive_verdict(tier, support["support"], intent),
        })
    return records


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------
def _load_rows(run_id: str) -> list[dict]:
    path = runs.run_paths(run_id)["per_question"]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _histogram(items: list[dict], key: str, labels: tuple) -> dict:
    counts = {label: 0 for label in labels}
    for it in items:
        v = it.get(key)
        if v in counts:
            counts[v] += 1
    return counts


def _slice_stats(spans: list[dict]) -> dict:
    """Rates for one slice of spans (overall, a category, or a lab/source pair).

    `paraphrase_rescue_rate` is the headline: the share of attributive spans
    that feature 2 marks ❌ but that this check finds grounded — how much of the
    old fabrication number was never fabrication.
    """
    attributive = [s for s in spans if s["intent"] == INTENT_ATTRIBUTIVE]
    n = len(attributive)
    out = {
        "n_spans": len(spans),
        "n_attributive": n,
        "tiers": _histogram(attributive, "match_tier", TIERS),
        "verdicts": _histogram(spans, "verdict", VERDICTS),
    }
    if not n:
        return out
    counts = out["verdicts"]
    # Spans feature 2 would fail (not exact) that this check nonetheless
    # grounds — the rescued ones.
    not_exact = [s for s in attributive if s["match_tier"] != TIER_EXACT]
    rescued = [s for s in not_exact if s["verdict"] in (
        VERDICT_ATTRIBUTED, VERDICT_PARAPHRASE_GROUNDED,
        VERDICT_MISQUOTE_BUT_TRUE)]
    out.update({
        "paraphrase_rescue_rate": round(len(rescued) / n, 4),
        "true_fabrication_rate": round(counts[VERDICT_FABRICATED] / n, 4),
        "misquote_but_true_rate": round(counts[VERDICT_MISQUOTE_BUT_TRUE] / n, 4),
        "contradicted_rate": round(
            sum(1 for s in attributive if s["support"] == SUPPORT_CONTRADICTED) / n, 4),
        "non_attributive_rate": round(
            (len(spans) - n) / len(spans), 4) if spans else 0.0,
    })
    return out


def run_quote_provenance(run_id: str | None = None, *,
                         adjudicate_verbatim: bool = False,
                         retriever=None) -> dict:
    """Grade every quoted span in a saved run. Returns the summary dict.

    `adjudicate_verbatim` also entailment-checks spans that DID match verbatim,
    which is the only way to populate the `misattributed` cell (a real span
    that does not support what it was cited for). Off by default because it
    turns a zero-generation-cost check into one call per verified span.
    """
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run profiles first")

    rows = _load_rows(run_id)
    if not rows:
        raise SystemExit(f"run {run_id} has no rows to check")

    # This stage replays evidence rather than retrieving it, but it still needs
    # the index the run was answered from. Say so plainly: the raw Chroma
    # "collection does not exist" is not an actionable message.
    if retriever is None:
        try:
            retriever = Retriever()
        except Exception as exc:
            raise SystemExit(
                "quote provenance needs the Chroma index the run was answered "
                f"from, but it could not be opened ({exc}). Run "
                "scripts/01_ingest.py first."
            ) from exc
    embeddings = _WindowEmbeddings()

    out_dir = runs.run_paths(run_id)["quote_provenance_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_spans: list[dict] = []
    per_row: list[dict] = []
    rows_without_evidence = 0
    for row in rows:
        # Replay the exact evidence this row was answered from. Ids can go
        # missing after a --fresh reingest; such a row cannot be audited at
        # all, so it is counted and skipped rather than silently scored 0.
        chunks = retriever.get_by_ids(row.get("retrieved_ids") or [])
        if not chunks:
            rows_without_evidence += 1
            continue
        spans = analyze_row(row, chunks, embeddings=embeddings,
                            adjudicate_verbatim=adjudicate_verbatim)
        if not spans:
            continue
        all_spans.extend(spans)
        per_row.append({
            "org": row.get("org"), "source_type": row.get("source_type"),
            "qid": row.get("qid"), "category": row.get("category"),
            # feature 2's row verdict, read and carried through UNCHANGED
            "quotes_verified": row.get("quotes_verified"),
            **row_verdicts(spans),
        })

    with open(out_dir / runs.QUOTE_SPANS_NAME, "w", encoding="utf-8") as f:
        for s in all_spans:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    by_category = {c: _slice_stats([s for s in all_spans if s["category"] == c])
                   for c in sorted({s["category"] for s in all_spans})}
    by_pair = {}
    for key in {f'{s["org"]}|{s["source_type"]}' for s in all_spans}:
        org, st = key.split("|")
        by_pair[key] = _slice_stats([s for s in all_spans
                                     if s["org"] == org and s["source_type"] == st])

    summary = {
        "run_id": run_id,
        "n_rows_analyzed": len(per_row),
        "n_rows_without_evidence": rows_without_evidence,
        "adjudicate_verbatim": adjudicate_verbatim,
        "overall": _slice_stats(all_spans),
        "intents": _histogram(all_spans, "intent", INTENTS),
        "by_category": by_category,
        "by_org_source": dict(sorted(by_pair.items())),
        "rows": per_row,
        "note": ("Provenance (is the TEXT in the sources?) and veracity (is its "
                 "CONTENT supported?) are scored independently; verdict is a "
                 "pure function of the two. Evidence is scoped by (org, "
                 "source_type) and there is no world-knowledge oracle, so "
                 "'unsupported' means unsupported BY THIS CORPUS, not false. "
                 "Only 'contradicted' is evidence against a span."),
    }
    (out_dir / runs.QUOTE_SUMMARY_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_report(summary, embeddings)
    print(f"outputs: {out_dir}")
    return summary


def _print_report(summary: dict, embeddings: "_WindowEmbeddings") -> None:
    o = summary["overall"]
    print(f"\n=== Quote provenance (run {summary['run_id']}) ===")
    print(f"spans: {o['n_spans']}  attributive: {o['n_attributive']}  "
          f"rows: {summary['n_rows_analyzed']}")
    if summary["n_rows_without_evidence"]:
        print(f"  ({summary['n_rows_without_evidence']} rows skipped: their "
              f"retrieved chunks are no longer in the index)")
    print("provenance tiers (attributive spans):")
    for tier, n in o["tiers"].items():
        share = n / o["n_attributive"] if o["n_attributive"] else 0.0
        print(f"  {tier:<15} {n:>4}  ({share:.0%})")
    print("intent of every quoted span:")
    for intent, n in summary["intents"].items():
        print(f"  {intent:<15} {n:>4}")
    print("verdicts:")
    for verdict, n in o["verdicts"].items():
        print(f"  {verdict:<20} {n:>4}")
    if o.get("paraphrase_rescue_rate") is not None:
        print(f"\nparaphrase rescue rate:  {o['paraphrase_rescue_rate']:.1%}  "
              f"(non-verbatim spans that are nonetheless grounded)")
        print(f"misquote-but-true rate:  {o['misquote_but_true_rate']:.1%}  "
              f"(invented quotation, sound content)")
        print(f"true fabrication rate:   {o['true_fabrication_rate']:.1%}")
        print(f"contradicted rate:       {o['contradicted_rate']:.1%}  "
              f"(the only label that is evidence AGAINST a span)")
    print(f"embedding calls: {embeddings.calls}")
