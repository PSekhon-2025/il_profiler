"""RAG question answering: retrieve scoped evidence, generate a grounded answer.

Design constraint (from the methodology): the answering model sees ONLY the
retrieved excerpts. It must never see the institutional-logics matrix or the
reference answers — those exist solely on the matcher's side. This separation
is what makes the subsequent answer matching meaningful: the answer reflects
the corpus, not the taxonomy.

The answer must also admit ignorance: when the excerpts don't address the
question, saying so explicitly lets the matcher abstain instead of grading a
hallucinated answer.

Two opt-in extensions (defaults preserve the original behavior exactly):
  - `chunks=` injects caller-supplied evidence instead of retrieving, so
    variant texts that don't exist in the index (metamorphic paraphrases,
    lab swaps) flow through the same answering path as production questions.
  - `require_quotes=True` asks the model to also return the verbatim excerpt
    spans its conclusion rests on. Each quote is verified IN CODE (substring
    check after whitespace normalization) — the model attests, the code
    audits, following GopherCite (Menick et al., 2022). The taxonomy
    separation above is untouched: quotes support the ANSWER, never a logic
    choice. Full literature basis and references: ARCHITECTURE.md §9.2.
"""
import re
from dataclasses import dataclass

from .config import TOP_K
from .json_utils import extract_json
from .llm import chat
from .retriever import Chunk, Retriever

SYSTEM = (
    "You are a careful analyst of AI labs' governance and organization. Answer "
    "the question using ONLY the provided source excerpts. Be concrete: name "
    "programs, structures, and practices found in the excerpts. If the excerpts "
    "do not contain enough information to answer, say exactly that."
)

TEMPLATE = """Question:
{question}

Source excerpts:
{context}

Answer the question from the excerpts above. State your conclusion first, then justify it briefly with specifics from the excerpts."""

QUOTE_TEMPLATE = """Question:
{question}

Source excerpts:
{context}

Answer the question from the excerpts above. In the "answer" field, state your conclusion first, then justify it briefly with specifics from the excerpts.

Return strictly this JSON object, nothing else:
{{"answer": "<your full answer>", "quotes": [{{"excerpt": <excerpt number>, "quote": "<span copied verbatim from that excerpt>"}}]}}

Rules for "quotes":
- 1 to 3 entries: the specific spans your conclusion rests on.
- Each "quote" must be copied character-for-character from the numbered excerpt it cites; never paraphrase inside a quote.
- If the excerpts do not contain enough information to answer, say exactly that in "answer" and return an empty "quotes" list."""


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.org}, {c.source_type}, {c.filename})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _verify_quotes(quotes, chunks: list[Chunk]) -> tuple[list[dict], bool]:
    """Check each claimed quote against the excerpts it could have come from.

        verified(q)     = norm(q) ≠ "" ∧ ∃ c ∈ chunks : norm(q) ⊑ norm(c)
        quotes_verified = |Q| > 0 ∧ ∀ q ∈ Q : verified(q)

    Matching is whitespace-normalized and case-insensitive but otherwise
    verbatim. A quote citing the wrong excerpt number still verifies if the
    span exists in ANY excerpt — the auditable claim is "this text is in the
    sources", not the model's bookkeeping of indices. Returns the normalized
    quote records (each with its own verified flag) and their conjunction;
    an empty or unusable quote list is unverified by definition (the |Q| > 0
    guard blocks the vacuous truth of an empty conjunction).
    """
    norm_chunks = [_norm_ws(c.text) for c in chunks]
    out = []
    for q in quotes if isinstance(quotes, list) else []:
        if not isinstance(q, dict):
            continue
        quote = str(q.get("quote", "")).strip()
        try:
            idx = int(q.get("excerpt", 0))
        except (TypeError, ValueError):
            idx = 0
        nq = _norm_ws(quote)
        found = bool(nq) and any(nq in nc for nc in norm_chunks)
        out.append({"excerpt": idx, "quote": quote, "verified": found})
    verified = bool(out) and all(q["verified"] for q in out)
    return out, verified


@dataclass
class RAGAnswer:
    question: str
    answer: str
    chunks: list[Chunk]
    # Populated only when require_quotes=True: each entry is
    # {"excerpt": int, "quote": str, "verified": bool}; quotes_verified is
    # their conjunction (False when the model returned no usable quotes).
    quotes: list[dict] | None = None
    quotes_verified: bool | None = None


def answer_question(retriever: Retriever | None, question: str, *, org: str,
                    source_type: str, k: int = TOP_K,
                    chunks: list[Chunk] | None = None,
                    require_quotes: bool = False) -> RAGAnswer:
    """Retrieve (org, source_type)-scoped evidence and answer from it.

    `chunks` bypasses retrieval with caller-supplied evidence (the retriever
    may then be None). `require_quotes` switches to the structured
    quote-grounded prompt; if the reply is unparseable even after a retry the
    raw text becomes the answer with quotes marked unverified, so one
    malformed JSON degrades a row instead of killing the run.
    """
    if chunks is None:
        chunks = retriever.retrieve(question, org=org, source_type=source_type, k=k)
    context = _format_context(chunks)

    if not require_quotes:
        user = TEMPLATE.format(question=question, context=context)
        # gpt-oss-120b is a reasoning model: max_tokens covers its hidden reasoning
        # PLUS the visible answer. Too small a cap silently truncates or empties the
        # answer (finish_reason=length), which then cascades into matcher abstains.
        answer = chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=2048,
        )
        return RAGAnswer(question=question, answer=answer.strip(), chunks=chunks)

    user = QUOTE_TEMPLATE.format(question=question, context=context)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    # Same reasoning-budget concern as above, plus the JSON envelope and the
    # copied quote spans; a truncated reply parses as nothing, so parse failure
    # retries once with a doubled budget (mirroring the matcher).
    raw = chat(messages, temperature=0.0, max_tokens=3072)
    parsed = extract_json(raw)
    if parsed is None:
        raw = chat(messages, temperature=0.0, max_tokens=6144)
        parsed = extract_json(raw)
    if parsed is None:
        return RAGAnswer(question=question, answer=raw.strip(), chunks=chunks,
                         quotes=[], quotes_verified=False)
    quotes, verified = _verify_quotes(parsed.get("quotes", []), chunks)
    answer = str(parsed.get("answer", "")).strip() or raw.strip()
    return RAGAnswer(question=question, answer=answer, chunks=chunks,
                     quotes=quotes, quotes_verified=verified)
