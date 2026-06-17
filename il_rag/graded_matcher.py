"""Graded multi-logic answer matching (generalization of Chandak et al. 2025).

Classic answer matching grades a free-form answer against one reference answer
as match / no-match. Here the reference space is the 7 institutional logics'
exemplar answers for the question's category, and the verdict is a weight
distribution over the logics rather than a binary call — institutional logics
co-exist in real organizations, so an answer may legitimately be part Market,
part Corporation.

Guarantees enforced in code (never trusted to the LLM):
  - weights are clamped non-negative and renormalized to sum to exactly 1;
  - an all-zero distribution is treated as an abstention even if the model
    forgot to set the flag;
  - on abstention all weights are zeroed, so "no evidence" can never leak
    weight into any logic.
"""
import json
import re

from .llm import chat
from .questionnaire import LOGICS, reference_answers

SYSTEM = (
    "You are a strict, consistent grader. You compare a candidate answer "
    "against a fixed set of reference answers and judge only what the candidate "
    "actually supports with evidence. Ignore style; never use prior knowledge "
    "about the organization; the reference answers are authoritative."
)

TEMPLATE = """A candidate answer describes how an AI lab operates along one institutional dimension: "{category}".

Reference answers — one per institutional logic (the only labels you may weight):
{references}

Question that was asked:
{question}

Candidate answer (grounded in the lab's corpus):
\"\"\"{candidate}\"\"\"

Task:
- Judge how strongly the EVIDENCE in the candidate answer matches each reference answer.
- Assign each logic a weight in [0, 1]; the seven weights must sum to 1.0.
- Concentrate weight on one logic, or split it across several when the evidence genuinely reflects more than one.
- Use ONLY evidence present in the candidate answer.
- If the candidate offers no usable evidence (it says the sources don't address this, refuses, or is empty), set "abstain" to true and all weights to 0.

Output strictly this JSON object, nothing else:
{{"abstain": false, "weights": {{{weight_keys}}}, "reasoning": "one short sentence"}}"""


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of an LLM reply, tolerating markdown fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _normalize(raw: dict) -> dict[str, float]:
    """Clamp to [0, inf), keep only known logics, renormalize to sum 1."""
    w = {}
    for logic in LOGICS:
        try:
            w[logic] = max(0.0, float(raw.get(logic, 0.0) or 0.0))
        except (TypeError, ValueError):
            w[logic] = 0.0
    total = sum(w.values())
    if total <= 0.0:
        return {logic: 0.0 for logic in LOGICS}
    return {logic: v / total for logic, v in w.items()}


def match_graded(*, question: str, candidate: str, category: str) -> dict:
    """Grade one candidate answer into a weight distribution over the 7 logics.

    Returns:
        {abstain: bool, weights: {logic: float}, reasoning: str, raw: str}
    """
    refs = reference_answers(category)
    references = "\n".join(f'- {logic}: "{refs[logic]}"' for logic in LOGICS)
    weight_keys = ", ".join(f'"{logic}": 0.0' for logic in LOGICS)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TEMPLATE.format(
            category=category, references=references,
            question=question, candidate=candidate, weight_keys=weight_keys,
        )},
    ]
    # gpt-oss-120b is a reasoning model: max_tokens covers its hidden reasoning
    # PLUS the JSON verdict. A small cap truncates the JSON mid-object, which
    # parses as nothing and silently becomes a spurious abstention. 1536 leaves
    # ample room; if parsing still fails, retry once with double the budget.
    raw = chat(messages, temperature=0.0, max_tokens=1536)
    parsed = _extract_json(raw)
    if parsed is None:
        raw = chat(messages, temperature=0.0, max_tokens=3072)
        parsed = _extract_json(raw) or {}
    weights = _normalize(parsed.get("weights", {}))
    abstain = bool(parsed.get("abstain", False)) or sum(weights.values()) <= 0.0
    if abstain:
        weights = {logic: 0.0 for logic in LOGICS}
    return {
        "abstain": abstain,
        "weights": weights,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "raw": raw,
    }
