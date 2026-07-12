"""Shared extraction of a JSON object from an LLM reply.

Lifted verbatim from graded_matcher so the quote-grounded answer path
(rag_qa) and the metamorphic paraphraser can reuse it instead of growing
three copies of the same fence-stripping regex.
"""
import json
import re


def extract_json(text: str) -> dict | None:
    """Pull a JSON object out of an LLM reply, tolerating markdown fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
