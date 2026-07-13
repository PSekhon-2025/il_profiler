"""The load-bearing guarantee: with all flags off, run_profiles writes rows
with exactly the original schema; flags only ADD fields.

The harness runs for real (run snapshots, aggregation, report) against a
temporary profiles directory, with the LLM/retriever boundary stubbed out.
"""
import json

import pytest

import il_rag.profile_harness as ph
import il_rag.runs as runs
from il_rag.questionnaire import LOGICS
from il_rag.rag_qa import RAGAnswer
from il_rag.retriever import Chunk

LEGACY_KEYS = {
    "org", "source_type", "qid", "category", "variant", "question", "answer",
    "retrieved_ids", "abstain", "weights", "reasoning",
}
GROUNDING_KEYS = {"retrieval_grounding_score", "retrieval_cosine_top", "grounding_bucket"}
QUOTE_KEYS = {"quotes", "quotes_verified"}


def _fake_answer(retriever, question, *, org, source_type, k=5,
                 chunks=None, require_quotes=False):
    ch = [Chunk(id="c1", text="charter obligations to humanity and its investors",
                org=org, source_type=source_type, filename="f.txt", score=0.9)]
    quotes = [{"excerpt": 1, "quote": "charter", "verified": True}] if require_quotes else None
    return RAGAnswer(question=question, answer="an answer", chunks=ch,
                     quotes=quotes,
                     quotes_verified=True if require_quotes else None)


def _fake_match(*, question, candidate, category, variant=None):
    w = {logic: 0.0 for logic in LOGICS}
    w["Market"] = 1.0
    return {"abstain": False, "weights": w, "reasoning": "r", "raw": "{}"}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    # Redirect ALL run-snapshot state into tmp so real data/profiles is never
    # touched (migrate_legacy would otherwise move the user's flat files).
    monkeypatch.setattr(runs, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runs, "CURRENT_PTR", tmp_path / "CURRENT")
    monkeypatch.setattr(ph, "Retriever", lambda: object())
    monkeypatch.setattr(ph, "answer_question", _fake_answer)
    monkeypatch.setattr(ph, "match_graded", _fake_match)
    return ph


def _rows_of_current():
    path = runs.run_paths(runs.get_current())["per_question"]
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_default_rows_keep_exact_legacy_schema(harness):
    harness.run_profiles(orgs=["OpenAI"], source_types=["published"], fresh=True)
    rows = _rows_of_current()
    assert len(rows) == 27  # 9 categories x 3 variants
    for r in rows:
        assert set(r) == LEGACY_KEYS


def test_flags_only_add_fields(harness):
    harness.run_profiles(orgs=["OpenAI"], source_types=["published"], fresh=True,
                         grounding=True, quotes=True)
    rows = _rows_of_current()
    assert len(rows) == 27
    for r in rows:
        assert set(r) == LEGACY_KEYS | GROUNDING_KEYS | QUOTE_KEYS
        assert 0.0 <= r["retrieval_grounding_score"] <= 1.0
        assert r["grounding_bucket"] in ("retrieval_missed", "abstained", "committed")
        assert r["quotes_verified"] is True


def test_grounding_only(harness):
    harness.run_profiles(orgs=["OpenAI"], source_types=["published"], fresh=True,
                         grounding=True)
    for r in _rows_of_current():
        assert set(r) == LEGACY_KEYS | GROUNDING_KEYS
