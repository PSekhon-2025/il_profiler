"""Offline tests for embedding_agreement: cosine math, truncation, and — the
regression that motivated them — per-variant reference_overrides resolution
(including the JSON-snapshot quirk of stringified override keys)."""
import json

import pytest

from il_rag import embedding_agreement as ea
from il_rag.questionnaire import LOGICS


def test_cosine_basics():
    assert ea._cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert ea._cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert ea._cosine([0, 0], [1, 1]) == 0.0  # zero vector defined as 0


def test_truncate_word_boundary_and_cap():
    long = "word " * 1000
    out = ea._truncate(long)
    assert len(out) <= ea.MAX_EMBED_CHARS
    assert not out.endswith(" wor")  # cut on a word boundary, not mid-word


def test_overrides_resolved_per_variant(tmp_path, monkeypatch):
    """A row's variant must select the overridden reference, and JSON-round-
    tripped override keys (ints -> strings) must still resolve."""
    base = {logic: f"base {logic}" for logic in LOGICS}
    snapshot = {
        "logics": LOGICS,
        "categories": ["Cat"],
        "questionnaire": {
            "Cat": {
                "questions": ["q1", "q2", "q3"],
                "reference_answers": base,
                "reference_overrides": {2: {"State": "OVERRIDDEN State"}},
            }
        },
    }
    # Round-trip through JSON exactly like runs.snapshot_questionnaire on disk:
    # the override key 2 becomes "2".
    snapshot = json.loads(json.dumps(snapshot))
    assert "2" in snapshot["questionnaire"]["Cat"]["reference_overrides"]

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "questionnaire.json").write_text(json.dumps(snapshot))
    rows = [
        {"org": "OpenAI", "source_type": "published", "qid": "Cat#2",
         "category": "Cat", "variant": 2, "abstain": False,
         "answer": "the answer text",
         "weights": {logic: (1.0 if logic == "State" else 0.0) for logic in LOGICS}},
    ]
    (run_dir / "per_question.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    monkeypatch.setattr(ea.runs, "run_dir", lambda rid: tmp_path / "runs" / rid)
    monkeypatch.setattr(ea.runs, "run_paths", lambda rid: {
        "per_question": tmp_path / "runs" / rid / "per_question.jsonl",
        "questionnaire": tmp_path / "runs" / rid / "questionnaire.json",
    })

    embedded: list[str] = []

    def fake_embed(texts):
        embedded.extend(texts)
        # Make "OVERRIDDEN State" the unique nearest neighbor of the answer.
        return [[1.0, 0.0] if ("OVERRIDDEN" in t or t == "the answer text")
                else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(ea, "embed", fake_embed)

    summary = ea.run_embedding_agreement(run_id="r1")

    # The overridden reference must have been embedded (i.e. not the base
    # State text for variant 2), and the row must agree via it.
    assert any("OVERRIDDEN" in t for t in embedded)
    assert summary["overall"] == {"n": 1, "agree": 1, "rate": 1.0}
    sim_rows = [json.loads(l) for l in  # noqa: E741
                (run_dir / "embedding_agreement" / "similarities.jsonl")
                .read_text().splitlines()]
    assert sim_rows[0]["embedding_nearest"] == "State"
    assert sim_rows[0]["agree"] is True
