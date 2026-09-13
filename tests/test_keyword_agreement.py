"""Offline tests for the keyword judge v2: the curated lexicon's invariants,
ladder-graded row scoring (exact / morphological / semantic), and the
end-to-end driver over fabricated files. No API, no lexicon files on disk —
the semantic rung is exercised through a fabricated vector space, exactly as
test_topic_keywords does."""
import json
import math

import pytest

from il_rag import keyword_agreement as ka
from il_rag import topic_keywords as tk
from il_rag.config import EMBEDDING_DIM
from il_rag.questionnaire import LOGICS


# ---------------------------------------------------------------------------
# Fabricated semantic space (the test_topic_keywords pattern): words are
# angles on a circle, cosine similarity = cos(a - b). "administration" sits
# close to "government"; "lettuce" is far from everything.
# ---------------------------------------------------------------------------
_ANGLES = {
    "government": 0.00,
    "administration": 0.10,
    "lettuce": 2.50,
}
_GRID = [-1.0, -0.5, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _vec(word: str) -> list:
    a = _ANGLES.get(word, 3.0 + (sum(ord(c) for c in word) % 50) / 100.0)
    return [math.cos(a), math.sin(a)] + [0.0] * (EMBEDDING_DIM - 2)


def _fake_embed(texts):
    return [_vec(t) for t in texts]


def _space(tmp_path):
    vectors = tk.WordVectors(tmp_path / "vec.npz", embed_fn=_fake_embed)
    return vectors, tk.Calibration(_GRID)


# ---------------------------------------------------------------------------
# Lexicon invariants — the curation rules, pinned
# ---------------------------------------------------------------------------
def test_lexicon_covers_exactly_the_seven_logics():
    assert set(ka.LOGIC_KEYWORDS) == set(LOGICS)
    for logic, kws in ka.LOGIC_KEYWORDS.items():
        assert kws, f"{logic} has an empty keyword list"


def test_lexicon_entries_are_normalized():
    """Lowercase, stripped, non-empty, unique within each list."""
    for logic, kws in ka.LOGIC_KEYWORDS.items():
        assert len(kws) == len(set(kws)), f"duplicate keyword in {logic}"
        for kw in kws:
            assert kw == kw.lower().strip(), f"unnormalized {kw!r} in {logic}"
            assert kw, f"empty keyword in {logic}"


def test_no_token_appears_in_two_logics():
    """Cross-logic credit must come from the answer, never from the lexicon."""
    seen: dict[str, str] = {}
    for logic, kws in ka.LOGIC_KEYWORDS.items():
        for kw in kws:
            for tok in kw.split():
                assert seen.setdefault(tok, logic) == logic, (
                    f"token {tok!r} appears in both {seen[tok]} and {logic}")


# ---------------------------------------------------------------------------
# Row scoring — the ladder, rung by rung (vectors=None disables semantic)
# ---------------------------------------------------------------------------
def _lex(**lists):
    """A minimal lexicon: every logic present, listed ones non-empty."""
    return {logic: lists.get(logic, []) for logic in LOGICS}


def test_exact_rung_scores_full_credit():
    lex = _lex(State=["government", "welfare"], Market=["profit", "pricing"])
    s = ka.score_row("The government funds welfare programs.", lex)
    assert s["no_overlap"] is False
    assert s["raw"]["State"] == pytest.approx(1.0)   # 2 of 2, both exact
    assert s["raw"]["Market"] == 0.0
    assert s["keyword_top"] == "State"
    assert sum(s["shares"].values()) == pytest.approx(1.0)
    assert s["matched"]["State"] == ["government", "welfare"]
    assert s["tiers"]["State"] == {"exact": 2}


def test_morphological_rung_earns_partial_credit():
    """'regulators' in the answer credits the keyword 'regulation'."""
    lex = _lex(State=["regulation"])
    s = ka.score_row("New rules from the regulators arrived.", lex)
    assert s["tiers"]["State"] == {"morphological": 1}
    assert 0.7 <= s["raw"]["State"] < 1.0            # ratio, not full credit
    assert s["matched"]["State"] == ["regulation~regulators"]


def test_phrase_keyword_needs_adjacency_for_exact():
    lex = _lex(Profession=["peer review"])
    s = ka.score_row("The work went through peer review.", lex)
    assert s["tiers"]["Profession"] == {"exact": 1}
    assert s["raw"]["Profession"] == pytest.approx(1.0)


def test_semantic_rung_fires_only_with_calibration(tmp_path):
    """'administration' is a neighbour of 'government' in the fabricated
    space: no credit without the calibration, graded credit with it."""
    lex = _lex(State=["government"])
    answer = "The administration announced its plans."

    plain = ka.score_row(answer, lex)                # no vectors/calibration
    assert plain["no_overlap"] is True
    assert plain["raw"]["State"] == 0.0

    vectors, cal = _space(tmp_path)
    sem = ka.score_row(answer, lex, vectors=vectors, calibration=cal)
    assert sem["tiers"]["State"] == {"semantic": 1}
    # cos(0.10) ~ 0.995 -> top of the grid, capped below exact's 1.0
    assert 0.9 <= sem["raw"]["State"] <= tk.KEYWORD_SEMANTIC_MAX_SCORE < 1.0
    assert sem["matched"]["State"] == ["government≈administration"]


def test_semantic_below_the_bar_scores_zero(tmp_path):
    """A word far from every keyword clears no rung and flags no_overlap."""
    vectors, cal = _space(tmp_path)
    s = ka.score_row("Mostly lettuce.", _lex(State=["government"]),
                     vectors=vectors, calibration=cal)
    assert s["no_overlap"] is True
    assert s["keyword_top"] is None
    assert all(v == 0.0 for v in s["shares"].values())


def test_graded_recall_is_a_mean_over_the_list():
    """One exact hit out of four keywords -> raw 0.25; recall beats count."""
    lex = _lex(State=["government", "welfare", "mandate", "sovereignty"],
               Market=["profit"])
    s = ka.score_row("Only the government appears, but profit does too.", lex)
    assert s["raw"]["State"] == pytest.approx(0.25)
    assert s["raw"]["Market"] == pytest.approx(1.0)
    assert s["keyword_top"] == "Market"              # mean, not raw count


# ---------------------------------------------------------------------------
# The run driver, end to end over fabricated files
# ---------------------------------------------------------------------------
def _run_dir(tmp_path, monkeypatch, rows):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "per_question.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(ka.runs, "run_dir", lambda rid: tmp_path / "runs" / rid)
    monkeypatch.setattr(ka.runs, "run_paths", lambda rid: {
        "per_question": tmp_path / "runs" / rid / "per_question.jsonl"})
    return run_dir


def test_run_driver_end_to_end_without_semantic(tmp_path, monkeypatch):
    w_state = {logic: (1.0 if logic == "State" else 0.0) for logic in LOGICS}
    rows = [
        {"org": "OpenAI", "source_type": "published", "qid": "Cat#1",
         "category": "Cat", "variant": 1, "abstain": False,
         "answer": "It complies with government regulation and oversight.",
         "weights": w_state},
        {"org": "OpenAI", "source_type": "published", "qid": "Cat#2",
         "category": "Cat", "variant": 2, "abstain": True,
         "answer": "", "weights": {logic: 0.0 for logic in LOGICS}},
    ]
    run_dir = _run_dir(tmp_path, monkeypatch, rows)
    # No calibration on disk in tests -> the semantic rung must stay off.
    monkeypatch.setattr(ka.Calibration, "load", classmethod(lambda cls: None))

    summary = ka.run_keyword_agreement(run_id="r1")

    assert summary["semantic_enabled"] is False
    assert summary["overall"]["n"] == 1              # abstained row excluded
    assert summary["overall"]["agree"] == 1
    assert summary["overall"]["rate"] == 1.0
    assert summary["tier_totals"].get("exact", 0) >= 2   # government, oversight

    out = [json.loads(line) for line in
           (run_dir / "keyword_agreement" / "rows.jsonl")
           .read_text().splitlines()]
    assert out[0]["keyword_top"] == "State"
    assert out[0]["agree"] is True
    assert "government" in out[0]["matched_words"]["State"]
    # regulation~ or exact, depending on surface form: the tag is recorded
    assert any(m.startswith("regulation") for m in out[0]["matched_words"]["State"])

    kw = json.loads((run_dir / "keyword_agreement" / "keywords.json").read_text())
    assert kw["semantic_enabled"] is False
    assert kw["lexicon"] == ka.LOGIC_KEYWORDS


def test_run_driver_uses_semantic_when_calibrated(tmp_path, monkeypatch):
    w_state = {logic: (1.0 if logic == "State" else 0.0) for logic in LOGICS}
    rows = [{"org": "OpenAI", "source_type": "published", "qid": "Cat#1",
             "category": "Cat", "variant": 1, "abstain": False,
             "answer": "The administration decides everything here.",
             "weights": w_state}]
    _run_dir(tmp_path, monkeypatch, rows)
    vectors, cal = _space(tmp_path)
    monkeypatch.setattr(ka.Calibration, "load", classmethod(lambda cls: cal))
    monkeypatch.setattr(ka, "WordVectors", lambda: vectors)
    # A tiny lexicon: the fabricated space parks every unknown word on a
    # tight arc, so the full curated lists would collide into spurious
    # semantic matches that say nothing about the driver under test.
    monkeypatch.setattr(ka, "LOGIC_KEYWORDS",
                        _lex(State=["government"], Market=["lettuce"]))

    summary = ka.run_keyword_agreement(run_id="r1")

    assert summary["semantic_enabled"] is True
    assert summary["tier_totals"].get("semantic", 0) >= 1
    assert summary["overall"]["agree"] == 1          # government ≈ administration


def test_run_driver_semantic_false_never_loads_calibration(tmp_path, monkeypatch):
    rows = [{"org": "OpenAI", "source_type": "published", "qid": "Cat#1",
             "category": "Cat", "variant": 1, "abstain": False,
             "answer": "The government said so.",
             "weights": {logic: (1.0 if logic == "State" else 0.0)
                         for logic in LOGICS}}]
    _run_dir(tmp_path, monkeypatch, rows)

    def _boom(cls):
        raise AssertionError("Calibration.load called despite semantic=False")
    monkeypatch.setattr(ka.Calibration, "load", classmethod(_boom))

    summary = ka.run_keyword_agreement(run_id="r1", semantic=False)
    assert summary["semantic_enabled"] is False
