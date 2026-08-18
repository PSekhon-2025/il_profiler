"""Offline tests for the lexical keyword judge: keyword derivation (the
distinctiveness filter is the load-bearing part), row scoring, and the
end-to-end driver over fabricated files."""
import json

import pytest

from il_rag import keyword_agreement as ka
from il_rag.questionnaire import LOGICS


def test_derive_keywords_drops_shared_framing_vocabulary():
    """Tokens in >max_df references are framing, not signal, and must go."""
    refs = {logic: f"the lab conduct governance {logic.lower()}ish"
            for logic in LOGICS}
    kws = ka.derive_keywords(refs, max_df=2)
    # 'lab', 'conduct', 'governance' appear in all seven -> dropped everywhere
    for logic in LOGICS:
        assert "conduct" not in kws[logic]
        assert "governance" not in kws[logic]
        # each logic keeps its unique token
        assert f"{logic.lower()}ish" in kws[logic]


def test_derive_keywords_respects_max_df_boundary():
    refs = {logic: "" for logic in LOGICS}
    refs["State"] = "regulators oversight shared"
    refs["Market"] = "customers revenue shared"
    refs["Corporation"] = "executives shared"
    # 'shared' is in 3 sets: kept at max_df=3, dropped at max_df=2
    assert "shared" in ka.derive_keywords(refs, max_df=3)["State"]
    assert "shared" not in ka.derive_keywords(refs, max_df=2)["State"]
    assert "regulators" in ka.derive_keywords(refs, max_df=2)["State"]


def test_score_row_recall_and_shares():
    kws = {logic: [] for logic in LOGICS}
    kws["State"] = ["regulators", "oversight", "lawful", "ministries"]
    kws["Market"] = ["customers", "revenue"]
    s = ka.score_row("The lab answers to regulators and its customers.", kws)
    assert s["no_overlap"] is False
    assert s["raw"]["State"] == pytest.approx(1 / 4)     # 1 of 4 keywords hit
    assert s["raw"]["Market"] == pytest.approx(1 / 2)    # 1 of 2 keywords hit
    assert s["keyword_top"] == "Market"                  # recall, not raw count
    assert sum(s["shares"].values()) == pytest.approx(1.0)
    assert s["matched"]["State"] == ["regulators"]
    assert s["matched"]["Market"] == ["customers"]


def test_score_row_no_overlap_flagged_not_forced():
    kws = {logic: ["zzzunseen"] for logic in LOGICS}
    s = ka.score_row("Nothing here matches at all.", kws)
    assert s["no_overlap"] is True
    assert s["keyword_top"] is None
    assert all(v == 0.0 for v in s["shares"].values())


def test_empty_keyword_set_scores_zero_not_crash():
    kws = {logic: [] for logic in LOGICS}
    kws["State"] = ["regulators"]
    s = ka.score_row("regulators appear", kws)
    assert s["keyword_top"] == "State"
    assert s["raw"]["Market"] == 0.0


def test_run_driver_end_to_end(tmp_path, monkeypatch):
    """Full path: snapshot with an override, one committed + one abstained row."""
    base = {logic: f"unique{logic.lower()} shared shared2" for logic in LOGICS}
    snapshot = {"logics": LOGICS, "categories": ["Cat"], "questionnaire": {
        "Cat": {"questions": ["q1", "q2", "q3"],
                "reference_answers": base,
                # override present AND stringified, as JSON round-trips store it
                "reference_overrides": {"2": {"State": "overridetoken only"}}}}}
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "questionnaire.json").write_text(json.dumps(snapshot))
    w_state = {logic: (1.0 if logic == "State" else 0.0) for logic in LOGICS}
    rows = [
        {"org": "OpenAI", "source_type": "published", "qid": "Cat#2",
         "category": "Cat", "variant": 2, "abstain": False,
         "answer": "the overridetoken appears here", "weights": w_state},
        {"org": "OpenAI", "source_type": "published", "qid": "Cat#3",
         "category": "Cat", "variant": 3, "abstain": True,
         "answer": "", "weights": {logic: 0.0 for logic in LOGICS}},
    ]
    (run_dir / "per_question.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(ka.runs, "run_dir", lambda rid: tmp_path / "runs" / rid)
    monkeypatch.setattr(ka.runs, "run_paths", lambda rid: {
        "per_question": tmp_path / "runs" / rid / "per_question.jsonl",
        "questionnaire": tmp_path / "runs" / rid / "questionnaire.json"})

    summary = ka.run_keyword_agreement(run_id="r1")

    # abstained row excluded; the committed row matched via the OVERRIDE keyword
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["agree"] == 1 and summary["overall"]["rate"] == 1.0
    out = [json.loads(l) for l in  # noqa: E741
           (run_dir / "keyword_agreement" / "rows.jsonl").read_text().splitlines()]
    assert out[0]["keyword_top"] == "State"
    assert "overridetoken" in out[0]["matched_words"]["State"]
    # derived keyword sets were persisted for audit
    kw = json.loads((run_dir / "keyword_agreement" / "keywords.json").read_text())
    assert "overridetoken" in kw["Cat#2"]["State"]
