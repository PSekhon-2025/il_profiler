"""Offline tests for replicate aggregation: the averaging math and the
questionnaire guard that stops two different measurements being mixed."""
import json

import pytest

from il_rag import replicates as rp
from il_rag.questionnaire import LOGICS


def _profile(pcts: dict, answered: int = 20) -> dict:
    """One (lab, source) profile with the given percentages."""
    full = {logic: 0.0 for logic in LOGICS} | pcts
    return {"logic_pct": full, "answered": answered, "abstained": 0,
            "by_category": {}}


def _runs(*pct_dicts) -> dict:
    return {f"run{i}": {"OpenAI": {"published": _profile(p)}}
            for i, p in enumerate(pct_dicts, 1)}


def test_mean_and_spread_across_replicates():
    agg = rp.aggregate_profiles(
        _runs({"Market": 30.0}, {"Market": 40.0}, {"Market": 50.0}),
        orgs=["OpenAI"], source_types=["published"])
    s = agg["profiles"]["OpenAI"]["published"]["logics"]["Market"]
    assert s["mean"] == pytest.approx(40.0)
    assert s["sd"] == pytest.approx(10.0)              # sample sd of 30/40/50
    assert s["sem"] == pytest.approx(10.0 / 3 ** 0.5, abs=0.01)
    assert (s["min"], s["max"], s["range"]) == (30.0, 50.0, 20.0)
    assert s["per_run"] == {"run1": 30.0, "run2": 40.0, "run3": 50.0}


def test_sd_withheld_below_three_replicates():
    """Two runs give the spread a single degree of freedom — not reported."""
    agg = rp.aggregate_profiles(_runs({"Market": 30.0}, {"Market": 50.0}),
                                orgs=["OpenAI"], source_types=["published"])
    s = agg["profiles"]["OpenAI"]["published"]["logics"]["Market"]
    assert s["sd"] is None and s["sem"] is None
    assert s["mean"] == pytest.approx(40.0)   # the mean is still meaningful
    assert s["range"] == 20.0                 # and so is the observed spread


def test_dominant_agreement_tracks_label_stability():
    """Two runs say Market, one says State -> the average is Market, 2/3."""
    agg = rp.aggregate_profiles(
        _runs({"Market": 60.0}, {"Market": 55.0}, {"State": 60.0}),
        orgs=["OpenAI"], source_types=["published"])
    d = agg["profiles"]["OpenAI"]["published"]
    assert d["mean_dominant"] == "Market"
    # stored rounded to 3dp by design
    assert d["dominant_agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert d["dominant_per_run"] == ["Market", "Market", "State"]


def test_profiles_with_no_answers_are_skipped():
    """A (lab, source) that abstained everywhere must not enter the average."""
    empty = {"logic_pct": {logic: 0.0 for logic in LOGICS}, "answered": 0,
             "abstained": 27, "by_category": {}}
    by_run = {"run1": {"OpenAI": {"published": _profile({"Market": 40.0})}},
              "run2": {"OpenAI": {"published": empty}}}
    agg = rp.aggregate_profiles(by_run, orgs=["OpenAI"],
                                source_types=["published"])
    d = agg["profiles"]["OpenAI"]["published"]
    assert d["replicates"] == 1
    assert d["logics"]["Market"]["mean"] == pytest.approx(40.0)


def test_fingerprint_is_content_based_not_formatting_based(tmp_path, monkeypatch):
    """Re-indented JSON is the SAME questionnaire; changed text is not."""
    q = {"logics": LOGICS, "categories": ["C"],
         "questionnaire": {"C": {"questions": ["q1"], "reference_answers": {}}}}
    for name, blob in (("a", json.dumps(q)),
                       ("b", json.dumps(q, indent=4, sort_keys=False)),
                       ("c", json.dumps({**q, "categories": ["D"]}))):
        d = tmp_path / name
        d.mkdir()
        (d / "questionnaire.json").write_text(blob)
    monkeypatch.setattr(rp.runs, "run_paths", lambda rid: {
        "questionnaire": tmp_path / rid / "questionnaire.json"})

    assert rp.questionnaire_fingerprint("a") == rp.questionnaire_fingerprint("b")
    assert rp.questionnaire_fingerprint("a") != rp.questionnaire_fingerprint("c")
    groups = rp.group_by_questionnaire(["a", "b", "c"])
    assert len(groups) == 2


def test_build_report_refuses_mixed_questionnaires(tmp_path, monkeypatch):
    """The guard is the point: averaging different question sets is invalid."""
    for name, cats in (("a", ["C"]), ("b", ["D"])):
        d = tmp_path / name
        d.mkdir()
        (d / "questionnaire.json").write_text(json.dumps({"categories": cats}))
        (d / "company_profiles.json").write_text(
            json.dumps({"OpenAI": {"published": _profile({"Market": 40.0})}}))
    monkeypatch.setattr(rp.runs, "run_paths", lambda rid: {
        "questionnaire": tmp_path / rid / "questionnaire.json",
        "profiles_json": tmp_path / rid / "company_profiles.json"})

    with pytest.raises(SystemExit, match="did NOT use the same questionnaire"):
        rp.build_report(["a", "b"])
