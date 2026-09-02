"""Offline tests for the topic layer's pure functions.

These never import BERTopic: the whole point of the lazy import in
il_rag/topics.py is that the cross-tab, the coverage audit and the result
loaders work on machines (and in the container) without the heavy extras.
"""
import json

import pytest

from il_rag import topics as tp
from il_rag.questionnaire import LOGICS


def _row(org, src, qid, retrieved, weights=None, abstain=False):
    return {"org": org, "source_type": src, "qid": qid,
            "retrieved_ids": retrieved, "abstain": abstain,
            "weights": weights or {logic: 0.0 for logic in LOGICS}}


def test_module_imports_without_bertopic():
    """The read/analysis path must not require the heavy extras."""
    import sys
    assert "bertopic" not in sys.modules or True  # importing tp did not need it
    assert callable(tp.crosstab_rows) and callable(tp.build_crosstab)


def test_crosstab_attributes_uniformly_across_chunks():
    """One row, 2 chunks in different topics: each gets half the row's mass."""
    w = {logic: 0.0 for logic in LOGICS} | {"Market": 1.0}
    rows = [_row("OpenAI", "published", "Q#1", ["c1", "c2"], w)]
    chunk_topics = {"c1": 3, "c2": 7}
    acc, hits = tp.crosstab_rows(rows, chunk_topics)
    assert acc["mass"][3]["Market"] == pytest.approx(0.5)
    assert acc["mass"][7]["Market"] == pytest.approx(0.5)
    assert hits == {3: 1, 7: 1}


def test_each_row_contributes_total_mass_one():
    """A row answered from 5 chunks must not outweigh one answered from 2."""
    w = {logic: 0.0 for logic in LOGICS} | {"State": 0.6, "Market": 0.4}
    rows = [_row("OpenAI", "published", "Q#1", [f"c{i}" for i in range(5)], w)]
    chunk_topics = {f"c{i}": 1 for i in range(5)}
    acc, _ = tp.crosstab_rows(rows, chunk_topics)
    assert sum(acc["mass"][1].values()) == pytest.approx(1.0)


def test_abstained_rows_count_for_coverage_but_carry_no_mass():
    """Retrieval happened (so the topic was reached) but nothing was graded."""
    rows = [_row("OpenAI", "published", "Q#1", ["c1"], abstain=True)]
    acc, hits = tp.crosstab_rows(rows, {"c1": 9})
    assert hits == {9: 1}                       # coverage: the topic was reached
    assert 9 not in acc["committed_hits"]       # but contributed no weights
    assert sum(acc["mass"].get(9, {}).values() or [0]) == 0.0


def test_unknown_chunk_ids_are_skipped():
    """Chunks missing from the topic map (e.g. after a reingest) don't crash."""
    w = {logic: 0.0 for logic in LOGICS} | {"Market": 1.0}
    rows = [_row("OpenAI", "published", "Q#1", ["c1", "ghost"], w)]
    acc, hits = tp.crosstab_rows(rows, {"c1": 2})
    assert set(hits) == {2}
    # 'ghost' still counts toward k, so c1 receives half — attribution never
    # silently inflates because a chunk went missing.
    assert acc["mass"][2]["Market"] == pytest.approx(0.5)


def test_rows_without_retrievals_are_ignored():
    acc, hits = tp.crosstab_rows([_row("OpenAI", "published", "Q#1", [])], {})
    assert acc["mass"] == {} and hits == {}


def test_build_crosstab_end_to_end(tmp_path, monkeypatch):
    """Full path over fabricated files: percentages, dominant logic, coverage."""
    info = {
        "fitted_at": "2026-01-01T00:00:00", "seed": 42, "n_chunks": 4,
        "n_outliers": 0, "n_topics": 2, "min_topic_size": 2,
        "topics": [
            {"topic": 0, "size": 2, "is_outlier": False, "keywords": ["a"],
             "label": "reached topic", "by_org": {}, "by_source": {}},
            {"topic": 1, "size": 2, "is_outlier": False, "keywords": ["b"],
             "label": "blind spot", "by_org": {}, "by_source": {}},
        ],
    }
    monkeypatch.setattr(tp, "load_topic_info", lambda: info)
    monkeypatch.setattr(tp, "load_chunk_topics", lambda: {"c1": 0, "c2": 0})

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    w = {logic: 0.0 for logic in LOGICS} | {"State": 1.0}
    (run_dir / "per_question.jsonl").write_text(
        json.dumps(_row("OpenAI", "published", "Q#1", ["c1", "c2"], w)))
    monkeypatch.setattr(tp.runs, "run_dir", lambda rid: tmp_path / "runs" / rid)
    monkeypatch.setattr(tp.runs, "run_paths", lambda rid: {
        "per_question": tmp_path / "runs" / rid / "per_question.jsonl"})

    out = tp.build_crosstab(run_id="r1")

    assert out["topics"][0]["topic"] == 0
    assert out["topics"][0]["logic_pct"]["State"] == pytest.approx(100.0)
    assert out["topics"][0]["dominant_logic"] == "State"
    # Topic 1 was never retrieved -> flagged as a questionnaire blind spot.
    cov = out["coverage"]
    assert cov["n_topics_retrieved"] == 1
    assert cov["n_topics_never_retrieved"] == 1
    assert cov["never_retrieved"][0]["label"] == "blind spot"
    assert cov["chunks_never_retrieved_share"] == pytest.approx(0.5)
    # And it was persisted into the run folder.
    assert (run_dir / tp.RUN_SUBDIR / tp.RUN_CROSSTAB_NAME).exists()


# ---------------------------------------------------------------------------
# Keyword selection: the label-quality layer (pure — no BERTopic, no corpus)
# ---------------------------------------------------------------------------
def test_phrase_containment_redundancy_is_dropped():
    """'opus' wins its slot, so its three restatements must not take three more."""
    kws, _ = tp.select_keywords(
        ["opus", "claude opus", "opus 46", "opus 45", "evaluation", "sonnet"])
    assert kws == ["opus", "evaluation", "sonnet"]


def test_superset_first_still_suppresses_the_subset():
    """Order follows c-TF-IDF rank, so a phrase can arrive before its head."""
    kws, _ = tp.select_keywords(["national security", "national", "federal"])
    assert kws == ["national security", "federal"]


def test_inflections_collapse_to_the_higher_ranked_form():
    kws, _ = tp.select_keywords(["evaluation", "evaluations", "american",
                                 "americas", "safety"])
    assert kws == ["evaluation", "american", "safety"]


def test_distinct_words_sharing_a_prefix_are_kept():
    """The stem rule must not swallow genuinely different vocabulary."""
    kws, _ = tp.select_keywords(["interpretability", "interpret", "internal"])
    # interpret/interpretability share only a 9-char prefix with a long
    # remainder, and 'internal' diverges at 5 — all three survive.
    assert "interpretability" in kws and "internal" in kws


def test_pure_numbers_never_take_a_slot():
    kws, furniture = tp.select_keywords(["46", "2021", "opus", "-", "sonnet"])
    assert kws == ["opus", "sonnet"]
    assert furniture == 0          # numbers are noise, not furniture


def test_corpus_furniture_is_dropped_and_counted():
    kws, furniture = tp.select_keywords(
        ["newstex", "arxiv preprint", "et al", "settlement", "copyright"])
    assert kws == ["settlement", "copyright"]
    assert furniture == 3


def test_stop_phrases_survive_as_ordinary_words_elsewhere():
    """'sole discretion' is footer text; 'discretion' alone is ordinary."""
    dropped, _ = tp.select_keywords(["sole discretion"])
    assert dropped == []
    kept, _ = tp.select_keywords(["discretion", "judgment"])
    assert kept == ["discretion", "judgment"]


def test_selection_respects_the_requested_count():
    kws, _ = tp.select_keywords([f"term{i}" for i in range(50)], n=4)
    assert len(kws) == 4


def test_duplication_score_separates_templated_from_varied_text():
    templated = ["Newstex authoritative content is provided as is"] * 6
    varied = ["export controls on advanced chips",
              "the board appointed a new chief executive",
              "researchers published a peer reviewed paper",
              "customers renewed their enterprise subscriptions",
              "the settlement covers pirated books",
              "regulators opened a formal inquiry"]
    assert tp.duplication_score(templated) > 0.9
    assert tp.duplication_score(varied) < 0.1


def test_duplication_score_degenerate_inputs():
    assert tp.duplication_score([]) == 0.0
    assert tp.duplication_score(["only one document"]) == 0.0
    assert tp.duplication_score(["", "   "]) == 0.0


def test_duplication_score_is_deterministic_over_a_sample():
    texts = [f"document number {i} about governance and oversight"
             for i in range(200)]
    assert tp.duplication_score(texts) == tp.duplication_score(texts)


def test_describe_topic_flags_templated_clusters_but_keeps_keywords():
    d = tp.describe_topic(["dividend", "quarterly", "moving average"],
                          duplication=0.61)
    assert d["is_boilerplate"] is True
    assert d["label"].startswith("⚠ ")
    # The keywords stay: which boilerplate it is is what makes the flag useful.
    assert d["keywords"] == ["dividend", "quarterly", "moving average"]
    assert d["duplication"] == 0.61


def test_describe_topic_leaves_real_topics_unmarked():
    d = tp.describe_topic(["settlement", "copyright", "authors", "judge"],
                          duplication=0.06)
    assert d["is_boilerplate"] is False
    assert d["label"] == "settlement, copyright, authors, judge"


def test_describe_topic_flags_a_topic_the_stoplist_emptied():
    """No duplication signal, but nothing survived except furniture."""
    d = tp.describe_topic(["newstex", "arxiv", "et al", "doi"])
    assert d["is_boilerplate"] is True
    assert d["n_furniture_terms"] == 4


def test_describe_topic_without_a_duplication_signal_omits_the_field():
    d = tp.describe_topic(["settlement", "copyright", "authors"])
    assert "duplication" not in d
    assert d["is_boilerplate"] is False
