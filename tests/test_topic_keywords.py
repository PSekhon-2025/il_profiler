"""Offline tests for the topic-keyword semantic matcher.

Everything here is fabricated and deterministic. The embedding space is eight
hand-placed unit vectors on a circle, so every cosine is exact arithmetic
rather than a guess about what a real model would say, and the calibration is
an eleven-point grid whose percentiles can be read off by eye. Nothing imports
BERTopic or Chroma, and one test asserts that embedding is never reached when
the semantic rung is disabled.
"""
import json
import math

import pytest

from il_rag import topic_keywords as tk

DIM = 8

# Angles in radians on the unit circle; cosine similarity between two words is
# cos(a - b). Chosen so the ordering the feature exists to produce is exact:
# manager is nearest hierarchy, then sales, then revenue, then lettuce.
_ANGLES = {
    "manager": 0.00,
    "management": 0.20,
    "hierarchy": 0.30,
    "governance": 0.35,
    "governing": 0.36,
    "oversight": 0.45,
    "export": 0.60,
    "controls": 0.65,
    "chips": 0.70,
    "safety": 0.80,
    "policy": 0.90,
    "sales": 1.00,
    "revenue": 1.20,
    "litigation": 2.00,
    "lettuce": 2.50,
}

# Ascending, eleven points, so grid[i] is the (i/10)-quantile and a
# percentile can be read straight off the index.
_GRID = [-1.0, -0.5, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _angle(word: str) -> float:
    if word in _ANGLES:
        return _ANGLES[word]
    # Stable across processes (unlike hash() for str, which is randomized),
    # and parked far from every named word.
    return 3.0 + (sum(ord(c) for c in word) % 50) / 100.0


def _vec(word: str) -> list:
    a = _angle(word)
    return [math.cos(a), math.sin(a)] + [0.0] * (DIM - 2)


def _fake_embed(texts):
    return [_vec(t) for t in texts]


def _space(tmp_path=None, *, calibrated=True):
    """(WordVectors, Calibration | None) over the fabricated space."""
    path = (tmp_path / "vec.npz") if tmp_path is not None else None
    vectors = tk.WordVectors(path, embed_fn=_fake_embed)
    vectors.ensure(sorted(_ANGLES))
    cal = tk.Calibration(_GRID, {"vocabulary": [[w, 10] for w in sorted(_ANGLES)]})
    return vectors, (cal if calibrated else None)


def _row(qid="Q#1", answer="", retrieved=("c1",), org="OpenAI",
         source_type="published", category="authority"):
    return {"org": org, "source_type": source_type, "qid": qid,
            "category": category, "variant": 1, "answer": answer,
            "retrieved_ids": list(retrieved), "abstain": False,
            "weights": {"Market": 1.0}}


# ---------------------------------------------------------------------------
# The exact rung
# ---------------------------------------------------------------------------
def test_exact_unigram_matches_whole_token_only():
    hit = tk.score_keyword("manager", "the manager said so")
    assert hit["tier"] == tk.TIER_EXACT
    assert hit["score"] == 1.0
    assert hit["rule"] == "token"
    assert hit["matched"] == "manager"
    # "management" contains "manager" as no whole token: that is the
    # morphological rung's job, not the exact rung's.
    near = tk.score_keyword("manager", "management structures matter")
    assert near["tier"] == tk.TIER_MORPH


def test_two_character_keyword_is_matchable():
    """The reason _words is not grounding.content_tokens."""
    assert tk.score_keyword("ai", "ai systems are here")["tier"] == tk.TIER_EXACT


def test_stopword_keyword_is_still_matchable_exactly():
    assert tk.score_keyword("all", "all systems")["tier"] == tk.TIER_EXACT


def test_exact_bigram_requires_adjacency():
    adjacent = tk.score_keyword("export controls", "new export controls on chips")
    assert adjacent["tier"] == tk.TIER_EXACT
    assert adjacent["rule"] == "phrase_adjacent"
    split = tk.score_keyword("export controls", "controls on the export of chips")
    assert split["tier"] != tk.TIER_EXACT


def test_bigram_falls_back_to_min_of_parts_capped_at_morph():
    """Both parts occur verbatim, but the PHRASE did not — so not exact."""
    out = tk.score_keyword("export controls", "controls on the export of chips")
    assert out["rule"] == "bigram_parts"
    assert out["tier"] == tk.TIER_MORPH
    assert out["score"] == 1.0            # min(1.0, 1.0) over the two parts
    assert out["matched"] == "export|controls"


def test_bigram_matched_words_are_deduplicated(tmp_path):
    """Both parts can land on the same answer word; "x|x" reads as a bug."""
    vectors, cal = _space(tmp_path)
    out = tk.score_keyword("manager management", "a flat hierarchy of teams",
                           vectors=vectors, calibration=cal)
    assert out["rule"] == "bigram_parts"
    assert out["matched"] == "hierarchy"


def test_bigram_score_is_the_weaker_part(tmp_path):
    """One part present, the other absent: the phrase is not present."""
    vectors, cal = _space(tmp_path)
    out = tk.score_keyword("export lettuce", "controls on the export of chips",
                           vectors=vectors, calibration=cal)
    assert out["rule"] == "bigram_parts"
    assert out["tier"] == tk.TIER_ABSENT   # min over parts, one of which missed
    assert out["score"] == 0.0             # so the phrase is not present at all


# ---------------------------------------------------------------------------
# The morphological rung
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("word,expected", [
    ("managers", 0.9333),
    ("managerial", 0.8235),
    ("management", 0.7059),
])
def test_morphological_tier_handles_inflections(word, expected):
    assert tk.morph_score("manager", word) == pytest.approx(expected, abs=1e-4)


def test_morphological_tier_handles_suffix_changes():
    assert tk.morph_score("hierarchy", "hierarchical") == pytest.approx(0.7619, abs=1e-4)
    # The prefix requirement drops to len(shorter), so a shorter root matches.
    assert tk.morph_score("safety", "safe") == pytest.approx(0.8, abs=1e-4)


def test_morphological_rejects_short_shared_prefix():
    assert tk.morph_score("manager", "mandate") is None   # prefix 3 < 5
    assert tk.morph_score("manager", "sales") is None      # prefix 0


def test_morphological_refuses_short_words():
    """A shared-prefix rule over three-letter words is noise."""
    assert tk.morph_score("ai", "aim") is None


def test_morphological_tier_picks_the_best_match():
    out = tk.score_keyword("manager", "management by managers of the managerial kind")
    assert out["tier"] == tk.TIER_MORPH
    assert out["matched"] == "managers"    # 0.933, the highest ratio available


# ---------------------------------------------------------------------------
# The semantic rung — the point of the feature
# ---------------------------------------------------------------------------
def test_manager_ranks_hierarchy_above_sales(tmp_path):
    """The requirement, asserted directly.

    manager -> hierarchy must beat manager -> sales, which must beat
    manager -> revenue, which must beat a wholly unrelated word.
    """
    vectors, cal = _space(tmp_path)

    def s(text):
        return tk.score_keyword("manager", text, vectors=vectors, calibration=cal)

    hierarchy = s("a flat hierarchy of teams")
    sales = s("a flat sales of teams")
    revenue = s("a flat revenue of teams")
    lettuce = s("a flat lettuce of teams")

    # Only hierarchy clears the bar, and it never scores a full 100% —
    # that is reserved for a literal occurrence.
    assert hierarchy["tier"] == tk.TIER_SEMANTIC
    assert hierarchy["rule"] == "embedding_percentile"
    assert hierarchy["matched"] == "hierarchy"
    assert tk.KEYWORD_SEMANTIC_MIN_PERCENTILE <= hierarchy["score"] < 1.0

    # The other three cleared nothing, so they score exactly 0 — a dropped
    # keyword must not earn partial credit toward retention. Their RANKING
    # survives in near_miss, which is what the explorer surfaces.
    for miss in (sales, revenue, lettuce):
        assert miss["tier"] == tk.TIER_ABSENT
        assert miss["score"] == 0.0
    assert (hierarchy["score"] > sales["near_miss"]
            > revenue["near_miss"] > lettuce["near_miss"])


def test_a_literal_hit_always_outscores_a_synonym(tmp_path):
    vectors, cal = _space(tmp_path)
    literal = tk.score_keyword("manager", "the manager decided",
                               vectors=vectors, calibration=cal)
    synonym = tk.score_keyword("manager", "the hierarchy decided",
                               vectors=vectors, calibration=cal)
    assert literal["score"] == 1.0 > synonym["score"]


def test_near_miss_is_recorded_but_never_scored(tmp_path):
    """A miss reports how close it got — as diagnosis, not as credit."""
    vectors, cal = _space(tmp_path)
    out = tk.score_keyword("manager", "quarterly sales figures",
                           vectors=vectors, calibration=cal)
    assert out["tier"] == tk.TIER_ABSENT
    assert out["score"] == 0.0                       # contributes nothing
    assert 0.0 < out["near_miss"] < tk.KEYWORD_SEMANTIC_MIN_PERCENTILE
    assert out["cosine"] is not None


def test_dropped_keywords_do_not_inflate_retention(tmp_path):
    """dropped_share and retention must tell the same story, not opposite ones."""
    vectors, cal = _space(tmp_path)
    row = _row(answer="quarterly sales figures", retrieved=["c1"])
    out = tk.score_row(row, {3: ["manager", "litigation"]}, {"c1": 3},
                       vectors=vectors, calibration=cal)
    assert out["dropped_share"] == 1.0
    assert out["retention"] == 0.0
    # ...while the near misses are still on the record for a recalibration.
    assert all(k["near_miss"] is not None for k in out["keywords"])


def test_ladder_is_pure_computation_without_vectors(monkeypatch):
    """vectors=None must never reach the embedder (quote_provenance's rule)."""
    def boom(*a, **k):
        raise AssertionError("the ladder embedded something with vectors=None")

    monkeypatch.setattr(tk, "_embed_batched", boom)
    out = tk.score_keyword("manager", "a flat hierarchy of teams", vectors=None)
    assert out["tier"] == tk.TIER_ABSENT
    assert out["score"] == 0.0
    assert out["cosine"] is None


def test_semantic_tier_requires_calibration(tmp_path):
    """A raw cosine is never used as a score — no calibration, no rung."""
    vectors, _ = _space(tmp_path)
    out = tk.score_keyword("manager", "a flat hierarchy of teams",
                           vectors=vectors, calibration=None)
    assert out["tier"] == tk.TIER_ABSENT
    assert out["score"] == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_percentile_is_bisect_exact():
    cal = tk.Calibration(_GRID)
    assert cal.percentile(-5.0) == 0.0            # below the grid
    assert cal.percentile(5.0) == 1.0             # above the grid
    assert cal.percentile(-1.0) == 0.0            # the grid minimum itself
    assert cal.percentile(1.0) == 1.0             # the grid maximum itself
    assert cal.percentile(0.5) == pytest.approx(0.5)
    prev = -1.0
    for i in range(51):
        p = cal.percentile(-1.0 + 2.0 * i / 50.0)
        assert p >= prev
        prev = p


def test_semantic_score_is_capped_below_one(tmp_path):
    """100% is reserved for a literal occurrence, never earned by a neighbour."""
    vectors, _ = _space(tmp_path)
    # A two-point grid puts every plausible cosine at the very top.
    ceiling_cal = tk.Calibration([-1.0, 0.0])
    out = tk.score_keyword("manager", "a flat hierarchy of teams",
                           vectors=vectors, calibration=ceiling_cal)
    assert out["tier"] == tk.TIER_SEMANTIC
    assert out["score"] == pytest.approx(tk.KEYWORD_SEMANTIC_MAX_SCORE)
    assert out["score"] < 1.0


def test_empty_calibration_is_inert():
    assert tk.Calibration([]).percentile(0.9) == 0.0


def test_build_calibration_over_a_fabricated_vocabulary(tmp_path):
    """The histogram + grid path, with no Chroma and no real model."""
    vectors = tk.WordVectors(tmp_path / "vec.npz", embed_fn=_fake_embed)
    vocab = [(w, 10) for w in sorted(_ANGLES)]
    out = tk.build_calibration(vocabulary=vocab, vectors=vectors, bins=2000,
                               grid_points=101, path=tmp_path / "cal.json")
    n = len(vocab)
    assert out["n_pairs"] == n * (n - 1) // 2     # every unordered pair, once
    grid = out["quantile_grid"]
    assert len(grid) == 101
    assert grid == sorted(grid)
    assert out["cos_min"] <= out["cos_p50"] <= out["cos_p99"] <= out["cos_max"]

    reloaded = tk.Calibration.load(tmp_path / "cal.json")
    assert reloaded is not None
    assert reloaded.vocabulary() == [w for w, _ in vocab]
    assert reloaded.percentile(out["cos_max"] + 1.0) == 1.0


def test_calibration_load_returns_none_when_absent(tmp_path):
    assert tk.Calibration.load(tmp_path / "nope.json") is None


# ---------------------------------------------------------------------------
# The word-vector cache
# ---------------------------------------------------------------------------
def test_word_vectors_cache_is_append_only(tmp_path):
    path = tmp_path / "vec.npz"
    v = tk.WordVectors(path, embed_fn=_fake_embed)
    assert v.ensure(["manager", "hierarchy"]) == 2
    assert v.ensure(["hierarchy", "sales"]) == 1        # only the miss
    assert v.ensure(["manager", "sales"]) == 0
    assert v.calls == 3
    v.save()

    reloaded = tk.WordVectors(path, embed_fn=_fake_embed)
    assert reloaded.ensure(["manager", "hierarchy", "sales"]) == 0
    assert reloaded.calls == 0
    assert reloaded.known() == {"manager", "hierarchy", "sales"}


def test_readonly_cache_never_embeds_and_never_writes(tmp_path, monkeypatch):
    path = tmp_path / "vec.npz"
    tk.WordVectors(path, embed_fn=_fake_embed).ensure(["manager"])

    def boom(*a, **k):
        raise AssertionError("a readonly cache embedded something")

    monkeypatch.setattr(tk, "_embed_batched", boom)
    v = tk.WordVectors(path, readonly=True)
    assert v.ensure(["hierarchy"]) == 0
    assert v.get("hierarchy") is None
    v.save()                                            # must be a no-op
    assert not path.exists()


def test_float16_roundtrip_preserves_cosine(tmp_path):
    """The cast is only safe because vectors are unit-normalized first."""
    import numpy as np

    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(12)]
    raw = {w: rng.normal(size=64) for w in words}
    path = tmp_path / "vec.npz"
    v = tk.WordVectors(path, embed_fn=lambda ws: [raw[w] for w in ws])
    v.ensure(words)
    before = {w: v.get(w).copy() for w in words}
    v.save()

    after = tk.WordVectors(path)
    for a in words:
        for b in words:
            exact = float(before[a] @ before[b])
            got = float(after.get(a) @ after.get(b))
            assert abs(exact - got) < 1e-3


def test_cosine_matrix_matches_scalar_cosine(tmp_path):
    """One definition of the notion, vectorized in exactly one place."""
    from il_rag.embedding_agreement import _cosine

    vectors, _ = _space(tmp_path)
    words = sorted(_ANGLES)
    kept, mat = vectors.matrix(words)
    query = vectors.get("manager")
    sims = tk._cosine_matrix(mat, query)
    for w, got in zip(kept, sims):
        assert float(got) == pytest.approx(
            _cosine(list(query), list(vectors.get(w))), abs=1e-6)


# ---------------------------------------------------------------------------
# Rollup
# ---------------------------------------------------------------------------
def test_row_topics_excludes_outliers_and_unknown_ids():
    ids = ["c1", "c2", "ghost", "c3", "c1"]
    chunk_topics = {"c1": 3, "c2": tk.topics_mod.OUTLIER_TOPIC, "c3": 7}
    assert tk.row_topics(ids, chunk_topics) == [3, 7]


def test_score_row_unions_the_keywords_of_its_topics():
    row = _row(answer="the manager approved export controls",
               retrieved=["c1", "c2"])
    out = tk.score_row(row, {3: ["manager", "export controls"], 7: ["manager", "chips"]},
                       {"c1": 3, "c2": 7})
    assert out["topics"] == [3, 7]
    # "manager" appears in both topics and is scored once.
    assert [k["keyword"] for k in out["keywords"]] == [
        "manager", "export controls", "chips"]
    assert out["n_keywords"] == 3


def test_rows_reaching_no_clustered_topic_are_skipped_not_guessed():
    row = _row(retrieved=["ghost"])
    out = tk.score_row(row, {3: ["manager"]}, {"c1": 3})
    assert out["skipped"] == "no_topics"
    assert out["keywords"] == []
    # and they are excluded from the rates rather than counted as zero
    s = tk._slice([out])
    assert s["n"] == 1 and s["n_scored"] == 0 and s["n_no_topics"] == 1


def test_row_tier_shares_partition():
    row = _row(answer="the manager approved chips and management review",
               retrieved=["c1"])
    out = tk.score_row(row, {3: ["manager", "chips", "governance", "litigation"]},
                       {"c1": 3})
    total = (out["verbatim_share"] + out["morph_share"]
             + out["semantic_share"] + out["dropped_share"])
    assert total == pytest.approx(1.0)
    assert out["retention"] == pytest.approx(
        sum(k["score"] for k in out["keywords"]) / 4)
    assert out["semantic_lift"] == pytest.approx(
        out["retention"] - out["verbatim_share"])


def test_verbatim_ceiling_is_free(monkeypatch):
    """The circularity baseline costs nothing — exact rung only."""
    def boom(*a, **k):
        raise AssertionError("the ceiling arm embedded something")

    monkeypatch.setattr(tk, "_embed_batched", boom)
    row = _row(answer="nothing relevant here", retrieved=["c1"])
    out = tk.score_row(row, {3: ["manager", "chips"]}, {"c1": 3},
                       chunk_texts={"c1": "the manager bought chips"})
    assert out["verbatim_ceiling"] == 1.0     # both keywords are in their chunk
    assert out["retention"] == 0.0            # but neither survived the answer


def test_null_arm_is_seeded_and_foreign():
    topic_keywords = {t: [f"w{t}"] for t in range(10)}
    chunk_topics = {"c1": 0}
    row = _row(answer="w3 w5 w7", retrieved=["c1"])
    a = tk.score_row(row, topic_keywords, chunk_topics, all_topics=list(range(10)))
    b = tk.score_row(row, topic_keywords, chunk_topics,
                     all_topics=list(reversed(range(10))))
    # Same row, same draw — regardless of the order topics are offered in.
    assert sorted(a["null_topics"]) == sorted(b["null_topics"])
    assert a["retention_null"] == b["retention_null"]
    assert len(a["null_topics"]) == tk.KEYWORD_NULL_DRAWS
    assert 0 not in a["null_topics"]          # never a topic the row retrieved

    # A different row gets a different draw, so the floor is not one lucky topic.
    other = tk.score_row(_row(qid="Q#9", answer="w3 w5 w7", retrieved=["c1"]),
                         topic_keywords, chunk_topics, all_topics=list(range(10)))
    assert other["null_topics"] != a["null_topics"]


def test_build_terms_counts_tiers_and_top_matches():
    rows = [
        tk.score_row(_row(qid="Q#1", answer="the manager left", retrieved=["c1"]),
                     {3: ["manager"]}, {"c1": 3}),
        tk.score_row(_row(qid="Q#2", answer="management left", retrieved=["c1"]),
                     {3: ["manager"]}, {"c1": 3}),
        tk.score_row(_row(qid="Q#3", answer="lettuce", retrieved=["c1"]),
                     {3: ["manager"]}, {"c1": 3}),
    ]
    terms = tk.build_terms(rows, labels={3: "a topic"})
    assert len(terms) == 1
    t = terms[0]
    assert t["keyword"] == "manager" and t["label"] == "a topic" and t["n_rows"] == 3
    assert t["tiers"] == {tk.TIER_EXACT: 1, tk.TIER_MORPH: 1,
                          tk.TIER_SEMANTIC: 0, tk.TIER_ABSENT: 1}
    assert dict(t["top_matches"])["manager"] == 1


def test_summary_shape_matches_the_keyword_agreement_judge():
    rows = [tk.score_row(_row(answer="the manager left"), {3: ["manager"]},
                         {"c1": 3})]
    summary = tk.summarize(rows, meta={"run_id": "r1"}, labels={3: "a topic"})
    for key in ("overall", "by_topic", "by_category", "by_org_source", "note"):
        assert key in summary
    assert summary["by_org_source"]["OpenAI|published"]["n_scored"] == 1
    assert summary["by_category"]["authority"]["mean_retention"] == 1.0
    # JSON-serializable, with no NaN smuggled in by a division.
    assert "NaN" not in json.dumps(summary)


# ---------------------------------------------------------------------------
# The neighborhood explorer
# ---------------------------------------------------------------------------
def test_neighbors_ranks_and_labels_tiers(tmp_path):
    vectors, cal = _space(tmp_path)
    got = tk.neighbors("manager", vectors=vectors, calibration=cal, top_n=5)
    words = [r["word"] for r in got]
    assert words[0] == "manager"                     # itself, cosine 1.0
    assert got[0]["tier"] == tk.TIER_EXACT
    # Descending by cosine, and the requirement's ordering holds here too.
    assert [r["cosine"] for r in got] == sorted(
        (r["cosine"] for r in got), reverse=True)
    assert words.index("hierarchy") < words.index("sales") if "sales" in words else True
    by_word = {r["word"]: r for r in got}
    assert by_word["management"]["tier"] == tk.TIER_MORPH   # an inflection
    assert by_word["hierarchy"]["tier"] == tk.TIER_SEMANTIC  # a related concept
    assert by_word["hierarchy"]["df"] == 10


def test_neighbors_of_an_unknown_word_is_empty_and_free(tmp_path, monkeypatch):
    vectors, cal = _space(tmp_path)

    def boom(*a, **k):
        raise AssertionError("the explorer embedded on demand")

    monkeypatch.setattr(tk, "_embed_batched", boom)
    assert tk.neighbors("zzzznotaword", vectors=vectors, calibration=cal) == []


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_run_end_to_end_over_a_temp_run(tmp_path, monkeypatch):
    """The only executable proof of the driver: files written, arithmetic right."""
    info = {
        "fitted_at": "2026-01-01T00:00:00", "seed": 42, "n_chunks": 3,
        "n_outliers": 1, "n_topics": 2,
        "topics": [
            {"topic": -1, "size": 1, "is_outlier": True, "keywords": ["junk"],
             "label": "(unclustered)"},
            {"topic": 3, "size": 1, "is_outlier": False,
             "keywords": ["manager", "chips"], "label": "manager, chips"},
            {"topic": 7, "size": 1, "is_outlier": False,
             "keywords": ["litigation"], "label": "litigation"},
        ],
    }
    monkeypatch.setattr(tk.topics_mod, "load_topic_info", lambda: info)
    monkeypatch.setattr(tk.topics_mod, "load_chunk_topics",
                        lambda: {"c1": 3, "c2": -1})

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    rows = [
        _row(qid="Q#1", answer="the manager reviewed management of chips",
             retrieved=["c1", "c2"]),
        # Abstained rows have no answer to score and must be dropped.
        {**_row(qid="Q#2", retrieved=["c1"]), "abstain": True},
    ]
    (run_dir / "per_question.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(tk.runs, "run_dir", lambda rid: tmp_path / "runs" / rid)
    monkeypatch.setattr(tk.runs, "run_paths", lambda rid: {
        "per_question": tmp_path / "runs" / rid / "per_question.jsonl"})
    monkeypatch.setattr(tk, "_fetch_chunk_texts",
                        lambda rs: {"c1": "chips and the manager"})

    def boom(*a, **k):
        raise AssertionError("the lexical-only run embedded something")

    monkeypatch.setattr(tk, "_embed_batched", boom)

    summary = tk.run_topic_keywords(run_id="r1", embeddings=False)

    out_dir = run_dir / tk.OUT_DIR_NAME
    assert (out_dir / "rows.jsonl").exists()
    assert (out_dir / "terms.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert summary["calibrated"] is False
    assert summary["cost"]["embed_calls"] == 0

    o = summary["overall"]
    assert o["n"] == 1 and o["n_scored"] == 1        # the abstained row is gone
    # Topic 3's two keywords: "manager" verbatim (1.0), "chips" verbatim (1.0).
    # The outlier chunk c2 contributes no topic, so topic 7 is never scored.
    assert o["mean_retention"] == 1.0
    assert o["mean_verbatim_share"] == 1.0
    assert o["mean_semantic_lift"] == 0.0
    assert o["mean_verbatim_ceiling"] == 1.0
    assert [t["topic"] for t in summary["by_topic"]] == [3]

    terms = tk.load_terms("r1")
    assert {t["keyword"] for t in terms} == {"manager", "chips"}
    assert tk.load_summary("r1")["run_id"] == "r1"
    assert len(tk.load_rows("r1")) == 1


def test_end_to_end_semantic_rung_lifts_retention(tmp_path, monkeypatch):
    """With calibration present, a synonym is rescued that the exact judge drops."""
    info = {"fitted_at": None, "seed": 1, "topics": [
        {"topic": 3, "size": 1, "is_outlier": False,
         "keywords": ["manager"], "label": "manager"}]}
    monkeypatch.setattr(tk.topics_mod, "load_topic_info", lambda: info)
    monkeypatch.setattr(tk.topics_mod, "load_chunk_topics", lambda: {"c1": 3})

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "per_question.jsonl").write_text(
        json.dumps(_row(answer="a flat hierarchy of teams")), encoding="utf-8")
    monkeypatch.setattr(tk.runs, "run_dir", lambda rid: tmp_path / "runs" / rid)
    monkeypatch.setattr(tk.runs, "run_paths", lambda rid: {
        "per_question": tmp_path / "runs" / rid / "per_question.jsonl"})
    monkeypatch.setattr(tk, "_fetch_chunk_texts", lambda rs: {})

    vectors, cal = _space(tmp_path)
    monkeypatch.setattr(tk.Calibration, "load", classmethod(lambda cls, path=None: cal))
    monkeypatch.setattr(tk, "WordVectors", lambda *a, **k: vectors)

    summary = tk.run_topic_keywords(run_id="r1")
    o = summary["overall"]
    assert summary["calibrated"] is True
    assert o["mean_verbatim_share"] == 0.0                     # nothing verbatim
    assert o["mean_semantic_share"] == 1.0                     # all rescued
    assert tk.KEYWORD_SEMANTIC_MIN_PERCENTILE <= o["mean_retention"] < 1.0
    assert o["mean_semantic_lift"] == pytest.approx(o["mean_retention"])


def test_module_imports_without_bertopic_or_chroma():
    """The read/score path must not need the heavy extras."""
    import sys

    assert "bertopic" not in sys.modules
    assert callable(tk.score_keyword) and callable(tk.run_topic_keywords)
