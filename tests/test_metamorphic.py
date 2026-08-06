import json

import il_rag.metamorphic as mm
from il_rag.questionnaire import CATEGORIES, LOGICS
from il_rag.retriever import Chunk


# ---------------------------------------------------------------------------
# Paraphrase fidelity — gate 1 (facts kept) and gate 2 (actually reworded)
# ---------------------------------------------------------------------------
def test_gates_accept_a_faithful_reword():
    v = mm.check_paraphrase_text(
        "OpenAI raised 6.6bn in 2024 under its Charter.",
        "Under its Charter, OpenAI secured 6.6bn of funding during 2024.")
    assert v["ok"] and v["reason"] is None


def test_gate1_rejects_a_dropped_number():
    v = mm.check_paraphrase_text("OpenAI raised 6.6bn in 2024.",
                                 "OpenAI secured funding in 2024.")
    assert not v["ok"]
    assert v["reason"] == "dropped_numbers"
    assert v["missing_numbers"] == ["6.6"]


def test_gate1_rejects_a_dropped_entity():
    v = mm.check_paraphrase_text("OpenAI raised 6.6bn in 2024.",
                                 "The lab raised 6.6bn in 2024.")
    assert not v["ok"]
    assert v["reason"] == "dropped_entities"
    assert v["missing_entities"] == ["OpenAI"]


def test_gate2_rejects_a_verbatim_copy():
    text = "The Anthropic board retains authority over deployment decisions."
    assert mm.check_paraphrase_text(text, text)["reason"] == "not_reworded"


def test_possessives_do_not_count_as_dropped_entities():
    # "OpenAI's" and "OpenAI" must normalize to the same token, or every
    # faithful rewrite that drops a possessive would be rejected.
    v = mm.check_paraphrase_text("OpenAI's charter governs the work.",
                                 "The charter of OpenAI governs how work is done.")
    assert v["missing_entities"] == []


def test_sentence_openers_do_not_count_as_entities():
    # "Safety" only starts a sentence; "safety" appears lowercase too.
    assert mm.entities_in("Safety is central. We treat safety as central.") == set()


# ---------------------------------------------------------------------------
# Paraphrase fidelity — gate 3 (still means the same)
# ---------------------------------------------------------------------------
def test_gate3_accepts_paraphrases_nearest_their_own_source():
    v = mm.check_paraphrase_semantics([[1, 0], [0, 1]],
                                      [[0.99, 0.14], [0.1, 0.99]])
    assert v["ok"] and v["reason"] is None


def test_gate3_rejects_a_paraphrase_nearer_a_sibling_excerpt():
    v = mm.check_paraphrase_semantics([[1, 0], [0, 1]],
                                      [[0.1, 0.99], [0.99, 0.1]])
    assert v["reason"] == "drifted_to_other_excerpt"


def test_gate3_rejects_low_similarity_even_when_nearest():
    v = mm.check_paraphrase_semantics([[1, 0]], [[0.6, 0.8]])  # cos = 0.6
    assert v["reason"] == "low_similarity"


# ---------------------------------------------------------------------------
# Ablation target selection
# ---------------------------------------------------------------------------
def _chunks(n):
    return [Chunk(id=f"c{i}", text=f"text {i}", org="OpenAI",
                  source_type="published", filename="f", score=0.0)
            for i in range(n)]


def test_ablation_prefers_the_verified_quote_chunk():
    row = {"retrieved_ids": ["c0", "c1", "c2"],
           "quotes": [{"excerpt": 3, "quote": "x", "verified": True}]}
    assert mm.ablation_target(row, _chunks(3)) == (2, "verified_quote")


def test_ablation_ignores_unverified_quotes():
    row = {"retrieved_ids": ["c0", "c1"],
           "quotes": [{"excerpt": 2, "quote": "x", "verified": False}]}
    assert mm.ablation_target(row, _chunks(2)) == (0, "top_retrieved")


def test_ablation_ignores_quotes_when_chunks_were_dropped():
    # Excerpt numbers index the ORIGINAL retrieved list; if a chunk went
    # missing from the index the indices no longer line up, so they are unsafe.
    row = {"retrieved_ids": ["c0", "c1", "c2"],
           "quotes": [{"excerpt": 3, "quote": "x", "verified": True}]}
    assert mm.ablation_target(row, _chunks(2)) == (0, "top_retrieved")


# ---------------------------------------------------------------------------
# Distractor donor selection and its relevance gate
# ---------------------------------------------------------------------------
def test_donor_category_is_deterministic_and_never_itself():
    for c in CATEGORIES:
        d = mm.donor_category(c)
        assert d in CATEGORIES and d != c
        assert mm.donor_category(c) == d


class _FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, *, org=None, source_type=None, k=5):
        return self.chunks


def _row(question="Who funds OpenAI and how does it sustain operations?"):
    return {"org": "OpenAI", "source_type": "published",
            "qid": "Basis of Norms#1", "category": "Basis of Norms",
            "variant": 1, "question": question, "retrieved_ids": ["c0"]}


def test_distractor_excludes_the_items_own_chunks():
    pool = [Chunk(id="c0", text="original evidence", org="OpenAI",
                  source_type="published", filename="f", score=0.9),
            Chunk(id="c9", text="Weather patterns over the Pacific vary.",
                  org="OpenAI", source_type="published", filename="g", score=0.5)]
    chunks, error, meta = mm.build_distractor_chunks(
        _FakeRetriever(pool), _row(), {})
    assert error is None
    assert [c.id for c in chunks] == ["c9"]
    assert meta["distractor_category"] == mm.donor_category("Basis of Norms")


def test_distractor_rejected_when_it_actually_answers_the_question():
    # This text covers the question's content words, so it is not a distractor.
    pool = [Chunk(id="c9",
                  text="OpenAI funds sustain operations through revenue.",
                  org="OpenAI", source_type="published", filename="g", score=0.5)]
    chunks, error, _ = mm.build_distractor_chunks(
        _FakeRetriever(pool), _row(), {})
    assert chunks is None
    assert error == "distractor_relevant"


def test_distractor_empty_when_everything_was_excluded():
    pool = [Chunk(id="c0", text="original", org="OpenAI",
                  source_type="published", filename="f", score=0.9)]
    chunks, error, _ = mm.build_distractor_chunks(
        _FakeRetriever(pool), _row(), {})
    assert chunks is None and error == "distractor_empty"


# ---------------------------------------------------------------------------
# Paraphrase parsing
# ---------------------------------------------------------------------------
def test_paraphrase_texts_parses_and_strips(monkeypatch):
    reply = json.dumps({"paraphrases": [" one ", "two"]})
    monkeypatch.setattr(mm, "chat", lambda *a, **k: reply)
    assert mm.paraphrase_texts(["a", "b"]) == ["one", "two"]


def test_paraphrase_texts_rejects_wrong_count(monkeypatch):
    calls = []

    def fake_chat(*a, **k):
        calls.append(k)
        return json.dumps({"paraphrases": ["only one"]})

    monkeypatch.setattr(mm, "chat", fake_chat)
    assert mm.paraphrase_texts(["a", "b"]) is None
    assert len(calls) == 2  # retried once with a larger budget


def test_build_paraphrase_discards_an_unfaithful_rewrite(monkeypatch):
    source = "OpenAI raised 6.6bn in 2024."
    monkeypatch.setattr(
        mm, "paraphrase_texts",
        lambda texts: ["The lab secured money that year."])  # loses name + numbers
    monkeypatch.setattr(mm, "embed", _must_not_be_called)
    out, fidelity = mm.build_paraphrase([source])
    assert out is None
    assert fidelity["reason"] == "dropped_numbers"
    assert fidelity["attempts"] == 2  # retried once, then gave up


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("embed must not be called once a text gate has failed")


def test_build_paraphrase_accepts_and_reports_quality(monkeypatch):
    source = "OpenAI raised 6.6bn in 2024."
    monkeypatch.setattr(
        mm, "paraphrase_texts",
        lambda texts: ["During 2024, 6.6bn of capital went to OpenAI."])
    monkeypatch.setattr(mm, "embed", lambda texts: [[1.0, 0.0], [0.99, 0.1]])
    out, fidelity = mm.build_paraphrase([source])
    assert out is not None and fidelity["ok"]
    assert fidelity["divergence"] is not None
    assert fidelity["min_cosine"] > 0.9


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def test_predicted_label_abstain_wins():
    assert mm.predicted_label(True, {"Market": 1.0}) == "abstain"


def test_predicted_label_argmax_and_deterministic_ties():
    w = {logic: 0.0 for logic in LOGICS}
    w["Market"] = 0.5
    w["State"] = 0.5
    # tie broken by LOGICS order: State precedes Market
    assert mm.predicted_label(False, w) == "State"
    w["Market"] = 0.6
    assert mm.predicted_label(False, w) == "Market"


# ---------------------------------------------------------------------------
# Stability math
# ---------------------------------------------------------------------------
def _src(qid, label="Market", bucket=None):
    w = {logic: 0.0 for logic in LOGICS}
    abstain = label == "abstain"
    if not abstain:
        w[label] = 1.0
    row = {"org": "OpenAI", "source_type": "published", "qid": qid,
           "category": "Basis of Norms", "variant": 1, "abstain": abstain,
           "weights": w}
    if bucket:
        row["grounding_bucket"] = bucket
    return row


def _var(qid, kind, idx, label=None, error=None, original="Market", **extra):
    v = {"org": "OpenAI", "source_type": "published", "qid": qid,
         "category": "Basis of Norms", "variant": 1,
         "variant_kind": kind, "variant_idx": idx, "original_label": original}
    v.update(extra)
    if error:
        v["error"] = error
        return v
    v["label"] = label
    v["label_matches_original"] = label == original
    return v


def test_paraphrase_flip_makes_an_item_unstable():
    src = [_src("q1")]
    variants = [
        _var("q1", "control", 1, label="Market"),
        _var("q1", "paraphrase", 1, label="Market"),
        _var("q1", "paraphrase", 2, label="Market"),
        _var("q1", "paraphrase", 3, label="State"),      # one flip
    ]
    per_item, summary = mm.compute_stability(src, variants)
    item = per_item[0]
    assert item["label_stability"] == round(2 / 3, 4)
    assert item["unstable"] is True                      # threshold 1.0
    assert item["control_flipped"] is False
    assert summary["control_flip_rate"] == 0.0
    assert summary["mean_label_stability"] == round(2 / 3, 4)
    assert summary["n_unstable"] == 1


def test_a_flipping_control_suppresses_the_unstable_verdict():
    # The pipeline moved on its own, so the paraphrase flip proves nothing.
    src = [_src("q1")]
    variants = [
        _var("q1", "control", 1, label="State"),         # noise
        _var("q1", "paraphrase", 1, label="State"),      # flip
    ]
    per_item, summary = mm.compute_stability(src, variants)
    assert per_item[0]["control_flipped"] is True
    assert per_item[0]["label_stability"] == 0.0
    assert per_item[0]["unstable"] is False
    assert summary["n_unstable"] == 0
    assert summary["control_flip_rate"] == 1.0


def test_rejected_paraphrases_leave_the_denominator_and_are_counted_by_reason():
    src = [_src("q1")]
    variants = [
        _var("q1", "paraphrase", 1, label="Market"),
        _var("q1", "paraphrase", 2, error="paraphrase_infidelity",
             fidelity={"ok": False, "reason": "dropped_entities"}),
        _var("q1", "paraphrase", 3, error="paraphrase_failed"),
    ]
    per_item, summary = mm.compute_stability(src, variants)
    item = per_item[0]
    assert item["n_paraphrases"] == 3
    assert item["n_paraphrases_ok"] == 1
    assert item["n_paraphrases_rejected"] == 2
    assert item["label_stability"] == 1.0 and item["unstable"] is False
    assert summary["rejected_by_reason"] == {"dropped_entities": 1,
                                             "paraphrase_failed": 1}


def test_ablation_survival_is_the_suspicious_outcome():
    src = [_src("q1")]
    kept = mm.compute_stability(
        src, [_var("q1", "ablation", 0, label="Market")])
    gone = mm.compute_stability(
        src, [_var("q1", "ablation", 0, label="abstain")])
    assert kept[0][0]["label_survived_ablation"] is True
    assert kept[1]["ablation_survival_rate"] == 1.0
    assert gone[0][0]["label_survived_ablation"] is False
    assert gone[1]["ablation_survival_rate"] == 0.0


def test_distractor_reproducing_the_original_label_is_prior_keyed():
    src = [_src("q1")]
    per_item, summary = mm.compute_stability(
        src, [_var("q1", "distractor", 0, label="Market",
                   distractor_category="Sources of Identity")])
    item = per_item[0]
    assert item["distractor_committed"] is True
    assert item["prior_keyed"] is True
    assert item["distractor_category"] == "Sources of Identity"
    assert summary["prior_leak_rate"] == 1.0


def test_distractor_abstention_is_the_correct_behaviour():
    src = [_src("q1")]
    per_item, summary = mm.compute_stability(
        src, [_var("q1", "distractor", 0, label="abstain")])
    assert per_item[0]["distractor_committed"] is False
    assert per_item[0]["prior_keyed"] is False
    assert summary["prior_leak_rate"] == 0.0
    assert summary["distractor_commit_rate"] == 0.0


def test_distractor_committing_to_a_different_label_is_not_prior_keyed():
    # It answered when it should have abstained (a lesser problem), but the
    # label did not come back, so this is not evidence of a prior on the lab.
    src = [_src("q1")]
    per_item, summary = mm.compute_stability(
        src, [_var("q1", "distractor", 0, label="State")])
    assert per_item[0]["distractor_committed"] is True
    assert per_item[0]["prior_keyed"] is False
    assert summary["distractor_commit_rate"] == 1.0
    assert summary["prior_leak_rate"] == 0.0


def test_directional_probes_never_fire_on_an_abstaining_original():
    # label0 is "abstain", so "the label survived" is meaningless.
    src = [_src("q1", label="abstain")]
    per_item, _ = mm.compute_stability(src, [
        _var("q1", "ablation", 0, label="abstain", original="abstain"),
        _var("q1", "distractor", 0, label="abstain", original="abstain"),
    ])
    assert per_item[0]["label_survived_ablation"] is False
    assert per_item[0]["prior_keyed"] is False


def test_probes_that_did_not_run_report_none_not_zero():
    src = [_src("q1")]
    per_item, summary = mm.compute_stability(
        src, [_var("q1", "paraphrase", 1, label="Market")])
    item = per_item[0]
    assert item["control_flipped"] is None
    assert item["label_survived_ablation"] is None
    assert item["prior_keyed"] is None
    assert summary["control_flip_rate"] is None
    assert summary["ablation_survival_rate"] is None
    assert summary["prior_leak_rate"] is None
    assert summary["n_distractor_evaluated"] == 0


def test_compute_stability_abstain_is_a_label():
    src = [_src("q1", label="abstain")]
    variants = [_var("q1", "paraphrase", 1, label="abstain", original="abstain")]
    per_item, _ = mm.compute_stability(src, variants)
    assert per_item[0]["original_label"] == "abstain"
    assert per_item[0]["label_stability"] == 1.0


def test_compute_stability_carries_grounding_bucket():
    src = [_src("q1", bucket="committed")]
    variants = [_var("q1", "paraphrase", 1, label="Market")]
    per_item, summary = mm.compute_stability(src, variants)
    assert per_item[0]["grounding_bucket"] == "committed"
    assert summary["by_grounding_bucket"]["committed"]["n"] == 1


def test_paraphrase_quality_is_averaged_into_the_summary():
    src = [_src("q1")]
    variants = [
        _var("q1", "paraphrase", 1, label="Market",
             fidelity={"ok": True, "divergence": 0.4, "min_cosine": 0.95}),
        _var("q1", "paraphrase", 2, label="Market",
             fidelity={"ok": True, "divergence": 0.2, "min_cosine": 0.91}),
    ]
    _, summary = mm.compute_stability(src, variants)
    assert summary["mean_paraphrase_divergence"] == 0.3
    assert summary["mean_paraphrase_cosine"] == 0.93


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------
def test_build_kinds_respects_the_requested_probes():
    assert mm._build_kinds(("control", "paraphrase"), paraphrases=2, controls=1) == [
        ("control", 1), ("paraphrase", 1), ("paraphrase", 2)]
    assert mm._build_kinds(("distractor",), paraphrases=3, controls=1) == [
        ("distractor", 0)]
    assert mm._build_kinds(mm.METAMORPHIC_PROBES, paraphrases=1, controls=1) == [
        ("control", 1), ("paraphrase", 1), ("ablation", 0), ("distractor", 0)]
