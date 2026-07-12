import json

import il_rag.metamorphic as mm
from il_rag.questionnaire import LOGICS


# ---------------------------------------------------------------------------
# Lab-name swap
# ---------------------------------------------------------------------------
def test_swap_handles_aliases_longest_first():
    text = "Google DeepMind and DeepMind's Gemini team"
    assert mm.swap_lab_text(text, "DeepMind", "Anthropic") == \
        "Anthropic and Anthropic's Gemini team"


def test_swap_is_case_insensitive():
    assert mm.swap_lab_text("OPENAI said; OpenAI did", "OpenAI", "DeepMind") == \
        "DeepMind said; DeepMind did"


def test_swap_respects_word_boundaries():
    assert mm.swap_lab_text("an OpenAIish approach", "OpenAI", "DeepMind") == \
        "an OpenAIish approach"


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


def _var(qid, kind, idx, label=None, error=None):
    v = {"org": "OpenAI", "source_type": "published", "qid": qid,
         "category": "Basis of Norms", "variant": 1,
         "variant_kind": kind, "variant_idx": idx, "original_label": "Market"}
    if error:
        v["error"] = error
        return v
    v["label"] = label
    v["label_matches_original"] = label == "Market"
    if kind == "lab_swap":
        v["swap_to"] = "DeepMind"
    return v


def test_compute_stability_paraphrases_and_swap():
    src = [_src("q1")]
    variants = [
        _var("q1", "paraphrase", 1, label="Market"),
        _var("q1", "paraphrase", 2, label="Market"),
        _var("q1", "paraphrase", 3, label="State"),      # one flip
        _var("q1", "lab_swap", 0, label="Corporation"),  # swap flip -> suspicious
    ]
    per_item, summary = mm.compute_stability(src, variants)
    item = per_item[0]
    assert item["label_stability"] == round(2 / 3, 4)
    assert item["unstable"] is True                      # threshold 1.0
    assert item["swap_label_changed"] is True
    assert summary["mean_label_stability"] == round(2 / 3, 4)
    assert summary["n_unstable"] == 1
    assert summary["swap_flip_rate"] == 1.0


def test_compute_stability_excludes_failed_variants():
    src = [_src("q1")]
    variants = [
        _var("q1", "paraphrase", 1, label="Market"),
        _var("q1", "paraphrase", 2, error="paraphrase_failed"),
        _var("q1", "lab_swap", 0, error="chunks_not_found"),
    ]
    per_item, summary = mm.compute_stability(src, variants)
    item = per_item[0]
    assert item["n_paraphrases"] == 2 and item["n_paraphrases_ok"] == 1
    assert item["label_stability"] == 1.0 and item["unstable"] is False
    assert item["swap_label_changed"] is None
    assert summary["n_swap_evaluated"] == 0 and summary["swap_flip_rate"] is None


def test_compute_stability_abstain_is_a_label():
    src = [_src("q1", label="abstain")]
    variants = [
        {**_var("q1", "paraphrase", 1), "label": "abstain",
         "label_matches_original": True},
    ]
    per_item, _ = mm.compute_stability(src, variants)
    assert per_item[0]["original_label"] == "abstain"
    assert per_item[0]["label_stability"] == 1.0


def test_compute_stability_carries_grounding_bucket():
    src = [_src("q1", bucket="committed")]
    variants = [_var("q1", "paraphrase", 1, label="Market")]
    per_item, summary = mm.compute_stability(src, variants)
    assert per_item[0]["grounding_bucket"] == "committed"
    assert summary["by_grounding_bucket"]["committed"]["n"] == 1
