import json

import il_rag.quote_provenance as qp
from il_rag.retriever import Chunk

CHUNK_TEXT = (
    "The charter commits to broadly distributed benefits. Our primary "
    "fiduciary duty is to humanity, and we will avoid uses of AI that harm "
    "the public. The safety team reports quarterly to the board."
)


def _chunk(text=CHUNK_TEXT, cid="c1"):
    return Chunk(id=cid, text=text, org="OpenAI", source_type="published",
                 filename="f.txt", score=0.9)


def _patch_chat(monkeypatch, replies):
    """chat() stub returning queued replies; records call count and kwargs."""
    calls = []

    def fake_chat(messages, **kw):
        calls.append({"messages": messages, **kw})
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(qp, "chat", fake_chat)
    return calls


def _support(label, **extra):
    return json.dumps({"support": label, "evidence_sentence": "s",
                       "grounded_fragment": extra.get("fragment", ""),
                       "reason": "r"})


# ---------------------------------------------------------------------------
# Stage C — the provenance ladder
# ---------------------------------------------------------------------------
def test_exact_tier_matches_feature_2_predicate():
    out = qp.locate_span("primary fiduciary duty is to humanity", [_chunk()])
    assert out["match_tier"] == qp.TIER_EXACT
    assert out["match_rule"] == "substring"
    assert out["best_chunk_id"] == "c1"


def test_curly_quote_drift_is_near_verbatim():
    chunk = _chunk('He called it a "safety-first" culture — reviewed quarterly.')
    out = qp.locate_span('He called it a “safety‐first” culture '
                         '— reviewed quarterly.', [chunk])
    assert out["match_tier"] == qp.TIER_NEAR_VERBATIM
    assert out["match_rule"] == "punctuation_insensitive"


def test_ellipsis_elision_is_near_verbatim():
    out = qp.locate_span(
        "The charter commits to broadly ... duty is to humanity", [_chunk()])
    assert out["match_tier"] == qp.TIER_NEAR_VERBATIM
    assert out["match_rule"] == "ellipsis_fragments_in_order"


def test_ellipsis_fragments_out_of_order_do_not_verify():
    # Same two fragments, swapped: the elision claims an order the source
    # does not have, so it must not pass as a copy.
    out = qp.locate_span(
        "duty is to humanity ... The charter commits to broadly", [_chunk()])
    assert out["match_tier"] != qp.TIER_NEAR_VERBATIM


def test_small_typo_is_near_verbatim_by_character_similarity():
    out = qp.locate_span("our primary fiducairy duty is to humanity, and we will",
                         [_chunk()])
    assert out["match_tier"] == qp.TIER_NEAR_VERBATIM
    assert out["match_rule"] == "character_similarity"
    assert out["match_score"] >= 0.9


def test_reworded_span_is_paraphrase_by_lexical_overlap():
    out = qp.locate_span("fiduciary duty to humanity and avoid harm to the public",
                         [_chunk()])
    assert out["match_tier"] == qp.TIER_PARAPHRASE
    assert out["match_rule"] == "lexical_overlap"


def test_invented_span_is_unsupported():
    out = qp.locate_span(
        "quarterly dividends are distributed to shareholders in perpetuity",
        [_chunk()])
    assert out["match_tier"] == qp.TIER_UNSUPPORTED


def test_cosine_route_reaches_paraphrase_tier(monkeypatch):
    """With lexical overlap below the bar, a near-identical vector still lands
    the span in the paraphrase tier."""
    monkeypatch.setattr(qp, "_embed_batched",
                        lambda texts: [[1.0, 0.0] for _ in texts])
    emb = qp._WindowEmbeddings()
    out = qp.locate_span("wholly unrelated phrasing entirely", [_chunk()], emb)
    assert out["match_tier"] == qp.TIER_PARAPHRASE
    assert out["match_rule"] == "embedding_cosine"


def test_ladder_is_pure_computation_without_embeddings(monkeypatch):
    """embeddings=None must never reach the embedding client."""
    def boom(_texts):
        raise AssertionError("embed must not be called")

    monkeypatch.setattr(qp, "_embed_batched", boom)
    assert qp.locate_span("nothing like the source at all here", [_chunk()],
                          None)["match_tier"] == qp.TIER_UNSUPPORTED


# ---------------------------------------------------------------------------
# Stage A/B — extraction and intent triage
# ---------------------------------------------------------------------------
def test_structured_entries_are_attributive_by_construction():
    row = {"answer": "", "quotes": [{"excerpt": 1, "quote": "a real span here",
                                     "verified": True}]}
    cands = qp.extract_candidates(row)
    assert len(cands) == 1 and cands[0]["source"] == qp.SOURCE_QUOTES_FIELD
    assert qp.classify_intent(cands[0]) == (
        qp.INTENT_ATTRIBUTIVE, "declared_in_quotes_field", "high")


def test_reporting_verb_marks_prose_span_attributive():
    cands = qp.extract_candidates(
        {"answer": 'The charter states that "our duty is to humanity first".'})
    intent, rule, conf = qp.classify_intent(cands[0])
    assert (intent, rule, conf) == (qp.INTENT_ATTRIBUTIVE, "reporting_verb", "high")


def test_scare_quote_is_not_attributive():
    cands = qp.extract_candidates(
        {"answer": 'They promote what it calls a "culture of safety review".'})
    intent, rule, _ = qp.classify_intent(cands[0])
    assert (intent, rule) == (qp.INTENT_SCARE_QUOTE, "mention_not_use")


def test_counterfactual_frame_beats_its_own_reporting_verb():
    """"might say" contains a reporting verb but attributes nothing — the
    counterfactual rule has to be tested first or it is swallowed."""
    cands = qp.extract_candidates(
        {"answer": 'A critic might say "they prioritise speed over safety".'})
    intent, rule, _ = qp.classify_intent(cands[0])
    assert (intent, rule) == (qp.INTENT_HYPOTHETICAL, "counterfactual_frame")


def test_example_framing_does_not_override_a_real_attribution():
    """"for example" only demotes a span when nothing else claims attribution."""
    cands = qp.extract_candidates(
        {"answer": 'For example, the charter states "our duty is to humanity '
                   'above all else".'})
    intent, rule, _ = qp.classify_intent(cands[0])
    assert (intent, rule) == (qp.INTENT_ATTRIBUTIVE, "reporting_verb")


def test_short_unpunctuated_span_is_term_of_art():
    cands = qp.extract_candidates(
        {"answer": 'The board reviews the "responsible scaling policy" annually.'})
    intent, rule, _ = qp.classify_intent(cands[0])
    assert (intent, rule) == (qp.INTENT_TERM_OF_ART, "short_unpunctuated_span")


def test_uncued_span_defaults_to_attributive_at_low_confidence():
    cands = qp.extract_candidates(
        {"answer": 'Their governance model. "The safety team reports quarterly '
                   'to the board and publishes findings."'})
    intent, rule, conf = qp.classify_intent(cands[0])
    assert (intent, rule, conf) == (qp.INTENT_ATTRIBUTIVE, "default_no_cue", "low")


def test_one_and_two_word_quotes_are_dropped_as_noise():
    assert qp.extract_prose_spans('They call it "alignment" internally.') == []


def test_prose_span_duplicating_a_structured_entry_is_not_counted_twice():
    row = {"answer": 'The charter states "our primary fiduciary duty is to '
                     'humanity" plainly.',
           "quotes": [{"excerpt": 1,
                       "quote": "our primary fiduciary duty is to humanity",
                       "verified": True}]}
    assert len(qp.extract_candidates(row)) == 1


# ---------------------------------------------------------------------------
# Stage D' — the verdict 2x2, as a pure function
# ---------------------------------------------------------------------------
def test_verdict_found_and_supported_is_attributed():
    assert qp.derive_verdict(qp.TIER_EXACT, None,
                             qp.INTENT_ATTRIBUTIVE) == qp.VERDICT_ATTRIBUTED


def test_verdict_found_but_contradicted_is_misattributed():
    assert qp.derive_verdict(qp.TIER_EXACT, qp.SUPPORT_CONTRADICTED,
                             qp.INTENT_ATTRIBUTIVE) == qp.VERDICT_MISATTRIBUTED


def test_verdict_absent_but_supported_is_misquote_but_true():
    """The cell this module exists for: the quotation was invented, the content
    it asserts is nonetheless carried by the evidence."""
    assert qp.derive_verdict(qp.TIER_UNSUPPORTED, qp.SUPPORT_SUPPORTED,
                             qp.INTENT_ATTRIBUTIVE) == qp.VERDICT_MISQUOTE_BUT_TRUE


def test_verdict_paraphrase_supported_is_graded_milder_than_misquote():
    assert qp.derive_verdict(qp.TIER_PARAPHRASE, qp.SUPPORT_PARTIAL,
                             qp.INTENT_ATTRIBUTIVE) == qp.VERDICT_PARAPHRASE_GROUNDED


def test_verdict_absent_and_unsupported_is_fabricated():
    for support in (qp.SUPPORT_NOT_ADDRESSED, qp.SUPPORT_CONTRADICTED, None):
        assert qp.derive_verdict(qp.TIER_UNSUPPORTED, support,
                                 qp.INTENT_ATTRIBUTIVE) == qp.VERDICT_FABRICATED


def test_non_attributive_spans_are_never_graded():
    for intent in (qp.INTENT_SCARE_QUOTE, qp.INTENT_TERM_OF_ART,
                   qp.INTENT_HYPOTHETICAL):
        assert qp.derive_verdict(qp.TIER_UNSUPPORTED, None,
                                 intent) == qp.VERDICT_NON_ATTRIBUTIVE


# ---------------------------------------------------------------------------
# Stage D — adjudication, evidence windows, and cost
# ---------------------------------------------------------------------------
def test_verbatim_clean_row_costs_zero_llm_calls(monkeypatch):
    calls = _patch_chat(monkeypatch, ["unused"])
    row = {"answer": "", "quotes": [
        {"excerpt": 1, "quote": "primary fiduciary duty is to humanity",
         "verified": True}]}
    spans = qp.analyze_row(row, [_chunk()])
    assert calls == []
    assert spans[0]["match_tier"] == qp.TIER_EXACT
    assert spans[0]["verdict"] == qp.VERDICT_ATTRIBUTED


def test_unsupported_span_is_adjudicated_against_the_full_retrieved_set(monkeypatch):
    calls = _patch_chat(monkeypatch, [_support("supported")])
    chunks = [_chunk(cid="c1"), _chunk("An unrelated passage about compute.", "c2")]
    row = {"answer": "", "quotes": [
        {"excerpt": 1, "quote": "dividends are paid to shareholders each quarter",
         "verified": False}]}
    spans = qp.analyze_row(row, chunks)
    prompt = calls[0]["messages"][1]["content"]
    assert "An unrelated passage about compute." in prompt
    assert CHUNK_TEXT in prompt
    assert spans[0]["verdict"] == qp.VERDICT_MISQUOTE_BUT_TRUE


def test_paraphrase_span_sees_only_the_aligned_passage(monkeypatch):
    calls = _patch_chat(monkeypatch, [_support("supported")])
    chunks = [_chunk(cid="c1"), _chunk("An unrelated passage about compute.", "c2")]
    row = {"answer": "", "quotes": [
        {"excerpt": 1,
         "quote": "fiduciary duty to humanity and avoid harm to the public",
         "verified": False}]}
    spans = qp.analyze_row(row, chunks)
    assert spans[0]["match_tier"] == qp.TIER_PARAPHRASE
    prompt = calls[0]["messages"][1]["content"]
    assert "An unrelated passage about compute." not in prompt


def test_verbatim_adjudication_is_opt_in(monkeypatch):
    calls = _patch_chat(monkeypatch, [_support("contradicted")])
    row = {"answer": "", "quotes": [
        {"excerpt": 1, "quote": "primary fiduciary duty is to humanity",
         "verified": True}]}
    spans = qp.analyze_row(row, [_chunk()], adjudicate_verbatim=True)
    assert len(calls) == 1
    assert spans[0]["verdict"] == qp.VERDICT_MISATTRIBUTED


def test_adjudicator_parse_failure_retries_then_degrades(monkeypatch):
    calls = _patch_chat(monkeypatch, ["not json", "still not json"])
    out = qp.adjudicate("some span", ["a passage"])
    assert len(calls) == 2
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
    assert out["support"] == qp.SUPPORT_NOT_ADDRESSED
    assert out["grounded_fragment"] is None


def test_unknown_support_label_degrades_to_not_addressed():
    assert qp._normalize_support(
        {"support": "definitely true"})["support"] == qp.SUPPORT_NOT_ADDRESSED


def test_partial_support_carries_the_grounded_fragment(monkeypatch):
    _patch_chat(monkeypatch, [_support("partial", fragment="reports quarterly")])
    out = qp.adjudicate("span", ["passage"])
    assert out["support"] == qp.SUPPORT_PARTIAL
    assert out["grounded_fragment"] == "reports quarterly"


# ---------------------------------------------------------------------------
# Stage E — row verdicts, and non-interference with feature 2
# ---------------------------------------------------------------------------
def test_row_grounded_requires_at_least_one_attributive_span():
    """Empty conjunctions are vacuously true — the same guard feature 2 uses."""
    assert qp.row_verdicts([])["quotes_grounded"] is False
    only_scare = [{"intent": qp.INTENT_SCARE_QUOTE,
                   "verdict": qp.VERDICT_NON_ATTRIBUTIVE}]
    assert qp.row_verdicts(only_scare)["quotes_grounded"] is False


def test_row_separates_fabrication_from_misquotation():
    spans = [
        {"intent": qp.INTENT_ATTRIBUTIVE, "verdict": qp.VERDICT_ATTRIBUTED},
        {"intent": qp.INTENT_ATTRIBUTIVE, "verdict": qp.VERDICT_MISQUOTE_BUT_TRUE},
    ]
    v = qp.row_verdicts(spans)
    assert v["quotes_misquoted"] is True
    assert v["quotes_fabricated"] is False
    assert v["quotes_grounded"] is False


def test_feature_2_verdict_is_carried_through_unchanged(monkeypatch):
    _patch_chat(monkeypatch, [_support("not_addressed")])
    row = {"answer": "", "quotes": [
        {"excerpt": 1, "quote": "invented text that appears nowhere at all",
         "verified": False}]}
    spans = qp.analyze_row(row, [_chunk()])
    assert spans[0]["verbatim_verified"] is False
    assert spans[0]["verdict"] == qp.VERDICT_FABRICATED
