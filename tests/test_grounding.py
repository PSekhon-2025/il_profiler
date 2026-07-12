from il_rag.grounding import (
    BUCKET_ABSTAINED,
    BUCKET_COMMITTED,
    BUCKET_RETRIEVAL_MISSED,
    bucket,
    grounding_scores,
    lexical_overlap,
    summarize_buckets,
)
from il_rag.retriever import Chunk


def _chunk(text, score=0.8):
    return Chunk(id="c", text=text, org="OpenAI", source_type="published",
                 filename="f.txt", score=score)


def test_lexical_overlap_full_and_none():
    q = "How is OpenAI funded and who are its investors?"
    assert lexical_overlap(q, "OpenAI is funded by investors like Microsoft") > 0.5
    assert lexical_overlap(q, "quarterly weather report for Lisbon") == 0.0


def test_lexical_overlap_ignores_stopwords_and_case():
    # "the", "of", "is" must not count as overlap
    assert lexical_overlap("the of is", "the of is completely different text") == 0.0
    assert lexical_overlap("SAFETY Board", "safety board minutes") == 1.0


def test_grounding_scores_takes_max_over_chunks():
    q = "Who controls deployment decisions at OpenAI?"
    chunks = [
        _chunk("unrelated cooking recipe", score=0.4),
        _chunk("OpenAI deployment decisions are controlled by the board", score=0.9),
    ]
    g = grounding_scores(q, chunks)
    assert g["score"] > 0.5
    assert g["cosine_top"] == 0.9


def test_grounding_scores_empty_chunks():
    assert grounding_scores("anything", []) == {"score": 0.0, "cosine_top": 0.0}


def test_bucket_precedence_retrieval_missed_beats_abstain():
    # low grounding + abstain -> retrieval failure, not model abstention
    assert bucket(abstain=True, grounding_score=0.0) == BUCKET_RETRIEVAL_MISSED
    assert bucket(abstain=True, grounding_score=0.9) == BUCKET_ABSTAINED
    assert bucket(abstain=False, grounding_score=0.9) == BUCKET_COMMITTED


def test_bucket_respects_threshold_argument():
    assert bucket(abstain=False, grounding_score=0.3, threshold=0.5) == BUCKET_RETRIEVAL_MISSED
    assert bucket(abstain=False, grounding_score=0.3, threshold=0.2) == BUCKET_COMMITTED


def test_summarize_buckets():
    rows = [
        {"grounding_bucket": BUCKET_COMMITTED, "abstain": False,
         "weights": {"Market": 0.8, "State": 0.2}},
        {"grounding_bucket": BUCKET_COMMITTED, "abstain": False,
         "weights": {"Market": 0.6, "State": 0.4}},
        {"grounding_bucket": BUCKET_RETRIEVAL_MISSED, "abstain": True, "weights": {}},
    ]
    s = summarize_buckets(rows)
    assert s[BUCKET_COMMITTED]["n"] == 2
    assert s[BUCKET_COMMITTED]["abstain_rate"] == 0.0
    assert s[BUCKET_COMMITTED]["mean_top_weight"] == 0.7
    assert s[BUCKET_RETRIEVAL_MISSED]["n"] == 1
    assert s[BUCKET_RETRIEVAL_MISSED]["abstain_rate"] == 1.0
    assert s[BUCKET_RETRIEVAL_MISSED]["mean_top_weight"] is None
    assert s[BUCKET_ABSTAINED]["n"] == 0
