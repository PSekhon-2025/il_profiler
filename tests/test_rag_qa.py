import json

import il_rag.rag_qa as rag_qa
from il_rag.retriever import Chunk


def _chunk(text="OpenAI's charter commits to broadly distributed benefits."):
    return Chunk(id="c1", text=text, org="OpenAI", source_type="published",
                 filename="f.txt", score=0.9)


def _patch_chat(monkeypatch, replies):
    """chat() stub returning queued replies; records call count and kwargs."""
    calls = []

    def fake_chat(messages, **kw):
        calls.append(kw)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(rag_qa, "chat", fake_chat)
    return calls


def test_default_path_unchanged(monkeypatch):
    calls = _patch_chat(monkeypatch, ["  free-form answer  "])
    out = rag_qa.answer_question(None, "q?", org="OpenAI", source_type="published",
                                 chunks=[_chunk()])
    assert out.answer == "free-form answer"
    assert out.quotes is None and out.quotes_verified is None
    assert len(calls) == 1 and calls[0]["max_tokens"] == 2048


def test_chunk_injection_bypasses_retriever(monkeypatch):
    _patch_chat(monkeypatch, ["a"])
    # retriever=None would explode if retrieval were attempted
    out = rag_qa.answer_question(None, "q?", org="OpenAI", source_type="published",
                                 chunks=[_chunk()])
    assert out.chunks[0].id == "c1"


def test_quotes_verified_verbatim(monkeypatch):
    chunk = _chunk("The  charter\ncommits to Broadly distributed benefits.")
    reply = json.dumps({
        "answer": "It commits to distributing benefits.",
        "quotes": [{"excerpt": 1, "quote": "charter commits to broadly distributed benefits"}],
    })
    _patch_chat(monkeypatch, [reply])
    out = rag_qa.answer_question(None, "q?", org="OpenAI", source_type="published",
                                 chunks=[chunk], require_quotes=True)
    assert out.answer == "It commits to distributing benefits."
    assert out.quotes_verified is True
    assert out.quotes[0]["verified"] is True


def test_fabricated_quote_fails_verification(monkeypatch):
    reply = json.dumps({
        "answer": "A claim.",
        "quotes": [{"excerpt": 1, "quote": "this span appears nowhere in the sources"}],
    })
    _patch_chat(monkeypatch, [reply])
    out = rag_qa.answer_question(None, "q?", org="OpenAI", source_type="published",
                                 chunks=[_chunk()], require_quotes=True)
    assert out.quotes_verified is False
    assert out.quotes[0]["verified"] is False


def test_empty_quote_list_is_unverified(monkeypatch):
    reply = json.dumps({"answer": "Insufficient information.", "quotes": []})
    _patch_chat(monkeypatch, [reply])
    out = rag_qa.answer_question(None, "q?", org="OpenAI", source_type="published",
                                 chunks=[_chunk()], require_quotes=True)
    assert out.quotes == [] and out.quotes_verified is False


def test_quote_parse_failure_retries_then_degrades(monkeypatch):
    calls = _patch_chat(monkeypatch, ["not json", "still not json"])
    out = rag_qa.answer_question(None, "q?", org="OpenAI", source_type="published",
                                 chunks=[_chunk()], require_quotes=True)
    assert len(calls) == 2
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
    assert out.answer == "still not json"      # raw text kept, row not lost
    assert out.quotes == [] and out.quotes_verified is False
