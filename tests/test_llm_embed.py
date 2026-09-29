"""embed() backend dispatch — offline; no model is loaded and no API is called."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import il_rag.llm as llm


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.ones((len(texts), 3), dtype=np.float32)


def test_local_backend_normalizes_and_returns_plain_lists(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(llm, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(llm, "_local_embedder", lambda: fake)
    out = llm.embed(["a", "b"])
    assert out == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert isinstance(out[0], list)          # callers (Chroma, json) need lists
    texts, kwargs = fake.calls[0]
    assert texts == ["a", "b"]
    # The stored index vectors are unit-length; local ones must be too.
    assert kwargs["normalize_embeddings"] is True


def test_local_backend_never_touches_the_together_client(monkeypatch):
    monkeypatch.setattr(llm, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(llm, "_local_embedder", lambda: _FakeModel())
    monkeypatch.setattr(llm, "client", lambda: pytest.fail("Together called"))
    llm.embed(["x"])


def test_together_backend_uses_the_api(monkeypatch):
    resp = SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.5])])
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda input, model: resp))
    monkeypatch.setattr(llm, "EMBEDDING_BACKEND", "together")
    monkeypatch.setattr(llm, "client", lambda: fake_client)
    monkeypatch.setattr(llm, "_local_embedder",
                        lambda: pytest.fail("local model loaded"))
    assert llm.embed(["x"]) == [[0.5, 0.5]]


def test_missing_sentence_transformers_gives_an_actionable_error(monkeypatch):
    monkeypatch.setattr(llm, "_local_model", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="requirements-topics.txt"):
        llm._local_embedder()
