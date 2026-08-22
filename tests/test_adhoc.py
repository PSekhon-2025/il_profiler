"""Offline tests for ad-hoc document analysis: extraction, chunking and the
brute-force retriever. All API calls are monkeypatched."""
import pytest

from il_rag import adhoc
from il_rag.retriever import Chunk


def test_extract_plain_text():
    doc = adhoc.extract_text("notes.txt", b"Hello world, this is a document.")
    assert doc.error is None
    assert "Hello world" in doc.text
    assert doc.n_chars == len(doc.text)


def test_extract_markdown_and_blank_line_collapse():
    doc = adhoc.extract_text("a.md", b"# Title\n\n\n\n\nBody text here.")
    assert doc.error is None
    assert "\n\n\n" not in doc.text          # runs of blank lines collapsed


def test_unsupported_type_is_reported_not_raised():
    doc = adhoc.extract_text("sheet.xlsx", b"\x00\x01binary")
    assert doc.error is not None and "unsupported" in doc.error
    assert doc.text == ""


def test_empty_document_flagged_as_no_text():
    doc = adhoc.extract_text("blank.txt", b"   \n\n  ")
    assert doc.error is not None and "no extractable text" in doc.error


def test_unreadable_file_degrades_only_itself():
    """A corrupt PDF must not raise — the batch keeps going."""
    doc = adhoc.extract_text("broken.pdf", b"not really a pdf")
    assert doc.error is not None
    assert doc.text == ""


def test_build_chunks_skips_errored_docs_and_tags_provenance():
    docs = [
        adhoc.AdhocDoc("good.txt", "word " * 800, 4000),
        adhoc.AdhocDoc("bad.txt", "", 0, error="unsupported"),
    ]
    chunks = adhoc.build_chunks(docs, subject="Acme")
    assert chunks, "expected chunks from the readable document"
    assert all(c.filename == "good.txt" for c in chunks)
    assert all(c.org == "Acme" for c in chunks)
    assert all(c.source_type == adhoc.ADHOC_SOURCE_TYPE for c in chunks)
    assert all(c.id.startswith("adhoc::") for c in chunks)


def test_retrieve_ranks_by_cosine_and_reports_score(monkeypatch):
    chunks = [
        Chunk(id="c1", text="alpha", org="A", source_type="uploaded",
              filename="f", score=0.0),
        Chunk(id="c2", text="beta", org="A", source_type="uploaded",
              filename="f", score=0.0),
        Chunk(id="c3", text="gamma", org="A", source_type="uploaded",
              filename="f", score=0.0),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    # The query embeds onto the first axis, so c1 is nearest, then c3.
    monkeypatch.setattr(adhoc, "embed", lambda texts: [[1.0, 0.0]])

    got = adhoc.retrieve("q", chunks, vectors, k=2)
    assert [c.id for c in got] == ["c1", "c3"]
    assert got[0].score == pytest.approx(1.0)
    assert 0.0 < got[1].score < 1.0            # similarity surfaced to the UI
    # originals untouched — retrieve returns copies
    assert chunks[0].score == 0.0


def test_retrieve_on_empty_corpus_returns_nothing():
    assert adhoc.retrieve("q", [], [], k=5) == []
