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


# ---------------------------------------------------------------------------
# Saving an analysis as a tagged run snapshot
# ---------------------------------------------------------------------------
def _result(subject="Acme", n_rows=2):
    """An analyze()-shaped result: one committed row, one abstention."""
    from il_rag.questionnaire import LOGICS
    zero = {logic: 0.0 for logic in LOGICS}
    rows = [{
        "org": subject, "source_type": adhoc.ADHOC_SOURCE_TYPE,
        "qid": "Basis of Norms#1", "category": "Basis of Norms", "variant": 1,
        "question": "What does Acme do?", "answer": "It answers to regulators.",
        "retrieved_ids": ["adhoc::a.txt::chunk0"],
        "retrieved": [{"id": "adhoc::a.txt::chunk0", "filename": "a.txt",
                       "score": 0.91, "text": "Acme answers to regulators."}],
        "abstain": False, "weights": zero | {"State": 1.0},
        "reasoning": "names a regulator",
    }]
    if n_rows > 1:
        rows.append({
            "org": subject, "source_type": adhoc.ADHOC_SOURCE_TYPE,
            "qid": "Basis of Norms#2", "category": "Basis of Norms",
            "variant": 2, "question": "And?", "answer": "",
            "retrieved_ids": [], "retrieved": [],
            "abstain": True, "weights": dict(zero), "reasoning": "no evidence",
        })
    profile = {"logic_pct": zero | {"State": 100.0}, "answered": 1,
               "abstained": n_rows - 1, "by_category": {}}
    return {"subject": subject, "rows": rows, "profile": profile}


@pytest.fixture()
def run_store(tmp_path, monkeypatch):
    """Point the runs module at a temp directory for the whole round trip."""
    from il_rag import runs as runs_mod
    monkeypatch.setattr(runs_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runs_mod, "CURRENT_PTR", tmp_path / "CURRENT")
    return tmp_path


def test_save_run_writes_a_full_snapshot(run_store):
    from il_rag import runs as runs_mod

    rid = adhoc.save_run(_result(), k=5, label="first pass",
                         documents=["a.txt"])
    paths = runs_mod.run_paths(rid)
    # Every file a corpus run writes, so every reader works unchanged.
    for key in ("per_question", "profiles_json", "profiles_csv",
                "questionnaire", "meta"):
        assert paths[key].exists(), f"{key} missing from the snapshot"

    meta = runs_mod.read_meta(rid)
    assert runs_mod.is_adhoc(meta) is True
    assert meta["subject"] == "Acme"
    assert meta["documents"] == ["a.txt"]
    assert meta["label"] == "first pass"
    assert meta["k"] == 5
    assert meta["source_types"] == [adhoc.ADHOC_SOURCE_TYPE]
    assert meta["answered"] == 1 and meta["abstained"] == 1
    assert meta["status"] == "complete"


def test_saving_never_moves_the_current_pointer(run_store):
    """CURRENT is the corpus pipeline's resume cursor — not ours to touch."""
    from il_rag import runs as runs_mod

    runs_mod.CURRENT_PTR.write_text("2026-01-01_000000\n", encoding="utf-8")
    adhoc.save_run(_result(), k=5)
    assert runs_mod.CURRENT_PTR.read_text(encoding="utf-8").strip() == \
        "2026-01-01_000000"


def test_round_trip_preserves_rows_profile_and_evidence(run_store):
    original = _result()
    rid = adhoc.save_run(original, k=5, documents=["a.txt"])
    loaded = adhoc.load_run(rid)

    assert loaded is not None
    assert loaded["subject"] == original["subject"]
    assert loaded["profile"] == original["profile"]
    assert len(loaded["rows"]) == len(original["rows"])
    # The evidence TEXT must survive: ad-hoc chunk ids resolve against nothing.
    assert loaded["rows"][0]["retrieved"][0]["text"] == \
        original["rows"][0]["retrieved"][0]["text"]
    assert loaded["run_id"] == rid


def test_profiles_json_uses_the_corpus_shape(run_store):
    """org -> source_type -> profile, so load_profiles reads it unchanged."""
    import json as _json
    from il_rag import runs as runs_mod

    rid = adhoc.save_run(_result(subject="Acme"), k=5)
    blob = _json.loads(
        runs_mod.run_paths(rid)["profiles_json"].read_text(encoding="utf-8"))
    assert set(blob) == {"Acme"}
    assert set(blob["Acme"]) == {adhoc.ADHOC_SOURCE_TYPE}
    assert blob["Acme"][adhoc.ADHOC_SOURCE_TYPE]["answered"] == 1


def test_load_run_refuses_a_corpus_run(run_store):
    """The picker is over ad-hoc runs; a corpus id reaching load_run is a bug."""
    from il_rag import runs as runs_mod

    rid = adhoc.save_run(_result(), k=5)
    runs_mod.update_meta(rid, kind=runs_mod.KIND_CORPUS)
    assert adhoc.load_run(rid) is None


def test_load_run_on_a_missing_run_returns_none(run_store):
    assert adhoc.load_run("2099-01-01_000000") is None


def test_list_runs_separates_the_two_kinds(run_store):
    from il_rag import runs as runs_mod

    adhoc_id = adhoc.save_run(_result(), k=5)
    corpus_id = adhoc.save_run(_result(subject="Other"), k=5)
    runs_mod.update_meta(corpus_id, kind=runs_mod.KIND_CORPUS)

    assert [m["run_id"] for m in runs_mod.list_runs(kind=runs_mod.KIND_ADHOC)] \
        == [adhoc_id]
    assert [m["run_id"] for m in runs_mod.list_runs(kind=runs_mod.KIND_CORPUS)] \
        == [corpus_id]
    assert len(runs_mod.list_runs()) == 2          # unfiltered sees both


def test_runs_predating_the_kind_field_read_as_corpus(run_store):
    """No backfill: an untagged snapshot must not vanish from the pickers."""
    from il_rag import runs as runs_mod

    rid = adhoc.save_run(_result(), k=5)
    meta = runs_mod.read_meta(rid)
    del meta["kind"]
    runs_mod.write_meta(rid, meta)

    assert runs_mod.run_kind(runs_mod.read_meta(rid)) == runs_mod.KIND_CORPUS
    assert runs_mod.is_adhoc(runs_mod.read_meta(rid)) is False
    assert [m["run_id"] for m in runs_mod.list_runs(kind=runs_mod.KIND_CORPUS)] \
        == [rid]


def test_display_name_marks_adhoc_runs(run_store):
    from il_rag import runs as runs_mod

    rid = adhoc.save_run(_result(), k=5, label="v1", documents=["a.txt"])
    name = runs_mod.display_name(runs_mod.read_meta(rid))
    assert name.startswith("📄 Acme")
    assert "1 doc" in name and "v1" in name
