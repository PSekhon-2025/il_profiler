"""Offline tests for the excerpt -> source-PDF resolution.

Everything here runs on synthetic trees under tmp_path: no dataset, no Chroma
index, no API key. The load-bearing group is `excerpt_citation`, which must
return the UI's pre-existing plain text on every path where a link cannot be
produced — that fallback is what makes the feature invisible when the corpus
is absent (a fresh checkout, and the cloud deployment).
"""
import pytest

from il_rag import pdf_sources as ps
from il_rag.rag_qa import _norm_ws as rag_qa_norm_ws

PLAIN = "\\[excerpt 1\\]"


def _pdf(root, *parts):
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def _url_for(filename, org, chunk_id, source_type="published"):
    return f"/app/static/corpus/{org}/{filename}"


def _evidence(**overrides):
    chunk = {"text": "the board retains final authority over deployment",
             "filename": "open AI charter.pdf", "org": "OpenAI",
             "source_type": "published"}
    chunk.update(overrides)
    return {"c1": chunk}


# --- name normalization -----------------------------------------------------

def test_normalize_name_absorbs_the_one_real_corpus_mismatch():
    # The corpus text file sanitized an apostrophe to an underscore; this is
    # the only document of 98 whose name does not match the disk exactly.
    assert (ps.normalize_name("Inside Google_s AGI Strategy.pdf")
            == ps.normalize_name("Inside Google's AGI Strategy.pdf"))


def test_normalize_name_ignores_case_spaces_and_punctuation():
    assert ps.normalize_name("Open AI - Charter (v2).pdf") == "openaicharterv2pdf"


# --- index building ---------------------------------------------------------

def test_build_index_walks_nested_dirs_and_ignores_non_pdfs(tmp_path):
    _pdf(tmp_path, "OpenAI", "Essays and Statements", "open AI charter.pdf")
    (tmp_path / "OpenAI" / "8 - Corpus").mkdir(parents=True)
    (tmp_path / "OpenAI" / "8 - Corpus" / "pdf_corpus.txt").write_text("x")
    (tmp_path / "OpenAI" / "9 - Articles").mkdir(parents=True)
    (tmp_path / "OpenAI" / "9 - Articles" / "O1.RTF").write_text("x")

    index = ps.build_index(tmp_path)
    assert list(index) == [ps.normalize_name("open AI charter.pdf")]


def test_build_index_groups_colliding_basenames_deterministically(tmp_path):
    _pdf(tmp_path, "DeepMind", "Model Cards", "Gemini-3-1-Pro-Model-Card.pdf")
    _pdf(tmp_path, "model cards", "Gemini-3-1-Pro-Model-Card.pdf")

    paths = ps.build_index(tmp_path)[ps.normalize_name("Gemini-3-1-Pro-Model-Card.pdf")]
    assert len(paths) == 2
    assert list(paths) == sorted(paths)


def test_build_index_empty_for_missing_or_unset_root(tmp_path):
    assert ps.build_index(None) == {}
    assert ps.build_index(tmp_path / "nope") == {}


# --- resolution -------------------------------------------------------------

def test_resolve_prefers_the_candidate_under_the_chunks_own_lab(tmp_path):
    # Regression guard for the two real collisions in the dataset.
    _pdf(tmp_path, "Anthropic", "05_Essays", "claudes-constitution.pdf")
    _pdf(tmp_path, "claudes-constitution.pdf")
    index = ps.build_index(tmp_path)

    got = ps.resolve("claudes-constitution.pdf", "Anthropic", index)
    assert got.parent.parent.name == "Anthropic"


def test_resolve_returns_none_for_unknown_basename(tmp_path):
    assert ps.resolve("nothing.pdf", "OpenAI", ps.build_index(tmp_path)) is None


# --- materialization --------------------------------------------------------

def test_materialize_publishes_and_is_idempotent(tmp_path):
    src = _pdf(tmp_path / "dataset", "OpenAI", "open AI charter.pdf")
    static_root = tmp_path / "static"

    url = ps.materialize(src, "OpenAI", static_root)
    dest = static_root / "corpus" / "OpenAI" / "open AI charter.pdf"
    assert dest.is_file()
    stamp = dest.stat().st_mtime_ns

    assert ps.materialize(src, "OpenAI", static_root) == url
    assert dest.stat().st_mtime_ns == stamp        # second call did no work
    assert not list(dest.parent.glob("*.part"))    # nothing half-written left


def test_materialize_percent_encodes_the_url(tmp_path):
    src = _pdf(tmp_path / "dataset", "OpenAI", "our principles - Sam Altman.pdf")
    url = ps.materialize(src, "OpenAI", tmp_path / "static")
    assert " " not in url
    assert url.startswith("/app/static/corpus/OpenAI/")


def test_materialize_appends_the_page_anchor(tmp_path):
    src = _pdf(tmp_path / "dataset", "OpenAI", "charter.pdf")
    assert ps.materialize(src, "OpenAI", tmp_path / "static", page=7).endswith("#page=7")


def test_materialize_refuses_a_file_the_static_route_would_reject(tmp_path,
                                                                 monkeypatch):
    # Streamlit 404s anything at or above its cap; shrink the cap rather than
    # writing 200 MB to disk.
    monkeypatch.setattr(ps, "MAX_SERVABLE_BYTES", 4)
    src = _pdf(tmp_path / "dataset", "OpenAI", "big.pdf")
    static_root = tmp_path / "static"

    assert ps.materialize(src, "OpenAI", static_root) is None
    assert not static_root.exists()


def test_materialize_returns_none_when_the_source_is_gone(tmp_path):
    missing = tmp_path / "dataset" / "gone.pdf"
    assert ps.materialize(missing, "OpenAI", tmp_path / "static") is None


# --- the fallback ladder: every path must render today's plain text ---------

def test_citation_links_a_published_chunk():
    got = ps.excerpt_citation(1, ["c1"], _evidence(), _url_for)
    assert got == f"[{PLAIN}](/app/static/corpus/OpenAI/open AI charter.pdf)"


def test_citation_plain_when_chunk_missing_from_index():
    # Empty index, or ids re-minted by a --fresh reingest since the run.
    assert ps.excerpt_citation(1, ["c1"], {}, _url_for) == PLAIN


def test_citation_links_a_thirdparty_chunk_and_routes_by_source_type():
    # Third-party evidence used to be out of scope; it now resolves through the
    # generated per-article PDFs. The source_type argument is what routes it,
    # so assert it was passed — without that check this test would still pass
    # if the routing argument were dropped.
    calls = []

    def spy(filename, org, chunk_id, source_type):
        calls.append((filename, org, chunk_id, source_type))
        return "/app/static/articles/OpenAI/O1__0025.pdf#page=2"

    evidence = _evidence(source_type="thirdparty", filename="O1.RTF")
    got = ps.excerpt_citation(1, ["c1"], evidence, spy)

    assert got == f"[{PLAIN}](/app/static/articles/OpenAI/O1__0025.pdf#page=2)"
    assert calls == [("O1.RTF", "OpenAI", "c1", "thirdparty")]


def test_citation_plain_when_the_article_pdf_was_never_generated():
    # The fallback the whole module exists for: no generated article, no link,
    # and the citation reads exactly as it did before this feature.
    evidence = _evidence(source_type="thirdparty", filename="O1.RTF")
    assert ps.excerpt_citation(1, ["c1"], evidence, lambda *a: None) == PLAIN


@pytest.mark.parametrize("excerpt", [None, "?", 0, 2, -1])
def test_citation_plain_for_unusable_excerpt_numbers(excerpt):
    got = ps.excerpt_citation(excerpt, ["c1"], _evidence(), _url_for)
    assert got == f"\\[excerpt {excerpt if excerpt is not None else '?'}\\]"


def test_citation_plain_when_retrieved_ids_are_missing():
    assert ps.excerpt_citation(1, None, _evidence(), _url_for) == PLAIN


def test_citation_plain_when_no_pdf_resolves():
    assert ps.excerpt_citation(1, ["c1"], _evidence(), lambda *a: None) == PLAIN


# --- mis-citation ------------------------------------------------------------

def test_norm_ws_matches_rag_qa():
    # containing_ids reports which chunk satisfies rag_qa's verification test,
    # so the two normalizations must not drift apart.
    for s in ["  A  B\tC\n", "Board Retains", "", "already normal"]:
        assert ps._norm_ws(s) == rag_qa_norm_ws(s)


def test_containing_ids_finds_the_chunk_holding_the_span():
    evidence = _evidence()
    assert ps.containing_ids("BOARD   retains final", ["c1"], evidence) == ["c1"]
    assert ps.containing_ids("nowhere at all", ["c1"], evidence) == []
    assert ps.containing_ids("", ["c1"], evidence) == []


def test_misattribution_note_silent_when_the_citation_is_right():
    evidence = _evidence()
    assert ps.misattribution_note("board retains", 1, ["c1"], evidence,
                                  _url_for) == ""


def test_misattribution_note_silent_when_the_span_is_nowhere():
    # An unverified quote is section 2's finding to report, not this one's.
    assert ps.misattribution_note("invented text", 1, ["c1"], _evidence(),
                                  _url_for) == ""


def test_misattribution_note_points_at_the_real_document():
    evidence = _evidence()
    evidence["c2"] = {"text": "a wholly different passage about governance",
                      "filename": "planning for agi and beyond.pdf",
                      "org": "OpenAI", "source_type": "published"}
    note = ps.misattribution_note("wholly different passage", 1, ["c1", "c2"],
                                  evidence, _url_for)
    assert note.startswith(" · found in [planning for agi and beyond.pdf](")


def test_misattribution_note_silent_when_the_span_is_elsewhere_in_the_same_article():
    # Chunks overlap by design and a press record contributes several, so a
    # span often sits in a different CHUNK of the document we just linked.
    # Saying "found in X" beside a link to X is noise, not a finding.
    evidence = {
        "OpenAI|thirdparty|O1|3|0": {
            "text": "the opening of the article", "filename": "O1.RTF",
            "org": "OpenAI", "source_type": "thirdparty", "label": "Singapore signs MOU"},
        "OpenAI|thirdparty|O1|3|2": {
            "text": "a later passage of the same article", "filename": "O1.RTF",
            "org": "OpenAI", "source_type": "thirdparty", "label": "Singapore signs MOU"},
    }
    note = ps.misattribution_note(
        "later passage of the same", 1,
        ["OpenAI|thirdparty|O1|3|0", "OpenAI|thirdparty|O1|3|2"], evidence, _url_for)
    assert note == ""


def test_misattribution_note_still_fires_across_two_articles():
    evidence = {
        "OpenAI|thirdparty|O1|3|0": {
            "text": "the cited article", "filename": "O1.RTF", "org": "OpenAI",
            "source_type": "thirdparty", "label": "Singapore signs MOU"},
        "OpenAI|thirdparty|O1|5|0": {
            "text": "Florida became the first state to sue", "filename": "O1.RTF",
            "org": "OpenAI", "source_type": "thirdparty", "label": "Florida sues OpenAI"},
    }
    note = ps.misattribution_note(
        "first state to sue", 1,
        ["OpenAI|thirdparty|O1|3|0", "OpenAI|thirdparty|O1|5|0"], evidence, _url_for)
    assert "Florida sues OpenAI" in note


def test_misattribution_note_deduplicates_overlapping_chunks():
    # Consecutive chunks of one document overlap by 150 chars, so a span near a
    # boundary really is in two chunks of the same file — name it once.
    shared = "the overlap region text"
    evidence = {
        "c1": {"text": "cited chunk with nothing in common", "filename": "a.pdf",
               "org": "OpenAI", "source_type": "published"},
        "c2": {"text": f"... {shared} ...", "filename": "b.pdf",
               "org": "OpenAI", "source_type": "published"},
        "c3": {"text": f"{shared} continues here", "filename": "b.pdf",
               "org": "OpenAI", "source_type": "published"},
    }
    note = ps.misattribution_note(shared, 1, ["c1", "c2", "c3"], evidence, _url_for)
    assert note.count("b.pdf](") == 1


# --- third-party article resolution -----------------------------------------

def _sources(**over):
    rec = {"file": "OpenAI/O1__0025.pdf", "headline": "Florida sues OpenAI",
           "publication": "The Center Square", "date": "June 6, 2026",
           "pages": {"0": 1, "1": 2}}
    rec.update(over)
    return {"OpenAI|thirdparty|O1|25": rec}


def test_article_key_parses_a_thirdparty_chunk_id():
    assert ps.article_key("OpenAI|thirdparty|O1|25|1") == ("OpenAI|thirdparty|O1|25", 1)


@pytest.mark.parametrize("cid", [
    "OpenAI|published|29|1",        # published ids must fall through
    "OpenAI|thirdparty|O1|25",      # too few fields
    "OpenAI|thirdparty|O1|25|x",    # non-integer chunk index
    "", None, 17,
])
def test_article_key_rejects_anything_else(cid):
    assert ps.article_key(cid) is None


def test_resolve_article_returns_path_page_and_record(tmp_path):
    pdf = tmp_path / "OpenAI" / "O1__0025.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")

    path, page, rec = ps.resolve_article("OpenAI|thirdparty|O1|25|1",
                                         _sources(), tmp_path)
    assert path == pdf
    assert page == 2
    assert rec["headline"] == "Florida sues OpenAI"


def test_resolve_article_defaults_to_page_one_for_an_unmapped_chunk(tmp_path):
    pdf = tmp_path / "OpenAI" / "O1__0025.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    _, page, _ = ps.resolve_article("OpenAI|thirdparty|O1|25|9", _sources(), tmp_path)
    assert page == 1


def test_resolve_article_none_when_the_pdf_was_not_generated(tmp_path):
    assert ps.resolve_article("OpenAI|thirdparty|O1|25|0", _sources(), tmp_path) is None


def test_resolve_article_none_for_an_unmapped_or_published_id(tmp_path):
    assert ps.resolve_article("OpenAI|thirdparty|O9|1|0", _sources(), tmp_path) is None
    assert ps.resolve_article("OpenAI|published|29|1", _sources(), tmp_path) is None
    assert ps.resolve_article("OpenAI|thirdparty|O1|25|0", _sources(), None) is None


def test_resolve_article_refuses_a_path_outside_the_article_root(tmp_path):
    # The map is a JSON artifact whose paths get published into an
    # unauthenticated static route, so traversal must be refused outright.
    outside = tmp_path.parent / "secrets.pdf"
    outside.write_bytes(b"%PDF-1.4")
    sources = _sources(file=f"../{outside.name}")
    assert ps.resolve_article("OpenAI|thirdparty|O1|25|0", sources, tmp_path) is None


def test_load_article_sources_empty_when_absent_or_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "ARTICLE_SOURCES_PATH", tmp_path / "nope.json")
    assert ps.load_article_sources() == {}
    bad = tmp_path / "article_sources.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(ps, "ARTICLE_SOURCES_PATH", bad)
    assert ps.load_article_sources() == {}


def test_article_label_and_sublabel():
    rec = _sources()["OpenAI|thirdparty|O1|25"]
    assert ps.article_label(rec) == "Florida sues OpenAI"
    assert ps.article_sublabel(rec) == "The Center Square · June 6, 2026"
    assert ps.article_label({"headline": "x" * 200}).endswith("…")
    assert ps.article_sublabel({}) == ""


def test_md_escape_survives_a_headline_with_brackets():
    # Nexis headlines really do contain brackets; unescaped they terminate the
    # surrounding [label](url) early and leak the raw url into the page.
    assert ps.md_escape("OpenAI [sic] launches (again)") == \
        "OpenAI \\[sic\\] launches (again)"


# --- mis-citation labelling for press records --------------------------------

def test_misattribution_note_names_the_article_not_the_rtf():
    # Every press chunk's `filename` is the same 45 MB bundle, so the bundle
    # name must never be what the reader is shown.
    evidence = {
        "OpenAI|thirdparty|O1|3|0": {
            "text": "cited chunk, unrelated", "filename": "O1.RTF",
            "org": "OpenAI", "source_type": "thirdparty", "label": "Cited piece"},
        "OpenAI|thirdparty|O1|9|0": {
            "text": "the span actually lives here", "filename": "O1.RTF",
            "org": "OpenAI", "source_type": "thirdparty",
            "label": "Anthropic wants to tame workplace AI"},
    }

    def url_for(filename, org, chunk_id, source_type="published"):
        return f"/app/static/articles/{org}/O1__0009.pdf"

    note = ps.misattribution_note(
        "span actually lives", 1,
        ["OpenAI|thirdparty|O1|3|0", "OpenAI|thirdparty|O1|9|0"], evidence, url_for)
    assert "Anthropic wants to tame workplace AI" in note
    assert "O1.RTF" not in note      # the bundle name must not reach the UI


def test_misattribution_note_falls_back_to_filename_without_a_label():
    evidence = _evidence()
    evidence["c2"] = {"text": "a wholly different passage about governance",
                      "filename": "planning for agi and beyond.pdf",
                      "org": "OpenAI", "source_type": "published"}
    note = ps.misattribution_note("wholly different passage", 1, ["c1", "c2"],
                                  evidence, _url_for)
    assert "planning for agi and beyond.pdf" in note


# --- page alignment ---------------------------------------------------------

def test_page_starts_aligns_pages_to_the_body():
    pages = ["Alpha beta gamma delta epsilon zeta eta theta iota kappa. ",
             "Lambda mu nu xi omicron pi rho sigma tau upsilon phi chi. ",
             "Psi omega again alpha beta gamma delta epsilon zeta eta. "]
    starts, rate = ps._page_starts("".join(pages), pages)

    assert rate == 1.0
    assert starts == sorted(starts)
    body, _ = ps._norm_alnum("".join(pages))
    for start, page in zip(starts, pages):
        assert body.startswith(ps._norm_alnum(page)[0][:20], start)


def test_page_starts_survives_a_printed_page_number_prefix():
    # The real failure this alignment exists for: pypdf emits the printed page
    # number as the first characters of each page, where the corpus converter
    # did not, so matching on a page's opening text alone misses every time.
    body = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. "
            "Mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
            "Second section here with quite different wording throughout it. ")
    third = len(body) // 3
    pages = [f"{i + 1}" + body[i * third:(i + 1) * third] for i in range(3)]

    starts, rate = ps._page_starts(body, pages)
    assert rate == 1.0
    assert starts == sorted(starts)
    assert starts[0] == 0


def test_page_starts_reports_a_low_rate_when_nothing_matches():
    # Body and PDF share no text: every page falls back to the proportional
    # estimate, and the low rate is what tells build_page_map to decline.
    body = "aaaaaaaaaa" * 40
    pages = ["totally unrelated wording one", "totally unrelated wording two"]
    starts, rate = ps._page_starts(body, pages)

    assert rate == 0.0
    assert starts == sorted(starts)


def test_page_starts_refuses_an_image_only_pdf():
    # pypdf extracts nothing from a scan; the corpus text came from elsewhere,
    # so there is nothing to align against and guessing would be worse.
    assert ps._page_starts("real text from some other converter",
                                    ["", "", ""]) is None


# --- page map ---------------------------------------------------------------

def test_load_chunk_pages_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PDF_PAGES_PATH", tmp_path / "nope.json")
    assert ps.load_chunk_pages() == {}


def test_load_chunk_pages_survives_a_corrupt_file(tmp_path, monkeypatch):
    bad = tmp_path / "pdf_pages.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(ps, "PDF_PAGES_PATH", bad)
    assert ps.load_chunk_pages() == {}
