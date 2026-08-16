"""Offline tests for the third-party article splitter and renderer helpers.

No dataset, no PDF library, no index. The load-bearing groups are
`pair_fragments` (which article a chunk belongs to) and `_parse_identity`
(what that article is called) — every link's correctness rests on those two,
and each has real edge cases taken verbatim from the corpus.
"""
from il_rag import article_pdfs as ap

EOD = "End of Document"


def _identity(headline, publication, date):
    return f"{EOD}\n\n\n{headline}\n{publication}\n{date}\nCopyright 2026 X"


def _body(words, text):
    return f"Length: {words} words\nBody\n\n{text}\nLoad-Date: June 8, 2026"


# --- fragment pairing --------------------------------------------------------

def test_pair_fragments_pairs_each_identity_with_its_body():
    frags = [
        _identity("Florida sues OpenAI", "The Center Square", "June 6, 2026 Saturday"),
        _body(394, "Florida became the first state this week to sue OpenAI."),
        _identity("OpenAI files for IPO", "The Irish Times", "June 10, 2026 Wednesday"),
        _body(442, "OpenAI has filed confidentially for an IPO."),
    ]
    articles = ap.pair_fragments(frags)

    assert len(articles) == 2
    assert articles[0]["fragments"] == [0, 1]
    assert articles[1]["fragments"] == [2, 3]
    assert articles[0]["identity"]["headline"] == "Florida sues OpenAI"
    assert articles[1]["identity"]["publication"] == "The Irish Times"


def test_pair_fragments_keeps_a_surviving_identity_fragment_with_its_article():
    # A1.RTF keeps 13 identity fragments (513 = 500 + 13). Their chunk ids are
    # real, so they must map to the same article as the body beside them —
    # which even/odd parity would get wrong.
    frags = [
        _identity("A", "Pub", "June 6, 2026"),
        _body(100, "body one"),
        _identity("B", "Pub", "June 7, 2026"),
        _body(100, "body two"),
    ]
    articles = ap.pair_fragments(frags)
    assert all(a["fragments"][0] % 2 == 0 for a in articles)
    assert [a["fragments"] for a in articles] == [[0, 1], [2, 3]]


def test_pair_fragments_drops_the_bare_trailing_terminator():
    frags = [
        _identity("A", "Pub", "June 6, 2026"),
        _body(100, "the body"),
        EOD,                       # 500th record's terminator, nothing after it
    ]
    articles = ap.pair_fragments(frags)
    assert len(articles) == 1
    assert articles[0]["fragments"] == [0, 1]


def test_pair_fragments_skips_the_job_header_but_keeps_its_identity_tail():
    # Fragment 0 is the Nexis job header plus a search-results index, with the
    # FIRST article's identity glued to its tail. Its chunks are table of
    # contents and belong to no article; the tail still names article 1.
    header = ("User Name: = \nDate and Time: = 2026-06-10\nJob Number: = 286864637\n"
              "\nDocuments (500)\n"
              "1. Some indexed headline\n2. Another indexed headline\n"
              "OpenAI launches Deployment Company\n"
              "CE Noticias Financieras English\n"
              "June 8, 2026 Monday\n"
              "Copyright 2026 Content Engine, LLC.")
    frags = [header, _body(402, "OpenAI announced the creation of Deployment Company.")]

    articles = ap.pair_fragments(frags)
    assert len(articles) == 1
    assert articles[0]["fragments"] == [1]          # fragment 0 gets no entry
    assert articles[0]["identity"]["headline"] == "OpenAI launches Deployment Company"
    assert articles[0]["identity"]["publication"] == "CE Noticias Financieras English"


def test_pair_fragments_handles_a_file_with_no_job_header():
    # The Anthropic exports carry no job header, so fragment 0 IS article 0's
    # identity block — a third shape, and the one that silently produced
    # identity-less articles until it was handled.
    frags = [
        "Anthropic wants to tame workplace AI\nAxios\n"
        "December 18, 2025 Thursday 5:00 PM EST\nCopyright 2025 Axios",
        _body(387, "Anthropic is trying to tame workplace AI."),
        _identity("Next one", "Pub", "December 19, 2025"),
        _body(200, "another body"),
    ]
    articles = ap.pair_fragments(frags)

    assert articles[0]["fragments"] == [0, 1]
    assert articles[0]["identity"]["headline"] == "Anthropic wants to tame workplace AI"
    assert articles[0]["identity"]["publication"] == "Axios"
    assert articles[1]["identity"]["headline"] == "Next one"


def test_pair_fragments_empty_input():
    assert ap.pair_fragments([]) == []


# --- identity parsing --------------------------------------------------------

def test_parse_identity_basic_three_line_block():
    got = ap._parse_identity(["Florida sues OpenAI", "The Center Square",
                              "June 6, 2026 Saturday 10:00 AM EST"])
    assert got == {"headline": "Florida sues OpenAI",
                   "publication": "The Center Square",
                   "date": "June 6, 2026 Saturday 10:00 AM EST"}


def test_parse_identity_joins_a_two_line_publication():
    # ~3% of records carry a distributor line above the publication.
    got = ap._parse_identity(["Florida sues OpenAI", "Newstex Blogs",
                              "The Center Square: Florida", "June 6, 2026"])
    assert got["headline"] == "Florida sues OpenAI"
    assert got["publication"] == "Newstex Blogs The Center Square: Florida"


def test_parse_identity_ignores_a_headline_that_starts_with_a_date():
    # The News Bites shape. Prefix-matching a date here mistakes the headline
    # for the date and shifts every field — it cost 6.2% of records.
    got = ap._parse_identity([
        "May 06, 2026: EVE Online and Google DeepMind Forge AI Partnership",
        "News Bites - Private Companies",
        "May 13, 2026 Wednesday"])
    assert got["headline"].startswith("May 06, 2026: EVE Online")
    assert got["date"] == "May 13, 2026 Wednesday"


def test_parse_identity_accepts_the_real_date_variants():
    variants = [
        "November 4, 2025 Tuesday 12:01 AM Eastern Time",
        "March 15, 2026, Sunday",
        "May 2026",
        "January, 2026",
        "2026-05-11",
        "June 10, 2026 Wednesday",
    ]
    for date in variants:
        got = ap._parse_identity(["Headline", "Publication", date])
        assert got is not None and got["date"] == date, date


def test_parse_identity_none_without_a_date():
    assert ap._parse_identity(["Headline", "Publication", "no date here"]) is None
    assert ap._parse_identity([]) is None


def test_parse_identity_from_tail_skips_the_search_index():
    lines = ["Job Number: = 1", "499. An indexed headline", "500. Another one",
             "The real headline", "The Publication", "June 8, 2026 Monday",
             "Copyright 2026"]
    got = ap._parse_identity(lines, from_tail=True)
    assert got == {"headline": "The real headline",
                   "publication": "The Publication",
                   "date": "June 8, 2026 Monday"}


# --- sanitization ------------------------------------------------------------

def test_sanitize_leaves_cp1252_text_untouched():
    text = "curly “quotes” em—dash bullet • euro € accent é tm ™"
    clean, subs = ap._sanitize(text)
    assert clean == text
    assert subs == {}


def test_sanitize_folds_the_table_characters():
    # NBSP -> space; ZWSP / word joiner / BOM deleted; non-breaking hyphen and
    # the fullwidth bar folded to ASCII.
    clean, subs = ap._sanitize("a b​c⁠d﻿e‑f｜g")
    assert clean == "a bcde-f|g"
    assert subs == {}          # table folds are not substitutions


def test_sanitize_counts_what_it_could_not_represent():
    # A rune and a CJK ideograph — both real, both in the 67-character residue.
    clean, subs = ap._sanitize("runes ᚨ and 郭 here")
    assert "ᚨ" not in clean and "郭" not in clean
    assert subs["ᚨ"] == 1 and subs["郭"] == 1
    clean.encode("cp1252")     # must not raise


def test_sanitize_folds_accents_rather_than_dropping_them():
    clean, _ = ap._sanitize("č")          # c with caron, not in cp1252
    assert clean == "c"


# --- offsets and pages -------------------------------------------------------

def test_alnum_offsets_counts_alphanumeric_prefixes():
    assert ap.alnum_offsets("ab, cd") == [0, 1, 2, 2, 2, 3, 4]


def test_page_for_offset_bisects_recorded_marks():
    marks = [(0, 1), (400, 1), (900, 2), (1500, 3)]
    assert ap.page_for_offset(marks, 0) == 1
    assert ap.page_for_offset(marks, 399) == 1
    assert ap.page_for_offset(marks, 900) == 2
    assert ap.page_for_offset(marks, 1499) == 2
    assert ap.page_for_offset(marks, 9999) == 3
    assert ap.page_for_offset([], 5) == 1


def test_chunk_pages_locates_each_chunk_in_its_own_fragment():
    fragment = "Alpha beta gamma. " * 40 + "Delta epsilon zeta. " * 40
    chunks = [fragment[:300], fragment[600:900]]
    # Pretend the second half of the fragment landed on page 2.
    half = len(ap._norm_alnum(fragment)[0]) // 2
    marks = [(0, 1), (half, 2)]

    pages = ap.chunk_pages(fragment, chunks, marks)
    assert pages["0"] == 1
    assert pages["1"] in (1, 2)


def test_chunk_pages_omits_a_chunk_it_cannot_locate():
    pages = ap.chunk_pages("some source text here", ["entirely unrelated span"], [(0, 1)])
    assert pages == {}


# --- wrapping ----------------------------------------------------------------

def _measure(s):
    """Stand-in for a font metric: one unit per character."""
    return float(len(s))


def test_wrap_offsets_index_back_into_the_source():
    text = ("The board retains final authority over deployment decisions, and "
            "may override commercial considerations when safety requires it.")
    for line, off in ap.wrap(text, 30, _measure):
        if line:
            assert text[off:off + len(line)] == line, (line, off)


def test_wrap_respects_the_column_width():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    assert all(_measure(line) <= 30 for line, _ in ap.wrap(text, 30, _measure))


def test_wrap_hard_breaks_a_token_wider_than_the_column():
    url = "https://example.com/" + "x" * 120
    lines = ap.wrap(url, 20, _measure)
    assert len(lines) > 1
    assert all(_measure(line) <= 20 for line, _ in lines)


def test_wrap_preserves_blank_lines_between_paragraphs():
    lines = ap.wrap("first para\n\nsecond para", 40, _measure)
    assert "" in [line for line, _ in lines]


# --- the map -----------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path, monkeypatch):
    # The builder writes it; pdf_sources reads it — the app never imports this
    # module, so the two halves of that contract are checked together here.
    from il_rag import pdf_sources

    path = tmp_path / "article_sources.json"
    monkeypatch.setattr(ap, "ARTICLE_SOURCES_PATH", path)
    monkeypatch.setattr(pdf_sources, "ARTICLE_SOURCES_PATH", path)

    articles = {"OpenAI|thirdparty|O1|25": {"file": "OpenAI/O1__0025.pdf",
                                            "headline": "Florida sues OpenAI",
                                            "pages": {"0": 1}}}
    ap.save_article_sources(articles)
    assert pdf_sources.load_article_sources() == articles


def test_article_dirs_uses_the_dataset_layout(tmp_path):
    dirs = ap.article_dirs(tmp_path)
    assert set(dirs) == {"OpenAI", "DeepMind", "Anthropic"}
    assert dirs["OpenAI"] == tmp_path / "OpenAI" / "9 - Articles"
    assert ap.article_dirs(None) == {}
