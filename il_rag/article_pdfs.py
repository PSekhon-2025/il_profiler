"""Render each third-party press record as its own PDF, so the audit trail's
`[excerpt N]` can link to the article a span came from.

LOCAL ONLY. The heavy dependency (fpdf2) and the raw corpus both stay on the
workstation; the app reads only the small JSON map this produces and never
imports this module. Same arrangement as the topic layer and the page map.

Why this exists: a published chunk carries its source PDF's filename, but a
third-party chunk carries only `O1.RTF` — one 45 MB Nexis export bundling 500
articles. Linking to that file would point at the bundle, not the article. So
we split the bundle the same way ingestion does and render one document per
record.

THE STRUCTURE WE DEPEND ON, and the bug that creates it
-------------------------------------------------------
`ingest.ARTICLE_SPLIT_RE` lists seven split markers, but five of them can never
fire: `Title:` / `Headline:` / `HEADLINE:` / `Byline:` / `Publication-Date:` are
each followed by `:` and then U+00A0, and the pattern ends with `\\b`, which
needs a word boundary that is not there. (`Byline:` occurs 140 times in O1.RTF
and causes zero splits.) Only `End of Document` and `Length: N words` ever
split.

That accident is what makes this module possible: every Nexis record splits into
exactly two fragments, in strict alternation —

    fragment 2k     "End of Document", then HEADLINE / PUBLICATION / DATE,
                    then Copyright..., Section:...        <- the article's identity
    fragment 2k+1   "Length: N words", Byline:, "Body", the prose, Load-Date:
                    <- the body, and the source of essentially every indexed chunk

so a chunk's article identity lives in the PRECEDING fragment. That is exactly
why third-party citations look anonymous in the UI today.

If `ARTICLE_SPLIT_RE` is ever fixed, this pairing dissolves AND every chunk id
is re-minted, so the map this builds must be regenerated. Do not fix one without
the other.

Two edge cases, both verified against all six files:

  * Fragment 0 is the Nexis job header (job metadata + a 500-item search-results
    index) with the FIRST article's identity block glued to its tail. In four of
    six files it is long enough to survive ingestion's filter, contributing ~36
    chunks of pure table-of-contents. Those chunks belong to no article, so they
    get no map entry and correctly keep rendering as plain text — but the tail is
    still read, so article 1 gets a proper header.
  * The last fragment is a bare `End of Document` terminator with nothing after
    it. Not an article.

Do not assume only odd fragments survive the filter: A1.RTF keeps 13 identity
fragments (513 = 500 + 13). Their chunk ids are real, so a map entry is emitted
per surviving FRAGMENT, both halves pointing at the same PDF.

Fonts: rendering uses a PDF core font under cp1252 rather than embedding a
Unicode TTF. Measured over all six files, after the small `_SANITIZE` table
below only 67 characters out of 15.8 million fall outside cp1252 — runes, three
CJK ideographs, arrows. Embedding a subsetted DejaVu would cost 12-25 KB per
file across ~3,100 files and dominate the output; core fonts keep each PDF at a
few KB. The residue is folded or replaced and COUNTED, never dropped silently.
"""
import json
import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from itertools import accumulate
from pathlib import Path

from .pdf_sources import ARTICLE_SOURCES_PATH, _norm_alnum

# Where each lab's press dumps sit inside the dataset tree. Read from
# PDF_DATASET_ROOT rather than config.ARTICLE_DIRS: those still point at the
# pre-reorganization layout and resolve to directories that do not exist.
# The map itself is defined by pdf_sources, which is what the app reads —
# nothing in the app imports this module.
ARTICLES_SUBDIR = "9 - Articles"

# Characters to fold before the cp1252 check. U+00A0 is Nexis's field separator
# and appears 15,307 times; the invisibles carry no glyph; U+FF5C is the pipe in
# their company-profile tables.
_SANITIZE = {
    0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x2009: " ",   # spaces
    0x00AD: "", 0x200B: "", 0x200C: "", 0x200D: "",       # invisible
    0x2060: "", 0xFEFF: "",
    0x2011: "-",                                          # non-breaking hyphen
    0xFF5C: "|",
}

# A Nexis date line. The whole line must match: News Bites headlines BEGIN with
# a date ("May 06, 2026: EVE Online and Google DeepMind Forge AI Partnership"),
# and a prefix match there mistakes the headline for the date and shifts every
# field by one. Measured: prefix matching parses 93.45% of records, whole-line
# matching 99.67%, and the variants below close the rest.
_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December")
_WEEKDAY = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_DATE_LINE_RE = re.compile(
    rf"^(?:"
    rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}"          # June 6, 2026
    rf"(?:,?\s+(?:{_WEEKDAY}))?"                       # [, ] Saturday
    rf"(?:\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?\s*(?:AM|PM|am|pm)?"
    rf"(?:\s+[\w.]+){{0,3}})?"                         # 12:01 AM Eastern Time
    rf"|(?:{_MONTHS}),?\s+\d{{4}}"                     # May 2026 / January, 2026
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"                       # 2026-05-11
    rf")\s*$"
)

# Fragment 0's giveaway: the Nexis job header preamble.
_JOB_HEADER_MARKERS = ("Job Number:", "User Name:", "Documents (")

_EOD = "End of Document"


def article_dirs(root: Path | None) -> dict[str, Path]:
    """{org: its press-dump directory}, mirroring pdf_sources.published_corpora.

    Keys must stay exactly ORGS — they are written verbatim into chunk metadata
    and are the first field of every chunk id.
    """
    from .config import ORGS
    if root is None:
        return {}
    return {org: root / org / ARTICLES_SUBDIR for org in ORGS}


def _sanitize(text: str) -> tuple[str, Counter]:
    """Fold `text` into cp1252, reporting every substitution it had to make.

    Returns (clean, counts-by-original-codepoint). A character that survives
    neither the table nor an NFKD fold becomes "?" — but it is counted, so the
    generator can report exactly which codepoints to add to `_SANITIZE` rather
    than quietly degrading the text.
    """
    text = text.translate(_SANITIZE)
    try:
        text.encode("cp1252")
        return text, Counter()
    except UnicodeEncodeError:
        pass
    out, subs = [], Counter()
    for ch in text:
        try:
            ch.encode("cp1252")
            out.append(ch)
            continue
        except UnicodeEncodeError:
            subs[ch] += 1
        folded = "".join(c for c in unicodedata.normalize("NFKD", ch)
                         if not unicodedata.combining(c))
        try:
            folded.encode("cp1252")
            out.append(folded)
        except UnicodeEncodeError:
            out.append("?")
    return "".join(out), subs


def _parse_identity(lines: list[str], from_tail: bool = False) -> dict | None:
    """Pull headline / publication / date out of an identity block.

    Forward form: the headline is the first line, the date is the first line
    that is ENTIRELY a date, and everything between is the publication — which
    is what handles the two-line publisher shape (`Newstex Blogs` /
    `The Center Square: Florida`).

    `from_tail` handles fragment 0, where the identity block is preceded by a
    500-item search-results index: scan backwards for the date and take only
    the two lines above it, since anything earlier is index junk.
    """
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return None

    if from_tail:
        for i in range(len(lines) - 1, 1, -1):
            if _DATE_LINE_RE.match(lines[i]):
                return {"headline": lines[i - 2], "publication": lines[i - 1],
                        "date": lines[i]}
        return None

    for i, line in enumerate(lines[:8]):
        if _DATE_LINE_RE.match(line):
            if i < 2:          # no room for both a headline and a publication
                continue
            return {"headline": lines[0],
                    "publication": " ".join(lines[1:i]),
                    "date": line}
    return None


def _is_job_header(fragment: str) -> bool:
    """Is this the Nexis job header rather than an article identity block?"""
    head = fragment[:2000]
    return any(marker in head for marker in _JOB_HEADER_MARKERS)


def pair_fragments(fragments: list[str]) -> list[dict]:
    """Group ingestion's fragment list into articles.

    Walks in order: a fragment starting with "End of Document" closes the
    previous article and opens a new one, carrying the new article's identity.
    Deliberately NOT even/odd parity — identity fragments that survive the
    filter would be mis-assigned by parity, and there are 13 of them in A1.

    Returns one dict per article: {"identity": {...} | None, "fragments":
    [indices]}. Articles with no fragments (the bare trailing terminator) are
    dropped. The job header contributes article 0's identity but never appears
    in any article's `fragments`, because its chunks are search-results index,
    not article text.
    """
    if not fragments:
        return []

    articles: list[dict] = []
    start = 0
    if _is_job_header(fragments[0]):
        # O1/O2/D1/D2: a job header with article 0's identity glued to its tail.
        # The header's own text is a search-results index, so it contributes no
        # fragment — its chunks belong to no article and stay unlinked.
        articles.append({
            "identity": _parse_identity(fragments[0].splitlines(), from_tail=True),
            "fragments": [],
        })
        start = 1
    elif not fragments[0].startswith(_EOD):
        # A1/A2: no job header at all, so fragment 0 IS article 0's identity
        # block. It contributes itself, in case it survives the filter.
        articles.append({
            "identity": _parse_identity(fragments[0].splitlines()),
            "fragments": [0],
        })
        start = 1

    for ai in range(start, len(fragments)):
        frag = fragments[ai]
        if frag.startswith(_EOD):
            body = frag.splitlines()[1:]          # drop the terminator line
            articles.append({"identity": _parse_identity(body), "fragments": []})
            if not [ln for ln in body if ln.strip()]:
                continue                          # bare terminator: no content
        if not articles:
            # Content before any identity block and without a job header —
            # not a shape we have seen, but never lose the fragment.
            articles.append({"identity": None, "fragments": []})
        articles[-1]["fragments"].append(ai)

    return [a for a in articles if a["fragments"]]


def wrap(text: str, max_width: float, measure) -> list[tuple[str, int]]:
    """Greedy word wrap that reports where each visual line began.

    Returns [(line_text, offset_into_text)]. The offsets are what make exact
    page anchors possible: they index back into the SOURCE string, so a chunk
    found in the source can be mapped to the line — and therefore the page —
    it was drawn on. A token wider than the column is hard-broken rather than
    left to overflow.
    """
    lines: list[tuple[str, int]] = []
    pos = 0
    for raw in text.split("\n"):
        if not raw.strip():
            lines.append(("", pos))
            pos += len(raw) + 1
            continue
        cursor = pos
        current, current_start = "", cursor
        for token in re.finditer(r"\S+\s*", raw):
            word = token.group()
            candidate = current + word
            if current and measure(candidate.rstrip()) > max_width:
                lines.append((current.rstrip(), current_start))
                current, current_start = word, pos + token.start()
            else:
                current = candidate
            # Hard-break a single token too wide for the column.
            while measure(current.rstrip()) > max_width and len(current.strip()) > 1:
                cut = len(current)
                while cut > 1 and measure(current[:cut]) > max_width:
                    cut -= 1
                lines.append((current[:cut], current_start))
                current_start += cut
                current = current[cut:]
        if current.strip() or not lines:
            lines.append((current.rstrip(), current_start))
        pos += len(raw) + 1
    return lines


def alnum_offsets(text: str) -> list[int]:
    """Prefix count of alphanumerics, so a source offset maps into _norm_alnum
    space in O(1). `out[i]` is how many alphanumerics precede `text[i]`."""
    return [0] + list(accumulate(1 if ch.isalnum() else 0 for ch in text))


def page_for_offset(marks: list[tuple[int, int]], at: int) -> int:
    """Page for a normalized offset, given [(offset, page)] recorded while
    drawing. `marks` is monotone by construction, so this is a bisect."""
    if not marks:
        return 1
    idx = bisect_right([m[0] for m in marks], at) - 1
    return marks[max(idx, 0)][1]


def chunk_pages(fragment: str, chunks: list[str],
                marks: list[tuple[int, int]]) -> dict[str, int]:
    """{chunk index: page} for one fragment's chunks.

    The chunk text comes from `chunk_text(strip_boilerplate(fragment))` while
    the PDF renders the FULL record, so a chunk is a line-SUBSEQUENCE of what
    was drawn, not a substring of it. We therefore never align the chunk
    against the rendered text — we align it against its own source fragment,
    whose offsets `marks` already ties to pages.

    Both sides go through the same normalization of the same string, so this is
    an exact match, not the fuzzy alignment the published page map needs. A
    chunk that still cannot be located (its probe straddles a boilerplate line
    that strip_boilerplate deleted) is OMITTED, so its link opens at page 1
    rather than at a guess.
    """
    frag_norm, _ = _norm_alnum(fragment)
    out: dict[str, int] = {}
    cursor = 0
    for ci, chunk in enumerate(chunks):
        probe, _ = _norm_alnum(chunk)
        at = -1
        for size in (120, 60, 30):
            if len(probe) < size:
                continue
            at = frag_norm.find(probe[:size], cursor)
            if at >= 0:
                break
        if at < 0:
            continue
        cursor = at
        out[str(ci)] = page_for_offset(marks, at)
    return out


# ---------------------------------------------------------------------------
# Rendering (LOCAL ONLY — fpdf2 is imported inside the functions)
# ---------------------------------------------------------------------------
# A4 in points, with generous margins: these documents are read on screen, and
# a narrower column is what keeps a wrapped line short enough to scan.
_PAGE_W, _PAGE_H = 595.28, 841.89
_MARGIN = 56.0
_TEXT_W = _PAGE_W - 2 * _MARGIN
_BOTTOM = _PAGE_H - _MARGIN
_LEADING = 13.0


def _render_record(identity: dict | None, blocks: list[tuple[int, str]]):
    """Render one press record; return (pdf_bytes, {fragment_index: marks}).

    `blocks` is [(fragment_index, raw_fragment_text)] in reading order. Marks
    are [(normalized_offset_within_that_fragment, page)], recorded as each
    visual line is drawn — which is what makes the page anchors exact rather
    than inferred. Automatic page breaks are switched OFF so that page numbers
    are ours to know, not fpdf2's to decide.
    """
    from fpdf import FPDF

    pdf = FPDF(format="A4", unit="pt")
    pdf.core_fonts_encoding = "cp1252"   # covers all but 67 chars of the corpus
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.add_page()
    if identity:
        pdf.set_title(_sanitize(identity.get("headline", ""))[0][:120])

    y = _MARGIN

    def new_page():
        nonlocal y
        pdf.add_page()
        y = _MARGIN

    def draw(text, size, style="", gap=0.0, record=None, base=None):
        """Wrap and draw `text`; if `record` is given, append (offset, page)."""
        nonlocal y
        pdf.set_font("helvetica", style, size)
        leading = size * 1.35
        for line, offset in wrap(text, _TEXT_W, pdf.get_string_width):
            if y + leading > _BOTTOM:
                new_page()
            if record is not None and base is not None:
                record.append((base[offset], pdf.page_no()))
            if line:
                pdf.set_xy(_MARGIN, y)
                pdf.cell(_TEXT_W, leading, line)
            y += leading
        y += gap

    # --- header: what this document IS -------------------------------------
    if identity:
        draw(_sanitize(identity["headline"])[0], 15, "B", gap=4)
        sub = " · ".join(p for p in (identity.get("publication"),
                                     identity.get("date")) if p)
        if sub:
            draw(_sanitize(sub)[0], 9.5, "I", gap=10)
    else:
        draw("(source metadata unavailable)", 9.5, "I", gap=10)

    # --- the record itself, verbatim ---------------------------------------
    marks: dict[int, list[tuple[int, int]]] = {}
    for ai, raw in blocks:
        clean, _ = _sanitize(raw)
        # Offsets are taken against the RAW fragment, because that is what the
        # chunks were derived from; _sanitize only substitutes characters
        # one-for-one or deletes zero-width ones, so alphanumeric counts —
        # the only thing these offsets index — are unchanged.
        base = alnum_offsets(raw)
        rec: list[tuple[int, int]] = []
        draw(clean, 10.5, "", gap=8, record=rec, base=base)
        marks[ai] = rec or [(0, 1)]

    return bytes(pdf.output()), marks


def _verify_pdf(data: bytes, rendered: str) -> float:
    """Fraction of the rendered text that survives a pypdf round trip.

    The generator's own audit: a font that silently drops glyphs, or a wrap bug
    that loses a line, shows up here rather than in the UI.
    """
    import io

    from pypdf import PdfReader

    try:
        pages = PdfReader(io.BytesIO(data)).pages
        got, _ = _norm_alnum("\n".join(p.extract_text() or "" for p in pages))
    except Exception:  # noqa: BLE001 — a bad PDF is a failed check, not a crash
        return 0.0
    want, _ = _norm_alnum(rendered)
    if not want:
        return 1.0
    # Count how much of the source is present, tolerating wrap-induced joins.
    hit = sum(1 for i in range(0, len(want), 200)
              if want[i:i + 60] and want[i:i + 60] in got)
    total = sum(1 for i in range(0, len(want), 200) if want[i:i + 60])
    return hit / total if total else 1.0


def build_articles(dirs: dict[str, Path], out_dir: Path, *, limit: int | None = None,
                   force: bool = False, verify: bool = True,
                   dry_run: bool = False) -> tuple[dict, dict]:
    """Convert every press record that produced chunks into its own PDF.

    Reuses ingestion's own splitting, stripping and chunking verbatim, and
    iterates the RTFs exactly as `_iter_thirdparty` does, so the ids in the
    returned map match the ids in the vector index by construction.

    Returns (map, stats). The map is keyed by fragment prefix
    `{org}|thirdparty|{stem}|{ai}` — one entry per surviving FRAGMENT, both
    halves of a record pointing at the same file, so the app never has to know
    about the identity/body pairing.
    """
    from striprtf.striprtf import rtf_to_text

    from .ingest import ARTICLE_SPLIT_RE, MIN_ARTICLE_CHARS, chunk_text, strip_boilerplate

    articles_map: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    substitutions: Counter = Counter()
    verify_scores: list[float] = []

    for org, dirpath in dirs.items():
        if not dirpath.is_dir():
            print(f"[warn] missing article dir for {org}: {dirpath}")
            continue
        s = stats.setdefault(org, {"records": 0, "written": 0, "skipped": 0,
                                   "no_identity": 0, "fragments": 0, "bytes": 0})
        for rtf in sorted(p for p in dirpath.iterdir() if p.suffix.lower() == ".rtf"):
            try:
                plain = rtf_to_text(rtf.read_text(encoding="utf-8", errors="ignore"),
                                    errors="ignore")
            except Exception as e:  # noqa: BLE001 — one bad file must not kill the run
                print(f"[warn] could not parse {rtf.name}: {e}")
                continue
            parts = [p.strip() for p in ARTICLE_SPLIT_RE.split(plain) if p.strip()]
            frags = parts if len(parts) > 1 else [plain.strip()]

            for n, article in enumerate(pair_fragments(frags)):
                if limit is not None and n >= limit:
                    break
                s["records"] += 1
                if article["identity"] is None:
                    s["no_identity"] += 1

                # Only fragments that survive ingestion's filter can ever be
                # cited, and only those need a map entry.
                kept = [(ai, frags[ai]) for ai in article["fragments"]
                        if len(strip_boilerplate(frags[ai])) >= MIN_ARTICLE_CHARS]
                if not kept:
                    s["skipped"] += 1
                    continue

                body_ai = max(ai for ai, _ in kept)
                rel = f"{org}/{rtf.stem}__{body_ai:04d}.pdf"
                dest = out_dir / rel

                for _, raw in kept:
                    substitutions.update(_sanitize(raw)[1])

                if dry_run:
                    s["written"] += 1
                    s["fragments"] += len(kept)
                    continue

                # Always render: `marks` is needed for the page map on every
                # run, and rendering is the cheap half (~1 ms). Only the write
                # is skipped when the file is already there.
                data, marks = _render_record(article["identity"], kept)
                if force or not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".pdf.part")
                    tmp.write_bytes(data)
                    tmp.replace(dest)
                    s["bytes"] += len(data)
                    if verify:
                        verify_scores.append(
                            _verify_pdf(data, "\n".join(r for _, r in kept)))

                s["written"] += 1
                for ai, raw in kept:
                    chunks = chunk_text(strip_boilerplate(raw))
                    entry = {"file": rel, "pages": chunk_pages(raw, chunks, marks[ai])}
                    entry.update(article["identity"] or {})
                    articles_map[f"{org}|thirdparty|{rtf.stem}|{ai}"] = entry
                    s["fragments"] += 1

    stats["_substitutions"] = dict(substitutions.most_common(20))
    stats["_substitution_total"] = sum(substitutions.values())
    if verify_scores:
        stats["_verify_min"] = min(verify_scores)
        stats["_verify_mean"] = sum(verify_scores) / len(verify_scores)
    return articles_map, stats


def save_article_sources(articles: dict) -> Path:
    """Write the map where load_article_sources() will find it."""
    from datetime import datetime
    ARTICLE_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTICLE_SOURCES_PATH.write_text(json.dumps(
        {"version": 1, "generated_at": datetime.now().isoformat(timespec="seconds"),
         "articles": articles}, ensure_ascii=False), encoding="utf-8")
    return ARTICLE_SOURCES_PATH
