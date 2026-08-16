"""Resolve a retrieved chunk back to the PDF it was extracted from, and render
the audit UI's `[excerpt N]` citations as links to that PDF.

The audit trail shows which spans an answer claims to rest on:

    ❌ [excerpt 3] "operates for the good of humanity (and can override any
                    for-profit interests)"

The bracketed number was a dead end — finding the document behind it meant
ticking the evidence checkbox, counting down the list, and hunting the file by
hand. This module closes that gap so the number becomes a link.

Three facts make the join non-trivial, and each maps to one function here:

  1. A chunk carries only a BARE BASENAME (`open AI charter.pdf`), never a
     path — see ingest.py's PDF_HEADER_RE, which reads it out of the corpus
     text file's `FILE:` header. The dataset, meanwhile, is a directory tree
     (`<root>/<Org>/<Category>/<name>.pdf`). `build_index` + `resolve` join the
     two on a normalized basename.
  2. Streamlit can only serve files beneath `<app dir>/static`, and its path
     check resolves symlinks before testing containment, so a junction into the
     dataset is rejected. `materialize` therefore hard-links (or copies) the
     cited PDF into that tree — lazily, so only documents actually cited are
     ever duplicated.
  3. The chunk's page number is not in the index (ingest keeps no offsets), so
     `load_chunk_pages` reads the offline-built map that scripts/09_build_pdf_pages.py
     produces, exactly as the app reads topics.load_chunk_topics().

Two kinds of evidence, resolved differently:

  PUBLISHED   the chunk names a PDF that exists in the dataset; we find it and
              publish it.
  THIRDPARTY  the chunk names an RTF bundle holding 500 press records, which
              identifies nothing. scripts/10_build_article_pdfs.py splits that
              bundle into one PDF per record ahead of time, and we look the
              chunk up in the map it leaves behind (see article_pdfs.py).

The second carries an extra caveat the UI must state: that link opens OUR
RENDERING of the press record the index was built from, not the publisher's
own page. There is no URL anywhere in the Nexis exports to link to instead.

Nothing here imports streamlit — the caching and the CLOUD_MODE guard live at
the call site in app.py, which keeps this module testable under pytest.

EPISTEMICS, stated once. `excerpt_citation` links the document the answer SAID
a span came from. It does not link a document the span was checked against:
rag_qa._verify_quotes deliberately verifies a span against ANY retrieved chunk
(a right span with a wrong number is sloppy citing, not fabrication), so a ✅
beside a link does not certify the span is in THAT pdf. `containing_ids` exists
to surface exactly that gap rather than let a link paper over it.
"""
import json
import os
import re
import shutil
from bisect import bisect_right
from pathlib import Path
from urllib.parse import quote

from .config import DATA_DIR, ORGS

# Where each lab's converted-PDF corpus sits inside the dataset tree. Separate
# from config.PUBLISHED_CORPUS, which still points at the pre-reorganization
# layout and is left alone because changing it re-mints every chunk id.
CORPUS_SUBDIR = "8 - Corpus"
CORPUS_FILENAME = "pdf_corpus.txt"

# Streamlit mounts <app dir>/static at this URL prefix when
# server.enableStaticServing is on (see .streamlit/config.toml).
APP_STATIC_URL_PREFIX = "/app/static"
STATIC_SUBDIR = "corpus"
# Generated press-record PDFs publish to their own subtree, so a Nexis headline
# can never collide with a corpus filename.
ARTICLES_SUBDIR = "articles"

# Streamlit's static route refuses anything at or above its own size cap
# (MAX_APP_STATIC_FILE_SIZE in streamlit/web/server/starlette/starlette_server_config.py)
# and returns 404. One corpus PDF is 161 MB, so this is close enough to matter:
# we check it ourselves and fall back to plain text, because a link that is
# guaranteed to 404 is worse than no link.
MAX_SERVABLE_BYTES = 200 * 1024 * 1024

# {chunk_id: 1-based page number}, built offline by scripts/09_build_pdf_pages.py.
PDF_PAGES_PATH = DATA_DIR / "pdf_pages.json"

# {fragment prefix: article record}, built offline by
# scripts/10_build_article_pdfs.py. Kept separate from PDF_PAGES_PATH rather
# than folded into it: that file's contract is a flat {chunk_id: int}, the two
# are built by different scripts from different halves of the dataset, and a
# missing one must degrade independently of the other.
ARTICLE_SOURCES_PATH = DATA_DIR / "article_sources.json"


def normalize_name(name: str) -> str:
    """Collapse a filename to a comparison key: lowercase, alphanumerics only.

    Deliberately aggressive, because the corpus text files were produced by an
    external converter that sanitized some names. Across all 98 documents there
    is exactly one such case — the corpus says `Inside Google_s AGI Strategy.pdf`
    where the disk says `Inside Google's AGI Strategy.pdf` — but the same
    normalization also absorbs any future stray hyphen, space, or case change,
    and it cannot introduce a collision that matters: the two basenames that DO
    collide in this dataset already collide exactly, and `resolve` separates
    them by org, not by name.
    """
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _norm_ws(s: str) -> str:
    """Whitespace-collapsed, lowercased text for substring comparison.

    Must stay identical to rag_qa._norm_ws — that function defines what
    "verified" means for a quote, and `containing_ids` below reports which
    chunk satisfies it. Duplicated rather than imported so this module stays
    dependency-free (rag_qa pulls in the Together client); tests assert the two
    agree so they cannot drift.
    """
    return re.sub(r"\s+", " ", s).strip().lower()


def build_index(root: Path | None) -> dict[str, tuple[str, ...]]:
    """Map every PDF under `root` by normalized basename.

    One walk of the dataset tree. Values are tuples because two basenames in
    this dataset are genuinely ambiguous (the same model card filed under both
    a lab folder and a loose top-level one); `resolve` breaks the tie. Paths are
    sorted so a tie broken by "first candidate" is at least deterministic.

    An unset or missing root yields {} — the feature is simply off, which is
    what a fresh checkout and the cloud deployment both want.
    """
    if root is None or not root.is_dir():
        return {}
    found: dict[str, list[str]] = {}
    for path in root.rglob("*.pdf"):
        if path.is_file():
            found.setdefault(normalize_name(path.name), []).append(str(path))
    return {key: tuple(sorted(paths)) for key, paths in found.items()}


def resolve(filename: str, org: str,
            index: dict[str, tuple[str, ...]]) -> Path | None:
    """Locate the dataset PDF a chunk's `filename` metadata refers to.

    On multiple candidates, prefer one filed under the chunk's own lab. That is
    not a heuristic here: every one of the 98 corpus documents resolves to a
    path inside its own org subtree, so the preference decides both real
    collisions correctly and never has to guess for the rest.
    """
    candidates = index.get(normalize_name(filename or ""))
    if not candidates:
        return None
    if len(candidates) > 1 and org:
        wanted = org.lower()
        scoped = [c for c in candidates
                  if any(part.lower() == wanted for part in Path(c).parts)]
        if scoped:
            candidates = tuple(scoped)
    return Path(candidates[0])


def static_url(org: str, name: str, page: int | None = None,
               subdir: str = STATIC_SUBDIR) -> str:
    """URL for a materialized PDF, percent-encoded segment by segment.

    Encoding is not optional: real corpus filenames contain spaces, an
    apostrophe, `&`, `$`, and an en-dash, and an unencoded `)` would terminate
    the surrounding markdown link early.
    """
    path = "/".join(quote(part) for part in (subdir, org, name))
    url = f"{APP_STATIC_URL_PREFIX}/{path}"
    return f"{url}#page={page}" if page else url


def _publish(src: Path, dest: Path) -> bool:
    """Put src's bytes at dest without ever exposing a partial file.

    A hard link goes straight to the final name: it is atomic by construction
    (the name either exists or it does not), costs no extra bytes, and gives a
    file-sync client nothing new to upload. It is tried FIRST and WITHOUT a
    temporary name on purpose — publishing a link via rename fails on this
    setup, because OneDrive refuses to rename a hard-linked file inside a
    synced folder ("the cloud operation cannot be performed on a file with
    incompatible hardlinks"), and the temporary it leaves behind then resists
    deletion too.

    A copy has no such guarantee, so it goes through a temporary and an atomic
    replace: a half-written 100 MB PDF must never be served.
    """
    try:
        os.link(src, dest)
        return True
    except OSError:
        pass
    tmp = dest.with_name(dest.name + ".part")
    try:
        tmp.unlink(missing_ok=True)
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def materialize(src: Path, org: str, static_root: Path,
                page: int | None = None,
                subdir: str = STATIC_SUBDIR) -> str | None:
    """Publish one PDF under `static_root` and return the URL that serves it.

    Streamlit resolves symlinks before checking that a requested file lives
    inside the static directory, so a junction pointing at the dataset is
    rejected — the bytes have to be reachable through a real path in the tree.
    Publishing happens once per document, and only for documents actually
    cited, so the tree stays a small fraction of the corpus.

    Returns None — never raises — on every failure path, because the caller is
    inside a render loop and the correct degradation is the plain text the UI
    showed before this feature existed.
    """
    dest = static_root / subdir / org / src.name
    try:
        size = src.stat().st_size
        if size >= MAX_SERVABLE_BYTES:
            return None  # the static route would 404 it
        if dest.is_file() and dest.stat().st_size == size:
            return static_url(org, src.name, page, subdir)  # already published
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if not _publish(src, dest):
        return None
    return static_url(org, src.name, page, subdir)


def load_chunk_pages() -> dict:
    """{chunk_id: page number}, or {} when the map has not been built.

    Mirrors topics.load_chunk_topics(): the expensive computation happens
    offline (scripts/09_build_pdf_pages.py, which needs pypdf) and the app only
    ever reads the small JSON result. Absent means links open at page 1.
    """
    if not PDF_PAGES_PATH.exists():
        return {}
    try:
        return json.loads(PDF_PAGES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Building the page map (LOCAL ONLY — needs pypdf and the raw dataset)
# ---------------------------------------------------------------------------
def _norm_alnum(text: str) -> tuple[str, list[int]]:
    """Alphanumeric-only projection of `text`, plus each kept char's origin.

    Page and chunk text come from two different extractors (the corpus files
    were converted externally, we read the PDFs with pypdf), so they disagree
    about whitespace, hyphenation, and ligatures. Stripping to alphanumerics
    makes them comparable; the index list carries the offsets back.
    """
    kept, origin = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            kept.append(ch.lower())
            origin.append(i)
    return "".join(kept), origin


# How far from a page's estimated start we will look for its actual text, and
# how much of the page to match on. The estimate is usually within a few dozen
# characters (see _page_starts), so the window only has to absorb drift.
_PAGE_SEARCH_WINDOW = 1500
_PAGE_PROBE_CHARS = 80

# Fraction of a document's pages that must be located by matching real text
# (rather than left on the proportional estimate) before we anchor links into
# it at all. Some documents' two extractions diverge far enough that no probe
# ever matches; there the estimate is unfalsifiable, so we decline to use it
# and the links open at page 1 — the document is still right, only the page is
# unknown, which is exactly what we should claim.
_PAGE_MIN_ANCHOR_RATE = 0.6


def _page_starts(doc_body: str,
                 page_texts: list[str]) -> tuple[list[int], float] | None:
    """Offset in the normalized document body where each PDF page begins.

    Anchoring on each page's opening text alone does not survive this corpus:
    pypdf emits the printed page number as the first characters of a page
    ("7highvolumedata..."), where the external converter put it elsewhere, so
    an opening-text search misses on every page of such a document and the
    whole alignment collapses onto one offset.

    So estimate first, then refine. The two extractions agree closely on total
    length (24,872 vs 24,881 normalized characters on one testimony PDF), which
    makes proportional cumulative page length a good first guess; we then look
    for real page text within a window around it, probing from a few points
    inside the page so a leading page number cannot spoil the match. A page
    that still can't be found keeps its estimate, and the sequence is forced
    monotonic so no chunk can be assigned to an earlier page than its
    predecessor.

    Returns (starts, anchor_rate), where anchor_rate is the fraction of pages
    placed by a real text match rather than left on the estimate — the caller's
    signal for whether this document's alignment can be trusted at all.

    Returns None when the PDF yields no extractable text at all. These are real
    and the dataset says so itself: the corpus manifests record `method: ocr`
    for 21 documents, and those are precisely the ones pypdf reads as empty —
    web pages saved as images, whose corpus text exists only because someone
    OCR'd them. There is nothing to align against, so this is a refusal rather
    than a fallback: guessing would send every link on the document to a wrong
    page, where declining just leaves it opening at page 1.
    """
    body_norm, _ = _norm_alnum(doc_body)
    page_norms = [_norm_alnum(t)[0] for t in page_texts]
    pdf_chars = sum(len(p) for p in page_norms)
    if not pdf_chars or not body_norm:
        return None

    scale = len(body_norm) / pdf_chars
    starts, cumulative, floor, anchored = [], 0, 0, 0
    for page in page_norms:
        estimate = int(cumulative * scale)
        # Probe from a few offsets into the page: the first skips whatever the
        # extractor put at the very top, the others give a short page or an
        # unlucky repeat another chance.
        found = None
        for frac in (0.05, 0.25, 0.5):
            offset = int(len(page) * frac)
            probe = page[offset:offset + _PAGE_PROBE_CHARS]
            if len(probe) < 20:
                continue
            lo = max(floor, estimate - _PAGE_SEARCH_WINDOW)
            at = body_norm.find(probe, lo, estimate + _PAGE_SEARCH_WINDOW
                                + len(probe))
            if at >= 0:
                # Step back to where the page began. This can go negative when
                # the page carries text the body does not — a printed page
                # number is exactly that — so clamp rather than treat a genuine
                # match as a miss.
                found = max(0, at - offset)
                break
        anchored += found is not None
        start = max(floor, found if found is not None else estimate)
        starts.append(start)
        floor = start
        cumulative += len(page)
    return starts, anchored / len(page_norms)


def published_corpora(root: Path | None) -> dict[str, Path]:
    """{org: path to its pdf_corpus.txt} for the current dataset layout.

    Keys must stay exactly ORGS: they are written verbatim into chunk metadata
    at ingest and are what the id prefix is built from.
    """
    if root is None:
        return {}
    return {org: root / org / CORPUS_SUBDIR / CORPUS_FILENAME for org in ORGS}


def build_page_map(corpora: dict, index: dict) -> tuple[dict, dict]:
    """Map every published chunk id to the PDF page it starts on.

    Reconstructs ids by re-running ingestion's own splitting and chunking over
    each lab's pdf_corpus.txt, so the ids match `{org}|published|{doc}|{chunk}`
    by construction — no vector index required (and none may exist yet). Chunks
    dropped at ingest by the near-duplicate filter simply produce map entries
    nothing ever looks up.

    Returns (map, stats). The map holds only chunks whose text was actually
    located in the document, so a miss means "no anchor", never a wrong page.
    `stats` reports the per-lab hit rate: the corpus text was produced by an
    unknown converter on another machine, so a low rate is a reason not to ship
    the file rather than something to paper over.
    """
    from pypdf import PdfReader  # local-only dep — see requirements-pdf.txt

    from .ingest import chunk_text, split_published_corpus

    page_map: dict[str, int] = {}
    stats: dict[str, dict] = {}
    for org, corpus_path in corpora.items():
        if not corpus_path.exists():
            print(f"[warn] missing published corpus for {org}: {corpus_path}")
            continue
        raw = corpus_path.read_text(encoding="utf-8", errors="ignore")
        hit = total = 0
        for doc_idx, (filename, body) in enumerate(split_published_corpus(raw)):
            src = resolve(filename, org, index)
            if src is None:
                print(f"[warn] {org}: no PDF on disk for {filename}")
                total += len(chunk_text(body))
                continue
            try:
                pages = [p.extract_text() or "" for p in PdfReader(str(src)).pages]
            except Exception as e:  # noqa: BLE001 — one bad PDF must not kill the run
                print(f"[warn] {org}: could not read {src.name}: {e}")
                total += len(chunk_text(body))
                continue
            aligned = _page_starts(body, pages)
            if aligned is None:
                print(f"[skip] {org}: {src.name} has no extractable text "
                      "(image-only scan?) — no page anchors for it")
                total += len(chunk_text(body))
                continue
            starts, anchor_rate = aligned
            if anchor_rate < _PAGE_MIN_ANCHOR_RATE:
                print(f"[skip] {org}: {src.name} aligned on only "
                      f"{anchor_rate:.0%} of pages — no page anchors for it")
                total += len(chunk_text(body))
                continue
            body_norm, _ = _norm_alnum(body)
            cursor = 0
            for ci, chunk in enumerate(chunk_text(body)):
                total += 1
                probe, _ = _norm_alnum(chunk)
                probe = probe[:120]
                at = body_norm.find(probe, cursor) if probe else -1
                if at < 0:
                    continue
                cursor = at
                # Last page starting at or before this chunk. `starts` is
                # monotonic by construction, so bisect gives the page number
                # directly (1-based, which is what #page= wants).
                page = bisect_right(starts, at) or 1
                page_map[f"{org}|published|{doc_idx}|{ci}"] = page
                hit += 1
        stats[org] = {"chunks": total, "mapped": hit,
                      "rate": (hit / total) if total else 0.0}
    return page_map, stats


def save_page_map(page_map: dict) -> Path:
    """Write the page map where load_chunk_pages() will find it."""
    PDF_PAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PDF_PAGES_PATH.write_text(json.dumps(page_map, ensure_ascii=False),
                              encoding="utf-8")
    return PDF_PAGES_PATH


def load_article_sources() -> dict:
    """{fragment prefix: article record}, or {} when never built.

    Mirrors load_chunk_pages(): the expensive work happens offline
    (scripts/10_build_article_pdfs.py, which needs fpdf2 and the raw press
    dumps) and the app only ever reads the small JSON result. Absent means
    third-party citations render as plain text, exactly as before the articles
    existed.
    """
    if not ARTICLE_SOURCES_PATH.exists():
        return {}
    try:
        data = json.loads(ARTICLE_SOURCES_PATH.read_text(encoding="utf-8"))
        return data.get("articles", {}) if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def article_key(chunk_id: str) -> tuple[str, int] | None:
    """Split a third-party chunk id into (fragment prefix, chunk index).

    `{org}|thirdparty|{stem}|{ai}|{ci}` — the article map is keyed by the first
    four fields, because only the page varies with `ci`. Anything that is not a
    well-formed third-party id returns None, so published ids and malformed
    bookkeeping both fall through to the published path.
    """
    if not isinstance(chunk_id, str):
        return None
    parts = chunk_id.split("|")
    if len(parts) != 5 or parts[1] != "thirdparty":
        return None
    try:
        return "|".join(parts[:4]), int(parts[4])
    except ValueError:
        return None


def resolve_article(chunk_id: str, sources: dict,
                    root: Path | None) -> tuple[Path, int, dict] | None:
    """Locate the generated PDF for a third-party chunk: (path, page, record).

    The `file` field is read from a JSON artifact and then joined to a
    directory whose contents get published into an unauthenticated static
    route, so a path escaping `root` is refused outright rather than trusted.
    A chunk with no page recorded opens at page 1 — the article is still right,
    only the offset within it is unknown.
    """
    key = article_key(chunk_id)
    if key is None or root is None:
        return None
    prefix, ci = key
    record = sources.get(prefix)
    if not isinstance(record, dict) or not record.get("file"):
        return None
    path = (root / record["file"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    if not path.is_file():
        return None
    page = record.get("pages", {}).get(str(ci), 1)
    return path, int(page or 1), record


def article_label(record: dict) -> str:
    """The article's headline, trimmed to fit a UI line."""
    headline = str(record.get("headline") or "").strip()
    return headline[:97] + "…" if len(headline) > 98 else headline


def article_sublabel(record: dict) -> str:
    """`Publication · Date`, for the evidence list."""
    return " · ".join(p for p in (str(record.get("publication") or "").strip(),
                                  str(record.get("date") or "").strip()) if p)


def md_escape(text: str) -> str:
    """Escape markdown link-text metacharacters.

    Required for article headlines and not for filenames: Nexis headlines
    genuinely contain brackets ("OpenAI [sic] files"), which would otherwise
    terminate the surrounding [label](url) early and leak raw url text.
    """
    return re.sub(r"([\\\[\]])", r"\\\1", text or "")


def _cited_id(excerpt, retrieved_ids) -> str | None:
    """Dereference the model's 1-based excerpt number to a chunk id.

    Same guards as metamorphic.ablation_target, for the same reason: the number
    is the model's own bookkeeping, so anything out of range is noise to be
    ignored rather than an error to raise.
    """
    if not isinstance(retrieved_ids, list):
        return None
    try:
        idx = int(excerpt) - 1
    except (TypeError, ValueError):
        return None
    return retrieved_ids[idx] if 0 <= idx < len(retrieved_ids) else None


def _document_key(chunk_id, chunk: dict) -> tuple:
    """What SOURCE a chunk came from, for "is this the same document?" tests.

    For a press record that is the article, not the RTF bundle — every chunk of
    every article in O1.RTF shares a filename, so filename alone would call 500
    unrelated articles one document.
    """
    key = article_key(chunk_id) if chunk_id else None
    if key is not None:
        return ("article", key[0])
    return ("file", chunk.get("org", ""), chunk.get("filename", ""))


def containing_ids(quote: str, retrieved_ids, evidence: dict) -> list[str]:
    """Which of a row's retrieved chunks actually contain `quote`.

    The same normalized substring test rag_qa._verify_quotes applies, run over
    the row's own handful of chunks — a few string operations per quote, so it
    is safe to call while rendering. Deliberately NOT the graded ladder in
    quote_provenance: that one uses difflib over 1400-character chunks and
    belongs in an offline stage, not in a render loop.
    """
    nq = _norm_ws(quote or "")
    if not nq or not isinstance(retrieved_ids, list):
        return []
    return [cid for cid in retrieved_ids
            if cid in evidence and nq in _norm_ws(evidence[cid].get("text", ""))]


def excerpt_citation(excerpt, retrieved_ids, evidence: dict, url_for) -> str:
    """Render `[excerpt N]` as markdown — a link to its PDF when one resolves.

    `url_for(filename, org, chunk_id)` is injected rather than imported so this
    module never learns about Streamlit's cache, and so tests can pass a stub.
    It takes the chunk id, not a page, because resolving a page means reading
    the offline page map — a cacheable concern that belongs on the app side.

    Returns markdown, never HTML: Streamlit gives every markdown link
    target="_blank" automatically, and keeping model-generated quote text out
    of raw HTML means it can never need escaping. The brackets are escaped so
    the unlinked fallback renders as the literal `[excerpt 3]` the UI has
    always shown, and the linked form puts the whole bracketed token inside the
    anchor.

    Falls back to that plain label whenever the chunk is missing from the index
    or has no resolvable document — a published PDF absent from disk, or a
    third-party chunk with no generated article. See the module docstring on
    what the link does and does not attest.
    """
    label = f"\\[excerpt {excerpt if excerpt is not None else '?'}\\]"
    cid = _cited_id(excerpt, retrieved_ids)
    if cid is None:
        return label
    chunk = evidence.get(cid)
    if not chunk:
        return label
    # No source-type test here on purpose: `url_for` already returns None for
    # anything it cannot resolve — a published doc missing from disk, a
    # third-party chunk with no generated article — and the plain label is the
    # fallback either way. Duplicating that policy here is what made adding
    # third-party support a two-place change the first time.
    url = url_for(chunk.get("filename", ""), chunk.get("org", ""), cid,
                  chunk.get("source_type", ""))
    return f"[{label}]({url})" if url else label


def misattribution_note(quote: str, excerpt, retrieved_ids, evidence: dict,
                        url_for) -> str:
    """Where a span was really found, when that is not where it was cited.

    A quote verifies against ANY retrieved chunk, so a ✅ can sit beside a
    citation pointing at a different document. Rather than let the new link
    quietly imply otherwise, say so: returns e.g.

        " · found in [our principles - Sam Altman.pdf](/app/static/...)"

    and returns "" in the ordinary case where the span is in the chunk it
    cites, is nowhere at all (an unverified quote — that is section 2's finding
    to report, not this function's), or has no other document to point at.
    """
    cited = _cited_id(excerpt, retrieved_ids)
    holders = containing_ids(quote, retrieved_ids, evidence)
    if not holders or cited in holders:
        return ""
    # Compare DOCUMENTS, not chunks. Consecutive chunks of one source overlap
    # by design, and a press record routinely contributes several — so a span
    # can sit in a different chunk of the very document we just linked. Saying
    # "found in X" beside a link to X is noise, not a finding.
    cited_doc = _document_key(cited, evidence.get(cited, {}))
    parts = []
    for cid in holders:
        chunk = evidence.get(cid, {})
        if _document_key(cid, chunk) == cited_doc:
            continue
        # Prefer the display label: for third-party evidence `filename` is the
        # RTF bundle ("O1.RTF"), which names 500 articles and identifies none
        # of them. `label` carries the headline when one is known.
        name = chunk.get("label") or chunk.get("filename", "")
        if not name:
            continue
        # Resolve with the filename, display with the label — they differ for
        # third-party evidence and the resolver wants the former.
        url = url_for(chunk.get("filename", ""), chunk.get("org", ""), cid,
                      chunk.get("source_type", ""))
        safe = md_escape(name)
        parts.append(f"[{safe}]({url})" if url else safe)
    # Duplicates are real: consecutive chunks of one document overlap by design,
    # so a span near a boundary is genuinely in two chunks of the same file.
    seen = list(dict.fromkeys(parts))
    return f" · found in {', '.join(seen)}" if seen else ""
