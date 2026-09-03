"""Topic-keyword semantic matching: does a topic's vocabulary survive into the answers?

The Topics tab discovers what the corpus is ABOUT (BERTopic clusters, each named
by its ten most distinctive c-TF-IDF terms). Those keywords were, until now,
purely descriptive labels — nothing scored them. This module scores them.

The question: when the pipeline answered a question from evidence belonging to
topic k, how much of topic k's vocabulary actually made it into the answer?
Verbatim, as an inflection, as a semantic neighbour, or not at all.

This is to keyword matching what quote_provenance.py is to quote verification.
Feature 2 asked "is this quote verbatim, yes/no"; feature 5 replaced the yes/no
with a graded ladder. keyword_agreement.py asks "does the answer contain this
keyword, yes/no", and its own docstring names the cost:

    it cannot see synonymy — an answer saying "government" earns no credit for
    a reference that says "state" — so its miss rate is structurally high

So here the yes/no becomes a four-rung ladder, cheapest first, first rung to
clear its bar wins:

    exact          the keyword occurs as a whole token (or, for a bigram, as
                   two adjacent tokens)                              -> 1.00
    morphological  some answer word shares its stem                  -> ratio
    semantic       some answer word is closer to it than tau_sem of
                   random corpus word pairs                          -> pctile
    absent         nothing cleared a bar                             -> 0

THE ONE HARD PROBLEM, and why this module is more than a cosine call. config.py
records the measurement that shapes everything here: e5 put a faithful reword at
0.849 and a wholly unrelated claim at 0.807 — 0.04 apart. Single WORDS are
worse; this corpus's whole vocabulary sits around 0.74-0.82. A raw cosine cannot
be shown to anyone as a "% match".

The semantic rung therefore scores a PERCENTILE against the distribution of
cosines between random word pairs drawn from this same corpus. That also
disposes of the domain-coherence objection: the corpus is entirely about AI
labs, so manager/sales genuinely IS fairly close in absolute cosine — but it is
a TYPICAL pair here and lands near the background median, while manager/
hierarchy lands in the tail. Corpus-relative calibration is what turns a 0.90
bar into a selective test instead of a rubber stamp.

Two reference arms are computed alongside, because a retention figure with no
referent is not a finding:

    ceiling C   the same keywords, exact rung only, against the retrieved CHUNKS.
                Keywords are derived FROM those chunks by c-TF-IDF, so this runs
                high by construction. It is the circularity concern stated as a
                number rather than caveated away.
    floor N     the full ladder against topics the row did NOT retrieve. If the
                measurement sits at the floor, the answers carry no more of the
                topic's vocabulary than a random topic's — and that is the
                finding.

Read as N <= V <= C.

Honest limitations (repeated verbatim in the UI):
  1. Circularity, as above — which is what the ceiling arm measures.
  2. e5 single-word vectors capture topical RELATEDNESS, not synonymy. manager
     and hierarchy are related, not synonyms. Antonyms embed close (permit /
     prohibit), so a high semantic score can mean the answer discusses the
     concept AND takes the opposite position. Read the matched word, never the
     score alone.
  3. Percentiles are readable, not absolute: a rank within THIS corpus under
     THIS embedding model. calibration.json records both.
  4. The keyword sets are vectorizer artifacts — ten c-TF-IDF terms under
     min_df=2, ngram_range=(1,2). Changing n_keywords changes every figure.
  5. The morphological rung is a prefix heuristic, not a stemmer: it over-fires
     on policy/police, under-fires on good/better. It never fabricates — the
     matched surface form is always stored.
  6. Per-row retention is noisy (ten keywords against one answer). Read the
     per-topic rollup.
  7. Attribution inherits the cross-tab's imprecision: a row's topics are the
     topics of ALL its retrieved chunks, including ones the answer never used
     (see the 1/k argument in topics.py).
  8. Every keyword counts equally. c-TF-IDF already filtered for
     distinctiveness once; weighting again would double-count it.
  9. This is a FOURTH judge, not a replacement. keyword_agreement.py is
     untouched: different keywords, different target, different question.

Cost: word vectors are cached to disk and append-only, so a first run costs a
few hundred embedded words and a rerun costs zero. Cosines are never cached —
they are a matmul over cached vectors, so retuning a threshold is free. Passing
vectors=None disables the semantic rung entirely, which makes the whole ladder
pure computation (what the offline tests use).

Outputs:
  data/lexicon/word_vectors.npz     corpus word vectors (float16, unit norm)
  data/lexicon/calibration.json     the background quantile grid + vocabulary
  <run>/topic_keywords/rows.jsonl   per answered row: every keyword's verdict
  <run>/topic_keywords/terms.jsonl  per (topic, keyword): retention + tiers
  <run>/topic_keywords/summary.json rates overall / per topic / per slice
"""
import bisect
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import runs
from . import topics as topics_mod
from .config import (
    CALIBRATION_BINS,
    CALIBRATION_GRID_POINTS,
    CALIBRATION_MIN_DF,
    CALIBRATION_VOCAB_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    KEYWORD_MAX_CANDIDATES,
    KEYWORD_MIN_WORD_CHARS,
    KEYWORD_MORPH_MIN_PREFIX,
    KEYWORD_MORPH_MIN_RATIO,
    KEYWORD_NULL_DRAWS,
    KEYWORD_NULL_SEED,
    KEYWORD_SEMANTIC_MAX_SCORE,
    KEYWORD_SEMANTIC_MIN_PERCENTILE,
)
from .embedding_agreement import _embed_batched
from .grounding import content_tokens
from .rag_qa import _norm_ws

LEXICON_DIR = DATA_DIR / "lexicon"
VECTORS_NAME = "word_vectors.npz"
CALIBRATION_NAME = "calibration.json"
OUT_DIR_NAME = "topic_keywords"

TIER_EXACT = "exact"
TIER_MORPH = "morphological"
TIER_SEMANTIC = "semantic"
TIER_ABSENT = "absent"
TIERS = (TIER_EXACT, TIER_MORPH, TIER_SEMANTIC, TIER_ABSENT)
# Ordering matters in one place only: a non-adjacent bigram takes the WEAKEST
# of its parts' tiers, and is then capped at morphological (the phrase itself
# did not occur, only its words did).
TIER_RANK = {TIER_EXACT: 3, TIER_MORPH: 2, TIER_SEMANTIC: 1, TIER_ABSENT: 0}
_TIER_BY_RANK = {v: k for k, v in TIER_RANK.items()}

_WORD_RE = re.compile(r"[a-z0-9]+")
# The npz stores words in a fixed-width unicode dtype so np.load needs no
# pickle; anything longer is not a word worth caching.
MAX_WORD_CHARS = 32
# Rows per block in the pairwise cosine sweep. 500 x 4000 float32 is ~8 MB,
# which is what keeps calibration inside the memory budget.
VECTOR_BLOCK = 500


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    """Ordered lowercased [a-z0-9]+ tokens, UNFILTERED.

    Deliberately not grounding.content_tokens: that drops stopwords and tokens
    of two characters or fewer, and BERTopic's vectorizer happily produces
    two-character keywords, so "ai" would be structurally unmatchable. A
    literal occurrence is a literal occurrence. The SEMANTIC rung does use
    content_tokens — there, a cosine against "the" is pure noise.
    """
    return _WORD_RE.findall(text.lower())


def candidate_words(text: str, limit: int = KEYWORD_MAX_CANDIDATES) -> list[str]:
    """The semantic rung's candidate set on the text side.

    Content words only, ranked by frequency within `text`, ties broken
    alphabetically so the set is reproducible, capped at `limit`. Unigrams
    only: embedding arbitrary answer bigrams multiplies the cached vocabulary
    for a gain the bigram-parts rule already approximates, and an e5 bigram
    vector is dominated by its head noun anyway.
    """
    keep = content_tokens(text)
    counts = Counter(t for t in _words(text) if t in keep)
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def morph_score(a: str, b: str) -> float | None:
    """Are these the same word in different clothes? The ratio if so, else None.

    Two conditions, BOTH required and both cheap:
      1. a shared leading stem of at least min(KEYWORD_MORPH_MIN_PREFIX,
         len(shorter)) characters
      2. a difflib ratio of at least KEYWORD_MORPH_MIN_RATIO

    Neither alone is safe: the prefix rule on its own accepts manage/mandate at
    three characters, the ratio on its own accepts pairs sharing no root. Words
    shorter than KEYWORD_MIN_WORD_CHARS are refused outright — a prefix rule
    over three-letter words is noise. See config.py for the calibration of both
    bars and for the documented false positives (policy/police).

    Identical words return 1.0 for well-definedness; the ladder never reaches
    this rung with them because the exact rung short-circuits first.
    """
    if a == b:
        return 1.0
    if len(a) < KEYWORD_MIN_WORD_CHARS or len(b) < KEYWORD_MIN_WORD_CHARS:
        return None
    if _shared_prefix(a, b) < min(KEYWORD_MORPH_MIN_PREFIX, len(a), len(b)):
        return None
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio if ratio >= KEYWORD_MORPH_MIN_RATIO else None


def _cosine_matrix(mat, vec):
    """Cosines of every row of a UNIT-normalized matrix against a unit vector.

    The vectorized form of embedding_agreement._cosine, which stays the scalar
    definition and is what the 1-vs-1 paths use. A test asserts the two agree
    to 1e-6, so there is still effectively one definition of the notion.
    """
    return np.asarray(mat, dtype=np.float32) @ np.asarray(vec, dtype=np.float32)


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


# ---------------------------------------------------------------------------
# Word vectors: a disk-backed, append-only cache
# ---------------------------------------------------------------------------
class WordVectors:
    """The analogue of quote_provenance._WindowEmbeddings, one level down.

    Keyed by WORD rather than chunk id, and PERSISTED, because the same few
    thousand words recur across every topic and every run — so a first run pays
    for a few hundred new words and every rerun pays nothing. `.calls` counts
    words sent to the embedder so a test can assert zero.

    readonly=True never embeds and never writes: the app browses the cache, and
    only the CLI stage is allowed to spend.

    Vectors are unit-normalized BEFORE being cast to float16, so every component
    lies in [-1, 1] where float16 carries ~3 decimal digits. Cosine error stays
    under 1e-3, an order of magnitude below the calibration grid's resolution,
    and the file stays ~8 MB instead of the ~40 MB JSON would cost.
    """

    def __init__(self, path: Path | None = None, *, embed_fn=None,
                 readonly: bool = False) -> None:
        self.path = Path(path) if path is not None else LEXICON_DIR / VECTORS_NAME
        self.readonly = readonly
        self.calls = 0
        self._embed = embed_fn
        self._index: dict[str, int] = {}
        self._rows: list = []
        self._dirty = False
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        with np.load(self.path, allow_pickle=False) as z:
            words = [str(w) for w in z["words"]]
            vecs = np.asarray(z["vectors"], dtype=np.float32)
        for i, w in enumerate(words):
            if w in self._index:
                continue
            self._index[w] = len(self._rows)
            self._rows.append(_unit(vecs[i]))

    def save(self) -> None:
        if self.readonly or not self._dirty or not self._rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._index, key=lambda w: self._index[w])
        words = np.array(ordered, dtype="U" + str(MAX_WORD_CHARS))
        vecs = np.stack(self._rows).astype(np.float16)
        np.savez_compressed(self.path, words=words, vectors=vecs)
        self._dirty = False

    # -- access ------------------------------------------------------------
    def known(self) -> set:
        return set(self._index)

    def get(self, word: str):
        i = self._index.get(word)
        return None if i is None else self._rows[i]

    def matrix(self, words: list):
        """(kept_words, unit matrix). Unknown words are dropped, not embedded."""
        kept = [w for w in words if w in self._index]
        if not kept:
            return [], np.zeros((0, 0), dtype=np.float32)
        return kept, np.stack([self._rows[self._index[w]] for w in kept])

    def ensure(self, words) -> int:
        """Embed only the words not already cached. Returns how many were new."""
        misses = []
        seen = set()
        for w in words:
            if (w and w not in self._index and w not in seen
                    and len(w) <= MAX_WORD_CHARS):
                seen.add(w)
                misses.append(w)
        if not misses or self.readonly:
            return 0
        # Module-global lookup at CALL time, so a test can monkeypatch
        # tk._embed_batched after the cache was constructed.
        embed_fn = self._embed or _embed_batched
        vecs = embed_fn(misses)
        for w, v in zip(misses, vecs):
            self._index[w] = len(self._rows)
            self._rows.append(_unit(v))
        self.calls += len(misses)
        self._dirty = True
        return len(misses)


# ---------------------------------------------------------------------------
# Percentile calibration
# ---------------------------------------------------------------------------
class Calibration:
    """An ascending quantile grid of corpus word-pair cosines.

    Stores CALIBRATION_GRID_POINTS quantiles rather than the ~8M cosines they
    summarize, so the file is small and the lookup is a bisect.
    """

    def __init__(self, grid: list, meta: dict | None = None) -> None:
        self.grid = [float(g) for g in grid]
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path | None = None):
        path = Path(path) if path is not None else LEXICON_DIR / CALIBRATION_NAME
        if not path.exists():
            return None
        meta = json.loads(path.read_text(encoding="utf-8"))
        grid = meta.get("quantile_grid") or []
        return cls(grid, meta) if grid else None

    def percentile(self, cosine: float) -> float:
        """Where this cosine falls in the background distribution, in [0, 1].

        grid[i] is the (i / (n-1))-quantile, so the answer is the largest i
        whose grid value this cosine reaches. Bisect, then clamp: ten
        comparisons, no interpolation. Below the grid reads 0.0, above it
        reads 1.0 — the clamp is what keeps a cosine above every observed pair
        from scoring more than 100%.
        """
        n = len(self.grid)
        if n < 2:
            return 0.0
        i = bisect.bisect_right(self.grid, float(cosine)) - 1
        return min(max(i, 0), n - 1) / (n - 1)

    def vocabulary(self) -> list:
        """The words the background was built from — the explorer's candidates."""
        return [w for w, _ in self.meta.get("vocabulary", [])]

    def document_frequency(self) -> dict:
        return {w: int(n) for w, n in self.meta.get("vocabulary", [])}


def corpus_vocabulary(min_df: int = CALIBRATION_MIN_DF,
                      limit: int = CALIBRATION_VOCAB_SIZE) -> list:
    """The corpus's most frequent content words, by document frequency.

    Pages Chroma for DOCUMENTS ONLY — the light half of
    topics.load_corpus_vectors(), so this never materializes embeddings. A set
    per chunk, so a word repeated inside one chunk counts once. Kept thin and
    separately monkeypatchable so build_calibration can be tested over a
    fabricated vocabulary on a machine with no index.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    # An absent index is the single most likely way this stage is run wrong, so
    # it gets an actionable message rather than a Chroma traceback — the same
    # courtesy topics.py extends for a missing topic model.
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception as e:  # noqa: BLE001 — the message is the point
        raise SystemExit(
            f"no vector index at {CHROMA_DIR} (collection '{COLLECTION_NAME}' "
            f"not found: {type(e).__name__}).\n"
            "Calibration is built FROM the corpus, so it needs the index that "
            "the corpus was embedded into. Either:\n"
            "  - build it here:  python scripts/01_ingest.py\n"
            "  - or copy an existing data/chroma/ into this checkout\n"
            "Nothing else in this stage runs without it."
        ) from e
    total = col.count()
    if total == 0:
        raise SystemExit(
            f"the index at {CHROMA_DIR} is empty (0 chunks). Ingest the corpus "
            "before calibrating: python scripts/01_ingest.py"
        )
    df: Counter = Counter()
    for offset in range(0, total, topics_mod.FETCH_PAGE):
        page = col.get(limit=topics_mod.FETCH_PAGE, offset=offset,
                       include=["documents"])
        for doc in page["documents"]:
            df.update(content_tokens(doc or ""))
    kept = [(w, n) for w, n in df.items()
            if n >= min_df and KEYWORD_MIN_WORD_CHARS <= len(w) <= MAX_WORD_CHARS]
    # The alphabetical tie-break is what makes the vocabulary reproducible.
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return kept[:limit]


def build_calibration(*, vocab_size: int = CALIBRATION_VOCAB_SIZE,
                      min_df: int = CALIBRATION_MIN_DF,
                      bins: int = CALIBRATION_BINS,
                      grid_points: int = CALIBRATION_GRID_POINTS,
                      vectors: WordVectors | None = None,
                      vocabulary: list | None = None,
                      path: Path | None = None) -> dict:
    """Build the background distribution of corpus word-pair cosines.

    ALL pairs, not a sample: the cosine matrix is swept in row blocks and
    accumulated into a fixed histogram, so ~8M pairs cost bounded memory.
    Computing exhaustively removes a random seed from the design entirely —
    same corpus and same model give the same file, byte for byte.
    """
    vocab = vocabulary if vocabulary is not None else corpus_vocabulary(
        min_df=min_df, limit=vocab_size)
    if not vocab:
        raise SystemExit("no corpus vocabulary — ingest the corpus first")
    df_by_word = {w: n for w, n in vocab}
    words = [w for w, _ in vocab]
    vectors = vectors if vectors is not None else WordVectors()
    print(f"vocabulary: {len(words)} words (df >= {min_df})")
    new = vectors.ensure(words)
    print(f"embedded {new} new words ({len(words) - new} already cached)")
    vectors.save()

    kept, mat = vectors.matrix(words)
    n = len(kept)
    if n < 2:
        raise SystemExit("need at least two embedded words to calibrate")

    hist = np.zeros(bins, dtype=np.int64)
    for start in range(0, n, VECTOR_BLOCK):
        block = mat[start:start + VECTOR_BLOCK] @ mat.T
        for r in range(block.shape[0]):
            # Upper triangle only: each unordered pair counted exactly once,
            # self-similarity (always 1.0) excluded.
            row = block[r, start + r + 1:]
            if row.size:
                hist += np.histogram(row, bins=bins, range=(-1.0, 1.0))[0]

    cum = np.cumsum(hist)
    total_pairs = int(cum[-1])
    edges = np.linspace(-1.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    targets = np.maximum(np.linspace(0.0, 1.0, grid_points) * total_pairs, 1.0)
    idx = np.clip(np.searchsorted(cum, targets, side="left"), 0, bins - 1)
    grid = np.maximum.accumulate(centers[idx])

    def _q(p: float) -> float:
        return round(float(grid[int(round(p * (grid_points - 1)))]), 4)

    out = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL,
        "vocab_size": n,
        "min_df": min_df,
        "bins": bins,
        "grid_points": grid_points,
        "n_pairs": total_pairs,
        "cos_min": _q(0.0), "cos_p50": _q(0.5),
        "cos_p90": _q(0.9), "cos_p99": _q(0.99), "cos_max": _q(1.0),
        "quantile_grid": [round(float(g), 6) for g in grid],
        # Carried so the neighborhood explorer has a bounded candidate
        # vocabulary (and its document frequencies) without needing Chroma —
        # which is what lets it run in the container.
        "vocabulary": [[w, int(df_by_word.get(w, 0))] for w in kept],
        "note": ("Percentiles are a rank within THIS corpus under THIS "
                 "embedding model. Refit the corpus or change the model and "
                 "every number moves; both are recorded here so a score can "
                 "never be read out of context."),
    }
    path = Path(path) if path is not None else LEXICON_DIR / CALIBRATION_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print("\n=== Calibration ===")
    print(f"pairs: {total_pairs:,} over {n} words")
    print(f"cosine band: min {out['cos_min']}  p50 {out['cos_p50']}  "
          f"p90 {out['cos_p90']}  p99 {out['cos_p99']}  max {out['cos_max']}")
    print("That whole band is why a raw cosine cannot be shown as a percentage.")
    print(f"outputs: {path.parent}")
    return out


# ---------------------------------------------------------------------------
# The ladder (pure computation when vectors/calibration are None)
# ---------------------------------------------------------------------------
class _Ctx:
    """One text, prepared once so N keywords can be scored against it cheaply.

    The candidate matrix is built once per text rather than once per keyword;
    without that, scoring 50 keywords would rebuild the same 120x1024 matrix
    50 times.
    """

    __slots__ = ("words", "norm", "cands", "cand_words", "cand_matrix",
                 "vectors", "calibration")

    def __init__(self, text: str, *, vectors=None, calibration=None,
                 keywords=None) -> None:
        self.words = set(_words(text))
        self.norm = _norm_ws(text)
        self.cands = candidate_words(text)
        self.vectors = vectors
        self.calibration = calibration
        self.cand_words: list = []
        self.cand_matrix = None
        if vectors is not None and calibration is not None:
            parts = {p for k in (keywords or []) for p in str(k).split()}
            vectors.ensure(sorted(parts | set(self.cands)))
            self.cand_words, self.cand_matrix = vectors.matrix(self.cands)


def _verdict(tier: str = TIER_ABSENT, score: float = 0.0, rule=None,
             matched=None, cosine=None, near_miss=None) -> dict:
    """One keyword's verdict.

    `score` is what retention averages, so a keyword that cleared NO bar
    contributes exactly 0 — otherwise "dropped" vocabulary would still earn
    most of a point and dropped_share would contradict retention. How close it
    came is kept separately in `near_miss`, which is diagnostic only: it is
    what tells someone retuning the bar whether a miss was nowhere near or a
    hair under.
    """
    return {
        "tier": tier,
        "score": round(float(score), 4),
        "rule": rule,
        "matched": matched,
        "cosine": None if cosine is None else round(float(cosine), 4),
        "near_miss": None if near_miss is None else round(float(near_miss), 4),
    }


def _score_unigram(kw: str, ctx: _Ctx) -> dict:
    """The three rungs, for a single-token keyword."""
    if kw in ctx.words:
        return _verdict(TIER_EXACT, 1.0, "token", kw)

    best_ratio, best_word = 0.0, None
    for w in ctx.words:
        m = morph_score(kw, w)
        if m is not None and m > best_ratio:
            best_ratio, best_word = m, w
    if best_word is not None:
        return _verdict(TIER_MORPH, best_ratio, "shared_stem", best_word)

    # Semantic. Disabled outright without BOTH a vector cache and a
    # calibration: a raw cosine rendered as a percentage would misrepresent
    # the whole feature (see the module docstring).
    if ctx.vectors is None or ctx.calibration is None or not ctx.cand_words:
        return _verdict()
    kvec = ctx.vectors.get(kw)
    if kvec is None:
        return _verdict()
    sims = _cosine_matrix(ctx.cand_matrix, kvec)
    i = int(np.argmax(sims))
    cos = float(sims[i])
    pct = ctx.calibration.percentile(cos)
    if pct >= KEYWORD_SEMANTIC_MIN_PERCENTILE:
        return _verdict(TIER_SEMANTIC, min(pct, KEYWORD_SEMANTIC_MAX_SCORE),
                        "embedding_percentile", ctx.cand_words[i], cos)
    # Scores 0 — it cleared no bar. The percentile rides along as `near_miss`
    # so "nowhere close" stays distinguishable from "just under the bar".
    return _verdict(TIER_ABSENT, 0.0, None, None, cos, near_miss=pct)


_PHRASE_CACHE: dict = {}


def _phrase_pattern(parts: list):
    key = " ".join(parts)
    pat = _PHRASE_CACHE.get(key)
    if pat is None:
        pat = re.compile(r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b")
        _PHRASE_CACHE[key] = pat
    return pat


def _score_phrase(parts: list, ctx: _Ctx) -> dict:
    """Multi-token keywords ("export controls").

    Adjacency is what makes it the phrase, so an adjacent occurrence in the
    normalized text is the only route to `exact`. Otherwise each part is scored
    on its own and the phrase takes the MINIMUM — the phrase is present only to
    the extent that BOTH concepts are; a mean would let "controls" alone carry
    "export controls". The result is then capped at `morphological`, because
    the phrase itself never occurred, only its words did.
    """
    if _phrase_pattern(parts).search(ctx.norm):
        return _verdict(TIER_EXACT, 1.0, "phrase_adjacent", " ".join(parts))
    results = [_score_unigram(p, ctx) for p in parts]
    score = min(r["score"] for r in results)
    rank = min(min(TIER_RANK[r["tier"]] for r in results), TIER_RANK[TIER_MORPH])
    # Deduplicated, order preserved: both parts can match the same answer word,
    # and "humanity|humanity" reads as a bug rather than as evidence.
    seen, hits = set(), []
    for r in results:
        if r["matched"] and r["matched"] not in seen:
            seen.add(r["matched"])
            hits.append(r["matched"])
    misses = [r["near_miss"] for r in results if r["near_miss"] is not None]
    return _verdict(_TIER_BY_RANK[rank], score, "bigram_parts",
                    "|".join(hits) or None,
                    near_miss=min(misses) if misses else None)


def _score_one(keyword: str, ctx: _Ctx) -> dict:
    parts = _words(str(keyword))
    if not parts:
        return _verdict()
    out = _score_unigram(parts[0], ctx) if len(parts) == 1 else _score_phrase(parts, ctx)
    return {"keyword": keyword, **out}


def score_keyword(keyword: str, text: str, *, vectors=None, calibration=None) -> dict:
    """Place one keyword on the ladder against one text.

    Standalone (it prepares its own context), so it is the convenient entry
    point for tests and one-off inspection. score_text is the batched form and
    is what the driver uses.
    """
    return _score_one(keyword, _Ctx(text, vectors=vectors, calibration=calibration,
                                    keywords=[keyword]))


def score_text(keywords: list, text: str, *, vectors=None, calibration=None) -> list:
    """Place every keyword on the ladder against one text, preparing it once."""
    ctx = _Ctx(text, vectors=vectors, calibration=calibration, keywords=keywords)
    return [_score_one(k, ctx) for k in keywords]


def _exact_share(keywords: list, text: str) -> float:
    """Share of keywords occurring VERBATIM in a text. No embeddings, ever.

    The ceiling arm: the same keywords against the chunks they were derived
    from. Exact rung only on purpose — the point is the tautological baseline,
    not a graded score.
    """
    if not keywords:
        return 0.0
    words = set(_words(text))
    norm = _norm_ws(text)
    hits = 0
    for k in keywords:
        parts = _words(str(k))
        if not parts:
            continue
        if len(parts) == 1:
            hits += 1 if parts[0] in words else 0
        else:
            hits += 1 if _phrase_pattern(parts).search(norm) else 0
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Rollup
# ---------------------------------------------------------------------------
def row_topics(retrieved_ids, chunk_topics: dict) -> list:
    """The distinct clustered topics a row's evidence belonged to, in order.

    The outlier topic (-1) is excluded — it is not a topic and has no keywords.
    Chunk ids missing from the map are skipped: a reingest after fitting
    renames ids, and that must degrade rather than crash (topics.py takes the
    same position).
    """
    seen, out = set(), []
    for cid in retrieved_ids or []:
        t = chunk_topics.get(cid)
        if t is None:
            continue
        t = int(t)
        if t == topics_mod.OUTLIER_TOPIC or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _shares(results: list) -> dict:
    """Retention plus the four tier shares (which partition, so they sum to 1)."""
    n = len(results)
    if not n:
        return {"retention": 0.0, "verbatim_share": 0.0, "morph_share": 0.0,
                "semantic_share": 0.0, "dropped_share": 0.0, "semantic_lift": 0.0}
    counts = Counter(r["tier"] for r in results)
    retention = sum(r["score"] for r in results) / n
    verbatim = counts[TIER_EXACT] / n
    return {
        "retention": round(retention, 4),
        "verbatim_share": round(verbatim, 4),
        "morph_share": round(counts[TIER_MORPH] / n, 4),
        "semantic_share": round(counts[TIER_SEMANTIC] / n, 4),
        "dropped_share": round(counts[TIER_ABSENT] / n, 4),
        # The headline: exactly what an exact-match judge cannot see.
        "semantic_lift": round(retention - verbatim, 4),
    }


def _null_topics(row: dict, retrieved: list, all_topics: list,
                 draws: int = KEYWORD_NULL_DRAWS) -> list:
    """Topics this row did NOT retrieve, drawn reproducibly.

    Seeded from the row's own identifiers rather than from a shared stream, so
    the draw is stable regardless of row order, resumption, or parallelism.
    """
    # Sorted, so the draw depends on the row's seed alone and not on the
    # order the caller happened to hand the topics over in.
    pool = sorted(t for t in all_topics if t not in set(retrieved))
    if not pool:
        return []
    rng = random.Random(
        f"{KEYWORD_NULL_SEED}:{row.get('org')}:{row.get('source_type')}:{row.get('qid')}")
    return rng.sample(pool, min(draws, len(pool)))


def score_row(row: dict, topic_keywords: dict, chunk_topics: dict, *,
              vectors=None, calibration=None, chunk_texts: dict | None = None,
              all_topics: list | None = None) -> dict:
    """Score one answered row against the keywords of its evidence's topics."""
    base = {
        "org": row.get("org"), "source_type": row.get("source_type"),
        "qid": row.get("qid"), "category": row.get("category"),
        "variant": row.get("variant") or 1,
    }
    ts = row_topics(row.get("retrieved_ids"), chunk_topics)
    if not ts:
        return {**base, "topics": [], "n_keywords": 0, "skipped": "no_topics",
                "keywords": []}

    pairs, seen = [], set()
    for t in ts:
        for k in topic_keywords.get(t, []):
            if k not in seen:
                seen.add(k)
                pairs.append((t, k))
    if not pairs:
        return {**base, "topics": ts, "n_keywords": 0, "skipped": "no_keywords",
                "keywords": []}

    answer = row.get("answer") or ""
    results = score_text([k for _, k in pairs], answer,
                         vectors=vectors, calibration=calibration)
    for (t, _k), r in zip(pairs, results):
        r["topic"] = t

    out = {**base, "topics": ts, "n_keywords": len(pairs), "skipped": None,
           **_shares(results)}

    # Ceiling: the same keywords against the chunks they were derived from.
    # Exact rung only, so it costs nothing.
    if chunk_texts is not None:
        evidence = " ".join(chunk_texts.get(cid, "")
                            for cid in (row.get("retrieved_ids") or []))
        out["verbatim_ceiling"] = round(
            _exact_share([k for _, k in pairs], evidence), 4)
    else:
        out["verbatim_ceiling"] = None

    # Floor: the full ladder against topics this row did not retrieve.
    if all_topics:
        nulls = _null_topics(row, ts, all_topics)
        scores = []
        for t in nulls:
            kws = topic_keywords.get(t, [])
            if not kws:
                continue
            scores.append(_shares(score_text(kws, answer, vectors=vectors,
                                             calibration=calibration))["retention"])
        out["retention_null"] = round(sum(scores) / len(scores), 4) if scores else None
        out["null_topics"] = nulls
    else:
        out["retention_null"] = None
        out["null_topics"] = []

    out["keywords"] = [
        {"topic": r["topic"], "keyword": r["keyword"], "tier": r["tier"],
         "score": r["score"], "rule": r["rule"], "matched": r["matched"],
         "cosine": r["cosine"], "near_miss": r["near_miss"]}
        for r in results
    ]
    return out


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _slice(items: list) -> dict:
    """One breakdown's stats. Same key discipline as keyword_agreement._slice."""
    scored = [r for r in items if not r.get("skipped")]
    out = {
        "n": len(items),
        "n_scored": len(scored),
        "n_no_topics": sum(1 for r in items if r.get("skipped") == "no_topics"),
    }
    if not scored:
        return out
    out.update({
        "mean_retention": _mean(r["retention"] for r in scored),
        "mean_verbatim_share": _mean(r["verbatim_share"] for r in scored),
        "mean_morph_share": _mean(r["morph_share"] for r in scored),
        "mean_semantic_share": _mean(r["semantic_share"] for r in scored),
        "mean_dropped_share": _mean(r["dropped_share"] for r in scored),
        "mean_semantic_lift": _mean(r["semantic_lift"] for r in scored),
        "mean_verbatim_ceiling": _mean(r.get("verbatim_ceiling") for r in scored),
        "mean_retention_null": _mean(r.get("retention_null") for r in scored),
    })
    if out["mean_retention"] is not None and out["mean_retention_null"] is not None:
        out["retention_over_null"] = round(
            out["mean_retention"] - out["mean_retention_null"], 4)
    else:
        out["retention_over_null"] = None
    return out


def build_terms(rows: list, labels: dict | None = None) -> list:
    """Per (topic, keyword) rollup — the drill-down that proves the feature.

    R_k = mean score of that keyword across the answered rows that retrieved
    evidence from its topic, plus the tier histogram and the words that most
    often produced the match.
    """
    acc: dict = defaultdict(lambda: {"scores": [], "tiers": Counter(),
                                     "matches": Counter()})
    for row in rows:
        if row.get("skipped"):
            continue
        for k in row.get("keywords", []):
            a = acc[(k["topic"], k["keyword"])]
            a["scores"].append(k["score"])
            a["tiers"][k["tier"]] += 1
            if k["matched"]:
                a["matches"][k["matched"]] += 1
    labels = labels or {}
    out = []
    for (topic, keyword), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append({
            "topic": topic,
            "label": labels.get(topic, str(topic)),
            "keyword": keyword,
            "n_rows": len(a["scores"]),
            "retention": round(sum(a["scores"]) / len(a["scores"]), 4),
            "tiers": {t: a["tiers"].get(t, 0) for t in TIERS},
            "top_matches": a["matches"].most_common(3),
        })
    return out


def build_by_topic(rows: list, labels: dict | None = None) -> list:
    """Per-topic retention, weighted so each answered row is one observation.

    Row-equal weighting, not keyword-instance weighting: otherwise a topic
    retrieved by many rows would dominate its own average.
    """
    acc: dict = defaultdict(list)
    for row in rows:
        if row.get("skipped"):
            continue
        by_topic: dict = defaultdict(list)
        for k in row.get("keywords", []):
            by_topic[k["topic"]].append(k)
        for topic, ks in by_topic.items():
            acc[topic].append((_shares(ks), row.get("verbatim_ceiling"),
                               row.get("retention_null")))
    labels = labels or {}
    out = []
    for topic, entries in acc.items():
        shares = [e[0] for e in entries]
        out.append({
            "topic": topic,
            "label": labels.get(topic, str(topic)),
            "n_rows": len(entries),
            "retention": _mean(s["retention"] for s in shares),
            "verbatim_share": _mean(s["verbatim_share"] for s in shares),
            "morph_share": _mean(s["morph_share"] for s in shares),
            "semantic_share": _mean(s["semantic_share"] for s in shares),
            "dropped_share": _mean(s["dropped_share"] for s in shares),
            "semantic_lift": _mean(s["semantic_lift"] for s in shares),
            "verbatim_ceiling": _mean(e[1] for e in entries),
            "retention_null": _mean(e[2] for e in entries),
        })
    out.sort(key=lambda r: (-(r["retention"] or 0.0), r["topic"]))
    return out


def summarize(rows: list, *, meta: dict, labels: dict | None = None) -> dict:
    """Overall / per-topic / per-category / per-(org, source) rollup."""
    by_category = {
        c: _slice([r for r in rows if r.get("category") == c])
        for c in sorted({r.get("category") for r in rows if r.get("category")})
    }
    by_pair = {}
    for key in sorted({(r.get("org"), r.get("source_type")) for r in rows}):
        by_pair["|".join(str(k) for k in key)] = _slice(
            [r for r in rows if (r.get("org"), r.get("source_type")) == key])
    return {
        **meta,
        "overall": _slice(rows),
        "by_topic": build_by_topic(rows, labels),
        "by_category": by_category,
        "by_org_source": by_pair,
        "note": ("Topic keywords are derived FROM the corpus chunks by "
                 "c-TF-IDF, so matching them against those same chunks is "
                 "near-tautological — that is what verbatim_ceiling measures, "
                 "and why it is reported rather than hidden. The non-trivial "
                 "comparison is against the ANSWERS. retention_null is the "
                 "same ladder against topics the row never retrieved: read "
                 "null <= retention <= ceiling. Semantic scores are "
                 "percentiles of this corpus's word-pair cosines under this "
                 "embedding model, and capture topical RELATEDNESS, not "
                 "synonymy — always read the matched word alongside the "
                 "score."),
    }


# ---------------------------------------------------------------------------
# Neighborhood explorer
# ---------------------------------------------------------------------------
def neighbors(word: str, *, vectors: WordVectors, calibration=None,
              vocabulary: list | None = None, top_n: int = 25) -> list:
    """The nearest corpus-vocabulary words to `word`, with calibrated scores.

    The interpretability proof: it is where "manager -> hierarchy outranks
    sales" is visible, and where the contrast between a raw-cosine band of a
    few hundredths and a percentile axis spanning tens of points makes the case
    for calibrating at all.

    Candidates default to the calibration vocabulary — the same words the
    background was built from, so a neighbour's percentile is exactly
    comparable to a scoring-time one, and all of them are already cached (zero
    API calls, so this works in the container with no key). An unknown query
    word returns [] rather than embedding on demand: the app is read-only over
    the cache.

    `tier` labels each neighbour the way the ladder would treat it, so an
    inflection is visibly a different KIND of neighbour from a related concept.
    """
    qvec = vectors.get(word)
    if qvec is None:
        return []
    if vocabulary is None:
        vocabulary = (calibration.vocabulary() if calibration is not None
                      else sorted(vectors.known()))
    kept, mat = vectors.matrix(list(vocabulary))
    if not kept:
        return []
    sims = _cosine_matrix(mat, qvec)
    df = calibration.document_frequency() if calibration is not None else {}
    order = np.argsort(-sims)[:top_n]
    out = []
    for i in order:
        i = int(i)
        w, cos = kept[i], float(sims[i])
        if w == word:
            tier = TIER_EXACT
        elif morph_score(word, w) is not None:
            tier = TIER_MORPH
        else:
            tier = TIER_SEMANTIC
        out.append({
            "word": w,
            "cosine": round(cos, 4),
            "percentile": (round(calibration.percentile(cos), 4)
                           if calibration is not None else None),
            "tier": tier,
            "df": df.get(w),
        })
    return out


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------
def _load_committed(run_id: str) -> list:
    path = runs.run_paths(run_id)["per_question"]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("abstain"):
                rows.append(r)
    return rows


def _fetch_chunk_texts(rows: list) -> dict:
    """Chunk id -> text for the ceiling arm. Degrades to {} without an index."""
    ids = sorted({cid for r in rows for cid in (r.get("retrieved_ids") or [])})
    if not ids:
        return {}
    try:
        from .retriever import Retriever
        chunks = Retriever().get_by_ids(ids)
    except Exception as e:  # noqa: BLE001 — a missing index must not kill the stage
        print(f"note: verbatim ceiling unavailable ({type(e).__name__}: {e})")
        return {}
    return {c.id: c.text for c in chunks}


def run_topic_keywords(run_id: str | None = None, *, embeddings: bool = True,
                       ceiling: bool = True) -> dict:
    """Score a saved run's answers against its evidence's topic keywords."""
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run profiles first")
    info = topics_mod.load_topic_info()
    chunk_topics = topics_mod.load_chunk_topics()
    if info is None or chunk_topics is None:
        raise SystemExit(
            "no topic model on disk — run `scripts/07_run_topics.py fit` first")

    topic_keywords = {r["topic"]: r.get("keywords", [])
                      for r in info["topics"] if not r.get("is_outlier")}
    labels = {r["topic"]: r.get("label", str(r["topic"])) for r in info["topics"]}
    all_topics = sorted(topic_keywords)

    rows = _load_committed(run_id)
    if not rows:
        raise SystemExit(f"run {run_id} has no committed rows to check")

    calibration = Calibration.load() if embeddings else None
    vectors = WordVectors() if (embeddings and calibration is not None) else None
    if embeddings and calibration is None:
        print("no calibration on disk — the semantic rung is DISABLED. Build it "
              "with `scripts/13_run_topic_keywords.py calibrate`.")
    cached_before = len(vectors.known()) if vectors is not None else 0

    chunk_texts = _fetch_chunk_texts(rows) if ceiling else None

    scored = []
    for r in rows:
        scored.append(score_row(r, topic_keywords, chunk_topics,
                                vectors=vectors, calibration=calibration,
                                chunk_texts=chunk_texts, all_topics=all_topics))
    if vectors is not None:
        vectors.save()

    out_dir = runs.run_dir(run_id) / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    terms = build_terms(scored, labels)
    with open(out_dir / "terms.jsonl", "w", encoding="utf-8") as f:
        for t in terms:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    meta = {
        "run_id": run_id,
        "fitted_at": info.get("fitted_at"),
        "seed": info.get("seed"),
        "calibrated": calibration is not None,
        "calibration": ({k: calibration.meta.get(k)
                         for k in ("built_at", "embedding_model", "vocab_size",
                                   "n_pairs", "cos_p50", "cos_p90", "cos_p99")}
                        if calibration is not None else None),
        "thresholds": {
            "morph_min_prefix": KEYWORD_MORPH_MIN_PREFIX,
            "morph_min_ratio": KEYWORD_MORPH_MIN_RATIO,
            "semantic_min_percentile": KEYWORD_SEMANTIC_MIN_PERCENTILE,
            "semantic_max_score": KEYWORD_SEMANTIC_MAX_SCORE,
            "max_candidates": KEYWORD_MAX_CANDIDATES,
            "null_draws": KEYWORD_NULL_DRAWS,
            "null_seed": KEYWORD_NULL_SEED,
        },
        "cost": {
            # embed_calls counts WORDS sent to the embedder, the same
            # accounting quote_provenance._WindowEmbeddings.calls uses.
            "embed_calls": vectors.calls if vectors is not None else 0,
            "vocab_cached": cached_before,
            "vocab_new": (len(vectors.known()) - cached_before
                          if vectors is not None else 0),
        },
    }
    summary = summarize(scored, meta=meta, labels=labels)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    o = summary["overall"]
    print(f"\n=== Topic vocabulary retention (run {run_id}) ===")
    print(f"rows: {o['n']}  scored: {o['n_scored']}  "
          f"no clustered topic retrieved: {o['n_no_topics']}")
    if o.get("mean_retention") is not None:
        print(f"retention:      {o['mean_retention']:.1%}")
        print(f"  verbatim only {o['mean_verbatim_share']:.1%}   "
              f"semantic lift {o['mean_semantic_lift']:.1%}")
        if o.get("mean_retention_null") is not None:
            print(f"  null floor    {o['mean_retention_null']:.1%}")
        if o.get("mean_verbatim_ceiling") is not None:
            print(f"  ceiling       {o['mean_verbatim_ceiling']:.1%}  "
                  f"(keywords vs their own chunks — the circularity baseline)")
    print(f"embedded {meta['cost']['vocab_new']} new words "
          f"({meta['cost']['vocab_cached']} already cached)")
    print(f"outputs: {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# Reading results (what the app uses)
# ---------------------------------------------------------------------------
def _run_file(run_id, name):
    if not run_id:
        return None
    path = runs.run_dir(run_id) / OUT_DIR_NAME / name
    return path if path.exists() else None


def load_summary(run_id):
    path = _run_file(run_id, "summary.json")
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_rows(run_id):
    path = _run_file(run_id, "rows.jsonl")
    return _load_jsonl(path) if path else None


def load_terms(run_id):
    path = _run_file(run_id, "terms.jsonl")
    return _load_jsonl(path) if path else None
