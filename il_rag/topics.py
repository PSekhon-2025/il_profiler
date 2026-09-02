"""Topic modeling layer: what the corpus talks about, discovered inductively.

Everything else in this pipeline is DEDUCTIVE: it starts from Thornton &
Ocasio's seven logics, writes reference answers for them, and scores the corpus
against that fixed taxonomy. The obvious objection is circularity — the
instrument was told what to look for, so of course it found it. This module is
the independent check: BERTopic (Grootendorst, 2022) clusters the corpus with
no knowledge of the taxonomy at all, and the result is compared against it
afterwards.

Three products, in increasing order of usefulness:

  1. TOPICS         what the corpus is about, per lab and source type.
  2. TOPIC x LOGIC  the cross-tabulation. Every answered question stored the
                    ids of the chunks it was answered from, so each retrieved
                    chunk carries both a topic (from here) and the logic weights
                    the resulting answer earned (from the run). That yields
                    statements like "evidence about export controls produces
                    State-weighted answers" — a link between the inductive and
                    deductive layers built from provenance already on disk.
  3. COVERAGE       topics that NO question ever retrieved: corpus regions the
                    questionnaire is structurally blind to. A limitation finder.

Read this correctly: **a topic is not a logic.** Topics are subject matter
(export controls, lawsuits, funding rounds); a logic is an ordering principle
(what confers legitimacy, who holds authority) that cuts ACROSS subject matter.
A clean topic->logic mapping is therefore not expected and its absence is not a
failure of either layer. The cross-tab is evidence about how the two relate, not
a validation that they are the same thing.

Design notes:
  - Runs LOCALLY, never in the deployed container. BERTopic pulls UMAP, HDBSCAN
    and scikit-learn, and UMAP on 15.5k x 1024 vectors peaks well above the
    1 GB cloud machine. The heavy import is therefore LAZY, inside fit_topics()
    alone, so that everything else here — loading results, the cross-tab, the
    coverage audit — imports and runs anywhere, including in the container.
    Install with: pip install -r requirements-topics.txt
  - Zero API cost: the chunk embeddings already sit in Chroma from ingest, so
    nothing is re-embedded.
  - Seeded. UMAP is stochastic; without a fixed random_state the topics move
    between runs. The seed is recorded in the output.

Naming a topic is its own problem, handled in select_keywords/describe_topic:
c-TF-IDF's top terms are heavily redundant (measured on the fitted model, 21%
of keyword slots held a restatement like "opus" -> "claude opus" -> "opus 46",
an inflection, or a bare number), and the corpus's own furniture — Newstex
licensing footers, citation fragments — ranks highly wherever it survived
ingest. Keywords are therefore selected from a deep candidate pool with
redundancy and furniture removed, and whole clusters that ARE furniture are
flagged by how much their chunks repeat one another rather than by vocabulary.
Known gap: reference sections cite different papers, so they do not repeat and
are not flagged; they surface as an ordinary "citations" topic.

`relabel_topics` recomputes all of that over the EXISTING clusters, so the
naming can be improved without a re-fit renumbering every topic and
invalidating the cross-tabs already saved in run folders.

Outputs:
  data/topics/topic_info.json    per-topic keywords, sizes, lab/source splits,
                                 duplication score and boilerplate flag
  data/topics/chunk_topics.json  {chunk_id: topic_id} for every indexed chunk
  <run>/topics/topic_logic.json  the cross-tab + coverage audit for one run
"""
import json
import re
from collections import defaultdict
from datetime import datetime

from . import runs
from .config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, ORGS, SOURCE_TYPES
from .grounding import content_tokens
from .questionnaire import LOGICS

TOPICS_DIR = DATA_DIR / "topics"
TOPIC_INFO_NAME = "topic_info.json"
CHUNK_TOPICS_NAME = "chunk_topics.json"
RUN_SUBDIR = "topics"
RUN_CROSSTAB_NAME = "topic_logic.json"

# HDBSCAN assigns unclustered points to -1. They are kept (dropping them would
# silently shrink the corpus) but are reported separately and never described as
# a topic.
OUTLIER_TOPIC = -1

DEFAULT_MIN_TOPIC_SIZE = 25
DEFAULT_N_KEYWORDS = 10
# c-TF-IDF is asked for this many times the keywords actually wanted, because
# select_keywords throws most of them away. Measured on the fitted model, 21%
# of stored slots were redundant or numeric, and filtering a 10-term list just
# leaves a shorter list — the pool is what refills the freed slots with real
# vocabulary.
KEYWORD_CANDIDATE_MULTIPLIER = 4

# Corpus furniture that survives ingest.strip_boilerplate and reaches c-TF-IDF.
# Every entry was found by mining the fitted model's own labels, not guessed:
# Newstex is the wire service whose licensing footer is appended to press
# records, and the citation block is reference sections in published PDFs.
# Deliberately narrow — only tokens with no plausible institutional reading, so
# ordinary words are never silenced. Passed to the vectorizer (which stops the
# bigrams too, since they cannot form without their tokens) AND re-checked in
# select_keywords, so a model fitted before this existed still benefits.
KEYWORD_STOP_TOKENS = frozenset({
    # Newstex press-feed licensing footer
    "newstex", "redistributors", "authoritative",
    # reference sections / citation furniture, including the fragments URLs
    # leave behind once punctuation is stripped (arxiv.org/abs/... -> org, abs)
    "arxiv", "preprint", "doi", "url", "http", "https", "isbn", "pp",
    "et", "al", "eprint", "bibtex", "org", "abs", "www", "html", "vol",
    # honorifics left by the press exports
    "mr", "mrs", "ms",
})

# Multi-word boilerplate whose individual tokens ARE ordinary English, so they
# must not go in KEYWORD_STOP_TOKENS. Matched as whole phrases only.
KEYWORD_STOP_PHRASES = frozenset({
    "sole discretion", "expressly reserve", "reserve right", "right delete",
    "delete stories", "stories sole", "discretion journal",
    "redistributors expressly", "authoritative content",
})

# A topic whose surviving keywords fall below this is reported as boilerplate
# rather than given a label that reads like a theme: if the stoplist ate almost
# everything, what is left is not a theme.
KEYWORD_MIN_USEFUL = 3

# Presentation only. A stored `label` is plain data — the marker is added when
# a topic is rendered, so nothing downstream (CSV exports, chart axes, saved
# cross-tabs) inherits a glyph it then has to strip.
BOILERPLATE_MARK = "\u26a0 "

# The stoplist cleans furniture OUT of real topics; this catches clusters that
# ARE furniture. Machine-generated text (Newstex licensing footers,
# BuySellSignals templated stock reports) repeats near-verbatim across
# documents, so a topic's chunks overlap each other far more than a real
# theme's do. Measured on the fitted model with mean pairwise Jaccard over
# content tokens: every hand-checked real topic scored <= 0.082 (opus 0.036,
# copyright lawsuit 0.063, interpretability 0.082) and every hand-checked
# boilerplate cluster >= 0.303 (Motley Fool 0.303, Newstex footer 0.619). The
# bar sits in that empty band. It does NOT catch reference sections, which
# cite different papers and so do not repeat — see the module docstring.
KEYWORD_BOILERPLATE_DUPLICATION = 0.25
# Chunks sampled per topic for that estimate. Pairwise cost is quadratic and
# the statistic is stable well before this, so the cap is what keeps
# relabelling a seconds-long operation on a 15.5k-chunk corpus.
DUPLICATION_SAMPLE = 25
DUPLICATION_SEED = 0
DEFAULT_SEED = 42
# Chroma is paged rather than read in one call: .get() with no limit
# materializes every embedding at once, which is the one place this module
# could spike memory even locally.
FETCH_PAGE = 2000


# ---------------------------------------------------------------------------
# Corpus loading (no BERTopic needed)
# ---------------------------------------------------------------------------
def load_corpus_vectors() -> tuple[list[str], list[str], list[dict], list]:
    """Read every indexed chunk's id, text, metadata and stored embedding.

    Returns (ids, documents, metadatas, embeddings). The embeddings were
    computed at ingest, so this costs no API calls.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    embs: list = []
    for offset in range(0, total, FETCH_PAGE):
        page = col.get(limit=FETCH_PAGE, offset=offset,
                       include=["documents", "metadatas", "embeddings"])
        ids.extend(page["ids"])
        docs.extend(page["documents"])
        metas.extend(page["metadatas"])
        embs.extend(page["embeddings"])
    return ids, docs, metas, embs


# ---------------------------------------------------------------------------
# Keyword selection (pure — no BERTopic, no corpus)
# ---------------------------------------------------------------------------
_NON_WORD_RE = re.compile(r"^[\d\W_]+$")


# What the two remainders after a shared prefix may be for the pair to count as
# one word inflected, keyed by the SHORTER remainder. A length bound alone is
# not enough: "compute"/"computer" leaves "" and "r", which is within any
# sane bound but is not an English inflection — and in this corpus "compute"
# is a term of art distinct from "computer", so collapsing them costs a real
# keyword. Requiring an actual inflectional ending also retires the
# policy/police collision the previous rule knowingly accepted ("y"/"e" is not
# a pair here). Pairs like american/americas are no longer collapsed either:
# they are different words, and spending a second slot on one is the cheaper
# error.
_INFLECTION_PAIRS = {
    "":  frozenset({"s", "es", "d", "ed", "ing"}),
    "e": frozenset({"es", "ed", "ing"}),
    "y": frozenset({"ies", "ied", "ying"}),
}


def _same_stem(a: str, b: str) -> bool:
    """Cheap inflection test for two single tokens: evaluation/evaluations.

    A shared prefix of >= 5 characters, and the two remainders must form one of
    _INFLECTION_PAIRS. Deliberately NOT topic_keywords.morph_score, which would
    be a circular import (that module imports this one); the rule is also
    tighter, because a false positive here silently costs a label slot.
    """
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    if n < 5:
        return False
    short, long = sorted((a[n:], b[n:]), key=len)
    return long in _INFLECTION_PAIRS.get(short, frozenset())


def _is_redundant(term: str, kept: list[str]) -> bool:
    """Does `term` restate something already kept?

    Two ways, both seen in the fitted model:
      - phrase containment: "opus" already kept makes "claude opus", "opus 46"
        and "opus 45" redundant (they are the same subject, spending three more
        of ten slots).
      - inflection: "evaluations" after "evaluation", "americas" after
        "american".
    Containment is checked on TOKEN SETS rather than substrings so that
    "national security" is caught by "national" but "inter" does not swallow
    "interpretability".
    """
    tokens = set(term.split())
    for k in kept:
        ktokens = set(k.split())
        if tokens <= ktokens or ktokens <= tokens:
            return True
        if len(tokens) == 1 and len(ktokens) == 1 and _same_stem(term, k):
            return True
    return False


def select_keywords(candidates: list[str],
                    n: int = DEFAULT_N_KEYWORDS) -> tuple[list[str], int]:
    """Pick up to `n` informative, non-redundant keywords from a ranked list.

    `candidates` is c-TF-IDF order (most distinctive first) and is walked in
    that order, so the strongest term always wins its slot and only weaker
    restatements are dropped. Returns (keywords, n_dropped_as_furniture) — the
    second value is what flags a topic as boilerplate, and it counts only
    stoplist hits, never redundancy, because a topic full of synonyms is a real
    topic while a topic full of licensing text is not.
    """
    kept: list[str] = []
    furniture = 0
    for raw in candidates:
        term = " ".join(str(raw).lower().split())
        if not term or _NON_WORD_RE.match(term):
            continue                                  # "46", "2021", "-"
        if term in KEYWORD_STOP_PHRASES:
            furniture += 1
            continue
        if any(tok in KEYWORD_STOP_TOKENS for tok in term.split()):
            furniture += 1
            continue
        if _is_redundant(term, kept):
            continue
        kept.append(term)
        if len(kept) >= n:
            break
    return kept, furniture


def duplication_score(texts: list[str], cap: int = DUPLICATION_SAMPLE,
                      seed: int = DUPLICATION_SEED) -> float:
    """Mean pairwise Jaccard overlap of a topic's chunks, in [0, 1].

    How much the same words recur across a topic's documents. A real theme
    discusses one subject in many different ways and scores low; templated
    text repeats itself and scores high. Sampled (seeded, so a rerun
    reproduces the number) because the pair count is quadratic.
    """
    import random as _random

    rnd = _random.Random(seed)
    sample = texts if len(texts) <= cap else rnd.sample(texts, cap)
    token_sets = [ts for ts in (content_tokens(x or "") for x in sample) if ts]
    if len(token_sets) < 2:
        return 0.0
    total = pairs = 0.0
    for i, a in enumerate(token_sets):
        for b in token_sets[i + 1:]:
            union = len(a | b)
            if union:
                total += len(a & b) / union
                pairs += 1
    return (total / pairs) if pairs else 0.0


def clean_label(label) -> str:
    """A stored label with any presentation marker stripped.

    Snapshots written before the marker moved out of the data carry a leading
    "\u26a0 ", including topic_info.json on the deployed volume and the labels
    copied into saved run cross-tabs. Stripping on the way out keeps those
    readable without a migration.
    """
    text = str(label or "")
    while text.startswith(BOILERPLATE_MARK):
        text = text[len(BOILERPLATE_MARK):]
    return text


def display_label(record: dict) -> str:
    """How a topic is named in the UI — the one place the marker is applied."""
    label = clean_label(record.get("label"))
    return BOILERPLATE_MARK + label if record.get("is_boilerplate") else label


def describe_topic(candidates: list[str], n: int = DEFAULT_N_KEYWORDS,
                   duplication: float | None = None) -> dict:
    """Keywords, display label and a boilerplate verdict for one topic.

    Two independent boilerplate signals, because they catch different things:
    a high `duplication` means the cluster IS templated text, while an
    exhausted keyword list means the stoplist removed nearly everything the
    cluster had to say. Either one marks the topic.
    """
    keywords, furniture = select_keywords(candidates, n)
    templated = (duplication is not None
                 and duplication >= KEYWORD_BOILERPLATE_DUPLICATION)
    starved = len(keywords) < KEYWORD_MIN_USEFUL and furniture > 0
    boilerplate = templated or starved
    if not keywords:
        label = "(no distinctive terms)"
    else:
        # The keywords are kept even when flagged — knowing WHICH boilerplate
        # a cluster is (a dividend template vs a licensing footer) is what
        # makes the flag actionable. The label stays clean data: `is_boilerplate`
        # is the signal, and display_label is where it becomes a marker.
        label = ", ".join(keywords[:4])
    out = {"keywords": keywords, "label": label,
           "is_boilerplate": boilerplate, "n_furniture_terms": furniture}
    if duplication is not None:
        out["duplication"] = round(float(duplication), 4)
    return out


# ---------------------------------------------------------------------------
# Fitting (the only part that needs BERTopic)
# ---------------------------------------------------------------------------
def fit_topics(min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
               seed: int = DEFAULT_SEED,
               n_keywords: int = DEFAULT_N_KEYWORDS) -> dict:
    """Cluster the corpus into topics from the stored embeddings.

    Heavy imports happen here and nowhere else, so this module stays importable
    on machines (and in containers) without BERTopic installed.
    """
    try:
        import numpy as np
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as e:  # noqa: BLE001 — actionable message beats a traceback
        raise SystemExit(
            f"Topic modeling needs extra packages ({e}). This is a LOCAL-only "
            "analysis; install with:\n"
            "    .venv/bin/pip install -r requirements-topics.txt"
        ) from e

    ids, docs, metas, embs = load_corpus_vectors()
    print(f"loaded {len(ids)} chunks with stored embeddings")
    # float32: Chroma hands back Python lists, and np.array would otherwise
    # default to float64 and double the matrix for no benefit.
    embeddings = np.asarray(embs, dtype="float32")

    # random_state makes UMAP deterministic (at the cost of parallelism) so a
    # rerun reproduces the same topics — the same reason the bootstrap is seeded.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    # Stopwords at the REPRESENTATION stage only: clustering already happened
    # in embedding space, this just decides which words NAME each topic. The
    # corpus furniture goes in here as well as in select_keywords, because a
    # stopped token cannot form a bigram either — which is what removes
    # "newstex authoritative" and "arxiv preprint" at the source.
    vectorizer = _keyword_vectorizer()

    model = BERTopic(
        embedding_model=None,          # embeddings are supplied, never computed
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        # Ask for far more terms than are kept: select_keywords discards the
        # redundant ones, and without a deep pool the freed slots stay empty.
        top_n_words=n_keywords * KEYWORD_CANDIDATE_MULTIPLIER,
        verbose=True,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topics = [int(t) for t in topics]

    # Per-topic slices by lab and source type, so a topic can be read as
    # "mostly OpenAI, mostly third-party" without another pass over the corpus.
    by_org: dict = defaultdict(lambda: defaultdict(int))
    by_source: dict = defaultdict(lambda: defaultdict(int))
    sizes: dict = defaultdict(int)
    for t, m in zip(topics, metas):
        sizes[t] += 1
        by_org[t][m.get("org", "?")] += 1
        by_source[t][m.get("source_type", "?")] += 1

    texts_by_topic: dict = defaultdict(list)
    for t, doc in zip(topics, docs):
        texts_by_topic[t].append(doc)

    topic_records = []
    for t in sorted(sizes):
        candidates = [w for w, _ in (model.get_topic(t) or [])]
        described = describe_topic(
            candidates, n_keywords,
            duplication=duplication_score(texts_by_topic[t]))
        topic_records.append({
            "topic": t,
            "size": sizes[t],
            "is_outlier": t == OUTLIER_TOPIC,
            **described,
            "label": ("(unclustered)" if t == OUTLIER_TOPIC
                      else described["label"]),
            "by_org": {o: by_org[t].get(o, 0) for o in ORGS},
            "by_source": {s: by_source[t].get(s, 0) for s in SOURCE_TYPES},
        })

    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "n_chunks": len(ids),
        "n_topics": sum(1 for r in topic_records if not r["is_outlier"]),
        "n_outliers": sizes.get(OUTLIER_TOPIC, 0),
        "min_topic_size": min_topic_size,
        "seed": seed,
        "topics": topic_records,
    }
    (TOPICS_DIR / TOPIC_INFO_NAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (TOPICS_DIR / CHUNK_TOPICS_NAME).write_text(
        json.dumps(dict(zip(ids, topics)), ensure_ascii=False), encoding="utf-8")

    print(f"\n{info['n_topics']} topics "
          f"({info['n_outliers']} unclustered chunks) -> {TOPICS_DIR}")
    for r in topic_records[:15]:
        if not r["is_outlier"]:
            print(f"  topic {r['topic']:>3}  n={r['size']:<5} {r['label']}")
    return info


def _keyword_vectorizer():
    """The CountVectorizer both fitting and relabelling name topics with.

    One definition so a relabel can never disagree with the fit that produced
    the clusters it is renaming.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    return CountVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | KEYWORD_STOP_TOKENS),
        min_df=2, ngram_range=(1, 2))


def load_corpus_documents() -> tuple[list[str], list[str]]:
    """Every indexed chunk's id and text — WITHOUT the embeddings.

    load_corpus_vectors' light half. Relabelling only needs the words, and
    pulling 15.5k x 1024 floats to count them would be the one thing that
    makes a cheap operation expensive.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    ids: list[str] = []
    docs: list[str] = []
    for offset in range(0, total, FETCH_PAGE):
        page = col.get(limit=FETCH_PAGE, offset=offset, include=["documents"])
        ids.extend(page["ids"])
        docs.extend(page["documents"])
    return ids, docs


def relabel_topics(n_keywords: int = DEFAULT_N_KEYWORDS) -> dict:
    """Recompute every topic's keywords WITHOUT re-clustering.

    The clusters are read from chunk_topics.json and left exactly as they are;
    only c-TF-IDF is recomputed, over the same corpus with the improved
    vectorizer, and re-selected. That matters because re-fitting would renumber
    every topic and silently invalidate the cross-tabs already saved inside run
    folders — whereas a label is just a name for a cluster that has not moved.

    Needs scikit-learn and BERTopic's c-TF-IDF (both local-only extras), plus
    the Chroma index for the text. No embeddings, no API calls, no UMAP.
    """
    try:
        import numpy as np
        from bertopic.vectorizers import ClassTfidfTransformer
        from scipy import sparse
    except ImportError as e:  # noqa: BLE001 — actionable message beats a traceback
        raise SystemExit(
            f"Relabelling needs the local-only extras ({e}); install with:\n"
            "    .venv/bin/pip install -r requirements-topics.txt") from e

    info = load_topic_info()
    chunk_topics = load_chunk_topics()
    if info is None or chunk_topics is None:
        raise SystemExit(
            "no topic model on disk — run `scripts/07_run_topics.py fit` first")

    ids, docs = load_corpus_documents()
    print(f"loaded {len(ids)} chunk texts (no embeddings)")
    known = [(i, chunk_topics.get(cid)) for i, cid in enumerate(ids)]
    missing = sum(1 for _, t in known if t is None)
    if missing:
        print(f"warning: {missing} indexed chunks are absent from the topic "
              "map (reingested since the fit?) and are skipped")

    # The vectorizer is fitted on CHUNKS, exactly as at fit time, so min_df
    # keeps its meaning ("appears in >= 2 chunks"). Fitting it on the 128
    # concatenated topic documents instead would silently reinterpret min_df
    # as "appears in >= 2 topics" and delete every topic-specific term.
    vectorizer = _keyword_vectorizer()
    X = vectorizer.fit_transform(docs)
    terms = np.array(vectorizer.get_feature_names_out())

    # Sum each topic's chunk rows with a sparse indicator matrix rather than
    # fancy-indexing per topic: one matmul instead of 129 slices, and the
    # result stays sparse, which is what ClassTfidfTransformer expects.
    order = sorted({t for _, t in known if t is not None})
    topic_row = {topic: r for r, topic in enumerate(order)}
    pairs = [(topic_row[t], i) for i, t in known if t is not None]
    indicator = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32),
         ([r for r, _ in pairs], [i for _, i in pairs])),
        shape=(len(order), X.shape[0]))
    counts = indicator @ X

    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    weights = np.asarray(ctfidf.fit_transform(counts).todense())

    texts_by_topic: dict = defaultdict(list)
    for i, topic in known:
        if topic is not None:
            texts_by_topic[topic].append(docs[i])

    pool = n_keywords * KEYWORD_CANDIDATE_MULTIPLIER
    described_by_topic = {}
    for row, topic in zip(weights, order):
        top = np.argsort(row)[::-1][:pool * 2]
        candidates = [terms[i] for i in top if row[i] > 0]
        described_by_topic[topic] = describe_topic(
            candidates, n_keywords,
            duplication=duplication_score(texts_by_topic[topic]))

    changed = n_boiler = 0
    for r in info["topics"]:
        d = described_by_topic.get(r["topic"])
        if d is None:
            continue
        before = list(r.get("keywords") or [])
        r.update(d)
        if r["topic"] == OUTLIER_TOPIC:
            r["label"] = "(unclustered)"
        changed += int(before != r["keywords"])
        n_boiler += int(r.get("is_boilerplate", False) and not r["is_outlier"])

    info["relabelled_at"] = datetime.now().isoformat(timespec="seconds")
    info["n_boilerplate_topics"] = n_boiler
    (TOPICS_DIR / TOPIC_INFO_NAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    refreshed = _refresh_crosstab_labels(info)

    print(f"\nrelabelled {changed} topic(s); {n_boiler} flagged as boilerplate "
          f"(templated text, duplication >= {KEYWORD_BOILERPLATE_DUPLICATION})")
    flagged = [r for r in info["topics"]
               if r.get("is_boilerplate") and not r["is_outlier"]]
    for r in sorted(flagged, key=lambda r: -r.get("duplication", 0))[:10]:
        print(f"    t{r['topic']:>3} n={r['size']:<5} "
              f"dup={r.get('duplication', 0):.2f}  {r['label'][:52]}")
    if refreshed:
        print(f"refreshed labels in {refreshed} saved cross-tab(s)")
    print(f"outputs: {TOPICS_DIR / TOPIC_INFO_NAME}")
    for r in sorted((r for r in info["topics"] if not r["is_outlier"]),
                    key=lambda r: -r["size"])[:12]:
        print(f"  topic {r['topic']:>3}  n={r['size']:<5} {r['label']}")
    return info


def _refresh_crosstab_labels(info: dict) -> int:
    """Copy new labels into cross-tabs already saved in run folders.

    build_crosstab denormalizes label/keywords into its own records, so without
    this a relabel would leave every saved cross-tab showing the old names —
    the exact drift that makes two artifacts disagree about the same topic.
    """
    from . import runs as runs_mod

    by_topic = {r["topic"]: r for r in info["topics"]}
    n = 0
    if not runs_mod.RUNS_DIR.exists():
        return 0
    for d in runs_mod.RUNS_DIR.iterdir():
        path = d / RUN_SUBDIR / RUN_CROSSTAB_NAME
        if not path.exists():
            continue
        try:
            xtab = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for rec in xtab.get("topics", []):
            src = by_topic.get(rec.get("topic"))
            if src:
                rec["label"] = src["label"]
                rec["keywords"] = src["keywords"]
                rec["is_boilerplate"] = src.get("is_boilerplate", False)
        for rec in xtab.get("coverage", {}).get("never_retrieved", []):
            src = by_topic.get(rec.get("topic"))
            if src:
                rec["label"] = src["label"]
                rec["keywords"] = src["keywords"]
        path.write_text(json.dumps(xtab, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Reading results (no BERTopic needed — this is what the app uses)
# ---------------------------------------------------------------------------
def load_topic_info() -> dict | None:
    path = TOPICS_DIR / TOPIC_INFO_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_chunk_topics() -> dict | None:
    path = TOPICS_DIR / CHUNK_TOPICS_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_crosstab(run_id: str | None) -> dict | None:
    if not run_id:
        return None
    path = runs.run_dir(run_id) / RUN_SUBDIR / RUN_CROSSTAB_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------
# Cross-tab + coverage (pure functions over saved data — no API, no BERTopic)
# ---------------------------------------------------------------------------
def crosstab_rows(rows: list[dict], chunk_topics: dict) -> tuple[dict, dict]:
    """Attribute each answered row's logic weights to its evidence's topics.

    A row was answered from k retrieved chunks, so each chunk receives 1/k of
    that row's credit — the row contributes a total mass of 1 no matter how many
    chunks it used, and no row can dominate by having been given more evidence:

        mass(topic t, logic l) += (1/k) * w_l   for each retrieved chunk of topic t

    Abstained rows carry no weights and are skipped, but their retrievals ARE
    counted for coverage (the questionnaire did reach that text, it just yielded
    nothing). Returns (per-topic accumulators, retrieval counts).
    """
    mass: dict = defaultdict(lambda: {logic: 0.0 for logic in LOGICS})
    hits: dict = defaultdict(int)          # retrievals, incl. abstained rows
    committed_hits: dict = defaultdict(int)  # retrievals that carried weights

    for r in rows:
        retrieved = r.get("retrieved_ids") or []
        if not retrieved:
            continue
        topics_here = [chunk_topics.get(cid) for cid in retrieved]
        for t in topics_here:
            if t is not None:
                hits[t] += 1
        if r.get("abstain"):
            continue
        k = len(retrieved)
        weights = r.get("weights") or {}
        for t in topics_here:
            if t is None:
                continue
            committed_hits[t] += 1
            for logic in LOGICS:
                mass[t][logic] += float(weights.get(logic, 0.0)) / k
    return {"mass": dict(mass), "committed_hits": dict(committed_hits)}, dict(hits)


def build_crosstab(run_id: str | None = None) -> dict:
    """Cross-tabulate a run's logic weights against corpus topics, and audit
    which topics the questionnaire never reached. Writes into the run folder."""
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run scripts/02_run_profiles.py first")
    info = load_topic_info()
    chunk_topics = load_chunk_topics()
    if info is None or chunk_topics is None:
        raise SystemExit(
            "no topic model on disk — run `scripts/07_run_topics.py fit` first")

    path = runs.run_paths(run_id)["per_question"]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        raise SystemExit(f"run {run_id} has no per-question rows")

    acc, hits = crosstab_rows(rows, chunk_topics)
    mass, committed_hits = acc["mass"], acc["committed_hits"]
    meta_by_topic = {r["topic"]: r for r in info["topics"]}

    records = []
    for t, m in sorted(mass.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(m.values())
        if total <= 0:
            continue
        tinfo = meta_by_topic.get(t, {})
        records.append({
            "topic": t,
            "label": tinfo.get("label", str(t)),
            "keywords": tinfo.get("keywords", []),
            "corpus_size": tinfo.get("size"),
            "retrievals": hits.get(t, 0),
            "committed_retrievals": committed_hits.get(t, 0),
            # Distribution over logics for answers grounded in this topic.
            "logic_pct": {logic: round(100.0 * m[logic] / total, 2)
                          for logic in LOGICS},
            "dominant_logic": max(m, key=m.get),
        })

    # Counts behind the coverage story: the same good chunks are retrieved by
    # many questions, so the number of DISTINCT chunks seen is far smaller than
    # the number of retrieval slots. Stored so the UI (and the download) can
    # state the chain rather than just the conclusion.
    all_retrieved = [cid for r in rows for cid in (r.get("retrieved_ids") or [])]
    distinct_retrieved = set(all_retrieved)

    retrieved_topics = {t for t, n in hits.items() if n > 0}
    never = [
        {"topic": r["topic"], "label": r["label"], "size": r["size"],
         "keywords": r["keywords"]}
        for r in info["topics"]
        if not r["is_outlier"] and r["topic"] not in retrieved_topics
    ]
    never.sort(key=lambda r: -r["size"])

    out = {
        "run_id": run_id,
        "fitted_at": info.get("fitted_at"),
        "seed": info.get("seed"),
        "attribution": "uniform 1/k across each row's retrieved chunks",
        "topics": records,
        "coverage": {
            "questions": len(rows),
            "retrieval_slots": len(all_retrieved),
            "distinct_chunks_retrieved": len(distinct_retrieved),
            "corpus_chunks": info["n_chunks"],
            "n_topics": info["n_topics"],
            "n_topics_retrieved": len([t for t in retrieved_topics
                                       if t != OUTLIER_TOPIC]),
            "n_topics_never_retrieved": len(never),
            "chunks_never_retrieved_share": round(
                sum(r["size"] for r in never)
                / max(1, info["n_chunks"] - info.get("n_outliers", 0)), 4),
            "never_retrieved": never,
        },
        "note": ("A topic is subject matter; a logic is an ordering principle. "
                 "This cross-tab describes how they co-occur in the evidence "
                 "actually used — it does not assert that topics ARE logics."),
    }

    out_dir = runs.run_dir(run_id) / RUN_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RUN_CROSSTAB_NAME).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    cov = out["coverage"]
    print(f"\n=== Topic x logic (run {run_id}) ===")
    for r in records[:12]:
        print(f"  topic {r['topic']:>3} n={r['retrievals']:<4} "
              f"{r['dominant_logic']:<11} {r['label'][:52]}")
    print(f"\ncoverage: {cov['n_topics_retrieved']}/{cov['n_topics']} topics "
          f"reached by the questionnaire; {cov['n_topics_never_retrieved']} never "
          f"retrieved ({cov['chunks_never_retrieved_share']:.1%} of clustered chunks)")
    print(f"outputs: {out_dir}")
    return out
