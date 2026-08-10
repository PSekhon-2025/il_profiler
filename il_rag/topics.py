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

Outputs:
  data/topics/topic_info.json    per-topic keywords, sizes, lab/source splits
  data/topics/chunk_topics.json  {chunk_id: topic_id} for every indexed chunk
  <run>/topics/topic_logic.json  the cross-tab + coverage audit for one run
"""
import json
from collections import defaultdict
from datetime import datetime

from . import runs
from .config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, ORGS, SOURCE_TYPES
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
# Fitting (the only part that needs BERTopic)
# ---------------------------------------------------------------------------
def fit_topics(min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
               seed: int = DEFAULT_SEED,
               n_keywords: int = 10) -> dict:
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
    # English stopwords at the REPRESENTATION stage only: clustering already
    # happened in embedding space, this just keeps "the/and/of" out of the
    # keywords that name each topic.
    vectorizer = CountVectorizer(stop_words="english", min_df=2,
                                 ngram_range=(1, 2))

    model = BERTopic(
        embedding_model=None,          # embeddings are supplied, never computed
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
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

    topic_records = []
    for t in sorted(sizes):
        words = [w for w, _ in (model.get_topic(t) or [])][:n_keywords]
        topic_records.append({
            "topic": t,
            "size": sizes[t],
            "is_outlier": t == OUTLIER_TOPIC,
            "keywords": words,
            "label": ("(unclustered)" if t == OUTLIER_TOPIC
                      else ", ".join(words[:4])),
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
