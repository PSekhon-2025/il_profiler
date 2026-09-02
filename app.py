r"""Streamlit GUI for the Institutional-Logics RAG profiler.

Run with:  .venv/bin/streamlit run app.py        (macOS/Linux)
           .venv\Scripts\streamlit run app.py    (Windows)
(or double-click "Launch IL Profiler.command" / "Launch IL Profiler.bat")

Areas:
  Run           — configure the API key, build the vector index, run profiles
                  (optionally with the grounding / quotes checks enabled).
                  Pipeline stages execute as subprocesses with live log
                  streaming, so the resumable behavior of the CLI scripts is
                  preserved and a closed browser tab never corrupts a run.
  Results       — the six alignment profiles as charts, the published-vs-
                  thirdparty comparison per lab, the Family/Religion sanity
                  check, downloads.
  Audit         — browse every question's RAG answer, graded weights, matcher
                  reasoning, and (when enabled) quotes + grounding bucket.
  Hallucination — the five opt-in checks for any saved run: retrieval-
                  grounding buckets, quote verification, quote provenance
                  (grading WHY a quote failed, and whether its content is
                  true anyway), and the metamorphic label-stability eval
                  (launchable from here), with alert banners when a
                  detection fires.
  Compare       — diff two run snapshots.
"""
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from il_rag import (
    adhoc as adhoc_mod,
    keyword_agreement as ka_mod,
    pdf_sources,
    runs,
    topic_keywords as tk_mod,
    topics as topics_mod,
)
from il_rag.config import (
    ARTICLE_PDF_DIR,
    CALIBRATION_VOCAB_SIZE,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    GROUNDING_LOW_THRESHOLD,
    KEYWORD_MAX_CANDIDATES,
    KEYWORD_MORPH_MIN_PREFIX,
    KEYWORD_MORPH_MIN_RATIO,
    KEYWORD_NULL_DRAWS,
    KEYWORD_SEMANTIC_MAX_SCORE,
    KEYWORD_SEMANTIC_MIN_PERCENTILE,
    METAMORPHIC_CONTROLS,
    METAMORPHIC_PARAPHRASES,
    METAMORPHIC_PARAPHRASE_TEMPERATURE,
    METAMORPHIC_PROBES,
    METAMORPHIC_STABILITY_THRESHOLD,
    ORGS,
    PARAPHRASE_MAX_TOKEN_OVERLAP,
    PARAPHRASE_MIN_COSINE,
    PDF_DATASET_ROOT,
    QUOTE_MIN_SPAN_TOKENS,
    QUOTE_NEAR_VERBATIM_THRESHOLD,
    QUOTE_PARAPHRASE_COS_THRESHOLD,
    QUOTE_PARAPHRASE_LEX_THRESHOLD,
    SOURCE_LINKS_DISABLED,
    SOURCE_TYPES,
    STATIC_DIR,
    TOP_K,
)
from il_rag.questionnaire import CATEGORIES, LOGICS


def _venv_python() -> str:
    """Path to the project venv's python (POSIX or Windows layout), else the
    interpreter running this app."""
    for cand in (PROJECT_ROOT / ".venv" / "bin" / "python",
                 PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    return sys.executable


PYTHON = _venv_python()
ENV_PATH = PROJECT_ROOT / ".env"

# Cloud deployments ship a prebuilt vector index but NOT the (copyrighted) raw
# corpus, so ingestion can't run there. Set IL_PROFILER_CLOUD=1 to hide the
# "Build the vector index" controls and expose only running profiles + viewing.
CLOUD_MODE = os.environ.get("IL_PROFILER_CLOUD") == "1"

# Fold any pre-snapshot flat outputs into a run so the app only ever deals with
# the runs/ layout. No-op once a run exists.
runs.migrate_legacy()

# Colors keep the same logic recognizable across every chart. Family/Religion
# are grey on purpose — they're the sanity-check logics expected near 0%.
LOGIC_COLORS = {
    "State": "#4C78A8", "Profession": "#54A24B", "Market": "#E45756",
    "Corporation": "#F58518", "Family": "#B0B0B0", "Religion": "#888888",
    "Community": "#72B7B2",
}

st.set_page_config(page_title="IL Profiler", page_icon="🏛️", layout="wide")


def _require_password() -> None:
    """Lightweight shared-password gate for hosted deployments.

    Enabled only when APP_PASSWORD is set (e.g. on Fly). This keeps the public
    *.fly.dev URL from being open to anyone while remaining trivial for the team.
    For per-reviewer identity, front the app with Cloudflare Access instead (see
    DEPLOY.md) and leave APP_PASSWORD unset.
    """
    expected = os.environ.get("APP_PASSWORD")
    if not expected:
        return  # no gate configured (local use)
    if st.session_state.get("authed"):
        return
    import hmac
    st.title("🏛️ IL Profiler")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        # compare_digest: constant-time comparison (no timing side-channel).
        if st.form_submit_button("Enter") and hmac.compare_digest(pw, expected):
            st.session_state["authed"] = True
            st.rerun()
    if st.session_state.get("_login_tried") and not st.session_state.get("authed"):
        st.error("Incorrect password.")
    st.session_state["_login_tried"] = True
    st.stop()


_require_password()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def api_key_present() -> bool:
    if not ENV_PATH.exists():
        return False
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("TOGETHER_API_KEY="):
            val = line.split("=", 1)[1].strip()
            return bool(val) and val != "your_together_api_key_here"
    return False


def save_api_key(key: str) -> None:
    ENV_PATH.write_text(f"TOGETHER_API_KEY={key.strip()}\n", encoding="utf-8")


@st.cache_data(ttl=300)
def fetch_chunks(chunk_ids: tuple[str, ...]) -> dict:
    """Look up the ACTUAL TEXT of retrieved chunks, by id, from Chroma.

    The audit rows persist only chunk ids, so without this the evidence behind
    every answer is unreadable. Fetched in ONE batched get for the whole
    filtered view (ids need no embedding, so this is cheap) and cached, rather
    than per row. Returns {id: {"text","filename","org","source_type"}};
    missing ids (e.g. after a --fresh reingest re-minted them) are simply absent.
    """
    if not chunk_ids:
        return {}
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        col = client.get_collection(COLLECTION_NAME)
        res = col.get(ids=list(chunk_ids), include=["documents", "metadatas"])
    except Exception:
        return {}
    # Display labels are attached HERE because this is the single place chunk
    # metadata is assembled for the UI — so the evidence list, the excerpt
    # links and the mis-citation note all get them without a second lookup.
    # For third-party evidence `filename` is the RTF bundle ("O1.RTF"), which
    # names 500 articles and identifies none; `label` carries the headline.
    articles = article_sources()
    out = {}
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        record = articles.get((pdf_sources.article_key(cid) or ("",))[0])
        out[cid] = {"text": doc, "filename": meta.get("filename", ""),
                    "org": meta.get("org", ""),
                    "source_type": meta.get("source_type", ""),
                    "label": (pdf_sources.article_label(record) if record
                              else meta.get("filename", "")),
                    "sublabel": pdf_sources.article_sublabel(record) if record else ""}
    return out


@st.cache_data(ttl=300)
def pdf_index() -> dict:
    """Normalized-basename -> dataset PDF paths, from one walk of the tree.

    Cached with a TTL rather than forever so a PDF dropped into the dataset is
    picked up without restarting the app.
    """
    return pdf_sources.build_index(PDF_DATASET_ROOT)


@st.cache_data(ttl=300)
def chunk_pages() -> dict:
    """{chunk_id: page number} for #page= deep links, or {} if never built."""
    return pdf_sources.load_chunk_pages()


@st.cache_data(ttl=300)
def article_sources() -> dict:
    """{fragment prefix: press-record metadata + page map}, or {} if never built."""
    return pdf_sources.load_article_sources()


@st.cache_data(ttl=300)
def pdf_url(filename: str, org: str, chunk_id: str | None = None,
            source_type: str = "published") -> str | None:
    """URL a browser can open for this chunk's source PDF, or None.

    Two sources, one return type. Published chunks resolve against the raw
    dataset; third-party chunks resolve against the per-article PDFs that
    scripts/10_build_article_pdfs.py generated, because their `filename` is a
    45 MB bundle of 500 records rather than a document.

    ACCESS NOTE, deliberately not a guard. Streamlit serves /app/static over
    HTTP with `Access-Control-Allow-Origin: *`, and that route is handled
    beneath the Python app, so it is NOT behind _require_password(). Anything
    published into static/ is therefore readable by anyone who reaches the
    hostname and guesses the path. A deployment that serves the corpus is
    relying on Cloudflare Access — which gates the whole hostname at the edge,
    static route included — or on the host being unlisted. `APP_PASSWORD` alone
    does NOT cover it. This was a deliberate choice for a private instance; see
    DEPLOY.md. `IL_PROFILER_DISABLE_SOURCE_LINKS=1` turns it all off again
    without a code change.

    There is intentionally no CLOUD_MODE test here any more. What gates the
    links is whether the source documents are actually present: on a machine
    without them every lookup below returns None, which is the same plain-text
    fallback the guard used to produce.

    Note this memoizes an idempotent SIDE EFFECT: the first call for a document
    publishes it into static/. Correctness never depends on the cache —
    materialize() re-checks the destination on every miss, and the cache only
    spares repeated stat() calls while a view renders hundreds of quotes.
    """
    if SOURCE_LINKS_DISABLED:
        return None
    if source_type == "thirdparty":
        # Needs no dataset root — the article PDFs were generated ahead of time.
        found = pdf_sources.resolve_article(chunk_id, article_sources(),
                                            ARTICLE_PDF_DIR)
        if found is None:
            return None
        src, page, _ = found
        return pdf_sources.materialize(src, org, STATIC_DIR, page=page,
                                       subdir=pdf_sources.ARTICLES_SUBDIR)
    if PDF_DATASET_ROOT is None:
        return None
    src = pdf_sources.resolve(filename, org, pdf_index())
    if src is None:
        return None
    page = chunk_pages().get(chunk_id) if chunk_id else None
    return pdf_sources.materialize(src, org, STATIC_DIR, page=page)


@st.cache_data(ttl=30)
def index_counts() -> pd.DataFrame | None:
    """Chunk counts per (org, source_type), or None if no index exists yet."""
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        return None
    rows = []
    for org in ORGS:
        for stype in SOURCE_TYPES:
            n = len(col.get(
                where={"$and": [{"org": org}, {"source_type": stype}]}, include=[]
            )["ids"])
            rows.append({"lab": org, "source": stype, "chunks": n})
    return pd.DataFrame(rows)


def load_profiles(run_id: str | None) -> dict | None:
    if not run_id:
        return None
    path = runs.run_paths(run_id)["profiles_json"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_per_question(run_id: str | None) -> pd.DataFrame | None:
    if not run_id:
        return None
    path = runs.run_paths(run_id)["per_question"]
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else None


def load_questionnaire(run_id: str | None) -> dict | None:
    """The questionnaire snapshot stored with a run (for wording diffs)."""
    if not run_id:
        return None
    path = runs.run_paths(run_id)["questionnaire"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_stability(run_id: str | None) -> dict | None:
    """The metamorphic eval's stability.json for a run, if it has been run."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "metamorphic" / "stability.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_variants(run_id: str | None) -> pd.DataFrame | None:
    """The metamorphic eval's per-variant audit rows for a run."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "metamorphic" / "variants.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else None


@st.cache_data(ttl=30)
def load_replicate_reports() -> dict:
    """All replicate-average reports on disk, newest group first."""
    from il_rag.replicates import REPLICATES_DIR, REPORT_JSON
    if not REPLICATES_DIR.exists():
        return {}
    out = {}
    for d in sorted(REPLICATES_DIR.iterdir(), reverse=True):
        p = d / REPORT_JSON
        if d.is_dir() and p.exists():
            try:
                out[d.name] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return out


def load_bootstrap_ci(run_id: str | None) -> dict | None:
    """The bootstrap-CI result for a run, if computed."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "bootstrap_ci" / "ci.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_embedding_summary(run_id: str | None) -> dict | None:
    """The embedding-agreement summary for a run, if computed."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "embedding_agreement" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_keyword_summary(run_id: str | None) -> dict | None:
    """The lexical keyword judge's summary for a run, if computed."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "keyword_agreement" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_keyword_rows(run_id: str | None) -> pd.DataFrame | None:
    """Per-question keyword scores + the matched words behind each verdict."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "keyword_agreement" / "rows.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else None


def load_embedding_rows(run_id: str | None) -> pd.DataFrame | None:
    """Per-row embedding similarities for a run, if computed."""
    if not run_id:
        return None
    path = runs.run_dir(run_id) / "embedding_agreement" / "similarities.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else None


def render_judge_profiles(rows: pd.DataFrame, share_col: str) -> None:
    """Per-lab grouped bars of the % profile a judge's shares imply.

    Mirrors the alignment-profile charts exactly — one chart per lab, the
    seven logics on x, a 0-100 scale, published vs thirdparty in the same
    source colors — so a judge's picture can be read directly against the
    matcher's profiles above it. Each bar is the mean share the judge put on
    that logic across the scored answers of one (lab, source), x100, and the
    per-source dominant-logic metrics underneath mirror the profile charts'.
    """
    import altair as alt

    recs = []
    for _, r in rows.iterrows():
        shares = r[share_col]
        if not isinstance(shares, dict):
            continue
        for logic in LOGICS:
            recs.append({"lab": r["org"], "source": r["source_type"],
                         "logic": logic,
                         "share": float(shares.get(logic) or 0.0)})
    if not recs:
        st.info("No per-logic shares found in this run's rows file.")
        return
    jl = (pd.DataFrame(recs)
          .groupby(["lab", "source", "logic"], as_index=False)
          .agg(pct=("share", "mean"), n=("share", "size")))
    jl["pct"] = 100.0 * jl["pct"]

    for org in [o for o in ORGS if o in jl["lab"].unique()]:
        st.subheader(org)
        sub = jl[jl["lab"] == org]
        chart = (
            alt.Chart(sub)
            .mark_bar()
            .encode(
                x=alt.X("logic:N", sort=LOGICS, title=None),
                xOffset=alt.XOffset("source:N"),
                y=alt.Y("pct:Q", title="% of judge profile",
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "source:N", title="source",
                    scale=alt.Scale(domain=SOURCE_TYPES,
                                    range=["#4C78A8", "#F58518"]),
                ),
                tooltip=["lab", "source", "logic",
                         alt.Tooltip("pct:Q", format=".1f"),
                         alt.Tooltip("n:Q", title="answers")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")

        sources_here = [s for s in SOURCE_TYPES if s in sub["source"].unique()]
        cols = st.columns(len(sources_here))
        for col, stype in zip(cols, sources_here):
            ssub = sub[sub["source"] == stype]
            top = ssub.loc[ssub["pct"].idxmax()]
            col.metric(
                f"{stype} — judge's dominant logic",
                f"{top['logic']} ({top['pct']:.0f}%)",
                help=f"mean share over {int(top['n'])} scored answer(s)",
            )



def load_quote_provenance(run_id: str | None) -> dict | None:
    """The quote-provenance summary for a run, if the stage has run."""
    if not run_id:
        return None
    path = runs.run_paths(run_id)["quote_provenance_summary"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_quote_spans(run_id: str | None) -> pd.DataFrame | None:
    """Per-span provenance records for a run, if the stage has run."""
    if not run_id:
        return None
    path = runs.run_paths(run_id)["quote_spans"]
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else None


def load_topic_keyword_summary(run_id: str | None) -> dict | None:
    """The topic-keyword retention summary for a run, if the stage has run."""
    return tk_mod.load_summary(run_id)


def load_topic_keyword_rows(run_id: str | None) -> pd.DataFrame | None:
    rows = tk_mod.load_rows(run_id)
    return pd.DataFrame(rows) if rows else None


def load_topic_keyword_terms(run_id: str | None) -> pd.DataFrame | None:
    rows = tk_mod.load_terms(run_id)
    return pd.DataFrame(rows) if rows else None


def lexicon_stamp() -> tuple | None:
    """(mtime, size) of the calibration file — the cache key for load_lexicon.

    cache_resource survives page reloads for the life of the server process,
    and `calibrate` is run from a terminal WHILE the app is open. Without a
    stamp the page would go on insisting there is no calibration long after
    one exists (and would keep serving the old grid after a recalibration).
    """
    path = tk_mod.LEXICON_DIR / tk_mod.CALIBRATION_NAME
    if not path.exists():
        return None
    info = path.stat()
    return (info.st_mtime_ns, info.st_size)


@st.cache_resource(show_spinner=False)
def load_lexicon(stamp: tuple | None):
    """(WordVectors, Calibration) over data/lexicon/, or (None, None).

    cache_resource rather than cache_data: this holds a live numpy matrix of a
    few thousand word vectors, not a small serializable payload. readonly=True
    so the app can never spend an embedding call or rewrite the cache — only
    the CLI stage is allowed to do either. `stamp` is not read; it exists to
    key the cache (see lexicon_stamp).
    """
    if stamp is None:
        return None, None
    cal = tk_mod.Calibration.load()
    if cal is None:
        return None, None
    return tk_mod.WordVectors(readonly=True), cal


# ---------------------------------------------------------------------------
# Hallucination export: one "decision + underlying data" table per check
# ---------------------------------------------------------------------------
# Every check reduces its inputs to a verdict per item; these builders lay that
# verdict beside the raw signals that produced it, so an auditor can re-derive
# the outcome offline. Decision columns are prefixed `decision_` throughout.
def _series(df: pd.DataFrame, name: str) -> pd.Series:
    """The named column, or an all-None column aligned to df (missing checks)."""
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _present(v) -> str | None:
    """The value as display text, or None if it is absent.

    Pandas turns a JSON null into NaN, which is TRUTHY — a bare `if row[col]`
    therefore renders the literal string "nan" into the page. Optional fields
    (an unaligned span, an empty grounded fragment) must go through this.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def _top_logic(weights) -> tuple:
    if not isinstance(weights, dict) or not weights:
        return (None, None)
    k = max(weights, key=weights.get)
    return (k, round(float(weights[k]), 4))


def _join_ids(x) -> str:
    return "|".join(map(str, x)) if isinstance(x, list) else ("" if x is None else str(x))


def build_grounding_export(dfh: pd.DataFrame) -> pd.DataFrame | None:
    """Check 1 — one row per bucketed question: the bucket + its overlap score."""
    if "grounding_bucket" not in dfh.columns or not dfh["grounding_bucket"].notna().any():
        return None
    g = dfh[dfh["grounding_bucket"].notna()].copy()
    tops = [_top_logic(w) for w in g["weights"]]
    return pd.DataFrame({
        "org": g["org"], "source_type": g["source_type"], "qid": g["qid"],
        "category": _series(g, "category"), "variant": _series(g, "variant"),
        "question": g["question"],
        "retrieval_grounding_score": g["retrieval_grounding_score"],
        "retrieval_cosine_top": _series(g, "retrieval_cosine_top"),
        "grounding_threshold": GROUNDING_LOW_THRESHOLD,
        "abstain": g["abstain"],
        "decision_grounding_bucket": g["grounding_bucket"],
        "top_logic": [t[0] for t in tops],
        "top_logic_weight": [t[1] for t in tops],
        "retrieved_ids": g["retrieved_ids"].apply(_join_ids),
    })


def build_quotes_exports(dfh: pd.DataFrame):
    """Check 2 — a per-answer verdict table and a per-span verification table."""
    if "quotes_verified" not in dfh.columns or not dfh["quotes_verified"].notna().any():
        return None, None
    q = dfh[dfh["quotes_verified"].notna()].copy()
    rows, spans = [], []
    for _, r in q.iterrows():
        quotes = r["quotes"] if isinstance(r["quotes"], list) else []
        verified = bool(r["quotes_verified"])
        verdict = ("no_quotes" if not quotes else "verified" if verified
                   else "unverified")
        rows.append({
            "org": r["org"], "source_type": r["source_type"], "qid": r["qid"],
            "category": r.get("category"), "question": r["question"],
            "answer": r["answer"], "n_quotes": len(quotes),
            "decision_quotes_verified": verified,
            "decision_verdict": verdict,
            "quotes_json": json.dumps(quotes, ensure_ascii=False),
        })
        for qq in quotes:
            spans.append({
                "org": r["org"], "source_type": r["source_type"], "qid": r["qid"],
                "excerpt": qq.get("excerpt"), "quote": qq.get("quote"),
                "decision_span_verified": qq.get("verified"),
            })
    return pd.DataFrame(rows), (pd.DataFrame(spans) if spans else None)


def build_quote_provenance_export(spans: pd.DataFrame | None) -> pd.DataFrame | None:
    """Check 5 — per span: the graded verdict beside BOTH axes that produced it.

    Feature 2's own bit for the same span travels in `verbatim_verified`, so a
    reader can see exactly which spans the graded check reclassified and why.
    """
    if spans is None or spans.empty:
        return None
    s = spans.copy()
    return pd.DataFrame({
        "org": s["org"], "source_type": s["source_type"], "qid": s["qid"],
        "category": _series(s, "category"),
        "quote": s["quote"], "span_source": s["source"],
        "feature2_verbatim_verified": _series(s, "verbatim_verified"),
        "decision_intent": s["intent"],
        "intent_rule": _series(s, "intent_rule"),
        "intent_confidence": _series(s, "intent_confidence"),
        "decision_match_tier": s["match_tier"],
        "match_rule": _series(s, "match_rule"),
        "match_score": _series(s, "match_score"),
        "decision_support": _series(s, "support"),
        "grounded_fragment": _series(s, "grounded_fragment"),
        "evidence_sentence": _series(s, "evidence_sentence"),
        "support_reason": _series(s, "support_reason"),
        "decision_verdict": s["verdict"],
        "best_chunk_id": _series(s, "best_chunk_id"),
        "best_span": _series(s, "best_span"),
    })


def build_metamorphic_exports(stab: dict | None, variants: pd.DataFrame | None):
    """Check 3 — per-item probe verdicts and the per-variant audit trail."""
    items = None
    if stab and stab.get("per_item"):
        items = pd.DataFrame(stab["per_item"])
        items["stability_threshold"] = METAMORPHIC_STABILITY_THRESHOLD
        items = items.rename(columns={
            "unstable": "decision_unstable",
            "control_flipped": "decision_control_flipped",
            "label_survived_ablation": "decision_label_survived_ablation",
            "prior_keyed": "decision_prior_keyed",
        })
    var = None
    if variants is not None and not variants.empty:
        keep = ["org", "source_type", "qid", "category", "variant",
                "variant_kind", "variant_idx", "original_label",
                "question", "answer", "abstain", "reasoning", "label",
                "label_matches_original", "ablation_basis",
                "ablation_removed_id", "distractor_category",
                "distractor_grounding", "error"]
        var = variants[[c for c in keep if c in variants.columns]].rename(
            columns={"label_matches_original": "decision_label_matches_original"})
        # The fidelity block is a dict per row; flatten the two fields a reader
        # of the CSV actually needs (why a rewrite was thrown out, how far it
        # moved) rather than dumping raw JSON into a cell.
        if "fidelity" in variants.columns:
            fid = variants["fidelity"].apply(
                lambda f: f if isinstance(f, dict) else {})
            var = var.assign(
                paraphrase_rejected_reason=fid.apply(lambda f: f.get("reason")),
                paraphrase_divergence=fid.apply(lambda f: f.get("divergence")),
                paraphrase_min_cosine=fid.apply(lambda f: f.get("min_cosine")),
            )
    return items, var


def build_embedding_export(erows: pd.DataFrame | None) -> pd.DataFrame | None:
    """Check 4 — per-answer agreement verdict + the 7 per-logic cosines/shares."""
    if erows is None or erows.empty:
        return None
    e = erows.copy()
    out = pd.DataFrame({
        "org": e["org"], "source_type": e["source_type"], "qid": e["qid"],
        "matcher_top": e["matcher_top"],
        "matcher_top_weight": _series(e, "matcher_top_weight"),
        "embedding_nearest": e["embedding_nearest"],
        "decision_agree": e["agree"],
        "margin": e["margin"],
        "share_on_matcher_top": _series(e, "share_on_matcher_top"),
        "overlap": _series(e, "overlap"),
    })
    for logic in LOGICS:
        out[f"sim_{logic}"] = e["similarities"].apply(
            lambda d, k=logic: d.get(k) if isinstance(d, dict) else None)
        if "embedding_shares" in e.columns:
            out[f"share_{logic}"] = e["embedding_shares"].apply(
                lambda d, k=logic: d.get(k) if isinstance(d, dict) else None)
    return out


def build_detection_bundle(run_id: str, dfh: pd.DataFrame,
                           stab: dict | None, emb: dict | None):
    """Assemble every check's decision table for a run.

    Returns (members, ran) where `members` is an ordered list of
    (filename, bytes) — the per-check decision CSVs plus the raw source files
    they derive from — and `ran` names the checks that actually produced data.
    """
    def csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    members: list[tuple[str, bytes]] = []
    ran: list[str] = []

    grounding = build_grounding_export(dfh)
    if grounding is not None:
        members.append(("1_retrieval_grounding.csv", csv_bytes(grounding)))
        ran.append("retrieval grounding")

    q_rows, q_spans = build_quotes_exports(dfh)
    if q_rows is not None:
        members.append(("2_quote_verification.csv", csv_bytes(q_rows)))
        if q_spans is not None:
            members.append(("2_quote_verification_spans.csv", csv_bytes(q_spans)))
        ran.append("quote verification")

    qp_spans = build_quote_provenance_export(load_quote_spans(run_id))
    if qp_spans is not None:
        members.append(("2b_quote_provenance.csv", csv_bytes(qp_spans)))
        ran.append("quote provenance")

    mm_items, mm_var = build_metamorphic_exports(stab, load_variants(run_id))
    if mm_items is not None:
        members.append(("3_metamorphic_items.csv", csv_bytes(mm_items)))
        if mm_var is not None:
            members.append(("3_metamorphic_variants.csv", csv_bytes(mm_var)))
        ran.append("metamorphic stability & evidence sensitivity")

    emb_rows = build_embedding_export(load_embedding_rows(run_id))
    if emb_rows is not None:
        members.append(("4_embedding_agreement.csv", csv_bytes(emb_rows)))
        ran.append("embedding agreement")

    # Raw source artifacts, verbatim, so nothing in the export is lossy.
    rd = runs.run_dir(run_id)
    for rel in ("per_question.jsonl", "meta.json",
                "metamorphic/stability.json", "metamorphic/variants.jsonl",
                "embedding_agreement/summary.json",
                "embedding_agreement/similarities.jsonl",
                "quote_provenance/summary.json", "quote_provenance/spans.jsonl"):
        p = rd / rel
        if p.exists():
            members.append((f"raw/{rel}", p.read_bytes()))

    members.insert(0, ("README.txt", _bundle_readme(run_id, ran).encode("utf-8")))
    return members, ran


def _bundle_readme(run_id: str, ran: list[str]) -> str:
    meta = runs.read_meta(run_id)
    return (
        "IL Profiler — hallucination & grounding detection export\n"
        "========================================================\n\n"
        f"Run:        {run_id}"
        + (f"  ({meta.get('label')})" if meta.get('label') else "") + "\n"
        f"Generated:  {datetime.now().isoformat(timespec='seconds')}\n"
        f"Checks with data in this export: "
        f"{', '.join(ran) if ran else '(none — no checks have run for this run)'}\n\n"
        "Each CSV is one decision per row: the `decision_*` column is the\n"
        "check's verdict, and the remaining columns are the underlying data\n"
        "used to reach it, so any outcome can be re-derived offline.\n\n"
        "Files\n"
        "-----\n"
        "1_retrieval_grounding.csv        per question: grounding bucket "
        "(decision) + question<->chunk overlap score, cosine, threshold,\n"
        f"                                 abstain flag. Threshold tau = "
        f"{GROUNDING_LOW_THRESHOLD}.\n"
        "2_quote_verification.csv         per answer: quotes_verified + verdict "
        "(verified / unverified / no_quotes) + the cited spans.\n"
        "2_quote_verification_spans.csv   per cited span: whether that exact "
        "span was found verbatim in the retrieved sources.\n"
        "2b_quote_provenance.csv          per quoted span, GRADED: intent "
        "(is it even a claim about a source?), provenance tier (verbatim /\n"
        "                                 drifted copy / paraphrase / absent), "
        "entailment support, and the combined verdict. Includes\n"
        "                                 feature2_verbatim_verified so you can "
        "see which spans the graded check reclassified, and why.\n"
        "                                 Note: 'unsupported' means unsupported "
        "BY THIS SCOPED CORPUS, not false in the world.\n"
        "3_metamorphic_items.csv          per item, one column per probe "
        "decision: control_flipped (pipeline noise), unstable (a paraphrase\n"
        "                                 changed the label), "
        "label_survived_ablation (the label held after its quoted excerpt was\n"
        "                                 removed), prior_keyed (the same label "
        "came back from text that doesn't answer the question).\n"
        f"                                 Stability threshold theta = "
        f"{METAMORPHIC_STABILITY_THRESHOLD}.\n"
        "3_metamorphic_variants.csv       per variant: its label, whether it "
        "matched the original, and why a rewrite was thrown out (the audit\n"
        "                                 trail).\n"
        "4_embedding_agreement.csv        per answer: agree (decision) + the "
        "matcher's top logic, the embedding-nearest logic, the margin, and\n"
        "                                 the 7 per-logic cosine similarities "
        "and closeness shares.\n"
        "raw/                             the untouched source artifacts these "
        "tables are derived from (per_question.jsonl, meta.json, and any\n"
        "                                 metamorphic / embedding outputs).\n\n"
        "Only checks that have actually run for this snapshot appear above.\n"
        "See ARCHITECTURE.md sections 9.1-9.5 for the full method of each check.\n"
    )


def zip_members(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def visible_runs() -> list[dict]:
    """Runs the corpus-facing pickers should offer.

    Saved document analyses are hidden by default: they are not part of the
    study, and the ORGS-keyed charts cannot render an arbitrary subject. The
    sidebar toggle (shown only once one exists) opts back in, so nothing is
    permanently unreachable.
    """
    if st.session_state.get("show_adhoc_runs"):
        return runs.list_runs()
    return runs.list_runs(kind=runs.KIND_CORPUS)


def run_selectbox(label: str, key: str, default_run_id: str | None = None) -> str | None:
    """Dropdown of selectable runs (newest first); returns the chosen run_id."""
    metas = visible_runs()
    if not metas:
        return None
    ids = [m["run_id"] for m in metas]
    names = {m["run_id"]: runs.display_name(m) for m in metas}
    idx = ids.index(default_run_id) if default_run_id in ids else 0
    return st.selectbox(label, ids, index=idx,
                        format_func=lambda r: names.get(r, r), key=key)


def symbol_glossary(pairs: list[tuple[str, str]], note: str | None = None) -> None:
    """Render the 'what the symbols mean' table under a formula block.

    Every explainer that states a formula defines its notation this way, so a
    reader never has to infer what a subscript stands for. Centralised so the
    presentation stays identical across all of them.
    """
    st.markdown("**What the symbols mean**")
    st.markdown("\n".join(
        ["| symbol | meaning |", "|---|---|"]
        + [f"| {sym} | {desc} |" for sym, desc in pairs]))
    if note:
        st.caption(note)


# Notation shared by several explainers, defined once.
_SET_NOTATION = [
    (r"$\lvert\,S\,\rvert$", "the **size** of set $S$ — how many items it holds"),
    (r"$\cap$", "**intersection** — the items two sets have in common"),
    (r"$\in$", "**is a member of**"),
    (r"$\sum$", "**sum over** whatever the subscript ranges across"),
    (r"$\arg\max$", "returns **which** item scores highest (not the score "
                    "itself — $\\max$ would give the value)"),
]


def word_diff_md(old: str, new: str) -> str:
    """Inline word-level diff as markdown: ~~removed~~ then **added**."""
    import difflib
    sm = difflib.SequenceMatcher(a=old.split(), b=new.split())
    out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(" ".join(old.split()[i1:i2]))
        elif tag == "delete":
            out.append("~~" + " ".join(old.split()[i1:i2]) + "~~")
        elif tag == "insert":
            out.append("**" + " ".join(new.split()[j1:j2]) + "**")
        elif tag == "replace":
            out.append("~~" + " ".join(old.split()[i1:i2]) + "~~ "
                       "**" + " ".join(new.split()[j1:j2]) + "**")
    return " ".join(p for p in out if p)


def stream_subprocess(args: list[str], log_box) -> int:
    """Run a pipeline stage as a subprocess, streaming output into the UI.

    Subprocesses (rather than in-process calls) preserve the scripts' resumable
    semantics and keep a mid-run browser refresh from corrupting state — the
    worst case is the UI loses the log while the run completes on its own.
    """
    # Force UTF-8 on the child's stdio so log streaming behaves identically on
    # macOS and Windows (whose console default is a legacy codepage).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        args, cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    lines: list[str] = []
    for raw in proc.stdout:  # tqdm progress arrives as \r-updates on one line
        part = raw.rstrip("\n").split("\r")[-1]
        if not part.strip():
            continue
        if lines and (part.startswith(("ingest", "profile", "metamorphic")) and
                      lines[-1].startswith(part.split(":")[0])):
            lines[-1] = part  # collapse progress-bar updates in place
        else:
            lines.append(part)
        log_box.code("\n".join(lines[-25:]), language=None)
    return proc.wait()


# ---------------------------------------------------------------------------
# Sidebar: status at a glance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏛️ IL Profiler")
    st.caption(
        "Profiles OpenAI, DeepMind, and Anthropic against the 7 institutional "
        "logics via RAG + graded answer matching."
    )

    st.subheader("Status")
    st.markdown(("✅" if api_key_present() else "❌") + " Together API key")

    counts = index_counts()
    if counts is None or counts["chunks"].sum() == 0:
        st.markdown("❌ Vector index — build it on the **Run** tab")
    else:
        st.markdown(f"✅ Vector index ({counts['chunks'].sum():,} chunks)")
        with st.expander("chunks per corpus"):
            st.dataframe(counts, hide_index=True, width="stretch")

    current_run = runs.get_current()
    dfq = load_per_question(current_run)
    total_q = len(ORGS) * len(SOURCE_TYPES) * runs.QUESTIONS_PER_ORG
    done_q = len(dfq) if dfq is not None else 0
    st.markdown(
        ("✅" if done_q >= total_q else "⏳" if done_q else "❌")
        + f" Profiles ({done_q}/{total_q} questions)"
    )

    all_metas = runs.list_runs()
    if all_metas:
        cur_meta = next((m for m in all_metas if m["run_id"] == current_run),
                        all_metas[0])
        n_adhoc = sum(1 for m in all_metas if runs.is_adhoc(m))
        n_corpus = len(all_metas) - n_adhoc
        st.caption(f"Active run: **{runs.display_name(cur_meta)}**  \n"
                   f"{n_corpus} run(s) saved"
                   + (f" · {n_adhoc} document analysis(es)" if n_adhoc else ""))
        if n_adhoc:
            st.checkbox(
                "Show document analyses in run pickers",
                key="show_adhoc_runs",
                help="Saved document analyses are real run snapshots, so the "
                     "Audit tab and the post-hoc judges work on them. They are "
                     "hidden from the run pickers by default because they are "
                     "not part of the six-profile study, and the Results "
                     "charts key on the three labs so they cannot plot an "
                     "uploaded subject.")

(tab_run, tab_results, tab_audit, tab_halluc, tab_topics, tab_adhoc,
 tab_compare) = st.tabs(
    ["▶️ Run", "📊 Results", "🔍 Audit", "🚨 Hallucination", "🧭 Topics",
     "📄 Analyse a document", "🆚 Compare runs"])

# ---------------------------------------------------------------------------
# Run tab
# ---------------------------------------------------------------------------
with tab_run:
    st.header("Setup & pipeline")

    # --- API key ---
    with st.expander("1 · Together API key", expanded=not api_key_present()):
        if api_key_present():
            st.success("API key is configured (.env).")
        key_in = st.text_input("Paste your TOGETHER_API_KEY", type="password",
                               placeholder="together_...")
        if st.button("Save key", disabled=not key_in):
            save_api_key(key_in)
            st.success("Saved to .env")
            st.rerun()

    # --- Ingest ---
    # Hidden in cloud mode: the deployed instance ships a prebuilt index but not
    # the raw corpus, so there is nothing to ingest there.
    if CLOUD_MODE:
        st.info(
            "This hosted instance ships with a prebuilt vector index. Index "
            "building is disabled here — it runs locally where the source "
            "corpus lives. Use the questionnaire below to run profiles.",
            icon="🏛️",
        )
    else:
        with st.expander("2 · Build the vector index",
                         expanded=counts is None or (counts is not None and counts["chunks"].sum() == 0)):
            st.caption(
                "Parses the published PDF corpora and third-party RTF dumps, chunks, "
                "embeds via Together, and stores everything in Chroma. Thousands of "
                "embedding calls — run once, it persists on disk. Resumable."
            )
            fresh_ingest = st.checkbox("Rebuild from scratch (--fresh)", value=False,
                                       key="fresh_ingest")
            if st.button("Build index", type="primary",
                         disabled=not api_key_present()):
                args = [PYTHON, "scripts/01_ingest.py"] + (["--fresh"] if fresh_ingest else [])
                with st.status("Building index…", expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    if rc == 0:
                        status.update(label="Index built ✅", state="complete")
                        index_counts.clear()
                    else:
                        status.update(label=f"Ingest failed (exit {rc})", state="error")

    # --- Profiles ---
    with st.expander("3 · Run the questionnaire", expanded=True):
        n_q = runs.QUESTIONS_PER_ORG
        st.caption(
            f"Runs the fixed {n_q}-question questionnaire per selected (lab, source) "
            "pair: RAG answer + graded matching per question. Resumable — "
            "completed questions are skipped on rerun."
        )
        with st.expander("ℹ️ How the pipeline computes each answer",
                         expanded=False):
            st.markdown(
                "What one question costs and how it is processed, end to end. "
                "Everything before the LLM is deterministic arithmetic; the "
                "two LLM calls run at **temperature 0** (greedy decoding)."
            )
            st.markdown(
                "**Stage 0 — chunking (done once, at ingest).** Documents are "
                "cut into a sliding window of "
                f"$L={CHUNK_SIZE}$ characters with $O={CHUNK_OVERLAP}$ "
                "characters of overlap, preferring sentence/newline breaks. "
                "Consecutive chunks therefore start $L-O$ apart:"
            )
            st.latex(rf"\text{{start}}_{{j+1}}=\text{{start}}_j+({CHUNK_SIZE}-{CHUNK_OVERLAP})")
            st.markdown(
                "**Stage 0b — near-duplicate removal.** The third-party press "
                "dumps are heavily syndicated, so each chunk is keyed by a "
                "normalized 240-character signature and only the first "
                "occurrence per (lab, source) is embedded:"
            )
            st.latex(
                r"\mathrm{sig}(c)=\mathrm{lower}\bigl(\mathrm{collapse\_ws}(c)\bigr)"
                r"[0:240]"
            )
            st.markdown(
                "**Stage 1 — retrieval.** The question is embedded and scored "
                "against the index by **cosine similarity**, restricted to one "
                "(lab, source) pair. Chroma returns a distance, which the "
                "retriever converts back to a similarity:"
            )
            st.latex(
                r"\mathrm{score}(q,c)=1-d_{\cos}(q,c)"
                r"=\frac{v_q\cdot v_c}{\lVert v_q\rVert\,\lVert v_c\rVert}"
            )
            st.markdown(
                f"To survive duplicates that slipped past ingest, retrieval "
                f"**over-fetches** $k\\times6$ candidates, drops repeated "
                f"signatures, and keeps the top $k={TOP_K}$ distinct chunks."
            )
            st.markdown(
                "**Stage 2 — the answer (LLM call 1).** The answering model "
                "sees *only* those $k$ excerpts — never the logics taxonomy or "
                "the reference answers. That separation is what makes the "
                "next stage meaningful: the answer reflects the corpus, not "
                "the classification scheme."
            )
            st.markdown(
                "**Stage 3 — graded matching (LLM call 2).** The matcher sees "
                "the answer and the seven reference answers for that question, "
                "and returns a raw weight per logic. The code then **clamps "
                "and renormalizes** — these guarantees are never trusted to "
                "the model:"
            )
            st.latex(
                r"\tilde{w}_k=\max(0,\,\hat{w}_k), \qquad "
                r"w_k=\frac{\tilde{w}_k}{\sum_{m}\tilde{w}_m}"
                r"\quad\text{if}\quad \textstyle\sum_m \tilde{w}_m>0"
            )
            st.markdown(
                "**Stage 4 — abstention.** If the model sets the abstain flag, "
                "*or* the weights sum to zero (a malformed or empty verdict), "
                "the row is recorded as an abstention with all weights forced "
                "to 0 — so 'no evidence' can never leak weight into any logic:"
            )
            st.latex(
                r"\mathrm{abstain}=\mathrm{flag}\ \vee\ "
                r"\Bigl[\textstyle\sum_m \tilde{w}_m \le 0\Bigr]"
                r"\;\Longrightarrow\; w=\mathbf{0}"
            )
            symbol_glossary([
                (r"$L$", f"the **chunk size** in characters "
                         f"(`CHUNK_SIZE` = {CHUNK_SIZE})"),
                (r"$O$", f"the **overlap** between consecutive chunks "
                         f"(`CHUNK_OVERLAP` = {CHUNK_OVERLAP})"),
                (r"$L-O$", "the **stride** — how far the window advances, so "
                           "consecutive chunks share $O$ characters"),
                (r"$k$", f"the number of **chunks retrieved per question** "
                         f"(`TOP_K` = {TOP_K})"),
                (r"$q,\;c$", "**a question** and **a chunk**"),
                (r"$v_q,\;v_c$", "their **embeddings** (1024 numbers each)"),
                (r"$d_{\cos}$", "**cosine distance**, which Chroma returns; "
                                "similarity is $1-d_{\\cos}$"),
                (r"$\hat{w}_k$", "the **raw weight** the matcher returned for "
                                 "logic $k$ — untrusted until checked"),
                (r"$\tilde{w}_k$", "the weight after **clamping** negatives to "
                                   "zero (tilde = *partly processed*)"),
                (r"$w_k$", "the **final weight** after renormalizing so the "
                           "seven sum to 1"),
                (r"$\mathbf{0}$", "the **all-zero vector** — what an "
                                  "abstention stores, so 'no evidence' can "
                                  "never leak weight into a logic"),
                (r"$\vee$", "**logical OR**"),
                (r"$\Longrightarrow$", "**implies** / therefore"),
                *_SET_NOTATION,
            ])
            st.markdown(
                "**Design decisions**\n"
                "- *Why two separate LLM calls rather than one?* Asking a "
                "single call to both read the evidence and pick a logic would "
                "let the taxonomy steer what it reads. Splitting them "
                "enforces the separation above.\n"
                "- *Why clamp and renormalize in code?* An LLM asked for "
                "numbers summing to 1 will occasionally return negatives, "
                "omissions, or sums like 0.97. The invariant the aggregation "
                "depends on is enforced deterministically instead.\n"
                "- *Cost.* Each question is exactly 1 embedding + 1 answer "
                "call + 1 matcher call, so a full six-profile run is "
                f"{6 * n_q} of each. Runs are resumable: completed questions "
                "are skipped, so an interrupted run is never re-billed."
            )
        c1, c2 = st.columns(2)
        sel_orgs = c1.multiselect("Labs", ORGS, default=ORGS)
        sel_sources = c2.multiselect("Source types", SOURCE_TYPES, default=SOURCE_TYPES)
        fresh_prof = st.checkbox(
            "Start a NEW run snapshot (--fresh) — keeps previous runs for comparison",
            value=True, key="fresh_prof")
        run_label = st.text_input(
            "Run label (optional)",
            placeholder="e.g. questionnaire v2 — rewrote Authority + Strategy",
            disabled=not fresh_prof,
            help="Names this snapshot so you can recognize it in Results and Compare.",
        )
        if not fresh_prof and runs.get_current():
            cur = runs.read_meta(runs.get_current())
            st.caption(f"↻ Will resume the active run **{runs.display_name(cur)}** "
                       "(only unanswered questions are run).")
        h1, h2 = st.columns(2)
        opt_grounding = h1.checkbox(
            "Grounding pre-check (--grounding)", value=False, key="opt_grounding",
            help="Scores question↔chunk overlap and buckets each row as "
                 "retrieval_missed / abstained / committed. No extra API calls. "
                 "Results appear on the Hallucination tab.")
        opt_quotes = h2.checkbox(
            "Quote-grounded answers (--quotes)", value=False, key="opt_quotes",
            help="Requires the answer model to return verbatim supporting quotes, "
                 "verified in code against the retrieved chunks. Same call count; "
                 "results appear on the Audit and Hallucination tabs.")
        n_pairs = len(sel_orgs) * len(sel_sources)
        st.caption(f"Selected: {n_pairs} profile(s) × {n_q} questions = "
                   f"{n_pairs * n_q} RAG + {n_pairs * n_q} matcher calls.")
        if st.button("Run profiles", type="primary",
                     disabled=not api_key_present() or not sel_orgs or not sel_sources):
            args = [PYTHON, "scripts/02_run_profiles.py",
                    "--orgs", *sel_orgs, "--sources", *sel_sources]
            if fresh_prof:
                args.append("--fresh")
            if run_label.strip():
                args += ["--label", run_label.strip()]
            if opt_grounding:
                args.append("--grounding")
            if opt_quotes:
                args.append("--quotes")
            with st.status("Running profiles…", expanded=True) as status:
                rc = stream_subprocess(args, st.empty())
                if rc == 0:
                    status.update(label="Profiles complete ✅", state="complete")
                else:
                    status.update(label=f"Run failed (exit {rc})", state="error")
            st.rerun()

# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------
with tab_results:
    # Imported here explicitly: the analysis expanders below chart with
    # altair before the profile charts further down would have imported it.
    import altair as alt

    res_run = run_selectbox("Run to view", key="results_run",
                            default_run_id=runs.get_current())
    res_meta = runs.read_meta(res_run) if res_run else {}
    profiles = load_profiles(res_run)
    if not profiles:
        st.info("No results yet — run the pipeline on the **Run** tab first.")
    else:
        st.header("Alignment profiles")
        if res_meta:
            st.caption(
                f"Run **{runs.display_name(res_meta)}** · "
                f"created {res_meta.get('created_at', '?')} · "
                f"{res_meta.get('answered', 0)} answered / "
                f"{res_meta.get('abstained', 0)} abstained"
            )

        with st.expander("🧾 Run provenance — exactly what produced these numbers",
                         expanded=False):
            st.caption(
                "Recorded in the run's meta.json when it was created. Kept "
                "because a percentage is only interpretable alongside the "
                "configuration that generated it."
            )
            prov = {
                "run id": res_meta.get("run_id", "—"),
                "label": res_meta.get("label") or "(none)",
                "status": res_meta.get("status", "—"),
                "created": res_meta.get("created_at", "—"),
                "last updated": res_meta.get("updated_at", "—"),
                "answering + matching model": res_meta.get("generation_model", "—"),
                "embedding model": res_meta.get("embedding_model", "—"),
                "chunks retrieved per question (k)": res_meta.get("k", "—"),
                "labs": ", ".join(res_meta.get("orgs") or []) or "—",
                "source types": ", ".join(res_meta.get("source_types") or []) or "—",
                "questions recorded": res_meta.get("questions", "—"),
                "answered / abstained": (f"{res_meta.get('answered', 0)} / "
                                         f"{res_meta.get('abstained', 0)}"),
            }
            # Values are deliberately stringified: the column mixes ints (k,
            # question counts) with text, and a mixed-type object column fails
            # Arrow serialization.
            st.dataframe(
                pd.DataFrame({"setting": list(prov),
                              "value": [str(v) for v in prov.values()]}),
                hide_index=True, width="stretch")
            st.caption(
                "All LLM calls ran at temperature 0. That is greedy decoding, "
                "not a bit-reproducibility guarantee: on shared GPU "
                "infrastructure batching and floating-point ordering can still "
                "flip a near-tie token, so a re-run of the identical "
                "questionnaire can move a profile by a few points. Compare any "
                "difference against the confidence intervals below before "
                "reading it as a real change."
            )

        with st.expander("ℹ️ How the profile percentages are computed",
                         expanded=False):
            st.markdown(
                "Aggregation happens in `il_rag/profile_harness.py`. Every "
                "**answered** question contributed a weight vector over the "
                "seven logics that the matcher normalized to sum to 1:"
            )
            st.latex(r"w^{(i)}=\bigl(w^{(i)}_1,\dots,w^{(i)}_7\bigr),\qquad "
                     r"\sum_{k=1}^{7} w^{(i)}_k = 1,\qquad w^{(i)}_k \ge 0")
            st.markdown(
                "**Step 1 — split answered from abstained.** For one "
                "(lab, source) pair, let $A$ be its **answered** questions and "
                "$B$ its abstentions. Abstained rows carry an all-zero weight "
                "vector and are removed from the denominator entirely:"
            )
            st.latex(r"n_{\text{answered}} = |A|, \qquad "
                     r"n_{\text{abstained}} = |B|")
            st.markdown(
                "**Step 2 — profile percentage per logic**: the mean weight "
                "across answered questions, ×100."
            )
            st.latex(r"P_k \;=\; \frac{100}{|A|}\sum_{i \in A} w^{(i)}_k")
            st.markdown(
                "Because every $w^{(i)}$ sums to 1, the seven $P_k$ sum to "
                "$\\approx 100$ (small deviations are display rounding to 2 "
                "decimals). **Step 3 — per-category breakdown** is the same "
                "formula restricted to the questions of one category $c$:"
            )
            st.latex(r"P^{(c)}_k \;=\; \frac{100}{|A_c|}\sum_{i \in A_c} "
                     r"w^{(i)}_k, \qquad A_c=\{i \in A: \mathrm{cat}(i)=c\}")
            symbol_glossary([
                (r"$i$", "**a question** — one row of the audit trail "
                         "(one lab × source × question)"),
                (r"$k$", "**a logic** — one of the seven"),
                (r"$w^{(i)}_k$", "the **weight** the matcher gave logic $k$ on "
                                 "question $i$; the superscript in brackets is "
                                 "a label (*which question*), not a power"),
                (r"$A$", "the set of **answered** questions for this "
                         "(lab, source) pair"),
                (r"$B$", "the set of **abstained** questions — counted, but "
                         "excluded from every denominator"),
                (r"$\lvert A\rvert$", "**how many** questions were answered — "
                                      "the denominator, and the $n$ that drives "
                                      "confidence-interval width"),
                (r"$P_k$", "the **profile percentage** for logic $k$ — what the "
                           "bar chart plots"),
                (r"$A_c$", "the answered questions **belonging to category "
                           "$c$** — used for the per-category breakdown"),
                (r"$P^{(c)}_k$", "the same percentage computed **within one "
                                 "category** (parenthesised superscript = a "
                                 "label, not a power)"),
                *_SET_NOTATION,
            ])
            st.markdown(
                "**Design decisions**\n"
                "- *Why exclude abstentions instead of scoring them as zeros?* "
                "Dividing by all 27 questions would let a silent corpus drag "
                "every logic toward 0 and silently rescale the profile. "
                "Excluding them means silence reduces **confidence** (a smaller "
                "$|A|$, hence wider confidence intervals) but never **shifts** "
                "the distribution.\n"
                "- *Why the mean and not a vote count?* Institutional logics "
                "co-exist; the matcher may legitimately split a question "
                "60/40 across two logics. Averaging the full weight vectors "
                "preserves those mixtures, whereas counting argmax winners "
                "would discard them.\n"
                "- *Every question weighs the same.* There is no confidence "
                "weighting: a question the matcher graded 0.99/0.01 counts "
                "exactly as much as one it graded 0.30/0.25/0.25/0.20."
            )
            st.markdown(
                "**Sanity-check banner thresholds.** The banner below reads the "
                "largest Family or Religion percentage across all six profiles, "
                "$m=\\max P_{\\text{Family}},P_{\\text{Religion}}$, and reports "
                "**passed** at $m \\le 5$, **borderline** at $5 < m \\le 15$, "
                "and **FAILED** above 15. These cutoffs are presentational "
                "heuristics, not statistical tests — Family and Religion have "
                "no natural place in an AI lab's institutional environment, so "
                "a high score means the instrument is misfiring."
            )

        # --- Bootstrap confidence intervals (optional, zero-API, post-hoc) ---
        ci_data = load_bootstrap_ci(res_run)
        with st.expander("Confidence intervals (bootstrap over questions)",
                         expanded=bool(ci_data)):
            st.caption(
                "Each % is a mean over the answered questions, so its error bar "
                "comes from resampling those questions with replacement. Wide "
                "bars mean the estimate leans on which questions were asked — "
                "expected with ~27 questions. Zero API cost, deterministic."
            )
            with st.expander("ℹ️ How the confidence intervals are computed",
                             expanded=False):
                st.markdown(
                    "The nonparametric bootstrap (Efron, 1979), implemented in "
                    "`il_rag/bootstrap_ci.py`. A profile percentage is a "
                    "**sample mean** over a finite questionnaire, so its "
                    "uncertainty is estimated by resampling that sample."
                )
                st.markdown(
                    "**Step 1 — the observed sample.** For one (lab, source) "
                    "pair, stack its $n=|A|$ answered weight vectors into a "
                    "matrix $W \\in \\mathbb{R}^{n \\times 7}$. The point "
                    "estimate is the column mean — identical to the percentage "
                    "shown on the chart:"
                )
                st.latex(r"\hat{P}_k = \frac{100}{n}\sum_{i=1}^{n} W_{ik}")
                st.markdown(
                    "**Step 2 — resample with replacement.** Draw $n$ row "
                    "indices uniformly *with replacement* (so a question may "
                    "appear twice or not at all) and recompute the mean. "
                    "Repeat $B$ times ($B = $ iterations, default 2000):"
                )
                st.latex(
                    r"I^{(b)}_1,\dots,I^{(b)}_n \overset{\text{iid}}{\sim} "
                    r"\mathrm{Uniform}\{1,\dots,n\}, \qquad "
                    r"\hat{P}^{*(b)}_k=\frac{100}{n}\sum_{j=1}^{n} W_{I^{(b)}_j k}"
                )
                st.markdown(
                    "**Step 3 — percentile interval.** Sort the $B$ bootstrap "
                    "means per logic and read the empirical quantiles. For a "
                    "$1-\\alpha$ interval (default 95%, so $\\alpha=0.05$):"
                )
                st.latex(
                    r"\mathrm{CI}_k=\Bigl[\,Q_{\alpha/2}\bigl(\hat{P}^{*}_k\bigr),"
                    r"\;Q_{1-\alpha/2}\bigl(\hat{P}^{*}_k\bigr)\,\Bigr]"
                )
                symbol_glossary([
                    (r"$n$", "the number of **answered questions** for this "
                             "(lab, source) pair — the sample being resampled"),
                    (r"$W$", "the $n \\times 7$ **matrix** of weight vectors: "
                             "one row per answered question, one column per "
                             "logic"),
                    (r"$W_{ik}$", "the weight in **row $i$, column $k$** — "
                                  "question $i$'s weight on logic $k$"),
                    (r"$\hat{P}_k$", "the **point estimate** — the percentage "
                                     "actually plotted (hat = *estimated from "
                                     "data*)"),
                    (r"$B$", "the number of **bootstrap resamples** "
                             "(`iterations`, default 2000)"),
                    (r"$b$", "**which** resample, from 1 to $B$"),
                    (r"$I^{(b)}_j$", "the $j$-th **randomly drawn row index** "
                                     "in resample $b$ — drawn *with "
                                     "replacement*, so rows can repeat"),
                    (r"$\hat{P}^{*(b)}_k$", "the percentage recomputed from "
                                            "resample $b$; the asterisk marks "
                                            "it as a **bootstrap replicate**, "
                                            "not a real observation"),
                    (r"$\alpha$", "**1 − confidence level** (0.05 for a 95% "
                                  "interval)"),
                    (r"$Q_p$", "the **$p$-th quantile** — the value below which "
                               "that fraction of the resamples fall"),
                    (r"$\mathrm{CI}_k$", "the resulting **interval** for logic "
                                         "$k$: the whisker on the chart"),
                    (r"$\sim \mathrm{Uniform}$", "**drawn at random**, every "
                                                 "row equally likely"),
                    (r"$\text{iid}$", "each draw is **independent** and from "
                                      "the same distribution"),
                ])
                st.markdown(
                    "The reported `std` is the standard deviation of those "
                    "same $B$ replicates (the bootstrap standard error). With "
                    "$n < 2$ the interval is undefined and is reported as "
                    "zero-width.\n\n"
                    "**Design decisions**\n"
                    "- *Why bootstrap the questions instead of re-running the "
                    "pipeline $N$ times?* Both are legitimate but answer "
                    "different questions. This one answers **“how much does "
                    "the profile depend on *which questions* we asked?”** — "
                    "the instrument's sampling uncertainty. A repeat-run study "
                    "would instead measure decoding variance.\n"
                    "- *Why percentile and not normal-approximation intervals?* "
                    "Weights are bounded in $[0,1]$ and often skewed (many "
                    "exact zeros), so a symmetric $\\hat{P}\\pm1.96\\,\\mathrm{SE}$ "
                    "interval can run past 0 or 100. Percentile bounds cannot.\n"
                    "- *Reproducibility.* The resampling uses a **seeded** RNG "
                    "(`numpy.random.default_rng(seed)`), so identical inputs "
                    "give byte-identical intervals every time.\n"
                    "- *Reading the width.* Interval width shrinks roughly as "
                    "$1/\\sqrt{n}$, so with ~20–27 answered questions per "
                    "profile the bars are genuinely wide. Dominant-logic "
                    "*rankings* are far more robust than the exact percentages."
                )
            if st.button("Compute / refresh confidence intervals"):
                args = [PYTHON, "scripts/05_run_bootstrap_ci.py",
                        "--run", res_run]
                with st.status("Bootstrapping…", expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    status.update(
                        label="Confidence intervals ready ✅" if rc == 0
                        else f"Failed (exit {rc})",
                        state="complete" if rc == 0 else "error")
                st.rerun()
            if ci_data:
                st.caption(f"{int(ci_data['ci'] * 100)}% CI · "
                           f"{ci_data['iterations']} resamples · seed "
                           f"{ci_data['seed']} — error bars shown on the charts "
                           "below; full table in the download.")
                ci_rows = []
                for _org, _by_st in ci_data["profiles"].items():
                    for _st, _by_logic in _by_st.items():
                        for _logic, s in _by_logic.items():
                            ci_rows.append({
                                "lab": _org, "source": _st, "logic": _logic,
                                "mean %": s["mean"], "CI low": s["lo"],
                                "CI high": s["hi"],
                                "width": round(s["hi"] - s["lo"], 2),
                                "std err": s["std"], "n questions": s["n"],
                            })
                if ci_rows:
                    st.markdown("**Full interval table** — the numbers behind "
                                "the whiskers, including the bootstrap "
                                "standard error and the sample size each "
                                "interval rests on:")
                    ci_df = pd.DataFrame(ci_rows)
                    fc1, fc2 = st.columns(2)
                    f_lab = fc1.multiselect("Lab", sorted(ci_df["lab"].unique()),
                                            key="ci_tbl_lab")
                    f_src = fc2.multiselect("Source",
                                            sorted(ci_df["source"].unique()),
                                            key="ci_tbl_src")
                    v = ci_df
                    if f_lab:
                        v = v[v["lab"].isin(f_lab)]
                    if f_src:
                        v = v[v["source"].isin(f_src)]
                    st.dataframe(v, hide_index=True, width="stretch")
                    st.caption(
                        "`width` is how many percentage points the interval "
                        "spans — the honest precision of that estimate. A "
                        "logic whose interval overlaps another's cannot be "
                        "ranked against it from this run alone."
                    )
                ci_csv = runs.run_dir(res_run) / "bootstrap_ci" / "ci.csv"
                if ci_csv.exists():
                    st.download_button("ci.csv", ci_csv.read_bytes(),
                                       file_name=f"bootstrap_ci_{res_run}.csv")

        # --- Replicate runs: decoding noise, averaged and measured ---
        with st.expander("Replicate runs (decoding noise)", expanded=False):
            st.caption(
                "Temperature 0 is greedy decoding, but not bit-reproducible on "
                "shared GPUs — re-asking the identical questionnaire moves the "
                "profiles. Averaging replicates cancels that; the spread across "
                "them measures how big it is."
            )
            with st.expander("ℹ️ How replicate averaging works", expanded=False):
                st.markdown(
                    "Implemented in `il_rag/replicates.py`. For one "
                    "(lab, source, logic) observed across $R$ replicate runs "
                    "as $P_1..P_R$:"
                )
                st.latex(r"\bar{P}=\frac{1}{R}\sum_{r=1}^{R}P_r, \qquad "
                         r"s=\sqrt{\frac{\sum_r (P_r-\bar{P})^2}{R-1}}, \qquad "
                         r"\mathrm{SEM}=\frac{s}{\sqrt{R}}")
                symbol_glossary([
                    (r"$R$", "the **number of replicate runs** (the slider "
                             "below; default 3)"),
                    (r"$r$", "**which** replicate, from 1 to $R$"),
                    (r"$P_r$", "the percentage this logic scored **in run $r$**"),
                    (r"$\bar{P}$", "the **mean** across replicates — the bar "
                                   "over a letter means *average of*"),
                    (r"$s$", "the **standard deviation** between runs: the "
                             "typical distance of a single run from the mean, "
                             "i.e. the size of the decoding noise"),
                    (r"$R-1$", "**degrees of freedom** — dividing by $R-1$ "
                               "rather than $R$ corrects the bias of estimating "
                               "spread around a mean you also estimated; at "
                               "$R=2$ this is 1, which is why the SD is "
                               "withheld"),
                    (r"$\mathrm{SEM}$", "the **standard error of the mean** — "
                                        "how precise the *average* is, which "
                                        "shrinks as $1/\\sqrt{R}$"),
                ])
                st.markdown(
                    "The mean's standard error shrinks as $1/\\sqrt{R}$, so "
                    "3–4 replicates roughly halve the jitter of a single run. "
                    "The SD is withheld below 3 replicates — with two runs it "
                    "has a single degree of freedom and would be misleading; "
                    "the observed range is shown instead."
                )
                st.markdown(
                    "**Two different uncertainties — report both.**\n\n"
                    "| | question |\n|---|---|\n"
                    "| **bootstrap CI** | how much would this move if we had "
                    "asked a *different sample of questions*? |\n"
                    "| **between-run SD** | how much does it move if we ask "
                    "the *same* questions again? |\n\n"
                    "Neither substitutes for the other. Combining them in "
                    "quadrature, $\\sqrt{\\mathrm{SE}_{\\text{boot}}^2 + "
                    "\\mathrm{SEM}^2}$, assumes they are independent — a "
                    "reasonable but stated assumption."
                )
                st.markdown(
                    "**Design decisions**\n"
                    "- *Only same-questionnaire runs may be averaged.* "
                    "Reference wording changes what the matcher grades "
                    "against, so mixing question sets averages two different "
                    "measurements. Each run stores the questionnaire that "
                    "produced it; the tool fingerprints it (SHA-256 of "
                    "canonical JSON) and refuses mismatches unless forced.\n"
                    "- *`dominant agreement`* is label-level stability: the "
                    "share of replicates whose top logic matched the top logic "
                    "of the averaged profile. It often holds at 100% even when "
                    "the percentages move — which is why rankings are the "
                    "safer claim.\n"
                    "- *Profiles that abstained everywhere are excluded* rather "
                    "than averaged in as zeros."
                )
            st.markdown(
                "**Run replicates.** Each replicate is a *complete* run of the "
                "questionnaire, so the cost is R× a normal run — scope it down "
                "while iterating.")
            rc1, rc2 = st.columns(2)
            rep_orgs = rc1.multiselect("Labs", ORGS, default=ORGS,
                                       key="rep_orgs")
            rep_sources = rc2.multiselect("Source types", SOURCE_TYPES,
                                          default=SOURCE_TYPES, key="rep_sources")
            n_reps = st.slider(
                "Number of replicate runs", min_value=2, max_value=10, value=3,
                key="rep_n",
                help="3 is the minimum that yields a standard deviation (2 runs "
                     "leave it a single degree of freedom, so only the range is "
                     "reported). The mean's standard error falls as 1/sqrt(R), "
                     "so the gain per extra run shrinks quickly.")
            rep_label = st.text_input(
                "Label (optional)", placeholder="e.g. finalized questionnaire",
                key="rep_label",
                help="Each replicate is saved as its own run snapshot, "
                     "numbered i/R.")
            _n_q = runs.QUESTIONS_PER_ORG
            _pairs = len(rep_orgs) * len(rep_sources)
            _calls = _pairs * _n_q * n_reps
            st.caption(
                f"**{n_reps} replicates × {_pairs} profile(s) × {_n_q} "
                f"questions = {_calls} answer + {_calls} matcher calls.** "
                f"Standard error of the mean shrinks to "
                f"{1 / n_reps ** 0.5:.0%} of a single run's."
            )
            if n_reps < 3:
                st.warning(
                    "With 2 replicates the standard deviation has one degree "
                    "of freedom and is withheld — you get the observed range "
                    "only. Use 3+ for an SD.", icon="⚠️")
            if st.button("Run replicates", type="primary",
                         disabled=not api_key_present() or not rep_orgs
                         or not rep_sources, key="rep_run_btn"):
                args = [PYTHON, "scripts/08_run_replicates.py", "run",
                        "--n", str(n_reps),
                        "--orgs", *rep_orgs, "--sources", *rep_sources]
                if rep_label.strip():
                    args += ["--label", rep_label.strip()]
                with st.status(f"Running {n_reps} replicates…",
                               expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    status.update(
                        label=("Replicates complete ✅" if rc == 0
                               else f"Failed (exit {rc})"),
                        state="complete" if rc == 0 else "error")
                load_replicate_reports.clear()
                st.rerun()
            st.caption(
                "Equivalent CLI: "
                f"`scripts/08_run_replicates.py run --n {n_reps}` · to average "
                "runs you already have: `scripts/08_run_replicates.py "
                "aggregate --runs <a> <b> <c>`")
            rep = load_replicate_reports()
            if not rep:
                st.info("No replicate report yet.", icon="🔁")
            else:
                pick = st.selectbox(
                    "Replicate group", list(rep),
                    format_func=lambda g: (
                        f"{g} — {rep[g]['n_replicates']} runs"),
                    key="rep_group")
                r = rep[pick]
                st.caption(
                    f"Runs averaged: {', '.join(r['run_ids'])} · "
                    f"questionnaire fingerprint `{r.get('questionnaire_fingerprint')}`"
                    + ("  ⚠️ MIXED questionnaires" if r.get("mixed_questionnaires")
                       else ""))
                rep_rows = []
                for _org, _by in r["profiles"].items():
                    for _st, d in _by.items():
                        for _logic, s in d["logics"].items():
                            rep_rows.append({
                                "lab": _org, "source": _st, "logic": _logic,
                                "mean %": s["mean"], "sd": s["sd"],
                                "sem": s["sem"], "min": s["min"],
                                "max": s["max"], "range": s["range"],
                                "runs": d["replicates"],
                            })
                st.dataframe(pd.DataFrame(rep_rows), hide_index=True,
                             width="stretch")
                dom = [{"lab": o, "source": s,
                        "averaged dominant": d["mean_dominant"],
                        "agreed in": (f"{d['dominant_agreement']:.0%}"
                                      if d.get("dominant_agreement") is not None
                                      else "—"),
                        "per run": ", ".join(d["dominant_per_run"])}
                       for o, by in r["profiles"].items() for s, d in by.items()]
                st.markdown("**Label stability** — did each replicate agree on "
                            "the dominant logic?")
                st.dataframe(pd.DataFrame(dom), hide_index=True, width="stretch")

        # --- Embedding agreement: the non-LLM second judge (post-hoc) ---
        emb = load_embedding_summary(res_run)
        with st.expander("Embedding agreement (non-LLM second judge)",
                         expanded=False):
            st.caption(
                "Embeds every committed answer and the run's own seven reference "
                "answers per category, ranks the references by cosine similarity, "
                "and checks whether the nearest reference's logic agrees with the "
                "LLM matcher's top logic. Deterministic and LLM-free. Absolute "
                "cosine values are NOT interpretable (e5 compresses them into a "
                "narrow band) — only the ranking and the top1–top2 margin are."
            )
            with st.expander("ℹ️ How embedding agreement is computed",
                             expanded=False):
                st.markdown(
                    "Implemented in `il_rag/embedding_agreement.py`. Every "
                    "**committed** answer is compared against the seven reference "
                    "answers *for its own question* (base text plus any "
                    "per-question override — the identical references the LLM "
                    "matcher saw). Abstained rows are skipped. No LLM is involved "
                    "here: embeddings and arithmetic only, so the result is "
                    "exactly reproducible."
                )
                st.markdown(
                    "**Step 1 — embed and measure cosine similarity.** Answers and "
                    "references are embedded with the same e5 model used for "
                    "retrieval (each truncated to 1400 characters to respect its "
                    "512-token limit; answers state their conclusion first, so the "
                    "head carries the signal). For each logic $k$:"
                )
                st.latex(r"s_k=\cos\bigl(v_{\text{answer}},\,v_{\text{ref}_k}\bigr)"
                         r"=\frac{v_{\text{answer}}\cdot v_{\text{ref}_k}}"
                         r"{\lVert v_{\text{answer}}\rVert\,\lVert v_{\text{ref}_k}\rVert}")
                st.markdown(
                    "**Step 2 — the binary verdict.** Take the nearest reference "
                    "and compare it with the matcher's top-weighted logic. The "
                    "**margin** records how decisive that pick was:"
                )
                st.latex(
                    r"\hat{k}=\arg\max_k s_k, \qquad "
                    r"\mathrm{agree}=\bigl[\hat{k}=\arg\max_k w_k\bigr], \qquad "
                    r"\mathrm{margin}=s_{(1)}-s_{(2)}"
                )
                st.markdown(
                    "**Step 3 — graded closeness shares.** Convert the seven "
                    "similarities into proportions by subtracting the *farthest* "
                    "reference's similarity, then normalizing (min-shifted "
                    "normalization). The farthest logic receives exactly 0; a row "
                    "with no spread at all degrades to uniform $1/7$:"
                )
                st.latex(
                    r"\sigma_k=\frac{s_k-\min_j s_j}{\sum_{m}\bigl(s_m-\min_j s_j\bigr)},"
                    r"\qquad \sum_k \sigma_k = 1"
                )
                st.markdown(
                    "**Step 4 — two continuous comparisons** against the matcher's "
                    "weight vector $w$. The first reads the share landing on the "
                    "matcher's single top pick; the second compares the *whole* "
                    "shape via the overlapping coefficient (histogram "
                    "intersection):"
                )
                st.latex(
                    r"\mathrm{share\_on\_top}=\sigma_{\arg\max_k w_k}, \qquad "
                    r"\mathrm{overlap}=\sum_{k=1}^{7}\min(\sigma_k, w_k)"
                )
                st.markdown(
                    "**Step 5 — baselines.** Each metric is reported against what "
                    "an uninformative judge would score, so the numbers can be "
                    "read as *above chance* rather than in the abstract:"
                )
                st.latex(
                    r"\text{chance share}=\tfrac{1}{7}\approx 0.143, \qquad "
                    r"\text{uniform overlap}=\sum_{k}\min\bigl(\tfrac{1}{7},\,w_k\bigr)"
                )
                symbol_glossary([
                    (r"$k$", "**a logic** — one of the seven"),
                    (r"$j,\;m$", "*other* logics — used when a formula holds $k$ "
                                 "fixed and ranges over all seven"),
                    (r"$v_{\text{answer}}$", "the answer's **embedding**: 1024 "
                                             "numbers encoding its meaning"),
                    (r"$v_{\mathrm{ref}_k}$", "the **embedding of logic $k$'s "
                                              "reference answer** for this question"),
                    (r"$s_k$", "**cosine similarity** between the answer and "
                               "reference $k$, in $[-1,1]$ (in practice ≈0.78–0.87)"),
                    (r"$s_{(1)},\,s_{(2)}$", "the **highest and second-highest** "
                                             "similarities, once sorted"),
                    (r"$\mathrm{margin}$", "$s_{(1)}-s_{(2)}$ — how **decisive** "
                                           "the winning pick was; near 0 = a tie"),
                    (r"$\sigma_k$", "the **closeness share** after min-shifting, so "
                                    "the seven sum to 1"),
                    (r"$\hat{k}$", "the **nearest** logic (the hat means *chosen*)"),
                    (r"$w_k$", "the **LLM matcher's weight** for logic $k$"),
                    (r"$\lVert v \rVert$", "the **length (norm)** of vector $v$ — "
                                           "dividing by it is what makes cosine "
                                           "measure direction, not magnitude"),
                    (r"$\cdot$", "the **dot product** of two vectors"),
                    *_SET_NOTATION,
                ])
                st.markdown(
                    "All five quantities are averaged for the overall summary and "
                    "for each category and lab×source slice.\n\n"
                    "**Design decisions**\n"
                    "- *Why min-shifted shares instead of softmax?* Softmax needs "
                    "a temperature constant that would have to be tuned and "
                    "justified. Min-shifting is parameter-free and "
                    "**scale-invariant**: adding a constant to every similarity "
                    "leaves the shares unchanged. That matters because e5 "
                    "compresses cosines into a narrow high band (empirically "
                    "≈0.78–0.87 here), so only the *relative spread within a row* "
                    "carries information.\n"
                    "- *Why absolute cosines are not reported as a score.* For the "
                    "same reason: a 0.84 similarity is not '84% similar'. Only "
                    "rankings, margins and shares are interpretable.\n"
                    "- *Why keep both the binary and graded views?* The binary "
                    "argmax is the conventional, recognizable number but is harsh: "
                    "a near-tie that falls the other way counts as a full "
                    "disagreement. The graded metrics show whether the answer "
                    "*leaned* toward the matcher's pick even when the argmax "
                    "flipped.\n"
                    "- *Why overlap needs a per-row baseline.* Overlap's floor "
                    "depends on how concentrated $w$ is — a matcher putting 1.0 on "
                    "one logic can score at most $1/7$ against a uniform judge, "
                    "while a diffuse $w$ scores much higher. Share, by contrast, "
                    "has the flat $1/7$ baseline.\n"
                    "- *What this check is for.* It is **triangulation, not ground "
                    "truth**: whole-answer embeddings track topic more than "
                    "institutional stance, so treat low agreement as a limit of "
                    "surface similarity rather than evidence the matcher is wrong."
                )
            if st.button("Compute embedding agreement",
                         disabled=not api_key_present(),
                         help="One embedding per committed row + 63 references, "
                              "batched — fractions of a cent. Recomputing "
                              "overwrites the previous result for this run."):
                args = [PYTHON, "scripts/04_run_embedding_agreement.py",
                        "--run", res_run]
                with st.status("Computing embedding agreement…",
                               expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    if rc == 0:
                        status.update(label="Embedding agreement complete ✅",
                                      state="complete")
                    else:
                        status.update(label=f"Check failed (exit {rc})",
                                      state="error")
                st.rerun()

            if emb:
                o = emb["overall"]
                e1, e2, e3 = st.columns(3)
                e1.metric("Agreement with matcher",
                          f"{o['rate']:.0%}" if o.get("rate") is not None else "—",
                          help="share of committed answers where the embedding-"
                               "nearest reference logic equals the matcher's top "
                               "logic")
                e2.metric("Rows compared", o.get("n", 0))
                e3.metric("Mean top1–top2 margin", f"{emb.get('mean_margin', 0):.3f}",
                          help="how decisively the nearest reference wins; small "
                               "margins mean the embedding judge itself was "
                               "uncertain")

                # Graded (ratio-of-closeness) metrics, present on newer summaries.
                if "mean_share_on_matcher_top" in emb:
                    g1, g2 = st.columns(2)
                    g1.metric(
                        "Closeness share on matcher's pick",
                        f"{emb['mean_share_on_matcher_top']:.3f}",
                        delta=f"chance {emb.get('share_chance_baseline', 1/7):.3f}",
                        delta_color="off",
                        help="mean share of embedding closeness the matcher's top "
                             "logic receives (min-shifted shares); above 0.143 = "
                             "above chance")
                    g2.metric(
                        "Distribution overlap",
                        f"{emb['mean_overlap']:.3f}",
                        delta=f"uniform judge {emb.get('mean_overlap_uniform_baseline', 0):.3f}",
                        delta_color="off",
                        help="overlap between embedding closeness shares and the "
                             "matcher's weights (1 = identical); compare against "
                             "what a totally uninformative judge would score")

                # --- The profile this judge implies, in the profile style ---
                erows = load_embedding_rows(res_run)
                if erows is not None and not erows.empty:
                    st.markdown(
                        "**The profile this judge implies.** The same chart "
                        "as the alignment profiles above — same axes, 0-100 "
                        "scale, same source colors — but each bar is the mean "
                        "**closeness share** the embedding judge put on that "
                        "logic, x100, over the scored answers. Expect it far "
                        "flatter than the matcher's: e5 compresses "
                        "similarities, so shares hover near the uniform "
                        "1/7 = 14% — read the ranking and the gaps, not the "
                        "absolute heights.")
                    if "embedding_shares" in erows.columns \
                            and erows["embedding_shares"].notna().any():
                        render_judge_profiles(
                            erows[erows["embedding_shares"].notna()],
                            "embedding_shares")
                    else:
                        st.info("Closeness shares are not in this run's file "
                                "— it predates the graded metrics. Recompute "
                                "the check above to add them.")

                cat_rows = [{"category": c, "rate": v["rate"], "n": v["n"],
                             "share": v.get("mean_share_on_matcher_top"),
                             "overlap": v.get("mean_overlap")}
                            for c, v in emb.get("by_category", {}).items()
                            if v.get("rate") is not None]
                if cat_rows:
                    edf = pd.DataFrame(cat_rows)
                    # Older summaries lack the graded per-category means; only
                    # offer the metrics the file actually carries.
                    metric_opts = {"binary agreement rate": ("rate", None)}
                    if edf["share"].notna().all():
                        metric_opts["closeness share on matcher's pick"] = (
                            "share", emb.get("share_chance_baseline", 1 / 7))
                    if edf["overlap"].notna().all():
                        metric_opts["distribution overlap"] = ("overlap", None)
                    sel_metric = st.radio(
                        "Per-category metric", list(metric_opts),
                        horizontal=True, key="emb_cat_metric",
                        help="Binary = strict argmax match. Closeness share and "
                             "overlap are the graded (ratio) views; the dashed "
                             "rule marks the chance baseline where applicable.")
                    field, baseline = metric_opts[sel_metric]
                    emb_chart = (
                        alt.Chart(edf)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{field}:Q", title=sel_metric,
                                    scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("category:N", title=None,
                                    sort=[c for c in CATEGORIES]),
                            color=alt.Color(f"{field}:Q", legend=None,
                                            scale=alt.Scale(scheme="redyellowgreen",
                                                            domain=[0, 1])),
                            tooltip=["category",
                                     alt.Tooltip("rate:Q", format=".2f",
                                                 title="binary rate"),
                                     alt.Tooltip("share:Q", format=".3f",
                                                 title="closeness share"),
                                     alt.Tooltip("overlap:Q", format=".3f"),
                                     "n"],
                        )
                        .properties(height=220)
                    )
                    if baseline:
                        rule = (alt.Chart(pd.DataFrame({"x": [baseline]}))
                                .mark_rule(strokeDash=[4, 3], color="#666")
                                .encode(x="x:Q"))
                        emb_chart = emb_chart + rule
                    st.altair_chart(emb_chart, width="stretch")

                if erows is not None and not erows.empty:
                    dis = erows[~erows["agree"]]
                    if dis.empty:
                        st.success("✅ The embedding judge agrees with the matcher "
                                   "on every committed answer.")
                    else:
                        st.markdown(f"**{len(dis)} disagreement(s)** — where the "
                                    "two judges split (low margins mean the "
                                    "embedding side was itself a coin toss):")
                        st.dataframe(
                            dis[["org", "source_type", "qid", "matcher_top",
                                 "matcher_top_weight", "embedding_nearest",
                                 "margin"]].rename(columns={
                                     "matcher_top": "matcher says",
                                     "matcher_top_weight": "weight",
                                     "embedding_nearest": "embedding says",
                                 }),
                            hide_index=True, width="stretch")

                if erows is not None and not erows.empty:
                    with st.expander("Per-question similarities (the raw numbers)",
                                     expanded=False):
                        st.caption(
                            "Every row's seven cosine similarities, and the "
                            "min-shifted closeness shares derived from them. These "
                            "are the inputs to every aggregate above — shown "
                            "because the aggregates are otherwise unauditable. "
                            "Remember the absolute cosines are not interpretable "
                            "on their own (e5 compresses them into a narrow band); "
                            "the ranking and the shares are."
                        )
                        which = st.radio(
                            "Show", ["cosine similarities", "closeness shares"],
                            horizontal=True, key="emb_raw_which")
                        col = ("similarities" if which == "cosine similarities"
                               else "embedding_shares")
                        if col not in erows.columns:
                            st.info(
                                "Closeness shares are not in this run's file — it "
                                "predates the graded metrics. Recompute the check "
                                "above to add them.")
                        else:
                            wide = pd.DataFrame([
                                {"lab": r["org"], "source": r["source_type"],
                                 "qid": r["qid"],
                                 "matcher": r["matcher_top"],
                                 "embedding": r["embedding_nearest"],
                                 "margin": r["margin"],
                                 **{logic: (r[col] or {}).get(logic)
                                    for logic in LOGICS}}
                                for _, r in erows.iterrows()
                            ])
                            st.dataframe(wide, hide_index=True, width="stretch")

                edir = runs.run_dir(res_run) / "embedding_agreement"
                dle1, dle2 = st.columns(2)
                if (edir / "summary.json").exists():
                    dle1.download_button("summary.json",
                                         (edir / "summary.json").read_bytes(),
                                         file_name=f"embedding_summary_{res_run}.json")
                if (edir / "similarities.jsonl").exists():
                    dle2.download_button("similarities.jsonl",
                                         (edir / "similarities.jsonl").read_bytes(),
                                         file_name=f"embedding_rows_{res_run}.jsonl")

        # --- Keyword agreement: the lexical third judge (post-hoc) ---
        with st.expander("Keyword agreement (lexical third judge)",
                         expanded=False):
            st.caption(
                "The most transparent judge: a **hand-curated keyword lexicon** "
                "per logic, scored against each RAG answer on a graded ladder — "
                "verbatim, as an inflection, or as a calibrated semantic "
                "neighbour. No LLM anywhere; every verdict is still a visible "
                "list of matched words, each annotated with the rung that "
                "matched it."
            )

            with st.expander("ℹ️ How keyword agreement is computed", expanded=False):
                st.markdown(
                    "Implemented in `il_rag/keyword_agreement.py`. No LLM "
                    "calls; without the local semantic lexicon it is pure, "
                    "deterministic computation, and with it the only cost is "
                    "embedding uncached answer words."
                )
                st.markdown(
                    "**Step 1 — the lexicon.** One hand-curated keyword list "
                    "per logic ($K_\\ell$, 12-17 entries), drawn from Thornton "
                    "& Ocasio's ideal types and the questionnaire's reference "
                    "vocabulary. Curation rules: institutional signal only "
                    "(*welfare*, not *made*); one surface form per stem (the "
                    "morphological rung credits inflections); no token in two "
                    "logics' lists; phrases allowed (*peer review*). v1 derived "
                    "keywords per question from the reference answers instead — "
                    "distinctiveness within one question let generic words "
                    "through, which is why the sets read as noise and were "
                    "replaced."
                )
                with st.expander("The lexicon — all seven lists", expanded=False):
                    st.dataframe(pd.DataFrame(
                        [{"logic": k, "n": len(v), "keywords": ", ".join(v)}
                         for k, v in ka_mod.LOGIC_KEYWORDS.items()]),
                        hide_index=True, width="stretch")
                st.markdown(
                    "**Step 2 — the ladder.** Each keyword is scored against "
                    "the answer by the same graded ladder as the topic-keyword "
                    "check (`il_rag/topic_keywords.py`), cheapest rung first:"
                )
                st.markdown(
                    "| rung | fires when | score |\n|---|---|---|\n"
                    "| **exact** | the keyword occurs as a token (phrases: "
                    "adjacent) | 1.00 |\n"
                    "| **morphological** | an answer word shares its stem "
                    "(*regulation* ← *regulators*) | difflib ratio |\n"
                    "| **semantic** | an answer word is closer than "
                    "$\\tau_{sem}$ of random corpus word pairs | percentile, "
                    "cap 0.99 |\n"
                    "| **absent** | nothing cleared a bar | 0 |"
                )
                st.markdown(
                    "The semantic rung scores a **percentile against this "
                    "corpus's own background distribution** of word-pair "
                    "cosines, never a raw cosine — e5 compresses word cosines "
                    "into a ~0.74-0.82 band, so no absolute cutoff "
                    "discriminates. It engages only when the local lexicon "
                    "files exist (`data/lexicon/`, built once by "
                    "`scripts/13_run_topic_keywords.py calibrate`); otherwise "
                    "the ladder stops at the morphological rung."
                )
                st.markdown(
                    "**Step 3 — graded recall per logic.** Each keyword earns "
                    "its ladder score $s_k$, and a logic's raw score is the "
                    "mean over its list:"
                )
                st.latex(
                    r"r_\ell=\frac{1}{\lvert K_\ell\rvert}"
                    r"\sum_{k \in K_\ell} s_k"
                )
                st.markdown(
                    "Mean, not count, so a logic with many keywords is not "
                    "rewarded merely for having a longer list."
                )
                st.markdown(
                    "**Step 4 — normalize to shares and pick a winner.** Same "
                    "shape as the embedding judge, so the two are directly "
                    "comparable:"
                )
                st.latex(
                    r"\sigma_\ell=\frac{r_\ell}{\sum_m r_m},\qquad "
                    r"\hat{\ell}=\arg\max_\ell r_\ell,\qquad "
                    r"\mathrm{overlap}=\sum_\ell \min(\sigma_\ell, w_\ell)"
                )
                st.markdown(
                    "**Rows with no overlap at all** — no keyword of any logic "
                    "cleared any rung — are reported separately as "
                    "`no_overlap` and **excluded from the rates** rather than "
                    "forced to a uniform guess."
                )
                symbol_glossary([
                    (r"$\ell$", "**a logic** — one of the seven (State, Profession, "
                                "Market, Corporation, Family, Religion, Community)"),
                    (r"$m$", "*another* logic — a second letter exists only so a "
                             "formula can hold $\\ell$ fixed while counting across "
                             "all the others"),
                    (r"$K_\ell$", "logic $\\ell$'s **curated keyword list**"),
                    (r"$k$", "**one keyword** from that list"),
                    (r"$s_k$", "keyword $k$'s **ladder score** in $[0,1]$: 1.0 "
                               "exact, the character ratio if morphological, the "
                               "capped percentile if semantic, 0 if absent"),
                    (r"$\tau_{sem}$", "the **semantic bar** (config "
                                      "`KEYWORD_SEMANTIC_MIN_PERCENTILE`, 0.90): "
                                      "an answer word must be closer than 90% of "
                                      "random corpus word pairs"),
                    (r"$r_\ell$", "**graded recall** — the mean ladder score of "
                                  "$K_\\ell$, in $[0,1]$"),
                    (r"$\sigma_\ell$", "the **normalized share** — $r_\\ell$ as a "
                                       "fraction of all seven, so they sum to 1"),
                    (r"$\hat{\ell}$", "the **predicted logic** (the hat means "
                                      "*estimated*, vs plain $\\ell$ = any logic)"),
                    (r"$w_\ell$", "the **LLM matcher's weight** for logic $\\ell$ "
                                  "— what this judge is compared against"),
                    *_SET_NOTATION,
                ])
                st.markdown(
                    "**Design decisions**\n"
                    "- *Curated, not derived (v2).* v1's per-question derivation "
                    "kept any token unique to one reference — including framing "
                    "noise like *made* or *fill*. A fixed lexicon is auditable "
                    "as a document, identical across runs, and changes only "
                    "deliberately.\n"
                    "- *One ladder for both keyword features.* The thresholds, "
                    "the word-vector cache and the corpus calibration are "
                    "shared with the topic-keyword check, so a rung means the "
                    "same thing wherever it appears.\n"
                    "- *Synonymy is now visible, not fixed.* The semantic rung "
                    "sees *government* ≈ *state*, but e5 word vectors capture "
                    "topical relatedness rather than true synonymy (antonyms "
                    "embed close), so a semantic match is graded evidence, and "
                    "the annotated word pair is stored so you read the match, "
                    "never the score alone.\n"
                    "- *Why keep a third judge at all.* The three fail in "
                    "different ways — the matcher can be swayed by rhetoric, "
                    "embeddings by topical similarity, keywords by wording. "
                    "Where all three agree, the classification is hard to "
                    "dismiss; where they split, the matched-word lists show "
                    "exactly why."
                )

            if st.button("Compute keyword agreement",
                         help="No LLM calls. Without the local semantic lexicon "
                              "this is pure computation; with it, only uncached "
                              "answer words are embedded (append-only cache). "
                              "Recomputing overwrites the previous result.",
                         key="kw_run_btn"):
                args = [PYTHON, "scripts/12_run_keyword_agreement.py",
                        "--run", res_run]
                with st.status("Computing keyword agreement…",
                               expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    status.update(
                        label=("Keyword agreement complete ✅" if rc == 0
                               else f"Failed (exit {rc})"),
                        state="complete" if rc == 0 else "error")
                st.rerun()

            kw = load_keyword_summary(res_run)
            if kw:
                o = kw["overall"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Agreement with matcher",
                          f"{o['rate']:.0%}" if o.get("rate") is not None else "—",
                          delta=f"chance {1 / len(LOGICS):.0%}", delta_color="off")
                m2.metric("Keyword share on matcher's pick",
                          f"{o.get('mean_share_on_matcher_top', 0):.3f}",
                          delta=f"chance {kw.get('share_chance_baseline', 1/7):.3f}",
                          delta_color="off")
                m3.metric("Distribution overlap",
                          f"{o.get('mean_overlap', 0):.3f}")
                m4.metric("No keyword overlap", o.get("n_no_overlap", 0),
                          help="answers where no keyword of any logic cleared "
                               "any rung — excluded from the rates rather than "
                               "guessed")
                if "semantic_enabled" in kw:
                    tiers = kw.get("tier_totals") or {}
                    tier_txt = " · ".join(
                        f"{tiers.get(k, 0)} {k}" for k in
                        ("exact", "morphological", "semantic") if tiers.get(k))
                    st.caption(
                        ("Semantic rung **on** (calibrated percentile ≥ 0.90)"
                         if kw["semantic_enabled"] else
                         "Semantic rung **off** — exact + morphological only. "
                         "Build the lexicon locally with "
                         "`scripts/13_run_topic_keywords.py calibrate` to "
                         "enable it.")
                        + (f" — matches by rung: {tier_txt}." if tier_txt else ""))

                # --- The profile this judge implies, in the profile style ---
                krows = load_keyword_rows(res_run)
                if krows is not None and not krows.empty:
                    st.markdown(
                        "**The profile this judge implies.** The same chart "
                        "as the alignment profiles above, but each bar is the "
                        "mean **keyword share** the lexical judge put on that "
                        "logic, x100, over the scored answers. Answers "
                        "matching no keyword of any logic are excluded, "
                        "exactly as they are from the agreement rates.")
                    kscored = krows[~krows["no_overlap"]]
                    if kscored.empty:
                        st.info("Every answer was no_overlap — nothing to "
                                "chart yet.")
                    else:
                        render_judge_profiles(kscored, "keyword_shares")

                # Head-to-head with the embedding judge when both have been run.
                if emb and emb["overall"].get("rate") is not None:
                    cmp_rows = []
                    for cat, ks in kw.get("by_category", {}).items():
                        es = emb.get("by_category", {}).get(cat, {})
                        if ks.get("rate") is not None and es.get("rate") is not None:
                            cmp_rows.append({"category": cat,
                                             "embedding": es["rate"],
                                             "keyword": ks["rate"]})
                    if cmp_rows:
                        st.markdown("**The two non-LLM judges, side by side.** They "
                                    "fail differently — where one collapses the "
                                    "other often holds, which is the reason to run "
                                    "both:")
                        cdf = pd.DataFrame(cmp_rows).melt(
                            "category", var_name="judge", value_name="rate")
                        ccht = (
                            alt.Chart(cdf).mark_bar().encode(
                                x=alt.X("rate:Q", title="agreement with matcher",
                                        scale=alt.Scale(domain=[0, 1])),
                                y=alt.Y("category:N", title=None,
                                        sort=[c for c in CATEGORIES]),
                                yOffset=alt.YOffset("judge:N"),
                                color=alt.Color("judge:N", title="judge"),
                                tooltip=["category", "judge",
                                         alt.Tooltip("rate:Q", format=".0%")],
                            ).properties(height=max(220, 30 * len(cmp_rows)))
                        )
                        st.altair_chart(ccht, width="stretch")

                if krows is not None and not krows.empty:
                    with st.expander("Matched words behind every verdict",
                                     expanded=False):
                        st.caption(
                            "The whole point of this judge: each verdict is "
                            "auditable by eye. Pick a row to see which of each "
                            "logic's keywords the answer matched — plain = "
                            "verbatim, `kw~word` = same stem, `kw≈word` = "
                            "semantic neighbour."
                        )
                        only_dis = st.checkbox("Only where it disagrees with the "
                                               "matcher", value=False, key="kw_dis")
                        kv = krows[krows["agree"] == False] if only_dis else krows  # noqa: E712
                        st.dataframe(
                            kv[["org", "source_type", "qid", "matcher_top",
                                "keyword_top", "agree", "share_on_matcher_top",
                                "overlap"]].rename(columns={
                                    "matcher_top": "matcher says",
                                    "keyword_top": "keywords say"}),
                            hide_index=True, width="stretch")
                        opts = [f"{r['org']} · {r['source_type']} · {r['qid']}"
                                for _, r in kv.iterrows()]
                        if opts:
                            sel = st.selectbox("Inspect a row", opts, key="kw_pick")
                            r = kv.iloc[opts.index(sel)]
                            mw = r["matched_words"] or {}
                            st.markdown(
                                f"**Matcher:** {r['matcher_top']} "
                                f"({100 * r['matcher_top_weight']:.0f}%) · "
                                f"**Keywords:** {r['keyword_top'] or '—'}")
                            if mw:
                                st.dataframe(pd.DataFrame(
                                    [{"logic": k,
                                      "keyword score": r["keyword_scores"].get(k),
                                      "matched words": ", ".join(v)}
                                     for k, v in mw.items()]),
                                    hide_index=True, width="stretch")
                            else:
                                st.info("This answer matched no keyword of any "
                                        "logic (`no_overlap`).")

                kdir = runs.run_dir(res_run) / "keyword_agreement"
                d1, d2, d3 = st.columns(3)
                for col, name, fname in (
                        (d1, "summary.json", f"keyword_summary_{res_run}.json"),
                        (d2, "rows.jsonl", f"keyword_rows_{res_run}.jsonl"),
                        (d3, "keywords.json", f"keywords_{res_run}.json")):
                    if (kdir / name).exists():
                        col.download_button(name, (kdir / name).read_bytes(),
                                            file_name=fname, key=f"kw_dl_{name}")

        # ci_long: lo/hi per (lab, source, logic) for error-bar overlays.
        ci_recs = []
        if ci_data:
            for org, by_st in ci_data["profiles"].items():
                for stype, by_logic in by_st.items():
                    for logic, s in by_logic.items():
                        ci_recs.append({"lab": org, "source": stype,
                                        "logic": logic,
                                        "lo": s["lo"], "hi": s["hi"]})
        ci_long = pd.DataFrame(ci_recs)

        # Long-form dataframe of every profile for charting.
        recs = []
        for org, by_st in profiles.items():
            for stype, p in by_st.items():
                if p["answered"] == 0:
                    continue
                for logic, pct in p["logic_pct"].items():
                    recs.append({"lab": org, "source": stype,
                                 "logic": logic, "pct": pct})
        long = pd.DataFrame(recs)

        if long.empty:
            st.warning("Profiles exist but contain no answered questions yet.")
        else:
            # --- Sanity check banner ---
            sanity = long[long["logic"].isin(["Family", "Religion"])]["pct"]
            worst = sanity.max() if not sanity.empty else 0.0
            if worst <= 5:
                st.success(f"Sanity check passed: Family/Religion peak at "
                           f"{worst:.1f}% (expected ≈0%).")
            elif worst <= 15:
                st.warning(f"Sanity check borderline: Family/Religion reach "
                           f"{worst:.1f}% somewhere — inspect the audit trail.")
            else:
                st.error(f"Sanity check FAILED: Family/Religion reach "
                         f"{worst:.1f}% — the method may be misfiring.")

            # --- One chart per lab: published vs thirdparty side by side ---
            for org in [o for o in ORGS if o in long["lab"].unique()]:
                st.subheader(org)
                sub = long[long["lab"] == org]
                bars = (
                    alt.Chart(sub)
                    .mark_bar()
                    .encode(
                        x=alt.X("logic:N", sort=LOGICS, title=None),
                        xOffset=alt.XOffset("source:N"),
                        y=alt.Y("pct:Q", title="% of profile",
                                scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color(
                            "source:N", title="source",
                            scale=alt.Scale(domain=SOURCE_TYPES,
                                            range=["#4C78A8", "#F58518"]),
                        ),
                        tooltip=["lab", "source", "logic",
                                 alt.Tooltip("pct:Q", format=".1f")],
                    )
                )
                layers = [bars]
                # Overlay bootstrap CI whiskers when available.
                if not ci_long.empty:
                    csub = ci_long[ci_long["lab"] == org]
                    if not csub.empty:
                        whiskers = (
                            alt.Chart(csub)
                            .mark_rule(strokeWidth=1.5, color="#333")
                            .encode(
                                x=alt.X("logic:N", sort=LOGICS, title=None),
                                xOffset=alt.XOffset("source:N"),
                                y=alt.Y("lo:Q", title="% of profile"),
                                y2="hi:Q",
                                tooltip=["lab", "source", "logic",
                                         alt.Tooltip("lo:Q", format=".1f"),
                                         alt.Tooltip("hi:Q", format=".1f")],
                            )
                        )
                        layers.append(whiskers)
                chart = alt.layer(*layers).properties(height=260)
                st.altair_chart(chart, width="stretch")

                cols = st.columns(len([s for s in SOURCE_TYPES
                                       if s in profiles.get(org, {})]))
                for col, stype in zip(cols, [s for s in SOURCE_TYPES
                                             if s in profiles.get(org, {})]):
                    p = profiles[org][stype]
                    if p["answered"]:
                        top = max(p["logic_pct"], key=p["logic_pct"].get)
                        col.metric(
                            f"{stype} — dominant logic",
                            f"{top} ({p['logic_pct'][top]:.0f}%)",
                            help=f"answered {p['answered']}, abstained {p['abstained']}",
                        )

            # --- Per-category breakdown ---
            st.subheader("Per-category breakdown")
            c1, c2 = st.columns(2)
            sel_org = c1.selectbox("Lab", [o for o in ORGS if o in profiles])
            sel_st = c2.selectbox(
                "Source", [s for s in SOURCE_TYPES
                           if s in profiles.get(sel_org, {})
                           and profiles[sel_org][s]["answered"]],
            )
            by_cat = profiles[sel_org][sel_st]["by_category"]
            if by_cat:
                cat_df = (
                    pd.DataFrame(by_cat).T
                    .reindex([c for c in CATEGORIES if c in by_cat])
                    [LOGICS]
                )
                st.dataframe(
                    cat_df.style.background_gradient(cmap="Blues", axis=None)
                    .format("{:.0f}%"),
                    width="stretch",
                )

            # --- Downloads ---
            st.subheader("Downloads")
            rp = runs.run_paths(res_run)
            d1, d2, d3 = st.columns(3)
            d1.download_button("company_profiles.json",
                               rp["profiles_json"].read_bytes(),
                               file_name=f"company_profiles_{res_run}.json")
            if rp["profiles_csv"].exists():
                d2.download_button("profiles_matrix.csv",
                                   rp["profiles_csv"].read_bytes(),
                                   file_name=f"profiles_matrix_{res_run}.csv")
            if rp["per_question"].exists():
                d3.download_button("per_question.jsonl",
                                   rp["per_question"].read_bytes(),
                                   file_name=f"per_question_{res_run}.jsonl")

# ---------------------------------------------------------------------------
# Audit tab
# ---------------------------------------------------------------------------
with tab_audit:
    aud_run = run_selectbox("Run to audit", key="audit_run",
                            default_run_id=runs.get_current())
    dfq = load_per_question(aud_run)
    if dfq is None or dfq.empty:
        st.info("No per-question results yet.")
    else:
        st.header("Audit trail")
        st.caption("Every question's RAG answer, graded weights, and matcher "
                   "reasoning — the evidence behind the percentages.")
        with st.expander("ℹ️ How to read a row", expanded=False):
            st.markdown(
                "Each row is one **(lab, source, question)** triple — the "
                "atomic unit everything else aggregates. The header shows the "
                "row's *dominant* logic, i.e. the argmax of its weight vector "
                "(ties break by the fixed logic order, so the label is "
                "deterministic):"
            )
            st.latex(r"\mathrm{label}=\begin{cases}"
                     r"\texttt{ABSTAINED} & \text{if the row abstained}\\[2pt]"
                     r"\arg\max_k w_k & \text{otherwise}\end{cases}")
            symbol_glossary([
                (r"$k$", "**a logic** — one of the seven"),
                (r"$w_k$", "the **weight** this row's answer earned on logic "
                           "$k$; the seven sum to 1"),
                (r"$\arg\max_k w_k$", "**which** logic carries the largest "
                                      "weight — the row's dominant label"),
                (r"$\mathrm{label}$", "what the row header displays"),
            ], note="Ties break by the fixed logic order, so the label is "
                    "deterministic for a given weight vector.")
            st.markdown(
                "**Fields**\n"
                "- **Weights** — the matcher's distribution over the seven "
                "logics, clamped non-negative and normalized to sum to 1. Only "
                "non-zero entries are listed. This exact vector is what the "
                "profile percentages average.\n"
                "- **Matcher reasoning** — the one-sentence justification the "
                "matcher returned. It is recorded for auditing and **never "
                "used in any calculation**.\n"
                "- **retrieved** — the ids of the chunks the answer was "
                "written from. These make the row replayable: the metamorphic "
                "and quote-provenance checks refetch exactly these chunks.\n"
                "- **Supporting quotes** (only when the run used `--quotes`) — "
                "spans the model claims to have copied, each ✅/❌ by a "
                "code-side verbatim check after whitespace normalization. The "
                "excerpt number links to the source PDF when the raw dataset "
                "is available locally; because a span verifies against *any* "
                "retrieved excerpt, a ✅ does not certify it is in the linked "
                "document, and where it isn't, the row names the one that "
                "actually holds it.\n"
                "- **grounding** (only when the run used `--grounding`) — the "
                "row's lexical grounding score, best cosine, and bucket.\n\n"
                "**Abstentions** carry an all-zero weight vector and are "
                "excluded from the profile denominator, so they reduce "
                "confidence without shifting the distribution. Filter to them "
                "with the *Abstentions only* checkbox to see where the corpus "
                "was silent."
            )
        f1, f2, f3, f4 = st.columns(4)
        orgs_f = f1.multiselect("Lab", sorted(dfq["org"].unique()))
        st_f = f2.multiselect("Source", sorted(dfq["source_type"].unique()))
        cat_f = f3.multiselect("Category", [c for c in CATEGORIES
                                            if c in set(dfq["category"])])
        only_abstain = f4.checkbox("Abstentions only")
        show_evidence = st.checkbox(
            "Show the retrieved evidence each answer was written from",
            value=False, key="audit_show_evidence",
            help="Fetches the actual chunk text by id from the vector index "
                 "(one batched lookup for the filtered rows — no API calls). "
                 "Off by default because it makes the page much longer.")

        view = dfq
        if orgs_f:
            view = view[view["org"].isin(orgs_f)]
        if st_f:
            view = view[view["source_type"].isin(st_f)]
        if cat_f:
            view = view[view["category"].isin(cat_f)]
        if only_abstain:
            view = view[view["abstain"]]

        st.caption(f"{len(view)} of {len(dfq)} rows "
                   f"({int(dfq['abstain'].sum())} abstentions overall)")

        # One batched lookup for every chunk in the filtered view, plus the
        # topic map if the inductive layer has been fitted — so each piece of
        # evidence can be labelled with the theme it belongs to.
        #
        # The chunk lookup now feeds TWO things: the evidence list below, and
        # the [excerpt N] links in the quote block, which need each cited
        # chunk's filename and org. So it is no longer gated on show_evidence —
        # it is gated on there being something to use it for, which keeps a run
        # without quotes exactly as cheap as before. Ids need no embedding, so
        # this stays one local sqlite get, cached (see fetch_chunks).
        evidence: dict = {}
        chunk_topic_map: dict = {}
        topic_label_map: dict = {}
        has_quote_rows = "quotes" in view.columns and view["quotes"].apply(
            lambda q: isinstance(q, list) and bool(q)).any()
        if len(view) and (show_evidence or has_quote_rows):
            ids = tuple(sorted({cid for r in view["retrieved_ids"]
                                if isinstance(r, list) for cid in r}))
            evidence = fetch_chunks(ids)
        if show_evidence and len(view):
            chunk_topic_map = topics_mod.load_chunk_topics() or {}
            tinfo_a = topics_mod.load_topic_info()
            if tinfo_a:
                topic_label_map = {r["topic"]: r["label"]
                                   for r in tinfo_a["topics"]}

        for _, row in view.iterrows():
            top = ("ABSTAINED" if row["abstain"] else
                   max(row["weights"], key=row["weights"].get))
            label = (f"{row['org']} · {row['source_type']} · {row['qid']} → "
                     f"{top}" + ("" if row["abstain"] else
                                 f" ({100 * row['weights'][top]:.0f}%)"))
            with st.expander(label):
                st.markdown(f"**Q:** {row['question']}")
                st.markdown(f"**RAG answer:**\n\n{row['answer']}")
                if not row["abstain"]:
                    wdf = pd.DataFrame(
                        [{"logic": k, "weight": v}
                         for k, v in row["weights"].items() if v > 0]
                    ).sort_values("weight", ascending=False)
                    st.dataframe(wdf, hide_index=True)
                st.markdown(f"**Matcher reasoning:** {row['reasoning']}")
                if isinstance(row.get("quotes"), list):
                    ok = bool(row.get("quotes_verified"))
                    st.markdown("**Supporting quotes:** "
                                + ("✅ all verified in sources" if ok
                                   else "⚠️ not verified"))
                    linked = False
                    for q in row["quotes"]:
                        mark = "✅" if q.get("verified") else "❌"
                        # The excerpt number becomes a link to the PDF it names;
                        # `note` calls out the case where the span is real but
                        # lives in a different document than the one cited.
                        cite = pdf_sources.excerpt_citation(
                            q.get("excerpt"), row.get("retrieved_ids"),
                            evidence, pdf_url)
                        note = pdf_sources.misattribution_note(
                            q.get("quote", ""), q.get("excerpt"),
                            row.get("retrieved_ids"), evidence, pdf_url)
                        # A linked citation opens the markdown link "[";
                        # the plain fallback opens the escaped bracket "\[".
                        linked = linked or cite.startswith("[")
                        st.markdown(f"> {mark} {cite} "
                                    f"“{q.get('quote', '')}”{note}")
                    if linked:
                        st.caption(
                            "Excerpt numbers link to the source the answer "
                            "CITED. Verification runs against every retrieved "
                            "excerpt, not just the cited one, so ✅ does not "
                            "certify the span is in the linked document — "
                            "where they differ, the real source is named "
                            "above. Press-clipping links open **our rendering "
                            "of the record the index was built from**, not the "
                            "publisher's page; the exports carry no URL.")
                gb = row.get("grounding_bucket")
                if isinstance(gb, str):
                    st.caption(f"grounding: {gb} · score "
                               f"{row.get('retrieval_grounding_score', 0):.2f} · "
                               f"cosine {row.get('retrieval_cosine_top', 0):.2f}")
                st.caption("retrieved: " + ", ".join(row["retrieved_ids"][:5]))
                if show_evidence:
                    st.markdown("**Evidence the answer was written from** "
                                "(the only text the answering model saw):")
                    for i, cid in enumerate(row["retrieved_ids"], 1):
                        ch = evidence.get(cid)
                        # `label` is the headline for press records and the
                        # filename for published docs — "O1.RTF" named 500
                        # articles and identified none of them.
                        tag = (f"[{i}] {ch['label'] or ch['filename']}" if ch
                               else f"[{i}] {cid}")
                        if ch and ch.get("sublabel"):
                            tag += f"  ·  {ch['sublabel']}"
                        if chunk_topic_map:
                            t = chunk_topic_map.get(cid)
                            if t is not None:
                                tag += f"  ·  topic {t}: {topic_label_map.get(t, '')}"
                        if ch:
                            with st.expander(tag, expanded=False):
                                # In the body, not the label: a click on a link
                                # inside an expander header also toggles it.
                                doc = pdf_url(ch["filename"], ch["org"], cid,
                                              ch.get("source_type", ""))
                                if doc:
                                    name = pdf_sources.md_escape(
                                        ch["label"] or ch["filename"])
                                    st.markdown(f"[📄 open {name}]({doc})")
                                st.text(ch["text"])
                        else:
                            st.caption(f"{tag} — not in the current index "
                                       "(re-ingested since this run?)")

# ---------------------------------------------------------------------------
# Hallucination tab — the five opt-in checks, with alerts when one fires
# ---------------------------------------------------------------------------
with tab_halluc:
    st.header("Hallucination & grounding checks")
    st.caption(
        "Four black-box checks: **retrieval grounding** (was there relevant "
        "text to answer from?), **quote verification** (does the cited support "
        "actually appear in the sources?), **quote provenance** (if it doesn't, "
        "is it a paraphrase, a figure of speech, or a fabrication — and is its "
        "content true anyway?), and **metamorphic stability & evidence "
        "sensitivity** (does the label survive a reword — and does it stop when "
        "the supporting evidence is taken away?). The two non-LLM judges — "
        "**embedding agreement** and **keyword agreement** — grade the same "
        "run and now sit on the **Results** tab, beside the profiles they "
        "judge."
    )
    hal_run = run_selectbox("Run to inspect", key="halluc_run",
                            default_run_id=runs.get_current())
    dfh = load_per_question(hal_run)
    stab = load_stability(hal_run)
    # Still loaded here although the check itself moved to the Results tab:
    # the alert banner and the download bundle below both read it.
    emb = load_embedding_summary(hal_run)
    prov = load_quote_provenance(hal_run)
    prov_spans = load_quote_spans(hal_run)

    if dfh is None or dfh.empty:
        st.info("No per-question results yet — run the pipeline on the **Run** "
                "tab first.")
    else:
        import altair as alt

        has_grounding = ("grounding_bucket" in dfh.columns
                         and dfh["grounding_bucket"].notna().any())
        has_quotes = ("quotes_verified" in dfh.columns
                      and dfh["quotes_verified"].notna().any())
        gdf = dfh[dfh["grounding_bucket"].notna()] if has_grounding else None
        qdf = dfh[dfh["quotes_verified"].notna()] if has_quotes else None
        # "Fabricated" = the model DID cite quotes but at least one span is not
        # in the sources. Rows with an empty quote list (typically abstentions
        # or parse fallbacks) are reported separately, not as fabrications.
        fab_rows = (qdf[qdf.apply(
            lambda r: isinstance(r["quotes"], list) and len(r["quotes"]) > 0
            and not bool(r["quotes_verified"]), axis=1)]
            if has_quotes else None)
        noq_rows = (qdf[qdf["quotes"].apply(
            lambda q: not (isinstance(q, list) and len(q) > 0))]
            if has_quotes else None)

        # --- Detection banner: loud when something fired, green when clean ---
        alerts: list[tuple[str, str]] = []
        if has_grounding:
            n_missed = int((gdf["grounding_bucket"] == "retrieval_missed").sum())
            if n_missed:
                alerts.append(("warning",
                               f"🔎 **Retrieval likely missed** on {n_missed} "
                               f"question(s) — their answers rest on weak evidence, "
                               f"whatever the model did next. See section 1."))
        if has_quotes and len(fab_rows):
            alerts.append(("error",
                           f"❌ **Unverified quotes** on {len(fab_rows)} answer(s): "
                           f"cited spans do not appear verbatim in the retrieved "
                           f"sources — possible fabricated support. See section 2."))
        if prov:
            po = prov["overall"]
            n_fab = po["verdicts"].get("fabricated", 0)
            n_misq = po["verdicts"].get("misquote_but_true", 0)
            if n_fab:
                alerts.append(("error",
                               f"🚨 **{n_fab} genuinely fabricated span(s)**: not in "
                               f"the sources, and the evidence does not support what "
                               f"they assert either. This is the number to act on — "
                               f"it is what survives after paraphrases and figures of "
                               f"speech are separated out. See section 2b."))
            if n_misq:
                alerts.append(("warning",
                               f"📎 **{n_misq} misquotation(s) with sound content**: "
                               f"the quotation was manufactured, but what it claims "
                               f"IS carried by the evidence — a citation-integrity "
                               f"failure, not a factual one. See section 2b."))
            if not n_fab and not n_misq and po["n_attributive"]:
                alerts.append(("success",
                               "✅ **No fabricated spans**: every attributive "
                               "quotation is either in the sources or a grounded "
                               "paraphrase of them. See section 2b."))
        if stab:
            s = stab["summary"]
            if s.get("n_unstable"):
                alerts.append(("error",
                               f"🎲 **{s['n_unstable']} unstable item(s)**: the "
                               f"predicted logic flipped under meaning-preserving "
                               f"paraphrase. See section 3."))
            if s.get("n_prior_keyed"):
                alerts.append(("error",
                               f"🧠 **{s['n_prior_keyed']} item(s) answered from a "
                               f"prior, not the text**: given real text from the same "
                               f"lab that does NOT address the question, the model "
                               f"returned the same logic anyway. See section 3."))
            if s.get("n_label_survived_ablation"):
                alerts.append(("warning",
                               f"🕳️ **{s['n_label_survived_ablation']} item(s) kept "
                               f"their label without their evidence**: removing the "
                               f"excerpt the answer quoted changed nothing. See "
                               f"section 3."))
        if emb and emb["overall"].get("rate") is not None \
                and emb["overall"]["rate"] < 0.5:
            alerts.append(("warning",
                           f"🧭 **Low embedding agreement** "
                           f"({emb['overall']['rate']:.0%}): the non-LLM judge "
                           f"often ranks a different logic's reference nearest "
                           f"than the matcher's top pick. See the "
                           f"**Results** tab."))
        if not (has_grounding or has_quotes or stab or emb or prov):
            st.info("None of the checks have run for this snapshot yet. Enable "
                    "**--grounding** / **--quotes** on the Run tab for the next "
                    "run, or launch the metamorphic eval / quote provenance "
                    "below (both work on any existing run).")
        elif alerts:
            for kind, msg in alerts:
                getattr(st, kind)(msg)
        else:
            st.success("✅ No hallucination signals fired on the checks that ran "
                       "for this snapshot.")

        # --- Download every check's decisions + underlying data ---
        with st.expander("⬇️ Download all detection results "
                         "(every decision + the data behind it)", expanded=False):
            members, ran = build_detection_bundle(hal_run, dfh, stab, emb)
            csv_members = [(n, b) for n, b in members
                           if n.endswith(".csv")]
            if not csv_members:
                st.caption(
                    "No checks have produced results for this snapshot yet. "
                    "Enable **--grounding** / **--quotes** on the Run tab, "
                    "launch the metamorphic eval below, or compute embedding "
                    "agreement on the Results tab, then come back here.")
            else:
                st.caption(
                    "One row per decision across the "
                    f"{len(ran)} check(s) that ran ({', '.join(ran)}). Each "
                    "table's `decision_*` column is the verdict; the other "
                    "columns are the data used to reach it. The ZIP also bundles "
                    "the raw source files under `raw/` and a README.")
                st.download_button(
                    "⬇️ Download everything (.zip)",
                    zip_members(members),
                    file_name=f"hallucination_results_{hal_run}.zip",
                    mime="application/zip", type="primary", key="dl_all_zip")
                st.caption("Or grab an individual check's decision table:")
                cols = st.columns(min(len(csv_members), 3))
                for i, (name, data) in enumerate(csv_members):
                    cols[i % len(cols)].download_button(
                        name, data, file_name=f"{name[:-4]}_{hal_run}.csv",
                        mime="text/csv", key=f"dl_csv_{name}")

        BUCKET_ORDER = ["committed", "abstained", "retrieval_missed"]
        BUCKET_COLORS = ["#54A24B", "#F58518", "#E45756"]

        # ---------------- 1 · Retrieval grounding ----------------
        st.subheader("1 · Retrieval grounding")
        with st.expander("ℹ️ How this score is computed", expanded=False):
            st.markdown(
                "Everything below is pure computation over the chunks the "
                "retriever already returned (`il_rag/grounding.py`) — no LLM "
                "call, zero extra API cost. The same notation is used in "
                "ARCHITECTURE.md §9.1 and the README."
            )
            st.markdown(
                "**Step 1 — content tokens.** Lowercase the text, split into "
                "alphanumeric tokens, and drop a small English stopword list "
                "plus tokens of ≤ 2 characters, so function words like "
                "*the / of / and* cannot inflate the overlap:"
            )
            st.latex(
                r"T(x)=\{\,t \in \mathrm{tokens}(\mathrm{lower}(x)) \mid "
                r"t \notin \mathrm{stopwords},\ |t|>2\,\}"
            )
            st.markdown(
                "**Step 2 — per-chunk lexical overlap** (ROUGE-1-recall-style "
                "set overlap): the fraction of the question's content tokens "
                "that appear in the chunk, always in $[0,1]$ "
                "(defined as $0$ when $T(q)$ is empty):"
            )
            st.latex(
                r"\mathrm{overlap}(q,c)=\frac{|\,T(q)\cap T(c)\,|}{|\,T(q)\,|}"
            )
            st.markdown(
                "**Step 3 — grounding score**: the **max** overlap across the "
                "retrieved chunks $R(q)$. Max, not mean — one genuinely "
                "relevant chunk is enough to ground an answer, so a strong hit "
                "should not be diluted by weak siblings:"
            )
            st.latex(
                r"g(q)=\max_{c\,\in\,R(q)} \mathrm{overlap}(q,c)"
            )
            st.markdown(
                "**Step 4 — bucketing** against the threshold "
                f"$\\tau = {GROUNDING_LOW_THRESHOLD}$ "
                "(`GROUNDING_LOW_THRESHOLD` in `il_rag/config.py`):"
            )
            st.latex(
                r"\mathrm{bucket}(q)=\begin{cases}"
                r"\texttt{retrieval\_missed} & g(q)<\tau\\[2pt]"
                r"\texttt{abstained} & g(q)\ge\tau \text{ and the matcher abstained}\\[2pt]"
                r"\texttt{committed} & \text{otherwise}"
                r"\end{cases}"
            )
            symbol_glossary([
                (r"$q$", "**the question** being scored"),
                (r"$c$", "**a retrieved chunk** — one excerpt of evidence"),
                (r"$R(q)$", "the **set of chunks retrieved** for question $q$ "
                            "(normally $k=5$ of them)"),
                (r"$T(x)$", "the **content tokens** of text $x$: lowercased "
                            "words, minus stopwords and ≤2-character tokens"),
                (r"$\mathrm{overlap}(q,c)$", "the share of the question's "
                                             "content words that appear in "
                                             "chunk $c$, in $[0,1]$"),
                (r"$g(q)$", "the **grounding score** — the *best* overlap "
                            "across all retrieved chunks"),
                (r"$\tau$", f"the **threshold** below which retrieval counts as "
                            f"having missed (`GROUNDING_LOW_THRESHOLD`, "
                            f"currently **{GROUNDING_LOW_THRESHOLD}**)"),
                (r"$\max$", "the **largest** value over whatever follows — here "
                            "over the retrieved chunks"),
                (r"$\cos_c$", "the retriever's **cosine score** for chunk $c$, "
                              "kept as a diagnostic but never thresholded"),
                (r"$\mathrm{clip}(x,0,1)$", "force $x$ into the range $[0,1]$"),
                *_SET_NOTATION,
            ])
            st.markdown(
                "`retrieval_missed` deliberately takes precedence over "
                "`abstained`: when retrieval never surfaced relevant text, "
                "abstaining was the *right* response, and the item's failure "
                "belongs to retrieval — not to the model's grounding.\n\n"
                "**Design decisions**\n"
                "- *Why threshold the lexical score and not the cosine?* The "
                "retriever's best cosine is kept as the `cosine_top` subscore, "
                "$\\max_{c} \\mathrm{clip}(\\cos_c, 0, 1)$, but never "
                "thresholded: e5 embeddings compress cosine into a narrow "
                "high band even for weak matches, so token overlap is the "
                "discriminative, interpretable signal. Cosine remains useful "
                "as a diagnostic — e.g. a low-$g$, high-cosine row hints at a "
                "paraphrased (not missing) match.\n"
                f"- *Where does $\\tau = {GROUNDING_LOW_THRESHOLD}$ come "
                "from?* It is a per-corpus heuristic set in `config.py`, not "
                "a learned or label-calibrated parameter; tune it against the "
                "histogram below.\n"
                "- *Why no accuracy column?* There are no gold labels in this "
                "pipeline, so each bucket reports its size, its abstention "
                "rate, and the mean top-logic weight of its committed answers "
                "($\\mathrm{mean}\\ \\max_k w_k$ — a proxy for how decisively "
                "the matcher graded them). The buckets separate *failure "
                "modes*, not correctness.\n\n"
                "**Basis in the literature** — the overlap is a set-based "
                "variant of ROUGE-1 recall "
                "([Lin, 2004](https://aclanthology.org/W04-1013/)); using "
                "lexical overlap with retrieved text as a groundedness signal "
                "follows Knowledge F1 ([Shuster et al., 2021]"
                "(https://aclanthology.org/2021.findings-emnlp.320/)); the "
                "retrieval-failure vs. model-failure split mirrors the "
                "\"Missing Content\" failure point of "
                "[Barnett et al., 2024](https://arxiv.org/abs/2401.05856); "
                "cosine is left unthresholded because of embedding anisotropy "
                "([Ethayarajh, 2019](https://aclanthology.org/D19-1006/)). "
                "Full reference list in ARCHITECTURE.md §9.1."
            )
        if not has_grounding:
            st.caption("Not scored for this run — check **Grounding pre-check** "
                       "on the Run tab (adds no API calls).")
        else:
            n_by = gdf["grounding_bucket"].value_counts()
            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 committed", int(n_by.get("committed", 0)),
                      help="retrieval looked plausible; answer graded into logics")
            m2.metric("🟠 abstained", int(n_by.get("abstained", 0)),
                      help="retrieval looked plausible but the model said the "
                           "excerpts don't answer — honest silence")
            m3.metric("🔴 retrieval missed", int(n_by.get("retrieval_missed", 0)),
                      help=f"grounding score < {GROUNDING_LOW_THRESHOLD}: the "
                           "question's content words barely appear in any "
                           "retrieved chunk")
            hist = (
                alt.Chart(gdf[["retrieval_grounding_score", "grounding_bucket"]])
                .mark_bar()
                .encode(
                    x=alt.X("retrieval_grounding_score:Q",
                            bin=alt.Bin(maxbins=20),
                            title="grounding score (question↔chunk overlap)"),
                    y=alt.Y("count()", title="questions"),
                    color=alt.Color("grounding_bucket:N", title="bucket",
                                    scale=alt.Scale(domain=BUCKET_ORDER,
                                                    range=BUCKET_COLORS)),
                )
                .properties(height=200)
            )
            rule = (
                alt.Chart(pd.DataFrame({"x": [GROUNDING_LOW_THRESHOLD]}))
                .mark_rule(color="#E45756", strokeDash=[6, 4], size=2)
                .encode(x="x:Q")
            )
            st.altair_chart(hist + rule, width="stretch")
            missed = gdf[gdf["grounding_bucket"] == "retrieval_missed"]
            if len(missed):
                with st.expander(f"🔴 {len(missed)} question(s) where retrieval "
                                 "likely missed", expanded=False):
                    for _, r in missed.iterrows():
                        st.markdown(
                            f"**{r['org']} · {r['source_type']} · {r['qid']}** — "
                            f"score {r['retrieval_grounding_score']:.2f}"
                            + ("  · model abstained ✅" if r["abstain"]
                               else "  · **model still committed** ⚠️"))
                        st.caption(r["question"])

        # ---------------- 2 · Quote verification ----------------
        st.subheader("2 · Quote verification")
        with st.expander("ℹ️ How quotes are verified", expanded=False):
            st.markdown(
                "The model **attests**, the code **audits**: nothing the model "
                "says about its own quotes is trusted — every claimed span is "
                "re-checked in pure code (`il_rag/rag_qa.py`), adding zero API "
                "calls beyond the answer call itself. The same notation is "
                "used in ARCHITECTURE.md §9.2 and the README."
            )
            st.markdown(
                "**Step 1 — the contract.** Alongside its answer, the model "
                "must return 1–3 spans *\"copied character-for-character from "
                "the numbered excerpt it cites; never paraphrase inside a "
                "quote\"* (verbatim prompt rule). If the excerpts can't answer "
                "the question, it must say so and return an **empty** quote "
                "list."
            )
            st.markdown(
                "**Step 2 — normalization.** Before comparing, both the quote "
                "and every retrieved chunk are normalized: collapse every "
                "whitespace run to a single space, strip the ends, lowercase. "
                "**Punctuation is not touched** — matching is tolerant to "
                "whitespace/case copying artifacts but otherwise verbatim:"
            )
            st.latex(
                r"\mathrm{norm}(s)=\mathrm{lowercase}(\mathrm{collapse\_ws}(s))"
            )
            st.markdown(
                "**Step 3 — per-quote check.** A quote $q$ verifies iff its "
                "normalized form is non-empty and appears as a substring "
                "($\\sqsubseteq$) of **any** normalized retrieved chunk in "
                "$R$:"
            )
            st.latex(
                r"\mathrm{verified}(q)=\big(\mathrm{norm}(q)\neq\text{``''}\big)"
                r"\ \wedge\ \exists\,c\in R:\ \mathrm{norm}(q)\sqsubseteq\mathrm{norm}(c)"
            )
            symbol_glossary([
                (r"$q$", "**a quoted span** the model claims to have copied"),
                (r"$Q$", "the **set of all quotes** returned for one answer"),
                (r"$c$", "**a retrieved chunk** — one piece of source evidence"),
                (r"$R$", "the **set of chunks** this answer was written from"),
                (r"$\mathrm{norm}(x)$", "**normalized** text: lowercased with "
                                        "runs of whitespace collapsed, so "
                                        "formatting differences don't cause "
                                        "false failures"),
                (r"$\sqsubseteq$", "**is a substring of** — appears verbatim "
                                   "inside"),
                (r"$\wedge$", "**logical AND** (both must hold)"),
                (r"$\exists\,c\in R$", "**there exists** at least one chunk $c$ "
                                       "in $R$ for which this is true"),
                (r"$\forall\,q\in Q$", "**for every** quote $q$ in $Q$"),
                (r"$\lvert Q\rvert>0$", "at least one quote was returned — the "
                                        "guard that stops an **empty** list "
                                        "counting as 'all verified'"),
            ])
            st.markdown(
                "The cited excerpt *number* is displayed but deliberately "
                "**not** used for matching: the auditable claim is \"this "
                "text is in the sources\", not the model's index bookkeeping "
                "— a right span with a wrong number is sloppy citing, not "
                "fabrication."
            )
            st.markdown(
                "**Step 4 — row verdict.** The row's `quotes_verified` is the "
                "conjunction over its quote set $Q$, guarded by $|Q|>0$:"
            )
            st.latex(
                r"\mathrm{quotes\_verified}=\big(|Q|>0\big)\ \wedge\ "
                r"\bigwedge_{q\in Q}\mathrm{verified}(q)"
            )
            st.markdown(
                "The guard exists because a conjunction over an empty set is "
                "vacuously true — an empty or unusable quote list is "
                "**unverified by definition**.\n\n"
                "**Design decisions**\n"
                "- *Fabricated ≠ no quotes.* The ❌ metric counts only rows "
                "that **did** cite quotes of which at least one span is not in "
                "the sources — possible fabricated support. Rows with an "
                "empty list (typically honest abstentions, or JSON parse "
                "fallbacks) are counted separately under ∅ — abstaining is "
                "not fabrication.\n"
                "- *Parse failure degrades, never crashes.* If the model's "
                "JSON doesn't parse, the call is retried once at a doubled "
                "token budget; if it still fails, the raw text is kept as the "
                "answer with an empty quote list (`quotes_verified = False`), "
                "so the run continues and the row stays auditable.\n"
                "- *No tunable constant.* Unlike grounding's $\\tau$, this "
                "check has no threshold — a span either occurs verbatim "
                "(after normalization) or it doesn't.\n\n"
                "**Limitations** — verbatim substring matching cannot credit "
                "a *paraphrased-but-faithful* quote, so a ❌ means \"not "
                "verbatim in the sources\", which is not identical to \"the "
                "answer is wrong\" (the check errs toward false alarms, never "
                "toward missed fabrications). Conversely a ✅ proves the span "
                "exists in the sources — not that the conclusion actually "
                "follows from it (attribution, not entailment).\n\n"
                "**Basis in the literature** — having the model support "
                "answers with verbatim quotes that are then mechanically "
                "verified against sources follows GopherCite "
                "([Menick et al., 2022](https://arxiv.org/abs/2203.11147)); "
                "the property audited is *attribution to identified sources* "
                "(AIS: [Rashkin et al., 2023]"
                "(https://aclanthology.org/2023.cl-4.2/); "
                "[Bohnet et al., 2022](https://arxiv.org/abs/2212.08037)); "
                "citation-quality evaluation of LLM output follows ALCE "
                "([Gao et al., 2023]"
                "(https://aclanthology.org/2023.emnlp-main.398/)). Full "
                "reference list in ARCHITECTURE.md §9.2."
            )
        if not has_quotes:
            st.caption("Not enabled for this run — check **Quote-grounded "
                       "answers** on the Run tab.")
        else:
            n_ok = int(qdf["quotes_verified"].astype(bool).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("✅ all quotes verified", n_ok)
            m2.metric("❌ unverified quotes", len(fab_rows),
                      help="the answer cited at least one span that is not in "
                           "the retrieved sources")
            m3.metric("∅ no quotes returned", len(noq_rows),
                      help="empty quote list — expected for abstentions")
            if len(fab_rows):
                # Same batched by-id lookup as the Audit tab, over the failing
                # rows only, so each [excerpt N] can link to the PDF it names.
                # fetch_chunks caches per id-tuple, so ids shared with the Audit
                # tab's view cost nothing.
                fab_ev = fetch_chunks(tuple(sorted(
                    {cid for r in fab_rows["retrieved_ids"]
                     if isinstance(r, list) for cid in r})))
                for _, r in fab_rows.iterrows():
                    with st.expander(f"❌ {r['org']} · {r['source_type']} · "
                                     f"{r['qid']}"):
                        st.markdown(f"**Q:** {r['question']}")
                        st.markdown(f"**Answer:** {r['answer']}")
                        for q in r["quotes"]:
                            mark = "✅" if q.get("verified") else "❌"
                            cite = pdf_sources.excerpt_citation(
                                q.get("excerpt"), r.get("retrieved_ids"),
                                fab_ev, pdf_url)
                            note = pdf_sources.misattribution_note(
                                q.get("quote", ""), q.get("excerpt"),
                                r.get("retrieved_ids"), fab_ev, pdf_url)
                            st.markdown(f"> {mark} {cite}"
                                        f" “{q.get('quote', '')}”{note}")
            else:
                st.caption("Every quoted span was found verbatim in its retrieved "
                           "sources.")

        # ------- 2b · Quote provenance & paraphrase grounding -------
        st.subheader("2b · Quote provenance & paraphrase grounding")
        st.caption(
            "Section 2 answers one question with one bit: is this span verbatim "
            "in the sources? This section grades **how** each quoted span "
            "relates to them, separates quotation marks that never claimed "
            "anything about a source, and asks whether a span's **content** "
            "holds up even when the span itself does not."
        )
        with st.expander("ℹ️ How quote provenance is computed", expanded=False):
            st.markdown(
                "A single ❌ in section 2 conflates four different things: a "
                "**copy that drifted** (curly quotes, an em-dash, an elided "
                "`…`), a **faithful paraphrase**, a **figure of speech** "
                "(scare quotes, terms of art, hypotheticals — quotation marks "
                "that never claimed anything about a source), and an actual "
                "**fabrication**. Only the last is a hallucination."
            )
            st.markdown(
                "**Step 1 — extraction.** Candidates come from the model's "
                "structured `quotes` entries *and* from quotation marks in the "
                "answer prose. The prose spans matter: they are unaudited by "
                "section 2, and they are where the figures of speech live. "
                f"Spans under {QUOTE_MIN_SPAN_TOKENS} content tokens are "
                "dropped as noise."
            )
            st.markdown(
                "**Step 2 — intent triage (no LLM).** Deterministic cue rules "
                "over the ~70 characters before the opening quote, in a fixed "
                "precedence: *counterfactual* (`a critic might say …`) → "
                "*reporting verb* (`the charter states …`) → *mention* "
                "(`so-called …`) → *example* → *shape* → default. The rule that "
                "fired is stored with the label, so every classification is "
                "inspectable. An unmatched span defaults to **attributive at "
                "low confidence** — wrongly auditing a scare quote is a false "
                "alarm you can dismiss, wrongly excusing a fabricated quotation "
                "hides what the check exists to find."
            )
            st.markdown(
                "**Step 3 — the provenance ladder (no LLM).** Cheapest-first; "
                "the first tier to clear its bar wins:"
            )
            st.latex(
                r"\begin{aligned}"
                r"\textbf{exact}&:\ \mathrm{norm}(s)\sqsubseteq\mathrm{norm}(c)\\[2pt]"
                r"\textbf{near\_verbatim}&:\ \mathrm{strip}(\mathrm{norm}(s))"
                r"\sqsubseteq\mathrm{strip}(\mathrm{norm}(c))\\"
                r"&\ \ \vee\ \text{elided fragments in order}\\"
                r"&\ \ \vee\ \textstyle\max_w \mathrm{ratio}(s,w)\geq"
                r"\tau_{\text{near}}\\[2pt]"
                r"\textbf{paraphrase}&:\ \mathrm{overlap}(s,c)\geq"
                r"\tau_{\text{lex}}\ \vee\ \textstyle\max_w\cos(s,w)\geq"
                r"\tau_{\text{cos}}\\[2pt]"
                r"\textbf{unsupported}&:\ \text{nothing cleared a bar}"
                r"\end{aligned}"
            )
            st.markdown(
                f"with $\\tau_{{\\text{{near}}}} = {QUOTE_NEAR_VERBATIM_THRESHOLD}$, "
                f"$\\tau_{{\\text{{lex}}}} = {QUOTE_PARAPHRASE_LEX_THRESHOLD}$, "
                f"$\\tau_{{\\text{{cos}}}} = {QUOTE_PARAPHRASE_COS_THRESHOLD}$. "
                "The `exact` tier is bit-identical to section 2's predicate, so "
                "**every span section 2 verifies lands in `exact`** — this "
                "check only ever adds resolution below that line, never "
                "reinterprets above it. Lexical overlap is the primary "
                "paraphrase signal rather than cosine, for the same reason "
                "section 1 thresholds lexical: e5 compresses cosine into a "
                "narrow high band (measured here: a faithful reword scored "
                "0.849, a wholly unrelated claim 0.807 — 0.04 apart), so a "
                "cosine gate is far less discriminative than it looks."
            )
            symbol_glossary([
                (r"$s$", "**a quoted span** being graded"),
                (r"$c$", "**a retrieved chunk** it might have come from"),
                (r"$w$", "a **sliding window** of $c$ the span is compared "
                         "against when hunting for a drifted copy"),
                (r"$\mathrm{norm}(x)$", "**normalized** text (lowercased, "
                                        "whitespace collapsed)"),
                (r"$\mathrm{strip}(x)$", "normalized text with **punctuation "
                                         "also removed** — catches curly "
                                         "quotes and stray dashes"),
                (r"$\sqsubseteq$", "**is a substring of**"),
                (r"$\mathrm{ratio}(s,w)$", "**character similarity** between "
                                           "span and window, in $[0,1]$ "
                                           "(difflib)"),
                (r"$\mathrm{overlap}(s,c)$", "**token overlap** — the share of "
                                             "the span's content words present "
                                             "in the chunk"),
                (r"$\cos(s,c)$", "**cosine similarity** of their embeddings"),
                (r"$\tau_{\text{near}}$", f"the **near-verbatim** bar "
                                          f"({QUOTE_NEAR_VERBATIM_THRESHOLD}) "
                                          "— above it, a drifted copy"),
                (r"$\tau_{\text{lex}}$", f"the **paraphrase** lexical bar "
                                         f"({QUOTE_PARAPHRASE_LEX_THRESHOLD})"),
                (r"$\tau_{\text{cos}}$", f"the paraphrase **cosine** bar "
                                         f"({QUOTE_PARAPHRASE_COS_THRESHOLD}), "
                                         "set high so it fires only on "
                                         "near-identity"),
                (r"$\vee$", "**logical OR** — any one route clears the tier"),
                *_SET_NOTATION,
            ], note="Tiers are tried cheapest-first and the first to clear its "
                    "bar wins, so a span is never graded by a more expensive "
                    "test than it needs.")
            st.markdown(
                "**Step 4 — veracity (LLM, flagged spans only).** Provenance "
                "asks *does this text exist in the sources?*; veracity asks "
                "*is what it asserts supported by them?* Spans reaching the "
                "paraphrase or unsupported tier get one entailment call "
                "returning `supported` / `partial` / `contradicted` / "
                "`not_addressed`, plus the fragment of the span the evidence "
                "does carry. The evidence window widens with the tier: a "
                "**paraphrase** is judged against the passage it aligned to "
                "(*did the model reword this faithfully?*), an **unsupported** "
                "span against the row's entire retrieved set (*the text isn't "
                "there, but is the claim?*). A run whose quotes are all "
                "verbatim costs **zero** LLM calls."
            )
            st.markdown(
                "**Step 5 — the verdict.** The two axes are independent, so "
                "the verdict is a 2×2 derived in pure code — no LLM in the "
                "derivation:"
            )
            st.markdown(
                "| | content supported | content not supported |\n"
                "|---|---|---|\n"
                "| **text in sources** | `attributed` | `misattributed` |\n"
                "| **text not in sources** | `paraphrase_grounded` · "
                "`misquote_but_true` | `fabricated` |\n"
            )
            st.markdown(
                "The bottom-left cell is the point of the whole section: a "
                "span that is not in the sources **as text** can still assert "
                "something the sources support. It splits by how far the text "
                "drifted — `paraphrase_grounded` (the model reworded a real "
                "passage) is a much milder failure than `misquote_but_true` "
                "(the model manufactured a quotation whose content happens to "
                "hold). `misattributed` requires entailment-checking spans "
                "that *did* match, which is opt-in "
                "(`--adjudicate-verbatim`) because it costs one call per "
                "verified span."
            )
            st.markdown(
                "**Row verdicts.** Only *attributive* spans count — a scare "
                "quote is not a claim about a source, so it can neither ground "
                "a row nor fabricate one. `quotes_grounded` carries the same "
                "$|A| > 0$ guard as section 2's `quotes_verified`, and for the "
                "same reason: a conjunction over an empty set is vacuously "
                "true, and a row that cited nothing has grounded nothing."
            )
            st.markdown(
                "**Design decisions**\n"
                "- *Section 2 is read, never rewritten.* `quotes_verified` "
                "keeps its strict all-spans-verbatim meaning, so runs from "
                "before this check stay directly comparable with runs after "
                "it. Both numbers are reported side by side.\n"
                "- *The model attests, the code audits.* An unrecognized or "
                "missing support label degrades to `not_addressed` — the "
                "neutral option — so a malformed reply can never manufacture "
                "a `supported` or a `contradicted` verdict.\n"
                "- *Post-hoc, so it is cheap to be wrong.* This stage replays "
                "a saved run's evidence instead of re-answering, so thresholds "
                "can be retuned and the check re-run for the cost of the "
                "flagged spans alone.\n"
                "- *A conservative tier boundary is nearly free.* A span that "
                "misses the paraphrase bar is not lost — it falls through to "
                "`unsupported` and is then adjudicated against the row's "
                "**full** evidence. The threshold decides which label a "
                "grounded span earns, never whether it gets checked."
            )
            st.markdown(
                "**Limitations** — the evidence is scoped by (lab, source "
                "type) and this pipeline has **no world-knowledge oracle** by "
                "design: the answerer only ever sees the corpus. So "
                "`unsupported` means *unsupported by this lab's scoped "
                "corpus*, **not** *false*. A claim can be perfectly true and "
                "still land there because the corpus is silent on it — which "
                "is exactly why `not_addressed` is a separate label from "
                "`contradicted`, and **only `contradicted` is evidence "
                "against a span**. Beyond that: the intent rules are English "
                "cue patterns, not a parser, and will miss unusual phrasings "
                "(they fail toward auditing, so the cost is false alarms); and "
                "the entailment judge is the same model family being audited, "
                "so it is a consistency check, not an independent oracle.\n\n"
                "**Basis in the literature** — the provenance/veracity split "
                "is the *attribution vs correctness* distinction from **AIS** "
                "([Rashkin et al., 2023]"
                "(https://aclanthology.org/2023.cl-4.2/); "
                "[Bohnet et al., 2022](https://arxiv.org/abs/2212.08037)); "
                "scoring a claim against retrieved evidence rather than "
                "against its surface string follows **FActScore** "
                "([Min et al., 2023]"
                "(https://aclanthology.org/2023.emnlp-main.741/)); verified "
                "quoting follows **GopherCite** "
                "([Menick et al., 2022](https://arxiv.org/abs/2203.11147)) and "
                "citation-quality evaluation follows **ALCE** "
                "([Gao et al., 2023]"
                "(https://aclanthology.org/2023.emnlp-main.398/)). Full "
                "reference list in ARCHITECTURE.md §9.5."
            )

        if st.button("Run quote provenance on this run", key="run_prov",
                     help="Replays the run's retrieved evidence and grades every "
                          "quoted span. Works on any saved run — it reads the "
                          "answers and quotes already on disk, and never rewrites "
                          "section 2's verdict. Cheapest-first: verbatim spans "
                          "cost nothing; only paraphrased or absent spans are sent "
                          "to the entailment judge."):
            args = [PYTHON, "scripts/06_run_quote_provenance.py",
                    "--run", hal_run]
            with st.status("Grading quoted spans…", expanded=True) as status:
                rc = stream_subprocess(args, st.empty())
                if rc == 0:
                    status.update(label="Quote provenance complete ✅",
                                  state="complete")
                else:
                    status.update(label=f"Check failed (exit {rc})",
                                  state="error")
            st.rerun()

        if not prov:
            st.caption("Not computed for this run yet — use the button above. "
                       "It works on any saved run, including ones answered "
                       "without **--quotes** (it reads quotation marks in the "
                       "answers themselves).")
        else:
            po = prov["overall"]
            pv = po["verdicts"]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("🚨 fabricated", pv.get("fabricated", 0),
                      help="not in the sources, and the evidence does not "
                           "support what they assert either — the number that "
                           "actually means hallucination")
            p2.metric("📎 misquoted, content sound",
                      pv.get("misquote_but_true", 0) + pv.get("misattributed", 0),
                      help="the quotation was manufactured or miscited, but "
                           "what it claims is carried by the evidence: a "
                           "citation-integrity failure, not a factual one")
            p3.metric("♻️ grounded paraphrases",
                      pv.get("paraphrase_grounded", 0),
                      help="not verbatim, so section 2 fails them, but they "
                           "faithfully reword a passage that IS in the sources")
            p4.metric("💬 figures of speech",
                      pv.get("non_attributive", 0),
                      help="scare quotes, terms of art, hypotheticals — "
                           "quotation marks that never claimed anything about "
                           "a source, so they are not graded at all")

            if po.get("paraphrase_rescue_rate") is not None:
                r1, r2 = st.columns(2)
                r1.metric("Paraphrase rescue rate",
                          f"{po['paraphrase_rescue_rate']:.0%}",
                          help="share of attributive spans that are NOT "
                               "verbatim — so section 2 marks them ❌ — but "
                               "that this check finds grounded anyway. How "
                               "much of the old fabrication number was never "
                               "fabrication.")
                r2.metric("True fabrication rate",
                          f"{po['true_fabrication_rate']:.0%}",
                          help="share of attributive spans that survive as "
                               "genuine fabrications")

            st.info(
                "**Reading `fabricated` correctly:** the evidence is scoped by "
                "(lab, source type) and there is no world-knowledge oracle "
                "here, so this means *not supported by this corpus* — **not** "
                "*false*. Only the `contradicted` support label is evidence "
                "**against** a span.", icon="⚠️")

            if prov_spans is not None and not prov_spans.empty:
                tier_rows = pd.DataFrame(
                    [{"tier": t, "n": n} for t, n in po["tiers"].items()])
                st.altair_chart(
                    alt.Chart(tier_rows).mark_bar().encode(
                        x=alt.X("n:Q", title="attributive spans"),
                        y=alt.Y("tier:N", title=None,
                                sort=["exact", "near_verbatim", "paraphrase",
                                      "unsupported"]),
                        tooltip=["tier", "n"],
                    ).properties(height=140),
                    width="stretch")

                rescued = prov_spans[
                    (prov_spans["verbatim_verified"] == False)  # noqa: E712
                    & (prov_spans["verdict"].isin(
                        ["attributed", "paraphrase_grounded"]))]
                if len(rescued):
                    st.markdown(
                        f"**{len(rescued)} span(s) section 2 failed that are "
                        f"actually grounded** — claimed text against the source "
                        f"it aligns to:")
                    for _, s in rescued.iterrows():
                        with st.expander(f"♻️ {s['org']} · {s['source_type']} · "
                                         f"{s['qid']} — {s['match_tier']}"):
                            st.markdown(f"**Claimed:** “{s['quote']}”")
                            # The lexical-overlap route aligns to a whole chunk
                            # rather than a span, so fall back to the sentence
                            # the entailment judge actually leaned on.
                            source = (_present(s.get("best_span"))
                                      or _present(s.get("evidence_sentence")))
                            if source:
                                st.markdown(f"**Source says:** “{source}”")
                            support = _present(s.get("support"))
                            st.caption(
                                f"tier `{s['match_tier']}` via "
                                f"`{s['match_rule']}` (score {s['match_score']})"
                                + (f" · entailment: `{support}`" if support else ""))

                misq = prov_spans[prov_spans["verdict"].isin(
                    ["misquote_but_true", "misattributed"])]
                if len(misq):
                    st.markdown(
                        f"**{len(misq)} manufactured quotation(s) whose content "
                        f"the evidence still carries** — the quotation marks are "
                        f"not defensible, the claim underneath them is:")
                    for _, s in misq.iterrows():
                        with st.expander(f"📎 {s['org']} · {s['source_type']} · "
                                         f"{s['qid']} — {s['verdict']}"):
                            st.markdown(f"**Claimed:** “{s['quote']}”")
                            evidence = _present(s.get("evidence_sentence"))
                            if evidence:
                                st.markdown(f"**Evidence:** “{evidence}”")
                            fragment = _present(s.get("grounded_fragment"))
                            if fragment:
                                st.markdown("**Part the evidence supports:** "
                                            f"“{fragment}”")
                            reason = _present(s.get("support_reason"))
                            if reason:
                                st.caption(reason)

                fabricated = prov_spans[prov_spans["verdict"] == "fabricated"]
                if len(fabricated):
                    st.markdown(f"**{len(fabricated)} fabricated span(s)** — "
                                "neither the text nor its content is in the "
                                "scoped evidence:")
                    for _, s in fabricated.iterrows():
                        with st.expander(f"🚨 {s['org']} · {s['source_type']} · "
                                         f"{s['qid']}"):
                            st.markdown(f"**Claimed:** “{s['quote']}”")
                            reason = _present(s.get("support_reason"))
                            st.caption(
                                f"support: `{_present(s.get('support')) or '—'}`"
                                + (f" — {reason}" if reason else ""))

                with st.expander("All graded spans", expanded=False):
                    st.dataframe(
                        prov_spans[["org", "source_type", "qid", "quote",
                                    "source", "intent", "intent_rule",
                                    "match_tier", "match_score", "support",
                                    "verdict"]],
                        hide_index=True, width="stretch")

            pdir = runs.run_paths(hal_run)["quote_provenance_dir"]
            dlp1, dlp2 = st.columns(2)
            if (pdir / "summary.json").exists():
                dlp1.download_button(
                    "summary.json", (pdir / "summary.json").read_bytes(),
                    file_name=f"quote_provenance_summary_{hal_run}.json")
            if (pdir / "spans.jsonl").exists():
                dlp2.download_button(
                    "spans.jsonl", (pdir / "spans.jsonl").read_bytes(),
                    file_name=f"quote_provenance_spans_{hal_run}.jsonl")

        # ------- 3 · Metamorphic stability & evidence sensitivity -------
        st.subheader("3 · Metamorphic stability & evidence sensitivity")
        with st.expander("ℹ️ How these checks work", expanded=False):
            st.markdown(
                "There are no gold labels in this pipeline, so correctness "
                "can't be checked directly. Instead we **change the evidence "
                "an answer was built from and watch what the label does**. "
                "Four probes run against each answered item, all through the "
                "same answer → match pipeline as the original run, so any "
                "change in the label comes from the perturbation and nothing "
                "else. The same notation is used in ARCHITECTURE.md §9.3 and "
                "the README."
            )
            st.markdown(
                "**Step 1 — the label.** Each item's label is the matcher's "
                "abstention, or else the top-weight logic (ties break "
                "deterministically by the fixed logic order). *Abstain is a "
                "label, not a gap* — a variant that abstains matches only if "
                "the original also abstained:"
            )
            st.latex(
                r"\mathrm{label}(v)=\begin{cases}"
                r"\texttt{abstain} & \text{the matcher abstained}\\[2pt]"
                r"\arg\max_{k}\, w_k & \text{otherwise}"
                r"\end{cases}"
            )
            st.markdown(
                "**Step 2 — the four probes.** Two ask whether the label "
                "*survives* something harmless; two ask whether it *stops* "
                "when it should.\n\n"
                "| Probe | In plain terms | What a grounded label should do |\n"
                "|---|---|---|\n"
                "| **Control** | run it again, change nothing | keep the same "
                "label — anything else is noise |\n"
                "| **Paraphrase** | say the same thing in different words | "
                "keep the same label |\n"
                "| **Ablation** | remove the excerpt the answer quoted | "
                "weaken, and lean toward *abstain* |\n"
                "| **Distractor** | ask the same question about real text from "
                "the same lab that doesn't answer it | say the excerpts don't "
                "address this |\n\n"
                "For the first two, a **changed** label is suspicious. For the "
                "last two it is the opposite: a label that **survives** is the "
                "warning sign, because the answer clearly didn't need the "
                "evidence it cited."
            )
            st.markdown(
                "**Step 3 — the control, and why it comes first.** The control "
                "re-runs an item on its *unchanged* evidence. Anything that "
                "flips here flipped on its own, so this is the floor every "
                "other number is read against — and an item whose own control "
                "flipped is never called unstable:"
            )
            st.latex(
                r"\mathrm{control\_flip\_rate}=\frac{\left|\{\,i:"
                r"\mathrm{label}(\mathrm{control}_i)\neq\mathrm{label}_0(i)\,\}"
                r"\right|}{n_{\mathrm{control}}}"
            )
            st.markdown(
                f"**Step 4 — paraphrases, and the three gates they must pass.** "
                f"$k = {METAMORPHIC_PARAPHRASES}$ rewrites of the item's "
                f"excerpts are generated by the LLM at temperature "
                f"{METAMORPHIC_PARAPHRASE_TEMPERATURE}. Because a rewrite that "
                f"quietly changed meaning would show up as a hallucination, "
                f"**fidelity is checked in code, not taken on trust.** Each "
                f"rewrite must pass all three:\n\n"
                f"1. **Facts kept** — every number, date and name in the "
                f"original still appears in the rewrite.\n"
                f"2. **Actually reworded** — word overlap with the original "
                f"stays at or below {PARAPHRASE_MAX_TOKEN_OVERLAP}, so a copy "
                f"can't pass as a paraphrase.\n"
                f"3. **Still means the same** — the rewrite must be closer "
                f"(by embedding) to its own excerpt than to any other excerpt "
                f"of the item, with a floor of {PARAPHRASE_MIN_COSINE}.\n\n"
                f"A rewrite that fails any gate is **thrown out and retried**, "
                f"never scored — a bad paraphrase says nothing about the "
                f"model. With $P_{{ok}}$ the rewrites that ran *and* passed, "
                f"and $\\mathrm{{label}}_0$ the original label:"
            )
            st.latex(
                r"\mathrm{label\_stability}="
                r"\frac{|\{\,v\in P_{ok} : \mathrm{label}(v)=\mathrm{label}_0\,\}|}"
                r"{|P_{ok}|}"
                r"\qquad\quad"
                r"\mathrm{unstable}\iff\mathrm{label\_stability}<\theta"
                r"\ \wedge\ \neg\,\mathrm{control\ flipped}"
            )
            st.markdown(
                f"with $\\theta = {METAMORPHIC_STABILITY_THRESHOLD}$ "
                "(`METAMORPHIC_STABILITY_THRESHOLD` in `il_rag/config.py`)."
            )
            st.markdown(
                "**Step 5 — the two probes that expect the label to stop.** "
                "Ablation removes the excerpt the answer quoted (the one check "
                "2 verified, else the top-ranked one). The distractor keeps the "
                "question but swaps in real excerpts retrieved for a different "
                "question of the *same lab and source type* — correct names, "
                "coherent text, wrong topic — and any set that turns out to be "
                "relevant to the original question (by check 1's grounding "
                "score) is rejected rather than scored. Answering anyway is a "
                "problem; answering with the **same logic as before** is the "
                "strongest hallucination signal this pipeline can produce:"
            )
            st.latex(
                r"\mathrm{ablation\_survival\_rate}=\frac{\left|\{\,i:"
                r"\mathrm{label}(\mathrm{abl}_i)=\mathrm{label}_0(i)\neq"
                r"\texttt{abstain}\,\}\right|}{n_{\mathrm{abl}}}"
            )
            st.latex(
                r"\mathrm{prior\_leak\_rate}=\frac{\left|\{\,i:"
                r"\mathrm{label}(\mathrm{dis}_i)=\mathrm{label}_0(i)\neq"
                r"\texttt{abstain}\,\}\right|}{n_{\mathrm{dis}}}"
            )
            symbol_glossary([
                (r"$i$", "**an item** — one answered question being re-tested"),
                (r"$\mathrm{label}_0(i)$", "the item's **original** label from "
                                           "the saved run (subscript 0 = "
                                           "*baseline*)"),
                (r"$\mathrm{label}(v)$", "the label a **variant** $v$ produced "
                                         "when pushed back through the same "
                                         "answer→match path"),
                (r"$P_{\mathrm{ok}}$", "the item's **usable paraphrase "
                                       "variants** — those that ran and passed "
                                       "the fidelity gates"),
                (r"$\mathrm{abl}_i,\ \mathrm{dis}_i$", "item $i$'s **ablation** "
                                                       "and **distractor** "
                                                       "variants"),
                (r"$n_{\mathrm{abl}},\ n_{\mathrm{dis}}$", "**how many** of "
                                                           "each ran usably — "
                                                           "the denominators"),
                (r"$\theta$", f"the **stability threshold** "
                              f"(`METAMORPHIC_STABILITY_THRESHOLD` = "
                              f"{METAMORPHIC_STABILITY_THRESHOLD}); below it an "
                              "item is flagged unstable"),
                (r"$\neq\texttt{abstain}$", "the variant **committed** to a "
                                            "logic rather than abstaining — "
                                            "which is what makes survival "
                                            "suspicious on these two probes"),
                *_SET_NOTATION,
            ], note="Ablation and distractor are DIRECTIONAL probes: a label "
                    "that survives them is the bad outcome, the opposite of "
                    "the paraphrase probe.")
            st.markdown(
                "**Design decisions**\n"
                "- *Failed and rejected variants are excluded from every "
                "denominator*, never counted as flips — a call that failed or "
                "a rewrite that broke the rules is evidence of nothing. Their "
                "counts are reported separately so each probe's health stays "
                "visible.\n"
                f"- *$\\theta = {METAMORPHIC_STABILITY_THRESHOLD}$ is the "
                "strictest setting*: any single paraphrase flip flags the "
                "item. Relax it in `config.py` if paraphrase noise is high.\n"
                "- *The noise floor gates the verdict, not just the reading.* "
                "An item whose control flipped is reported but not counted as "
                "unstable, because its paraphrase result is uninterpretable.\n"
                "- *Gate 3 is rank-based on purpose.* e5 squeezes absolute "
                "cosines into a narrow band (the same reason check 4 only "
                "reads rankings), so the real test is \"nearest to its own "
                f"excerpt\"; {PARAPHRASE_MIN_COSINE} is a coarse guard, not a "
                "calibrated threshold.\n"
                "- When the run also has grounding scores (check 1), stability "
                "is broken down by grounding bucket — the checks "
                "cross-validate each other.\n"
            )
            st.markdown(
                "**Why there is no lab-name swap any more.** An earlier "
                "version renamed the lab throughout the evidence (OpenAI → "
                "DeepMind, and so on) and read a label change as proof the "
                "model was going on its prior about the lab. That test was "
                "dropped: renaming a lab changes what the question is asking, "
                "and the rename only caught the lab's name — product names, "
                "people and events stayed put — so the model was being shown a "
                "document that contradicted itself. A flip could not be pinned "
                "on the name. The **distractor** probe answers the same "
                "underlying question with text that is real and correctly "
                "named, so its number means what it says."
            )
            st.markdown(
                "**Limitations** — the fidelity gates check facts, wording and "
                "topic, but not subtle shifts in emphasis, so a rewrite can "
                "still pass and read slightly differently. The distractor "
                "probe shows how the model behaves on *unrelated* evidence, "
                "which is a harder test than the merely thin evidence it meets "
                "in a real run. And at "
                f"$k = {METAMORPHIC_PARAPHRASES}$ per-item stability is "
                "coarse-grained (steps of "
                f"$1/{METAMORPHIC_PARAPHRASES}$) — the aggregates are the "
                "readable numbers. Flagged items are worth a look by hand; the "
                "originals and rewrites are shown below for exactly that.\n\n"
                "**Basis in the literature** — testing output *relations* "
                "under input transformations when no oracle exists is "
                "metamorphic testing "
                "([Chen et al., 1998](https://arxiv.org/abs/2002.12543); "
                "survey: [Segura et al., 2016]"
                "(https://doi.org/10.1109/TSE.2016.2532875)); "
                "label-preserving perturbations follow CheckList's invariance "
                "tests ([Ribeiro et al., 2020]"
                "(https://aclanthology.org/2020.acl-main.442/)); prediction "
                "consistency under paraphrase follows "
                "([Elazar et al., 2021]"
                "(https://aclanthology.org/2021.tacl-1.60/)); metamorphic "
                "relations for LLM hallucination detection follow MetaQA "
                "([Yang et al., FSE 2025]"
                "(https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/48/Hallucination-Detection-in-Large-Language-Models-with-Metamorphic-Relations)); "
                "the distractor probe's expected behaviour — abstaining when "
                "the evidence doesn't support an answer — follows "
                "([Rajpurkar et al., 2018]"
                "(https://aclanthology.org/P18-2124/)), and reading a surviving "
                "label as the model preferring what it already believed over "
                "what it was shown follows ([Longpre et al., 2021]"
                "(https://aclanthology.org/2021.emnlp-main.565/)); "
                "re-running an unchanged input to establish a noise floor "
                "follows sampling-based consistency checking "
                "([Manakul et al., 2023]"
                "(https://aclanthology.org/2023.emnlp-main.557/)). "
                "Full reference list in ARCHITECTURE.md §9.3."
            )
        with st.expander("Run the metamorphic eval for this snapshot",
                         expanded=stab is None):
            st.caption(
                "Pick which probes to run. Every variant is re-answered and "
                "re-graded through the production path. Resumable — turning a "
                "probe on later only runs the variants you added. Results land "
                "inside this run's folder."
            )
            p1, p2, p3, p4 = st.columns(4)
            use_control = p1.checkbox(
                "Control", value=True, key="mm_p_control",
                help="Same evidence, run again. The noise floor — keep this on "
                     "unless you already know it.")
            use_para = p2.checkbox(
                "Paraphrase", value=True, key="mm_p_para",
                help="Same meaning, different words. A changed label is "
                     "suspicious.")
            use_abl = p3.checkbox(
                "Ablation", value=True, key="mm_p_abl",
                help="Removes the excerpt the answer quoted. A label that "
                     "survives is suspicious.")
            use_dis = p4.checkbox(
                "Distractor", value=True, key="mm_p_dis",
                help="Real same-lab text that doesn't answer the question. "
                     "The right response is to abstain.")
            probes = [p for p, on in (("control", use_control),
                                      ("paraphrase", use_para),
                                      ("ablation", use_abl),
                                      ("distractor", use_dis)) if on]

            c1, c2, c3 = st.columns(3)
            n_para = c1.number_input("Paraphrases per item", 1, 10,
                                     METAMORPHIC_PARAPHRASES, key="mm_para",
                                     disabled=not use_para)
            mm_sample = c2.number_input("Sample size (0 = all items)", 0, 500, 30,
                                        key="mm_sample")
            mm_seed = c3.number_input("Sample seed", 0, 9999, 0, key="mm_seed")

            n_items = len(dfh) if not mm_sample else min(int(mm_sample), len(dfh))
            n_variants = (METAMORPHIC_CONTROLS * use_control
                          + int(n_para) * use_para + use_abl + use_dis)
            # Each variant costs an answer call plus a matching call; each
            # paraphrase costs one more call to generate the rewrite.
            n_calls = n_items * (2 * n_variants + int(n_para) * use_para)
            if not probes:
                st.warning("Select at least one probe.")
            else:
                st.caption(f"≈ {n_items} item(s) × {n_variants} variant(s) "
                           f"≈ {n_calls} chat calls.")
            if st.button("Run metamorphic eval", type="primary",
                         disabled=not api_key_present() or not hal_run
                         or not probes,
                         key="mm_go"):
                args = [PYTHON, "scripts/03_run_metamorphic_eval.py",
                        "--run", hal_run,
                        "--probes", *probes,
                        "--paraphrases", str(int(n_para)),
                        "--seed", str(int(mm_seed))]
                if mm_sample:
                    args += ["--sample", str(int(mm_sample))]
                with st.status("Running metamorphic eval…", expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    if rc == 0:
                        status.update(label="Metamorphic eval complete ✅",
                                      state="complete")
                    else:
                        status.update(label=f"Eval failed (exit {rc})",
                                      state="error")
                st.rerun()

        if stab:
            s = stab["summary"]
            items = pd.DataFrame(stab["per_item"])

            def _flag(col) -> pd.Series:
                """A column as a boolean mask, tolerating older result files."""
                series = items.get(col)
                if series is None:
                    return pd.Series(False, index=items.index)
                return series.fillna(False).astype(bool)

            ran = s.get("probes") or ["paraphrase"]
            st.caption(f"Evaluated {s['items']} item(s) — probes: "
                       + ", ".join(ran)
                       + (f", sample={s['sample']}" if s.get("sample") else ""))

            def _rate(key, fmt="{:.2f}"):
                v = s.get(key)
                return "—" if v is None else fmt.format(v)

            m1, m2, m3 = st.columns(3)
            m1.metric("🎛️ Control flip rate", _rate("control_flip_rate"),
                      help="how often the label changed with NOTHING changed. "
                           "This is the noise floor — read every number below "
                           "against it.")
            m2.metric("🧠 Prior leak rate", _rate("prior_leak_rate"),
                      help="how often the original label came back from text "
                           "that doesn't answer the question — the model went "
                           "on what it already believed. Higher is worse.")
            m3.metric("🕳️ Survived ablation", _rate("ablation_survival_rate"),
                      help="how often the label held after the excerpt the "
                           "answer quoted was removed — the answer didn't need "
                           "its evidence. Higher is worse.")

            m4, m5, m6 = st.columns(3)
            ms = s.get("mean_label_stability")
            m4.metric("Mean label stability",
                      "—" if ms is None else f"{ms:.2f}",
                      help="fraction of paraphrases keeping the original "
                           "label, averaged over items. Higher is better.")
            pf = s.get("pct_fully_stable")
            m5.metric("Fully stable items", "—" if pf is None else f"{pf:.0f}%")
            m6.metric("🎲 Unstable items", s.get("n_unstable", 0),
                      delta=None if not s.get("n_unstable") else "detection",
                      delta_color="inverse",
                      help="a paraphrase changed the label, and the item's own "
                           "control did not — so it isn't just noise.")

            if s.get("mean_paraphrase_divergence") is not None \
                    or s.get("n_paraphrases_rejected"):
                q1, q2, q3 = st.columns(3)
                rejected = s.get("n_paraphrases_rejected", 0)
                reasons = s.get("rejected_by_reason") or {}
                q1.metric("Rewrites thrown out", rejected,
                          help="failed a fidelity gate or the call itself — "
                               "discarded, never counted as a flip. "
                               + (", ".join(f"{r}: {n}"
                                            for r, n in reasons.items())
                                  if reasons else "none"))
                q2.metric("Wording changed",
                          _rate("mean_paraphrase_divergence"),
                          help="how far the rewrites moved from the original "
                               "wording (0 = identical, 1 = no words in "
                               "common).")
                q3.metric("Meaning kept", _rate("mean_paraphrase_cosine"),
                          help="similarity between each rewrite and its own "
                               "excerpt. Near 1 is what a faithful paraphrase "
                               "looks like.")

            cat_rows = [{"category": c, "stability": v}
                        for c, v in s.get("by_category", {}).items()
                        if v is not None]
            if cat_rows:
                cdf = pd.DataFrame(cat_rows)
                cat_chart = (
                    alt.Chart(cdf)
                    .mark_bar()
                    .encode(
                        x=alt.X("stability:Q", title="mean label stability",
                                scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("category:N", title=None,
                                sort=[c for c in CATEGORIES]),
                        color=alt.Color("stability:Q", legend=None,
                                        scale=alt.Scale(scheme="redyellowgreen",
                                                        domain=[0, 1])),
                        tooltip=["category",
                                 alt.Tooltip("stability:Q", format=".2f")],
                    )
                    .properties(height=220)
                )
                st.altair_chart(cat_chart, width="stretch")

            if "by_grounding_bucket" in s:
                st.caption("Stability by grounding bucket (from this run's "
                           "--grounding scores):")
                st.dataframe(pd.DataFrame(s["by_grounding_bucket"]).T,
                             width="stretch")

            flagged = items[_flag("unstable")
                            | _flag("prior_keyed")
                            | _flag("label_survived_ablation")]
            if flagged.empty:
                st.success("✅ No item was flagged: labels held under "
                           "paraphrase, and none of them came back from "
                           "evidence that didn't support them.")
            else:
                st.markdown(f"**⚠️ {len(flagged)} flagged item(s)** — the "
                            "detection firing, item by item:")
                vdf = load_variants(hal_run)
                for _, it in flagged.iterrows():
                    badges = []
                    if it.get("unstable"):
                        badges.append("🎲 unstable")
                    if it.get("prior_keyed"):
                        badges.append("🧠 prior-keyed")
                    if it.get("label_survived_ablation"):
                        badges.append("🕳️ unsupported")
                    ls = it.get("label_stability")
                    title = (f"{' + '.join(badges)} · {it['org']} · "
                             f"{it['source_type']} · {it['qid']} — original "
                             f"label: {it['original_label']}"
                             + (f", stability {ls:.2f}"
                                if ls is not None and not pd.isna(ls) else ""))
                    with st.expander(title):
                        if it.get("prior_keyed"):
                            st.markdown(
                                f"**Prior-keyed:** asked the same question over "
                                f"excerpts retrieved for *"
                                f"{it.get('distractor_category', '?')}* — text "
                                f"from the same lab that doesn't address it — "
                                f"and it answered **{it['original_label']}** "
                                f"again instead of saying the excerpts don't "
                                f"cover this.")
                        if it.get("label_survived_ablation"):
                            st.markdown(
                                f"**Unsupported:** removing the excerpt the "
                                f"answer rested on "
                                f"(*{it.get('ablation_basis', '?')}*) left the "
                                f"label at **{it['original_label']}**.")
                        if it.get("control_flipped"):
                            st.info("This item's control also flipped, so its "
                                    "paraphrase result is noise — read the "
                                    "directional probes only.")
                        if vdf is None:
                            continue
                        sub = vdf[(vdf["org"] == it["org"])
                                  & (vdf["source_type"] == it["source_type"])
                                  & (vdf["qid"] == it["qid"])]
                        if sub.empty:
                            continue
                        if "label" in sub.columns:
                            disp = sub[["variant_kind", "variant_idx", "label",
                                        "label_matches_original"]].copy()
                            disp["label_matches_original"] = disp[
                                "label_matches_original"].map(
                                {True: "✅ kept", False: "❌ changed"})
                            st.dataframe(
                                disp.rename(columns={
                                    "variant_kind": "probe",
                                    "variant_idx": "#",
                                    "label_matches_original": "vs original",
                                }), hide_index=True, width="stretch")
                        for _, v in sub.iterrows():
                            kind = v.get("variant_kind")
                            changed = v.get("label_matches_original") is False
                            directional = kind in ("ablation", "distractor")
                            if not changed and not directional:
                                continue
                            st.markdown(f"**{kind} #{v.get('variant_idx')} → "
                                        f"{v.get('label', '?')}** — answer:")
                            st.caption(v.get("answer") or "(no answer)")
                            # For a paraphrase flip, the thing worth auditing is
                            # whether the rewrite really did say the same thing.
                            src_ctx = v.get("source_context")
                            new_ctx = v.get("context")
                            if kind == "paraphrase" and isinstance(src_ctx, list) \
                                    and isinstance(new_ctx, list):
                                with st.popover("Compare the rewritten "
                                                "excerpts"):
                                    for n, (before, after) in enumerate(
                                            zip(src_ctx, new_ctx), 1):
                                        st.markdown(f"**Excerpt {n} — original**")
                                        st.caption(before)
                                        st.markdown(f"**Excerpt {n} — rewrite**")
                                        st.caption(after)
                                        st.divider()

            dl1, dl2 = st.columns(2)
            mdir = runs.run_dir(hal_run) / "metamorphic"
            if (mdir / "stability.json").exists():
                dl1.download_button("stability.json",
                                    (mdir / "stability.json").read_bytes(),
                                    file_name=f"stability_{hal_run}.json")
            if (mdir / "variants.jsonl").exists():
                dl2.download_button("variants.jsonl",
                                    (mdir / "variants.jsonl").read_bytes(),
                                    file_name=f"variants_{hal_run}.jsonl")

# ---------------------------------------------------------------------------
# Topics tab — the inductive layer: what the corpus talks about, and how those
# topics relate to the deductive logic scores. Read-only: fitting happens
# locally (BERTopic is not installed in the container).
# ---------------------------------------------------------------------------
with tab_topics:
    # Imported here explicitly: other tabs import altair inside conditional
    # branches, so it is not guaranteed to be bound when this tab renders.
    import altair as alt

    st.header("Topic layer (inductive)")
    st.caption(
        "Everything else here is deductive — it scores the corpus against "
        "Thornton & Ocasio's seven logics. This layer clusters the corpus "
        "with **no knowledge of the taxonomy**, then compares the two."
    )

    with st.expander("ℹ️ How the topic layer is computed", expanded=False):
        st.markdown(
            "Fitted with **BERTopic** (Grootendorst, 2022) in "
            "`il_rag/topics.py`, reusing the chunk embeddings already stored "
            "in Chroma — so it costs **no API calls** and nothing is "
            "re-embedded. Three stages:"
        )
        st.markdown(
            "**Stage 1 — reduce.** UMAP projects each 1024-dim chunk embedding "
            "to 5 dimensions under cosine distance, because density clustering "
            "degrades badly in high dimensions. UMAP is stochastic, so a fixed "
            "`random_state` is used and recorded — without it the topics move "
            "between runs.\n\n"
            "**Stage 2 — cluster.** HDBSCAN groups the reduced points. It does "
            "**not** force every chunk into a topic: points in no dense region "
            "are labelled $-1$ (*unclustered*) and are reported separately "
            "rather than being padded into a topic they don't belong to.\n\n"
            "**Stage 3 — name.** Each cluster is described by **c-TF-IDF**: "
            "term frequency computed per *topic* (all its chunks concatenated) "
            "rather than per document, so the words that come out are the ones "
            "distinctive to that topic rather than common across the corpus:"
        )
        st.latex(
            r"\mathrm{c\text{-}TF\text{-}IDF}(t, k)=f_{t,k}\;\cdot\;"
            r"\log\!\left(1+\frac{A}{\sum_{j} f_{t,j}}\right)"
        )
        st.markdown(
            "where $f_{t,k}$ is the frequency of term $t$ in topic $k$ and $A$ "
            "the average number of words per topic."
        )
        st.markdown(
            "**Topic × logic cross-tab.** Every answered question recorded the "
            "ids of the $k$ chunks it was answered from, and each chunk now has "
            "a topic. So a row's logic weights can be attributed back to the "
            "topics of its evidence. Each chunk receives $1/k$ of the row's "
            "credit, so every row contributes a total mass of exactly 1 "
            "regardless of how much evidence it used:"
        )
        st.latex(
            r"M_{k,\ell} \;=\; \sum_{i \in A}\;\sum_{c \,\in\, R(i)}"
            r"\frac{1}{|R(i)|}\; w^{(i)}_{\ell}\;"
            r"\mathbb{1}\bigl[\mathrm{topic}(c)=k\bigr]"
        )
        st.markdown(
            "Each topic's row is then normalized to a percentage distribution "
            "over the logics — *when evidence of this topic was used, the "
            "resulting answers scored like this*:"
        )
        st.latex(r"P^{\text{topic }k}_{\ell}=\frac{100\,M_{k,\ell}}"
                 r"{\sum_{m} M_{k,m}}")
        symbol_glossary([
            (r"$t$", "**a term** (word) appearing in a topic"),
            (r"$k$", "**a topic** — one cluster of chunks"),
            (r"$f_{t,k}$", "how often term $t$ occurs **in topic $k$** (all its "
                           "chunks treated as one document)"),
            (r"$A$", "the **average number of words per topic** — the "
                     "yardstick that stops a large topic winning on volume"),
            (r"$\ell$", "**a logic** — one of the seven"),
            (r"$m$", "*another* logic, used for the summing denominator"),
            (r"$i$", "**a question** (an answered audit row)"),
            (r"$A$ (in the cross-tab)", "the set of **answered** questions — "
                                        "note this $A$ is unrelated to the $A$ "
                                        "in c-TF-IDF above"),
            (r"$R(i)$", "the **chunks retrieved** for question $i$"),
            (r"$\lvert R(i)\rvert$", "**how many** — the $k$ in the $1/k$ "
                                     "credit split"),
            (r"$w^{(i)}_{\ell}$", "question $i$'s **matcher weight** on logic "
                                  "$\\ell$"),
            (r"$\mathbb{1}[\cdot]$", "the **indicator**: 1 when the condition "
                                     "inside holds, 0 otherwise — here it "
                                     "selects only chunks belonging to topic $k$"),
            (r"$M_{k,\ell}$", "the **accumulated mass**: how much logic-$\\ell$ "
                              "credit topic $k$ collected across the whole run"),
            (r"$P^{\text{topic }k}_{\ell}$", "that mass as a **percentage** of "
                                             "topic $k$'s row"),
            *_SET_NOTATION,
        ], note="Two different quantities are both conventionally written $A$; "
                "the c-TF-IDF one is a word count, the cross-tab one is a set "
                "of questions.")
        st.markdown(
            "**Coverage audit.** Topics that appear in **no** row's retrieved "
            "evidence are corpus regions the questionnaire never reaches — a "
            "structural blind spot of the instrument, not of the corpus."
        )
        st.markdown(
            "**Design decisions**\n"
            "- *A topic is not a logic.* Topics are **subject matter** (export "
            "controls, lawsuits, funding); a logic is an **ordering principle** "
            "(what confers legitimacy, who holds authority) that cuts across "
            "subject matter. A clean topic→logic mapping is therefore not "
            "expected, and its absence falsifies neither layer. This cross-tab "
            "describes how they co-occur; it does not claim they are the same "
            "thing.\n"
            "- *Why this layer exists.* It is the answer to the circularity "
            "objection: the deductive pipeline was told which seven logics to "
            "look for. BERTopic was told nothing, so where the two agree, the "
            "agreement is **convergent validity** from an independent method.\n"
            "- *Why uniform $1/k$ attribution?* The pipeline does not record "
            "which retrieved chunk actually drove the answer, so any weighting "
            "by rank or cosine would invent precision that isn't there. Uniform "
            "is the honest default.\n"
            "- *Why it runs locally.* UMAP on 15.5k × 1024 vectors peaks well "
            "above the 1 GB cloud machine, and the dependencies (UMAP, HDBSCAN, "
            "scikit-learn) would bloat every deploy. Topics are fitted on a "
            "workstation; only the small JSON results are shipped and read here."
        )

    tinfo = topics_mod.load_topic_info()
    if tinfo is None:
        st.info(
            "No topic model on disk yet. It is fitted **locally** (not in this "
            "hosted app):\n\n"
            "```bash\n"
            ".venv/bin/pip install -r requirements-topics.txt\n"
            ".venv/bin/python scripts/07_run_topics.py fit\n"
            ".venv/bin/python scripts/07_run_topics.py crosstab\n"
            "```\n"
            "Then ship `data/topics/` alongside the index (see DEPLOY.md).",
            icon="🧭")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Topics found", tinfo["n_topics"])
        c2.metric("Chunks clustered",
                  f"{tinfo['n_chunks'] - tinfo['n_outliers']:,}",
                  delta=f"{tinfo['n_outliers']:,} unclustered",
                  delta_color="off")
        c3.metric("Min topic size", tinfo["min_topic_size"])
        c4.metric("Seed", tinfo["seed"],
                  help="UMAP random_state — fixed so topics reproduce")
        st.caption(f"Fitted {tinfo.get('fitted_at', '?')}")

        tdf = pd.DataFrame([
            {"topic": r["topic"], "size": r["size"], "label": r["label"],
             **{f"n_{o}": r["by_org"].get(o, 0) for o in ORGS},
             **{f"n_{s}": r["by_source"].get(s, 0) for s in SOURCE_TYPES}}
            for r in tinfo["topics"] if not r["is_outlier"]
        ])

        st.subheader("What the corpus is about")
        show_n = st.slider("Show top N topics by size", 5,
                           min(60, max(5, len(tdf))), min(20, len(tdf)),
                           key="topics_n")
        top_tdf = tdf.nlargest(show_n, "size")
        chart = (
            alt.Chart(top_tdf)
            .mark_bar()
            .encode(
                x=alt.X("size:Q", title="chunks"),
                y=alt.Y("label:N", title=None, sort="-x"),
                tooltip=["topic", "label", "size",
                         *[f"n_{o}" for o in ORGS],
                         *[f"n_{s}" for s in SOURCE_TYPES]],
            )
            .properties(height=min(24 * show_n + 40, 900))
        )
        st.altair_chart(chart, width="stretch")
        with st.expander("All topics (table)", expanded=False):
            st.dataframe(tdf.sort_values("size", ascending=False),
                         hide_index=True, width="stretch")

        # --- cross-tab against a run ---
        st.subheader("Topic × logic")
        topic_run = run_selectbox("Run to cross-tab", key="topics_run",
                                  default_run_id=runs.get_current())
        xtab = topics_mod.load_crosstab(topic_run)
        if xtab is None:
            st.info(
                "No cross-tab for this run yet. Locally:\n\n"
                "```bash\n"
                f".venv/bin/python scripts/07_run_topics.py crosstab --run {topic_run}\n"
                "```", icon="🔗")
        else:
            recs = xtab["topics"]
            st.caption(
                f"{len(recs)} topics appeared in this run's retrieved evidence. "
                "Each row shows how answers grounded in that topic scored "
                "across the logics — attribution is "
                f"{xtab.get('attribution', 'uniform')}."
            )
            min_hits = st.slider(
                "Minimum retrievals (filters out thinly-evidenced topics)",
                1, 50, 5, key="topics_min_hits")
            shown = [r for r in recs if r["retrievals"] >= min_hits
                     and r["topic"] != topics_mod.OUTLIER_TOPIC]
            if not shown:
                st.warning("No topic meets that retrieval threshold.")
            else:
                heat = pd.DataFrame([
                    {"topic": f"{r['topic']}: {r['label'][:34]}",
                     "logic": logic, "pct": r["logic_pct"][logic],
                     "retrievals": r["retrievals"]}
                    for r in shown for logic in LOGICS
                ])
                hchart = (
                    alt.Chart(heat)
                    .mark_rect()
                    .encode(
                        x=alt.X("logic:N", sort=LOGICS, title=None),
                        y=alt.Y("topic:N", title=None,
                                sort=[f"{r['topic']}: {r['label'][:34]}"
                                      for r in shown]),
                        color=alt.Color("pct:Q", title="% of logic mass",
                                        scale=alt.Scale(scheme="blues")),
                        tooltip=["topic", "logic",
                                 alt.Tooltip("pct:Q", format=".1f"),
                                 "retrievals"],
                    )
                    .properties(height=min(26 * len(shown) + 40, 900))
                )
                st.altair_chart(hchart, width="stretch")
                st.caption(
                    "Read a row, not a column: each row sums to 100%. A row "
                    "concentrated on one logic means evidence of that topic "
                    "consistently produced answers scored that way."
                )

            # --- coverage audit ---
            cov = xtab["coverage"]
            st.subheader("Coverage audit — what the questionnaire never asks about")
            k1, k2, k3 = st.columns(3)
            k1.metric("Topics reached",
                      f"{cov['n_topics_retrieved']}/{cov['n_topics']}")
            k2.metric("Never retrieved", cov["n_topics_never_retrieved"])
            k3.metric("Of clustered corpus",
                      f"{cov['chunks_never_retrieved_share']:.1%}",
                      help="share of clustered chunks in topics no question "
                           "ever retrieved")
            with st.expander("ℹ️ What “topics reached” actually means",
                             expanded=False):
                n_q = cov.get("questions")
                slots = cov.get("retrieval_slots")
                distinct = cov.get("distinct_chunks_retrieved")
                corpus_n = cov.get("corpus_chunks")
                st.markdown(
                    "**Step 1 — the corpus was grouped into themes.** BERTopic "
                    f"clustered all {corpus_n:,} chunks into "
                    f"**{cov['n_topics']} topics**, each a bundle of chunks "
                    "about the same thing (one is export controls, another "
                    "copyright settlements, another chain-of-thought "
                    "faithfulness)."
                    if corpus_n else
                    "**Step 1 — the corpus was grouped into themes** by "
                    "BERTopic."
                )
                if n_q and slots and distinct and corpus_n:
                    st.markdown(
                        "**Step 2 — the questionnaire only ever sees what "
                        f"retrieval hands it.** This run asked **{n_q} "
                        f"questions**, each retrieving evidence — "
                        f"**{slots:,} retrieval slots** in total. But the same "
                        "useful chunks get pulled repeatedly across questions, "
                        f"so those slots were filled by only **{distinct:,} "
                        f"distinct chunks — {distinct / corpus_n:.1%} of the "
                        "corpus.**"
                    )
                st.markdown(
                    "**Step 3 — map those chunks back to their topics.** They "
                    f"come from **{cov['n_topics_retrieved']} of "
                    f"{cov['n_topics']}** topics. So chunks belonging to the "
                    f"other **{cov['n_topics_never_retrieved']} topics were "
                    "never retrieved even once** by any question. That is what "
                    "“topics reached” counts: a topic is *reached* if it "
                    "contributed at least one chunk of evidence anywhere in "
                    "the run."
                )
                st.markdown(
                    "**Why the topic count is the meaningful number, not the "
                    "chunk percentage.** “Only a small % of chunks were read” "
                    "is trivially true — 27 questions physically cannot touch "
                    "a corpus this size, and a questionnaire is *supposed* to "
                    "be selective. The topic view says something sharper: it "
                    "is not that the corpus was sampled thinly and evenly, it "
                    "is that **entire coherent subject areas are invisible to "
                    "the instrument** — zero chunks from them, not a few."
                )
                st.markdown(
                    "**Why it happens.** Retrieval is driven by question "
                    "wording. These questions ask about governance — "
                    "obligations, authority, legitimacy, funding, oversight — "
                    "so they retrieve governance-flavoured evidence. A chunk "
                    "about a technical safety method sits far from all of them "
                    "in embedding space and never lands in any top-$k$."
                )
                st.markdown(
                    "**Why it matters.** It *bounds the claim*: the profiles "
                    "do not characterise “the corpus”, they characterise the "
                    "part of it this questionnaire reaches. That is a "
                    "limitation worth stating outright — and it doubles as a "
                    "roadmap, since the list below names exactly which subject "
                    "areas new questions would have to target."
                )
            if cov["never_retrieved"]:
                st.markdown(
                    "These topics exist in the corpus but **no question ever "
                    "retrieved them** — the instrument is structurally blind "
                    "to them. This is a limitation of the 27-question "
                    "questionnaire, not of the corpus:")
                st.dataframe(
                    pd.DataFrame(cov["never_retrieved"])[
                        ["topic", "size", "label"]],
                    hide_index=True, width="stretch")
            else:
                st.success("Every topic was reached by at least one question.")

            st.caption(xtab.get("note", ""))

        # --- Keyword retention: the semantic keyword matcher ---------------
        # Sits outside the cross-tab branch on purpose: it needs `tinfo` and a
        # run, not `topic_logic.json`, and must not vanish when that is absent.
        st.subheader("Keyword retention — does a topic's vocabulary reach the answers?")
        st.caption(
            "The keywords above are BERTopic's description of each topic. This "
            "section scores them: when the pipeline answered a question from "
            "evidence belonging to a topic, how much of that topic's vocabulary "
            "actually made it into the answer — **verbatim**, as an "
            "**inflection**, as a **semantic neighbour**, or not at all. It is "
            "the graded counterpart of the exact-match keyword judge on the "
            "Results tab, which by construction cannot see synonymy."
        )

        with st.expander("ℹ️ How keyword retention is scored", expanded=False):
            st.markdown(
                "Implemented in `il_rag/topic_keywords.py`. Same shape as the "
                "quote-provenance ladder: four rungs, cheapest first, and the "
                "**first rung to clear its bar wins**."
            )
            st.markdown(
                "**Rung 1 — exact.** The keyword occurs as a whole token. A "
                "bigram (*export controls*) must occur **adjacently**, because "
                "adjacency is what makes it the phrase; if only its words "
                "appear, the phrase takes the **weaker** of the two and is "
                "capped one rung down.\n\n"
                "**Rung 2 — morphological.** Some answer word shares its stem "
                "(*managers*, *managerial*, *management* for *manager*). Two "
                "conditions, both required: a shared leading stem of at least "
                f"**{KEYWORD_MORPH_MIN_PREFIX}** characters *and* a character "
                f"similarity of at least **{KEYWORD_MORPH_MIN_RATIO}**. Neither "
                "alone is safe — the prefix rule on its own accepts "
                "*manage*/*mandate*.\n\n"
                "**Rung 3 — semantic.** Some answer word is closer to the "
                "keyword than $\\tau_{\\text{sem}}$ of random corpus word "
                "pairs. This is the rung the feature exists for.\n\n"
                "**Rung 4 — absent.** Nothing cleared a bar. The best near-miss "
                "percentile is still recorded, so *nowhere close* is "
                "distinguishable from *just under the bar*."
            )
            st.latex(
                r"S(\kappa, a)=\begin{cases}"
                r"1.0 & \kappa \sqsubseteq_{\text{tok}} a \\[4pt]"
                r"\mathrm{ratio}(\kappa, w) & \exists\, w \in W(a):\ "
                r"\mathrm{pre}(\kappa,w)\ge p^{*}\ \wedge\ "
                r"\mathrm{ratio}(\kappa,w)\ge \tau_{\text{morph}} \\[4pt]"
                r"\min\bigl(\pi(\cos(\kappa,w^{*})),\, \sigma_{\max}\bigr) & "
                r"\pi(\cos(\kappa,w^{*})) \ge \tau_{\text{sem}},\ \ "
                r"w^{*}=\arg\max_{w \in C(a)} \cos(\kappa, w) \\[4pt]"
                r"0 & \text{otherwise}"
                r"\end{cases}"
            )
            st.markdown(
                "**Why a raw cosine is not a percentage here.** `config.py` "
                "records the measurement that forces this design: on this "
                "stack a faithful reword scored **0.849** while a wholly "
                "unrelated claim scored **0.807** — 0.04 apart. Single *words* "
                "are worse; this corpus's whole vocabulary sits in a band "
                "roughly 0.74–0.82 wide. So the semantic rung reports a "
                "**percentile** against the cosines of every pair of the "
                "corpus's most frequent words:"
            )
            st.latex(
                r"\pi(x)\;=\;\frac{\bigl|\{\,g \in G\ :\ g \le x\,\}\bigr|}{|G|}"
                r"\qquad "
                r"G=\Bigl\{\cos(u,v)\ :\ \{u,v\}\subset V,\ u\neq v\Bigr\}"
            )
            st.markdown(
                "That also disposes of the obvious objection. The corpus is "
                "entirely about AI labs, so *manager*/*sales* genuinely **is** "
                "fairly close in absolute cosine — but it is a *typical* pair "
                "here and lands near the background median, while "
                "*manager*/*hierarchy* lands in the tail. Calibrating against "
                "**this** corpus is what turns "
                f"$\\tau_{{\\text{{sem}}}}={KEYWORD_SEMANTIC_MIN_PERCENTILE}$ "
                "into a selective test instead of a rubber stamp. The "
                "explorer at the bottom of this page shows the effect directly."
            )
            st.markdown(
                "**Rolling up.** A row's keywords are the union over the topics "
                "of the chunks it was answered from. Retention is the mean "
                "score; the per-topic figure weights each answered row equally, "
                "so a heavily-retrieved topic cannot dominate its own average:"
            )
            st.latex(
                r"V_i=\frac{1}{|K(i)|}\sum_{\kappa \in K(i)} S(\kappa, a_i)"
                r"\qquad "
                r"V_k=\frac{1}{|A_k|}\sum_{i \in A_k}\frac{1}{|K_k|}"
                r"\sum_{\kappa \in K_k} S(\kappa, a_i)"
            )
            st.markdown(
                "**Semantic lift** is the headline — the part of retention an "
                "exact-match judge is blind to — and the two reference arms are "
                "what give a retention figure a referent at all:"
            )
            st.latex(
                r"L_i = V_i - \sigma^{\text{exact}}_i"
                r"\qquad\qquad "
                r"N_i \;\le\; V_i \;\le\; C_i"
            )
            st.markdown(
                "$C_i$ is the same keywords, **exact rung only, against the "
                "retrieved chunks**. Keywords are derived *from* those chunks "
                "by c-TF-IDF, so it runs high by construction — it is the "
                "circularity concern stated as a number rather than caveated "
                f"away. $N_i$ is the full ladder against **{KEYWORD_NULL_DRAWS} "
                "topics the row never retrieved**. If $V_i$ sits at $N_i$, the "
                "answers carry no more of the topic's vocabulary than a random "
                "topic's — and that is the finding."
            )
            symbol_glossary([
                (r"$\kappa$", "**a topic keyword** — one c-TF-IDF term, "
                              "unigram or bigram"),
                (r"$a_i$", "the **RAG answer** of row $i$"),
                (r"$W(a)$", "every **word** in the answer, unfiltered — a "
                            "literal hit is a literal hit, so two-character "
                            "keywords and stopwords are matchable here"),
                (r"$C(a)$", "the answer's **candidate words** for the semantic "
                            "rung: content words only, ranked by frequency, "
                            f"capped at **{KEYWORD_MAX_CANDIDATES}**"),
                (r"$\sqsubseteq_{\text{tok}}$", "**occurs as a whole token** "
                                                "(for a bigram: as two "
                                                "adjacent tokens)"),
                (r"$\mathrm{pre}(\kappa,w)$", "the length of their **shared "
                                              "leading stem**, in characters"),
                (r"$p^{*}$", f"the **stem bar** — {KEYWORD_MORPH_MIN_PREFIX} "
                             "characters, or the length of the shorter word "
                             "if that is less"),
                (r"$\mathrm{ratio}$", "**character similarity** in $[0,1]$ "
                                      "(difflib)"),
                (r"$\tau_{\text{morph}}$", f"the **morphological bar**: "
                                           f"**{KEYWORD_MORPH_MIN_RATIO}**"),
                (r"$\cos(u,v)$", "**cosine similarity** of two word vectors"),
                (r"$\pi(x)$", "the **percentile** of cosine $x$ in the corpus "
                              "background distribution — *not* a similarity"),
                (r"$G$", "the **background**: the cosine of every pair of "
                         "corpus vocabulary words"),
                (r"$V$", f"the **vocabulary** the background is built from — "
                         f"the corpus's {CALIBRATION_VOCAB_SIZE} most frequent "
                         "content words"),
                (r"$\tau_{\text{sem}}$", "the **semantic bar**: "
                                         f"**{KEYWORD_SEMANTIC_MIN_PERCENTILE}"
                                         "**, i.e. *closer than 90% of random "
                                         "corpus word pairs*"),
                (r"$\sigma_{\max}$", "the **synonym cap**: "
                                     f"**{KEYWORD_SEMANTIC_MAX_SCORE}**. A "
                                     "neighbour never scores a full 100% — "
                                     "that is reserved for a literal "
                                     "occurrence"),
                (r"$K_k$", "topic $k$'s **keyword set** (its c-TF-IDF terms)"),
                (r"$K(i)$", "the **union** of the keyword sets of the topics "
                            "row $i$'s evidence belonged to"),
                (r"$A_k$", "the **answered rows** that retrieved evidence from "
                           "topic $k$"),
                (r"$V_i,\;V_k$", "**retention** — per row, and per topic"),
                (r"$\sigma^{\text{exact}}_i$", "the share of row $i$'s "
                                               "keywords matched **verbatim**"),
                (r"$L_i$", "**semantic lift** — the part of retention an "
                           "exact-match judge cannot see"),
                (r"$C_i$", "the **verbatim ceiling**: the same keywords "
                           "against the chunks they were derived from"),
                (r"$N_i$", "the **null floor**: the same ladder against topics "
                           "the row never retrieved"),
                *_SET_NOTATION,
            ], note="$w^{*}$ is the answer word that produced the match — the "
                    "star means *the one that won*, not a footnote. It is "
                    "stored with every verdict, which is why the drill-down "
                    "can show you which word did it.")
            st.markdown(
                "**Design decisions**\n"
                "- *Why percentile, not raw cosine.* See the 0.849/0.807 "
                "measurement above. The band e5 puts single words in is too "
                "tight to gate on, and far too tight to render as a "
                "percentage.\n"
                "- *Why the rungs short-circuit.* A keyword is never graded by "
                "a more expensive test than it needs, and a machine with no "
                "calibration on disk still gets the two lexical rungs rather "
                "than nothing.\n"
                "- *Why the weaker part carries a split bigram.* “export "
                "controls” is present only to the extent that **both** "
                "concepts are; a mean would let *controls* alone carry it.\n"
                "- *Why the maximum over candidate words.* The grounding "
                "check's argument: one genuinely relevant hit is enough, and a "
                "strong hit must not be diluted by unrelated siblings.\n"
                "- *Why the ceiling and floor arms exist.* A retention figure "
                "with no referent is not a finding. The ceiling states the "
                "circularity out loud; the floor is the noise level, the same "
                "role the metamorphic control arm plays for flip rates.\n"
                "- *Why this does not replace the keyword judge on the Results "
                "tab.* Different keywords (topic c-TF-IDF vs per-logic "
                "reference), different target (answers vs logic "
                "classification), different question. That judge stays exactly "
                "as it is — the value of a blunt, fully visible baseline is "
                "that it is blunt."
            )
            st.markdown(
                "**Limitations, stated plainly**\n"
                "1. **Circularity.** Topic keywords are derived *from* the "
                "corpus chunks, so scoring them against those same chunks is "
                "near-tautological — which is what the ceiling arm measures "
                "and why it is shown rather than hidden. The non-trivial "
                "comparison is against the **answers**.\n"
                "2. **These vectors capture topical relatedness, not "
                "synonymy.** *manager* and *hierarchy* are related, not "
                "synonyms. Antonyms embed close together (*permit* / "
                "*prohibit*), so a high semantic score can mean the answer "
                "discusses the concept **and takes the opposite position**. "
                "Read the matched word, never the score alone.\n"
                "3. **Percentiles are readable, not absolute** — a rank within "
                "*this* corpus under *this* embedding model. Both are recorded "
                "in `calibration.json`.\n"
                "4. **The keyword sets are vectorizer artifacts**: ten "
                "c-TF-IDF terms under one particular set of settings. "
                "Changing how many keywords are stored changes every figure "
                "here.\n"
                "5. **The morphological rung is a prefix heuristic, not a "
                "stemmer.** It over-fires on *policy*/*police* and under-fires "
                "on *good*/*better*. It never fabricates — the matched surface "
                "form is always shown.\n"
                "6. **Per-row retention is noisy** (ten keywords against one "
                "answer). Read the per-topic rollup.\n"
                "7. **Attribution inherits the cross-tab's imprecision**: a "
                "row's topics are the topics of *all* its retrieved chunks, "
                "including ones the answer legitimately ignored.\n"
                "8. **Every keyword counts equally.** c-TF-IDF already "
                "filtered for distinctiveness once; weighting again would "
                "double-count it."
            )

        lex_vectors, lex_cal = load_lexicon(lexicon_stamp())
        if lex_cal is None:
            st.warning(
                "No percentile calibration on disk, so the **semantic rung is "
                "disabled** — only verbatim and morphological matches can be "
                "scored. A raw cosine is deliberately not used as a fallback: "
                "shown as a percentage it would misrepresent the whole "
                "measure. Build the calibration locally (one time, a few "
                "hundred embedded words, then free forever):\n\n"
                "```bash\n"
                ".venv/bin/python scripts/13_run_topic_keywords.py calibrate\n"
                "```\n"
                "Then ship `data/lexicon/` alongside `data/topics/`.",
                icon="📐")
        else:
            cm = lex_cal.meta
            st.caption(
                f"Calibrated {cm.get('built_at', '?')} on "
                f"{cm.get('vocab_size', '?'):,} corpus words "
                f"({cm.get('n_pairs', 0):,} word pairs) with "
                f"`{cm.get('embedding_model', '?')}`. Median pair cosine "
                f"**{cm.get('cos_p50', '?')}**, 99th percentile "
                f"**{cm.get('cos_p99', '?')}** — that narrow band is what the "
                "percentile scale is stretching out."
            )

        if topic_run:
            if st.button(
                    "Compute topic vocabulary retention",
                    key="tk_run_btn",
                    help="Unlike the other buttons on this page, this one "
                         "COSTS: it embeds any answer word not already in "
                         "data/lexicon/ — a few hundred words the first time, "
                         "and ~0 on a rerun, because the cache is append-only. "
                         "Recomputing overwrites the previous result."):
                args = [PYTHON, "scripts/13_run_topic_keywords.py", "score",
                        "--run", topic_run]
                with st.status("Scoring topic vocabulary…", expanded=True) as status:
                    rc = stream_subprocess(args, st.empty())
                    status.update(
                        label=("Keyword retention complete ✅" if rc == 0
                               else f"Failed (exit {rc})"),
                        state="complete" if rc == 0 else "error")
                st.rerun()

        tks = load_topic_keyword_summary(topic_run)
        if tks is None:
            st.caption("Not computed for this run yet — use the button above.")
        elif not tks["overall"].get("n_scored"):
            st.warning(
                "No answered row retrieved evidence from a **clustered** "
                "topic. Every retrieval landed either in the outlier topic "
                "($-1$) or on chunk ids missing from the topic map — which is "
                "what a reingest after fitting does, since chunk ids are "
                "re-minted. Refit topics against the current index:\n\n"
                "```bash\n"
                ".venv/bin/python scripts/07_run_topics.py fit\n"
                "```", icon="🧭")
        else:
            o = tks["overall"]
            if not tks.get("calibrated"):
                st.info(
                    "This run was scored **without** the semantic rung, so "
                    "retention below is verbatim + morphological only. "
                    "Calibrate and recompute to see the semantic lift.",
                    icon="⚠️")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Vocabulary retention",
                      f"{o['mean_retention']:.0%}",
                      delta=(f"null floor {o['mean_retention_null']:.0%}"
                             if o.get("mean_retention_null") is not None
                             else None),
                      delta_color="off",
                      help="mean score of a topic's keywords against the "
                           "answers grounded in that topic")
            k2.metric("Verbatim only", f"{o['mean_verbatim_share']:.0%}",
                      help="what an exact-match judge would see on its own")
            k3.metric("Semantic lift", f"{o['mean_semantic_lift']:.0%}",
                      help="the part this matcher adds — inflections and "
                           "neighbours the exact judge is blind to. The "
                           "headline number for this section.")
            k4.metric("Verbatim ceiling",
                      (f"{o['mean_verbatim_ceiling']:.0%}"
                       if o.get("mean_verbatim_ceiling") is not None else "—"),
                      help="the same keywords against the chunks they were "
                           "DERIVED from — the circularity baseline, not an "
                           "achievement")

            # Floor / measurement / ceiling, so "is this better than chance?"
            # is answerable at a glance.
            arms = [{"arm": "null floor", "value": o.get("mean_retention_null")},
                    {"arm": "retention", "value": o.get("mean_retention")},
                    {"arm": "verbatim ceiling", "value": o.get("mean_verbatim_ceiling")}]
            arms = [a for a in arms if a["value"] is not None]
            if len(arms) > 1:
                st.altair_chart(
                    alt.Chart(pd.DataFrame(arms)).mark_bar().encode(
                        x=alt.X("value:Q", title=None,
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%")),
                        y=alt.Y("arm:N", title=None,
                                sort=["null floor", "retention",
                                      "verbatim ceiling"]),
                        color=alt.Color(
                            "arm:N", legend=None,
                            scale=alt.Scale(
                                domain=["null floor", "retention",
                                        "verbatim ceiling"],
                                range=[BUCKET_COLORS[2], BUCKET_COLORS[1],
                                       "#9E9E9E"])),
                        tooltip=["arm", alt.Tooltip("value:Q", format=".1%")],
                    ).properties(height=120),
                    width="stretch")
                st.caption(
                    "Read left to right. Retention sitting on the floor means "
                    "the answers carry no more of the topic's vocabulary than "
                    "a random topic's; sitting near the ceiling means they "
                    "carry nearly everything the evidence itself contained."
                )

            tk_rows = load_topic_keyword_rows(topic_run)
            tk_terms = load_topic_keyword_terms(topic_run)

            # Tier histogram over every scored keyword instance.
            if tk_terms is not None and not tk_terms.empty:
                tier_counts = {t: 0 for t in tk_mod.TIERS}
                for d in tk_terms["tiers"]:
                    for t, n in (d or {}).items():
                        tier_counts[t] = tier_counts.get(t, 0) + int(n)
                st.altair_chart(
                    alt.Chart(pd.DataFrame(
                        [{"tier": t, "n": n} for t, n in tier_counts.items()]
                    )).mark_bar().encode(
                        x=alt.X("n:Q", title="keyword × answer pairs"),
                        y=alt.Y("tier:N", title=None, sort=list(tk_mod.TIERS)),
                        tooltip=["tier", "n"],
                    ).properties(height=140),
                    width="stretch")

            # Per-topic composition.
            by_topic = [t for t in tks.get("by_topic", []) if t.get("n_rows")]
            if by_topic:
                st.markdown("**Where the vocabulary survives, and where it drops**")
                min_rows = st.slider(
                    "Minimum answered rows", 1,
                    max(2, max(t["n_rows"] for t in by_topic)), 1,
                    key="tk_min_rows",
                    help="topics reached by fewer answers than this are hidden "
                         "— a one-row topic average is not a measurement")
                shown = [t for t in by_topic if t["n_rows"] >= min_rows]
                if not shown:
                    st.warning("No topic meets that threshold.")
                else:
                    labels = [f"{t['topic']}: {str(t['label'])[:34]}" for t in shown]
                    heat = pd.DataFrame([
                        {"topic": f"{t['topic']}: {str(t['label'])[:34]}",
                         "tier": tier,
                         "share": t.get(col) or 0.0,
                         "rows": t["n_rows"]}
                        for t in shown
                        for tier, col in (("exact", "verbatim_share"),
                                          ("morphological", "morph_share"),
                                          ("semantic", "semantic_share"),
                                          ("absent", "dropped_share"))
                    ])
                    st.altair_chart(
                        alt.Chart(heat).mark_rect().encode(
                            x=alt.X("tier:N", sort=list(tk_mod.TIERS), title=None),
                            y=alt.Y("topic:N", title=None, sort=labels),
                            color=alt.Color("share:Q", title="share of keywords",
                                            scale=alt.Scale(scheme="blues")),
                            tooltip=["topic", "tier",
                                     alt.Tooltip("share:Q", format=".0%"),
                                     "rows"],
                        ).properties(height=min(26 * len(shown) + 40, 900)),
                        width="stretch")
                    st.caption(
                        "Read a row, not a column: each row sums to 100% of "
                        "that topic's keywords. A row weighted toward "
                        "`absent` is a topic whose distinctive vocabulary the "
                        "answers do not carry, in any form."
                    )

                dropped = sorted(shown, key=lambda t: -(t.get("dropped_share") or 0))
                worst = dropped[0] if dropped else None
                if worst and (worst.get("dropped_share") or 0) > 0.75:
                    st.warning(
                        f"Topic **{worst['topic']}** "
                        f"({str(worst['label'])[:60]}) loses "
                        f"{worst['dropped_share']:.0%} of its vocabulary: the "
                        "answers grounded in it do not carry those words "
                        "verbatim, as inflections, or as neighbours. Worth "
                        "reading a few of those answers directly.", icon="🕳️")
                elif worst:
                    st.success(
                        "No topic loses more than three quarters of its "
                        "vocabulary — every topic's distinctive words reach "
                        "the answers in some form.")

            # Per-keyword drill-down: the audit trail.
            if tk_terms is not None and not tk_terms.empty:
                with st.expander("Every keyword's verdict, and the word that "
                                 "produced it", expanded=False):
                    st.caption(
                        "The point of a graded matcher is that each score is "
                        "checkable. Pick a topic to see how each of its "
                        "keywords fared, then a keyword to see the exact "
                        "answer word that matched it."
                    )
                    topic_opts = sorted({int(t) for t in tk_terms["topic"]})
                    label_by_topic = {int(r["topic"]): str(r["label"])
                                      for _, r in tk_terms.iterrows()}
                    sel_topic = st.selectbox(
                        "Topic", topic_opts, key="tk_topic_pick",
                        format_func=lambda t: f"{t}: {label_by_topic.get(t, '')[:50]}")
                    tv = tk_terms[tk_terms["topic"] == sel_topic].copy()
                    tv["exact"] = [d.get("exact", 0) for d in tv["tiers"]]
                    tv["morph"] = [d.get("morphological", 0) for d in tv["tiers"]]
                    tv["semantic"] = [d.get("semantic", 0) for d in tv["tiers"]]
                    tv["absent"] = [d.get("absent", 0) for d in tv["tiers"]]
                    tv["top matches"] = [
                        ", ".join(f"{w} ({n})" for w, n in (m or []))
                        for m in tv["top_matches"]]
                    st.dataframe(
                        tv[["keyword", "retention", "exact", "morph",
                            "semantic", "absent", "top matches", "n_rows"]],
                        hide_index=True, width="stretch")

                    if tk_rows is not None and not tk_rows.empty:
                        kw_opts = list(tv["keyword"])
                        sel_kw = st.selectbox("Keyword", kw_opts, key="tk_kw_pick")
                        detail = []
                        for _, r in tk_rows.iterrows():
                            for k in (r.get("keywords") or []):
                                if k["topic"] == sel_topic and k["keyword"] == sel_kw:
                                    detail.append({
                                        "org": r["org"],
                                        "source_type": r["source_type"],
                                        "qid": r["qid"],
                                        "tier": k["tier"],
                                        "score": k["score"],
                                        "matched": _present(k["matched"]) or "—",
                                        "rule": _present(k["rule"]) or "—",
                                        # null for exact/morph matches; without
                                        # _present pandas renders literal "nan"
                                        "cosine": _present(k["cosine"]) or "—",
                                        "near miss": _present(
                                            k.get("near_miss")) or "—",
                                    })
                        if detail:
                            st.dataframe(pd.DataFrame(detail), hide_index=True,
                                         width="stretch")
                            st.caption(
                                f"Every row is one answer graded against "
                                f"`{sel_kw}`. Where `tier` is *semantic*, "
                                "`matched` is the answer word that earned the "
                                "score — read it before trusting the number: "
                                "these vectors capture relatedness, and an "
                                "antonym is closely related. Where `tier` is "
                                "*absent* the score is **0** — a keyword that "
                                "cleared no bar earns nothing — and `near "
                                "miss` says how close it came, which is what "
                                "you would retune the bar against."
                            )

            kdir = runs.run_dir(topic_run) / tk_mod.OUT_DIR_NAME
            d1, d2, d3 = st.columns(3)
            for col, name in ((d1, "summary.json"), (d2, "rows.jsonl"),
                              (d3, "terms.jsonl")):
                if (kdir / name).exists():
                    col.download_button(
                        name, (kdir / name).read_bytes(),
                        file_name=f"topic_keywords_{name.split('.')[0]}_"
                                  f"{topic_run}.{name.split('.')[-1]}",
                        key=f"tk_dl_{name}")

            st.caption(tks.get("note", ""))

        # --- Keyword neighborhood explorer --------------------------------
        # Depends only on the lexicon: no run, no cross-tab, no API key. This
        # is the interpretability proof — it is where the percentile scale
        # visibly earns its place against the raw cosine it is derived from.
        st.subheader("Keyword neighborhood explorer")
        if lex_cal is None or lex_vectors is None:
            st.info(
                "The explorer reads `data/lexicon/`, which is built by the "
                "`calibrate` command above. Once it exists, pick any topic "
                "keyword here and see its nearest corpus words with calibrated "
                "scores — no API calls, because every vector is already "
                "cached.", icon="🔎")
        else:
            st.caption(
                "Pick a keyword and see what the corpus considers close to it. "
                "This is the check on everything above: the **percentile** "
                "column should separate words that a **raw cosine** barely "
                "distinguishes."
            )
            all_kw = sorted({k for r in tinfo["topics"] if not r["is_outlier"]
                             for k in r.get("keywords", [])})
            c_pick, c_free = st.columns([2, 2])
            picked = c_pick.selectbox(
                "A topic keyword", all_kw,
                index=(all_kw.index("manager") if "manager" in all_kw else 0),
                key="tk_nb_pick") if all_kw else None
            typed = c_free.text_input("…or any corpus word", value="",
                                      key="tk_nb_free").strip().lower()
            query = typed or picked
            top_n = st.slider("Neighbours to show", 5, 50, 25, key="tk_nb_n")

            # Bigram keywords are not single vectors; score their head word.
            probe = (query or "").split()[-1] if query else ""
            nbrs = tk_mod.neighbors(probe, vectors=lex_vectors,
                                    calibration=lex_cal, top_n=top_n) if probe else []
            if not nbrs:
                st.info(
                    f"`{probe or '—'}` is not in the cached lexicon (the "
                    f"{lex_cal.meta.get('vocab_size', '?')} most frequent "
                    "corpus content words). The explorer never embeds on "
                    "demand — it is read-only over the cache.", icon="🔎")
            else:
                ndf = pd.DataFrame(nbrs)
                st.altair_chart(
                    alt.Chart(ndf).mark_bar().encode(
                        x=alt.X("percentile:Q", title="calibrated closeness",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%")),
                        y=alt.Y("word:N", title=None,
                                sort=list(ndf["word"])),
                        color=alt.Color("tier:N", title="how the ladder would "
                                                        "treat it",
                                        scale=alt.Scale(
                                            domain=list(tk_mod.TIERS))),
                        tooltip=["word", alt.Tooltip("cosine:Q", format=".4f"),
                                 alt.Tooltip("percentile:Q", format=".1%"),
                                 "tier", "df"],
                    ).properties(height=min(22 * len(ndf) + 40, 900)),
                    width="stretch")
                band = float(ndf["cosine"].max() - ndf["cosine"].min())
                st.caption(
                    f"These {len(ndf)} neighbours span a raw cosine range of "
                    f"just **{band:.4f}** (from {ndf['cosine'].min():.4f} to "
                    f"{ndf['cosine'].max():.4f}), while the percentile axis "
                    f"spreads them across "
                    f"{(ndf['percentile'].max() - ndf['percentile'].min()):.0%}. "
                    "That gap is the entire argument for calibrating: the raw "
                    "numbers are not a scale anyone could read."
                )
                st.dataframe(
                    ndf[["word", "cosine", "percentile", "tier", "df"]],
                    hide_index=True, width="stretch")
                st.caption(
                    "`exact` is the word itself, `morphological` an "
                    "inflection, `semantic` a different word the corpus places "
                    "nearby. A neighbour being close does **not** mean it "
                    "means the same thing — antonyms are close too."
                )



# ---------------------------------------------------------------------------
# Ad-hoc tab — drop in documents and run the questionnaire against them alone.
# Deliberately isolated from the corpus: nothing here is written to Chroma.
# ---------------------------------------------------------------------------
with tab_adhoc:
    import altair as alt

    st.header("Analyse a document")
    st.caption(
        "Drop in one or more documents and run the same 27-question "
        "questionnaire against **only those files**. Nothing is added to the "
        "vector index — the six research profiles cannot be affected by "
        "anything you upload here."
    )

    with st.expander("ℹ️ How this differs from a corpus run", expanded=False):
        st.markdown(
            "**Same measurement, different evidence.** From retrieval onward "
            "this is the production path — the identical questionnaire, the "
            "identical answering prompt, the identical graded matcher, the "
            "identical aggregation. Only the source of evidence changes, so a "
            "percentage here means what it means on the Results tab."
        )
        st.markdown(
            "**Retrieval is exhaustive, not indexed.** Uploaded text is "
            "chunked with the same splitter as ingest, embedded once, and held "
            "**in memory**. Each question then scores every chunk by cosine "
            "similarity and takes the top $k$:"
        )
        st.latex(r"\mathrm{score}(q,c)=\frac{v_q\cdot v_c}"
                 r"{\lVert v_q\rVert\,\lVert v_c\rVert},\qquad "
                 r"R(q)=\operatorname*{top-}k_c\ \mathrm{score}(q,c)")
        symbol_glossary([
            (r"$q,\;c$", "**a question** and **an uploaded chunk**"),
            (r"$v_q,\;v_c$", "their **embeddings** (1024 numbers each)"),
            (r"$k$", "how many chunks are given to the answering model"),
            (r"$R(q)$", "the **evidence set** for question $q$"),
        ])
        st.markdown(
            "**Design decisions**\n"
            "- *Why no Chroma.* The six profiles are the study's result, and a "
            "stray upload must never be able to contaminate the index they are "
            "computed from. Not writing at all is a stronger guarantee than "
            "writing carefully and cleaning up afterwards.\n"
            "- *Why exhaustive search is fine.* A handful of documents is a few "
            "hundred chunks; comparing against all of them is instant and "
            "needs no collection lifecycle.\n"
            "- *Why a subject name is required.* The questions literally ask "
            "what \"{org}\" does. Answering them about an unnamed entity would "
            "change what is being measured, so the name is an input, not a "
            "label.\n"
            "- *Scope.* A result lives in this browser session until you "
            "**save** it. Saving writes a real run snapshot — the same files a "
            "corpus run writes — tagged as a document analysis, so the Audit "
            "tab and the post-hoc judges read it unchanged. It stays out of "
            "the run pickers by default (a sidebar toggle opts in): an "
            "uploaded document is not part of the six-profile study, and the "
            "Results charts key on the three labs, so they cannot plot an "
            "arbitrary subject."
        )

    saved = runs.list_runs(kind=runs.KIND_ADHOC)
    if saved:
        with st.expander(f"📂 Saved analyses ({len(saved)})", expanded=False):
            st.caption(
                "Each one is a run snapshot on disk. Reloading shows it below "
                "exactly as it looked when saved — including the evidence text, "
                "which travels with the row because uploaded chunks are never "
                "indexed."
            )
            ids = [m["run_id"] for m in saved]
            names = {m["run_id"]: runs.display_name(m) for m in saved}
            pick = st.selectbox("Analysis", ids, key="adhoc_saved_pick",
                                format_func=lambda r: names.get(r, r))
            pm = next((m for m in saved if m["run_id"] == pick), {})
            docs_in = pm.get("documents") or []
            st.caption(
                f"Saved {pm.get('created_at', '?')} · k={pm.get('k', '?')} · "
                f"{pm.get('answered', 0)} answered / {pm.get('abstained', 0)} "
                "abstained"
                + (f" · files: {', '.join(docs_in)}" if docs_in else ""))
            if st.button("Load this analysis", key="adhoc_load"):
                loaded = adhoc_mod.load_run(pick)
                if loaded is None:
                    st.error("That snapshot could not be read.")
                else:
                    st.session_state["adhoc_result"] = loaded
                    st.session_state["adhoc_saved_as"] = pick
                    st.rerun()

    ad1, ad2 = st.columns([2, 1])
    subject = ad1.text_input(
        "Subject name — the organisation these documents are about",
        placeholder="e.g. OpenAI, or Acme Corp",
        help="Substituted into every question, so it must name the entity the "
             "documents describe.")
    adhoc_k = ad2.slider("Chunks per question (k)", 3, 10, TOP_K, key="adhoc_k")

    uploads = st.file_uploader(
        "Drag and drop documents here",
        type=["pdf", "txt", "md", "rtf"], accept_multiple_files=True,
        help="PDF, TXT, Markdown or RTF. Scanned PDFs need OCR first — there "
             "is no text layer to extract.")

    if uploads:
        docs = [adhoc_mod.extract_text(f.name, f.getvalue()) for f in uploads]
        ok = [d for d in docs if not d.error]
        bad = [d for d in docs if d.error]
        for d in bad:
            st.warning(f"**{d.filename}** — {d.error}", icon="⚠️")
        if ok:
            chunks_preview = adhoc_mod.build_chunks(ok, subject or "SUBJECT")
            n_q = runs.QUESTIONS_PER_ORG
            st.dataframe(
                pd.DataFrame([{"file": d.filename, "characters": d.n_chars,
                               "chunks": sum(1 for c in chunks_preview
                                             if c.filename == d.filename)}
                              for d in ok]),
                hide_index=True, width="stretch")
            st.caption(
                f"**{len(chunks_preview)} chunks** to embed, then {n_q} "
                f"questions × (1 answer + 1 matcher call) = **{2 * n_q} LLM "
                f"calls**, run sequentially — expect **3-5 minutes**. "
                f"Keep this tab open; closing it loses the run."
            )
            can_run = bool(subject.strip()) and api_key_present()
            if not subject.strip():
                st.info("Enter a subject name above to enable the run.",
                        icon="✏️")
            if st.button("Run the questionnaire on these documents",
                         type="primary", disabled=not can_run,
                         key="adhoc_run"):
                chunks = adhoc_mod.build_chunks(ok, subject.strip())
                bar = st.progress(0.0, text="Embedding uploaded text…")
                try:
                    vecs = adhoc_mod.embed_chunks(
                        chunks,
                        progress=lambda i, n: bar.progress(
                            i / n, text=f"Embedding {i}/{n} chunks…"))
                    result = adhoc_mod.analyze(
                        chunks, vecs, subject.strip(), k=adhoc_k,
                        progress=lambda i, n: bar.progress(
                            i / n, text=f"Question {i}/{n}…"))
                except Exception as e:  # noqa: BLE001 — surface, don't crash the tab
                    bar.empty()
                    st.error(f"Analysis failed: {e}")
                    result = None
                else:
                    bar.empty()
                    result["documents"] = [d.filename for d in ok]
                    result["k"] = adhoc_k
                    st.session_state["adhoc_result"] = result
                    st.session_state.pop("adhoc_saved_as", None)
                    st.success("Done.")

    res = st.session_state.get("adhoc_result")
    if res:
        prof = res["profile"]
        st.subheader(f"Profile — {res['subject']}")
        st.caption(f"{prof['answered']} answered · {prof['abstained']} "
                   "abstained (abstentions are excluded from the percentages)")
        if prof["answered"]:
            pdf_ = pd.DataFrame([{"logic": k, "pct": v}
                                 for k, v in prof["logic_pct"].items()])
            st.altair_chart(
                alt.Chart(pdf_).mark_bar().encode(
                    x=alt.X("logic:N", sort=LOGICS, title=None),
                    y=alt.Y("pct:Q", title="% of profile",
                            scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("logic:N", legend=None,
                                    scale=alt.Scale(
                                        domain=list(LOGIC_COLORS),
                                        range=list(LOGIC_COLORS.values()))),
                    tooltip=["logic", alt.Tooltip("pct:Q", format=".1f")],
                ).properties(height=280), width="stretch")
            sanity = max(prof["logic_pct"].get("Family", 0),
                         prof["logic_pct"].get("Religion", 0))
            if sanity > 15:
                st.warning(f"Family/Religion reach {sanity:.1f}% — with a small "
                           "document set this is easily noise, but worth "
                           "reading the answers below before trusting it.")
            st.caption(
                "One document set is a much smaller sample than a corpus "
                "profile, so treat the ranking as indicative and the exact "
                "percentages as soft."
            )

        with st.expander("Every question, its answer and its evidence",
                         expanded=False):
            for row in res["rows"]:
                top = ("ABSTAINED" if row["abstain"]
                       else max(row["weights"], key=row["weights"].get))
                with st.expander(f"{row['qid']} → {top}"):
                    st.markdown(f"**Q:** {row['question']}")
                    st.markdown(f"**Answer:**\n\n{row['answer']}")
                    if not row["abstain"]:
                        st.dataframe(pd.DataFrame(
                            [{"logic": k, "weight": round(v, 3)}
                             for k, v in row["weights"].items() if v > 0]
                        ).sort_values("weight", ascending=False),
                            hide_index=True)
                    st.markdown(f"**Matcher reasoning:** {row['reasoning']}")
                    st.markdown("**Evidence used:**")
                    for i, ev in enumerate(row["retrieved"], 1):
                        with st.expander(f"[{i}] {ev['filename']} "
                                         f"(similarity {ev['score']:.3f})"):
                            st.text(ev["text"])

        # --- Save this analysis as a run snapshot ---
        saved_as = st.session_state.get("adhoc_saved_as")
        if saved_as:
            st.success(
                f"Saved as **{saved_as}**. It is on disk under "
                "`data/profiles/runs/`, reloadable above, and the Audit tab "
                "and post-hoc judges can read it — enable **Show document "
                "analyses in run pickers** in the sidebar to select it there.",
                icon="💾")
        else:
            with st.container(border=True):
                st.markdown("**Save this analysis**")
                st.caption(
                    "Writes a run snapshot so the result survives closing this "
                    "tab. Tagged as a document analysis, so it stays out of the "
                    "study's run pickers unless you opt in from the sidebar."
                )
                sv1, sv2 = st.columns([2, 1])
                save_label = sv1.text_input(
                    "Label (optional)", key="adhoc_save_label",
                    placeholder="e.g. Acme 2026 annual report")
                sv2.markdown("&nbsp;", unsafe_allow_html=True)
                if sv2.button("💾 Save analysis", type="primary",
                              key="adhoc_save"):
                    try:
                        rid = adhoc_mod.save_run(
                            res, k=res.get("k", TOP_K),
                            label=save_label.strip() or None,
                            documents=res.get("documents"))
                    except Exception as e:  # noqa: BLE001 — surface, don't crash
                        st.error(f"Could not save: {e}")
                    else:
                        st.session_state["adhoc_saved_as"] = rid
                        st.rerun()

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "profile.json",
            json.dumps({"subject": res["subject"], "profile": prof},
                       ensure_ascii=False, indent=2),
            file_name=f"adhoc_profile_{res['subject']}.json")
        dl2.download_button(
            "per_question.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in res["rows"]),
            file_name=f"adhoc_rows_{res['subject']}.jsonl")
        if st.button("Clear result", key="adhoc_clear"):
            del st.session_state["adhoc_result"]
            st.session_state.pop("adhoc_saved_as", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Compare tab — diff two run snapshots (the point of saving runs)
# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Compare two runs")
    metas = visible_runs()
    if len(metas) < 2:
        st.info(
            "Need at least two saved runs to compare. After you change the "
            "questionnaire, run again on the **Run** tab with **Start a NEW run "
            "snapshot** checked — the previous run is preserved, and both will "
            "show up here."
        )
    else:
        ids = [m["run_id"] for m in metas]
        names = {m["run_id"]: runs.display_name(m) for m in metas}
        c1, c2 = st.columns(2)
        # Default A = second-newest (old), B = newest (new): the common case is
        # "did my latest questionnaire change anything vs the previous run".
        a_run = c1.selectbox("Baseline · A (old)", ids, index=min(1, len(ids) - 1),
                             format_func=lambda r: names[r], key="cmp_a")
        b_run = c2.selectbox("Compare · B (new)", ids, index=0,
                             format_func=lambda r: names[r], key="cmp_b")

        if a_run == b_run:
            st.warning("Pick two different runs.")
        else:
            prof_a, prof_b = load_profiles(a_run), load_profiles(b_run)
            q_a, q_b = load_questionnaire(a_run), load_questionnaire(b_run)
            dq_a, dq_b = load_per_question(a_run), load_per_question(b_run)

            sub_delta, sub_words, sub_perq = st.tabs(
                ["📈 Profile deltas", "✏️ Question wording", "🔬 Per-question diff"])

            # --- 1. Profile % deltas (B − A) ------------------------------
            with sub_delta:
                with st.expander("ℹ️ How the deltas are computed",
                                 expanded=False):
                    st.markdown(
                        "A plain arithmetic difference of the two runs' "
                        "profile percentages, per (lab, source, logic) — run B "
                        "minus run A:"
                    )
                    st.latex(r"\Delta_k = P^{B}_k - P^{A}_k")
                    symbol_glossary([
                        (r"$k$", "**a logic** — one of the seven"),
                        (r"$P^{A}_k$", "logic $k$'s percentage in run **A** "
                                       "(the baseline); the superscript is a "
                                       "label, not a power"),
                        (r"$P^{B}_k$", "the same logic's percentage in run **B**"),
                        (r"$\Delta_k$", "the **change** in percentage points — "
                                        "delta is the standard symbol for a "
                                        "difference"),
                        (r"$\lvert\Delta_k\rvert$", "its **magnitude**, "
                                                    "ignoring direction — what "
                                                    "the bars are sorted by"),
                    ])
                    st.markdown(
                        "Values are **percentage points**, not percentages of "
                        "a percentage: a bar reading $+8$ means that logic's "
                        "share rose from e.g. 30% to 38%. Since each run's "
                        "seven percentages sum to ~100, the seven deltas of a "
                        "profile sum to ~0 — weight moves *between* logics, so "
                        "a rise in one is necessarily a fall in others. Bars "
                        "are sorted by $|\\Delta_k|$ so the largest movements "
                        "appear first, and pairs answered in only one of the "
                        "two runs are skipped.\n\n"
                        "**How to read a delta**\n"
                        "- Deltas are **not** significance-tested. Compare "
                        "each against the bootstrap confidence interval on the "
                        "Results tab: a shift well inside the CI is "
                        "indistinguishable from question-sampling noise.\n"
                        "- Two runs of the *same* questionnaire still differ, "
                        "because decoding at temperature 0 is greedy but not "
                        "bit-reproducible on shared GPU infrastructure. "
                        "Attribute a delta to a questionnaire change only if "
                        "it exceeds that baseline wobble.\n"
                        "- The denominators can differ too: if a question "
                        "abstained in one run and committed in the other, "
                        "$|A|$ changes, which moves every percentage slightly "
                        "even for untouched questions."
                    )
                if not prof_a or not prof_b:
                    st.info("One of the runs has no aggregated profiles yet.")
                else:
                    rows = []
                    for org in ORGS:
                        for stype in SOURCE_TYPES:
                            pa = prof_a.get(org, {}).get(stype)
                            pb = prof_b.get(org, {}).get(stype)
                            if not pa or not pb:
                                continue
                            if not pa.get("answered") or not pb.get("answered"):
                                continue
                            for logic in LOGICS:
                                av = pa["logic_pct"].get(logic, 0.0)
                                bv = pb["logic_pct"].get(logic, 0.0)
                                rows.append({"lab": org, "source": stype,
                                             "logic": logic, "a_pct": av,
                                             "b_pct": bv, "delta": round(bv - av, 2)})
                    if not rows:
                        st.info("No (lab, source) pair was answered in BOTH runs.")
                    else:
                        ddf = pd.DataFrame(rows)
                        pairs = sorted({(r["lab"], r["source"]) for r in rows})
                        cc1, cc2 = st.columns(2)
                        lab_sel = cc1.selectbox(
                            "Lab", sorted({p[0] for p in pairs}), key="cmp_dlab")
                        src_opts = [s for (l, s) in pairs if l == lab_sel]  # noqa: E741
                        src_sel = cc2.selectbox("Source", src_opts, key="cmp_dsrc")
                        view = ddf[(ddf.lab == lab_sel) & (ddf.source == src_sel)].copy()
                        view = view.sort_values(
                            "delta", key=lambda s: s.abs(), ascending=False)

                        import altair as alt
                        chart = (
                            alt.Chart(view)
                            .mark_bar()
                            .encode(
                                x=alt.X("delta:Q", title="change in % (B − A)"),
                                y=alt.Y("logic:N", sort=LOGICS, title=None),
                                color=alt.condition(
                                    alt.datum.delta > 0,
                                    alt.value("#54A24B"), alt.value("#E45756")),
                                tooltip=[
                                    "logic",
                                    alt.Tooltip("a_pct:Q", title="A %", format=".1f"),
                                    alt.Tooltip("b_pct:Q", title="B %", format=".1f"),
                                    alt.Tooltip("delta:Q", title="Δ", format="+.1f"),
                                ],
                            )
                            .properties(height=240)
                        )
                        st.altair_chart(chart, width="stretch")

                        disp = view.rename(columns={
                            "a_pct": "A %", "b_pct": "B %", "delta": "Δ"})[
                            ["logic", "A %", "B %", "Δ"]]
                        st.dataframe(
                            disp.style
                            .format("{:.1f}", subset=["A %", "B %", "Δ"])
                            .background_gradient(cmap="RdYlGn", subset=["Δ"],
                                                 vmin=-30, vmax=30),
                            hide_index=True, width="stretch")

                        with st.expander("All labs × sources (full delta table)"):
                            full = ddf.rename(columns={
                                "a_pct": "A %", "b_pct": "B %", "delta": "Δ"})
                            st.dataframe(full, hide_index=True, width="stretch")

            # --- 2. Question-wording diff ---------------------------------
            with sub_words:
                if not q_a or not q_b:
                    st.info("A questionnaire snapshot is missing for one run "
                            "(older runs created before snapshots may lack it).")
                else:
                    qa, qb = q_a["questionnaire"], q_b["questionnaire"]
                    cats = q_b.get("categories") or list(qb.keys())
                    st.caption("Legend:  ~~strikethrough~~ = removed in B · "
                               "**bold** = added in B.")
                    show_refs = st.checkbox(
                        "Also show reference-answer changes", value=False,
                        key="cmp_show_refs")
                    changed = 0
                    for cat in cats:
                        a_block, b_block = qa.get(cat, {}), qb.get(cat, {})
                        a_qs = a_block.get("questions", [])
                        b_qs = b_block.get("questions", [])
                        for v in range(max(len(a_qs), len(b_qs))):
                            ta = a_qs[v] if v < len(a_qs) else ""
                            tb = b_qs[v] if v < len(b_qs) else ""
                            if ta.strip() != tb.strip():
                                changed += 1
                                with st.expander(f"✏️ {cat}#{v + 1}", expanded=False):
                                    st.markdown("**A (old):** " + (ta or "_(none)_"))
                                    st.markdown("**B (new):** " + (tb or "_(none)_"))
                                    st.markdown("**Diff:** " + word_diff_md(ta, tb))
                        if show_refs:
                            # Diff the RESOLVED reference per (question, logic):
                            # base + per-variant override, exactly as the matcher
                            # sees it. Diffing only the base block would miss
                            # reference_overrides, which now carry most of the
                            # per-question reference content. Identical changes
                            # across variants are grouped so a base-only edit
                            # shows once, not three times.
                            def _resolved(block: dict) -> dict:
                                base = block.get("reference_answers", {})
                                ov = block.get("reference_overrides", {})
                                out = {}
                                for v in (1, 2, 3):
                                    # JSON snapshots stringify override keys.
                                    o = ov.get(v) or ov.get(str(v)) or {}
                                    for logic in LOGICS:
                                        out[(v, logic)] = o.get(
                                            logic, base.get(logic, ""))
                                return out

                            a_res, b_res = _resolved(a_block), _resolved(b_block)
                            grouped: dict[tuple, list[int]] = {}
                            for v in (1, 2, 3):
                                for logic in LOGICS:
                                    ra_, rb_ = a_res[(v, logic)], b_res[(v, logic)]
                                    if ra_.strip() != rb_.strip():
                                        grouped.setdefault(
                                            (logic, ra_, rb_), []).append(v)
                            for (logic, ra_, rb_), vs in grouped.items():
                                changed += 1
                                qtag = ("Q1–3" if len(vs) == 3
                                        else "Q" + "/".join(str(v) for v in vs))
                                with st.expander(
                                        f"📐 {cat} {qtag} · reference[{logic}]"):
                                    st.markdown("**A:** " + (ra_ or "_(none)_"))
                                    st.markdown("**B:** " + (rb_ or "_(none)_"))
                                    st.markdown("**Diff:** "
                                                + word_diff_md(ra_, rb_))
                    if changed == 0:
                        st.success("No wording changes — both runs used identical "
                                   "questionnaires.")
                    else:
                        st.caption(f"{changed} item(s) differ between A and B.")

            # --- 3. Per-question answer / weight diff ---------------------
            with sub_perq:
                if dq_a is None or dq_b is None:
                    st.info("Per-question data missing for one of the runs.")
                else:
                    keycols = ["org", "source_type", "qid"]
                    a = dq_a.drop_duplicates(subset=keycols, keep="last").set_index(keycols)
                    b = dq_b.drop_duplicates(subset=keycols, keep="last").set_index(keycols)
                    common = sorted(set(a.index).intersection(set(b.index)))
                    if not common:
                        st.info("The two runs share no (lab, source, question) keys.")
                    else:
                        g1, g2, g3 = st.columns(3)
                        orgf = g1.multiselect("Lab", sorted({i[0] for i in common}),
                                              key="cmp_pq_lab")
                        srcf = g2.multiselect("Source", sorted({i[1] for i in common}),
                                              key="cmp_pq_src")
                        changed_only = g3.checkbox(
                            "Only where the verdict changed", value=True,
                            key="cmp_pq_changed")

                        def _top(w, abstain):
                            if abstain:
                                return "ABSTAIN"
                            return max(w, key=w.get) if w else "—"

                        shown = 0
                        for idx in common:
                            org, src, qid = idx
                            if orgf and org not in orgf:
                                continue
                            if srcf and src not in srcf:
                                continue
                            ra, rb = a.loc[idx], b.loc[idx]
                            wa = ra["weights"] if isinstance(ra["weights"], dict) else {}
                            wb = rb["weights"] if isinstance(rb["weights"], dict) else {}
                            ta, tb = _top(wa, ra["abstain"]), _top(wb, rb["abstain"])
                            verdict_changed = (ta != tb) or (ra["abstain"] != rb["abstain"])
                            q_changed = ra["question"] != rb["question"]
                            if changed_only and not verdict_changed:
                                continue
                            shown += 1
                            flag = "🔀" if verdict_changed else ("✏️" if q_changed else "•")
                            with st.expander(f"{flag} {org} · {src} · {qid}   "
                                             f"{ta} → {tb}"):
                                if q_changed:
                                    st.markdown("**Question changed:** "
                                                + word_diff_md(ra["question"], rb["question"]))
                                else:
                                    st.markdown(f"**Q:** {rb['question']}")
                                ca, cb = st.columns(2)
                                ca.markdown(f"**A · {ta}**")
                                ca.markdown(ra["answer"] or "_(no answer)_")
                                cb.markdown(f"**B · {tb}**")
                                cb.markdown(rb["answer"] or "_(no answer)_")
                                wdf = pd.DataFrame({
                                    "A %": {k: round(100 * v) for k, v in wa.items() if v > 0},
                                    "B %": {k: round(100 * v) for k, v in wb.items() if v > 0},
                                }).fillna(0).astype(int)
                                if not wdf.empty:
                                    st.dataframe(wdf.reindex(
                                        [l for l in LOGICS if l in wdf.index]),  # noqa: E741
                                        width="stretch")
                        st.caption(f"{shown} question(s) shown of {len(common)} "
                                   "shared between the runs.")
