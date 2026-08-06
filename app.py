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
  Hallucination — the four opt-in checks for any saved run: retrieval-
                  grounding buckets, quote verification, and the metamorphic
                  label-stability eval (launchable from here), with alert
                  banners when a detection fires.
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

from il_rag import runs
from il_rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    GROUNDING_LOW_THRESHOLD,
    METAMORPHIC_CONTROLS,
    METAMORPHIC_PARAPHRASES,
    METAMORPHIC_PARAPHRASE_TEMPERATURE,
    METAMORPHIC_PROBES,
    METAMORPHIC_STABILITY_THRESHOLD,
    ORGS,
    PARAPHRASE_MAX_TOKEN_OVERLAP,
    PARAPHRASE_MIN_COSINE,
    SOURCE_TYPES,
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
                "embedding_agreement/similarities.jsonl"):
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
        "See ARCHITECTURE.md sections 9.1-9.4 for the full method of each check.\n"
    )


def zip_members(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def run_selectbox(label: str, key: str, default_run_id: str | None = None) -> str | None:
    """Dropdown of all runs (newest first); returns the chosen run_id."""
    metas = runs.list_runs()
    if not metas:
        return None
    ids = [m["run_id"] for m in metas]
    names = {m["run_id"]: runs.display_name(m) for m in metas}
    idx = ids.index(default_run_id) if default_run_id in ids else 0
    return st.selectbox(label, ids, index=idx,
                        format_func=lambda r: names.get(r, r), key=key)


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
        st.caption(f"Active run: **{runs.display_name(cur_meta)}**  \n"
                   f"{len(all_metas)} run(s) saved")

tab_run, tab_results, tab_audit, tab_halluc, tab_compare = st.tabs(
    ["▶️ Run", "📊 Results", "🔍 Audit", "🚨 Hallucination", "🆚 Compare runs"])

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
                ci_csv = runs.run_dir(res_run) / "bootstrap_ci" / "ci.csv"
                if ci_csv.exists():
                    st.download_button("ci.csv", ci_csv.read_bytes(),
                                       file_name=f"bootstrap_ci_{res_run}.csv")

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
            import altair as alt
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
        f1, f2, f3, f4 = st.columns(4)
        orgs_f = f1.multiselect("Lab", sorted(dfq["org"].unique()))
        st_f = f2.multiselect("Source", sorted(dfq["source_type"].unique()))
        cat_f = f3.multiselect("Category", [c for c in CATEGORIES
                                            if c in set(dfq["category"])])
        only_abstain = f4.checkbox("Abstentions only")

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
                    for q in row["quotes"]:
                        mark = "✅" if q.get("verified") else "❌"
                        st.markdown(f"> {mark} [excerpt {q.get('excerpt', '?')}] "
                                    f"“{q.get('quote', '')}”")
                gb = row.get("grounding_bucket")
                if isinstance(gb, str):
                    st.caption(f"grounding: {gb} · score "
                               f"{row.get('retrieval_grounding_score', 0):.2f} · "
                               f"cosine {row.get('retrieval_cosine_top', 0):.2f}")
                st.caption("retrieved: " + ", ".join(row["retrieved_ids"][:5]))

# ---------------------------------------------------------------------------
# Hallucination tab — the four opt-in checks, with alerts when one fires
# ---------------------------------------------------------------------------
with tab_halluc:
    st.header("Hallucination & grounding checks")
    st.caption(
        "Four black-box checks: **retrieval grounding** (was there relevant "
        "text to answer from?), **quote verification** (does the cited support "
        "actually appear in the sources?), **metamorphic stability & evidence "
        "sensitivity** (does the label survive a reword — and does it stop when "
        "the supporting evidence is taken away?), and **embedding agreement** "
        "(does a non-LLM judge rank the same reference nearest?)."
    )
    hal_run = run_selectbox("Run to inspect", key="halluc_run",
                            default_run_id=runs.get_current())
    dfh = load_per_question(hal_run)
    stab = load_stability(hal_run)
    emb = load_embedding_summary(hal_run)

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
                           f"than the matcher's top pick. See section 4."))
        if not (has_grounding or has_quotes or stab or emb):
            st.info("None of the checks have run for this snapshot yet. Enable "
                    "**--grounding** / **--quotes** on the Run tab for the next "
                    "run, or launch the metamorphic eval / embedding agreement "
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
                    "Enable **--grounding** / **--quotes** on the Run tab, or "
                    "launch the metamorphic eval / embedding agreement below, "
                    "then come back here.")
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
                for _, r in fab_rows.iterrows():
                    with st.expander(f"❌ {r['org']} · {r['source_type']} · "
                                     f"{r['qid']}"):
                        st.markdown(f"**Q:** {r['question']}")
                        st.markdown(f"**Answer:** {r['answer']}")
                        for q in r["quotes"]:
                            mark = "✅" if q.get("verified") else "❌"
                            st.markdown(f"> {mark} [excerpt {q.get('excerpt', '?')}]"
                                        f" “{q.get('quote', '')}”")
            else:
                st.caption("Every quoted span was found verbatim in its retrieved "
                           "sources.")

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

        # ---------------- 4 · Embedding agreement ----------------
        st.subheader("4 · Embedding agreement (second judge)")
        st.caption(
            "Embeds every committed answer and the run's own seven reference "
            "answers per category, ranks the references by cosine similarity, "
            "and checks whether the nearest reference's logic agrees with the "
            "LLM matcher's top logic. Deterministic and LLM-free. Absolute "
            "cosine values are NOT interpretable (e5 compresses them into a "
            "narrow band) — only the ranking and the top1–top2 margin are."
        )
        if st.button("Compute embedding agreement",
                     disabled=not api_key_present(),
                     help="One embedding per committed row + 63 references, "
                          "batched — fractions of a cent. Recomputing "
                          "overwrites the previous result for this run."):
            args = [PYTHON, "scripts/04_run_embedding_agreement.py",
                    "--run", hal_run]
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

            cat_rows = [{"category": c, "rate": v["rate"], "n": v["n"]}
                        for c, v in emb.get("by_category", {}).items()
                        if v.get("rate") is not None]
            if cat_rows:
                edf = pd.DataFrame(cat_rows)
                emb_chart = (
                    alt.Chart(edf)
                    .mark_bar()
                    .encode(
                        x=alt.X("rate:Q", title="agreement rate",
                                scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("category:N", title=None,
                                sort=[c for c in CATEGORIES]),
                        color=alt.Color("rate:Q", legend=None,
                                        scale=alt.Scale(scheme="redyellowgreen",
                                                        domain=[0, 1])),
                        tooltip=["category",
                                 alt.Tooltip("rate:Q", format=".2f"), "n"],
                    )
                    .properties(height=220)
                )
                st.altair_chart(emb_chart, width="stretch")

            erows = load_embedding_rows(hal_run)
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

            edir = runs.run_dir(hal_run) / "embedding_agreement"
            dle1, dle2 = st.columns(2)
            if (edir / "summary.json").exists():
                dle1.download_button("summary.json",
                                     (edir / "summary.json").read_bytes(),
                                     file_name=f"embedding_summary_{hal_run}.json")
            if (edir / "similarities.jsonl").exists():
                dle2.download_button("similarities.jsonl",
                                     (edir / "similarities.jsonl").read_bytes(),
                                     file_name=f"embedding_rows_{hal_run}.jsonl")

# ---------------------------------------------------------------------------
# Compare tab — diff two run snapshots (the point of saving runs)
# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Compare two runs")
    metas = runs.list_runs()
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
