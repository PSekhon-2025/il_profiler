"""Streamlit GUI for the Institutional-Logics RAG profiler.

Run with:  .venv/bin/streamlit run app.py
(or double-click "Launch IL Profiler.command")

Three areas:
  Run      — configure the API key, build the vector index, run profiles.
             Pipeline stages execute as subprocesses with live log streaming,
             so the resumable behavior of the CLI scripts is preserved and a
             closed browser tab never corrupts a run.
  Results  — the six alignment profiles as charts, the published-vs-thirdparty
             comparison per lab, the Family/Religion sanity check, downloads.
  Audit    — browse every question's RAG answer, graded weights, and matcher
             reasoning from per_question.jsonl.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from il_rag.config import CHROMA_DIR, COLLECTION_NAME, ORGS, PROFILES_DIR, SOURCE_TYPES
from il_rag.questionnaire import CATEGORIES, LOGICS

PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
ENV_PATH = PROJECT_ROOT / ".env"
PER_QUESTION = PROFILES_DIR / "per_question.jsonl"
PROFILES_JSON = PROFILES_DIR / "company_profiles.json"
PROFILES_CSV = PROFILES_DIR / "profiles_matrix.csv"

# Colors keep the same logic recognizable across every chart. Family/Religion
# are grey on purpose — they're the sanity-check logics expected near 0%.
LOGIC_COLORS = {
    "State": "#4C78A8", "Profession": "#54A24B", "Market": "#E45756",
    "Corporation": "#F58518", "Family": "#B0B0B0", "Religion": "#888888",
    "Community": "#72B7B2",
}

st.set_page_config(page_title="IL Profiler", page_icon="🏛️", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def api_key_present() -> bool:
    if not ENV_PATH.exists():
        return False
    for line in ENV_PATH.read_text().splitlines():
        if line.strip().startswith("TOGETHER_API_KEY="):
            val = line.split("=", 1)[1].strip()
            return bool(val) and val != "your_together_api_key_here"
    return False


def save_api_key(key: str) -> None:
    ENV_PATH.write_text(f"TOGETHER_API_KEY={key.strip()}\n")


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


def load_profiles() -> dict | None:
    if not PROFILES_JSON.exists():
        return None
    return json.loads(PROFILES_JSON.read_text(encoding="utf-8"))


def load_per_question() -> pd.DataFrame | None:
    if not PER_QUESTION.exists():
        return None
    rows = []
    with open(PER_QUESTION, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else None


def stream_subprocess(args: list[str], log_box) -> int:
    """Run a pipeline stage as a subprocess, streaming output into the UI.

    Subprocesses (rather than in-process calls) preserve the scripts' resumable
    semantics and keep a mid-run browser refresh from corrupting state — the
    worst case is the UI loses the log while the run completes on its own.
    """
    proc = subprocess.Popen(
        args, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: list[str] = []
    for raw in proc.stdout:  # tqdm progress arrives as \r-updates on one line
        part = raw.rstrip("\n").split("\r")[-1]
        if not part.strip():
            continue
        if lines and (part.startswith(("ingest", "profile")) and
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

    dfq = load_per_question()
    total_q = len(ORGS) * len(SOURCE_TYPES) * 27
    done_q = len(dfq) if dfq is not None else 0
    st.markdown(
        ("✅" if done_q >= total_q else "⏳" if done_q else "❌")
        + f" Profiles ({done_q}/{total_q} questions)"
    )

tab_run, tab_results, tab_audit = st.tabs(["▶️ Run", "📊 Results", "🔍 Audit"])

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
        st.caption(
            "Runs the fixed 27-question questionnaire per selected (lab, source) "
            "pair: RAG answer + graded matching per question. Resumable — "
            "completed questions are skipped on rerun."
        )
        c1, c2 = st.columns(2)
        sel_orgs = c1.multiselect("Labs", ORGS, default=ORGS)
        sel_sources = c2.multiselect("Source types", SOURCE_TYPES, default=SOURCE_TYPES)
        fresh_prof = st.checkbox("Discard previous results and start over (--fresh)",
                                 value=False, key="fresh_prof")
        n_pairs = len(sel_orgs) * len(sel_sources)
        st.caption(f"Selected: {n_pairs} profile(s) × 27 questions = "
                   f"{n_pairs * 27} RAG + {n_pairs * 27} matcher calls.")
        if st.button("Run profiles", type="primary",
                     disabled=not api_key_present() or not sel_orgs or not sel_sources):
            args = [PYTHON, "scripts/02_run_profiles.py",
                    "--orgs", *sel_orgs, "--sources", *sel_sources]
            if fresh_prof:
                args.append("--fresh")
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
    profiles = load_profiles()
    if not profiles:
        st.info("No results yet — run the pipeline on the **Run** tab first.")
    else:
        st.header("Alignment profiles")

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
                chart = (
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
                    .properties(height=260)
                )
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
            d1, d2, d3 = st.columns(3)
            d1.download_button("company_profiles.json",
                               PROFILES_JSON.read_bytes(),
                               file_name="company_profiles.json")
            if PROFILES_CSV.exists():
                d2.download_button("profiles_matrix.csv",
                                   PROFILES_CSV.read_bytes(),
                                   file_name="profiles_matrix.csv")
            if PER_QUESTION.exists():
                d3.download_button("per_question.jsonl",
                                   PER_QUESTION.read_bytes(),
                                   file_name="per_question.jsonl")

# ---------------------------------------------------------------------------
# Audit tab
# ---------------------------------------------------------------------------
with tab_audit:
    dfq = load_per_question()
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
                st.caption("retrieved: " + ", ".join(row["retrieved_ids"][:5]))
