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
  Hallucination — the three opt-in checks for any saved run: retrieval-
                  grounding buckets, quote verification, and the metamorphic
                  label-stability eval (launchable from here), with alert
                  banners when a detection fires.
  Compare       — diff two run snapshots.
"""
import json
import os
import subprocess
import sys
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
    ORGS,
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
    st.title("🏛️ IL Profiler")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter") and pw == expected:
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
# Hallucination tab — the three opt-in checks, with alerts when one fires
# ---------------------------------------------------------------------------
with tab_halluc:
    st.header("Hallucination & grounding checks")
    st.caption(
        "Three black-box checks: **retrieval grounding** (was there relevant "
        "text to answer from?), **quote verification** (does the cited support "
        "actually appear in the sources?), and **metamorphic stability** (does "
        "the label survive paraphrase and a lab-name swap?)."
    )
    hal_run = run_selectbox("Run to inspect", key="halluc_run",
                            default_run_id=runs.get_current())
    dfh = load_per_question(hal_run)
    stab = load_stability(hal_run)

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
            if s.get("n_swap_label_changed"):
                alerts.append(("error",
                               f"🏷️ **{s['n_swap_label_changed']} lab-swap flip(s)**: "
                               f"the label changed when only the lab's NAME changed — "
                               f"the model may be keyed on its prior about the lab, "
                               f"not the text. See section 3."))
        if not (has_grounding or has_quotes or stab):
            st.info("None of the checks have run for this snapshot yet. Enable "
                    "**--grounding** / **--quotes** on the Run tab for the next "
                    "run, or launch the metamorphic eval below (works on any "
                    "existing run).")
        elif alerts:
            for kind, msg in alerts:
                getattr(st, kind)(msg)
        else:
            st.success("✅ No hallucination signals fired on the checks that ran "
                       "for this snapshot.")

        BUCKET_ORDER = ["committed", "abstained", "retrieval_missed"]
        BUCKET_COLORS = ["#54A24B", "#F58518", "#E45756"]

        # ---------------- 1 · Retrieval grounding ----------------
        st.subheader("1 · Retrieval grounding")
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

        # ---------------- 3 · Metamorphic label stability ----------------
        st.subheader("3 · Metamorphic label stability")
        with st.expander("Run the metamorphic eval for this snapshot",
                         expanded=stab is None):
            st.caption(
                "For each item: k LLM paraphrases of its retrieved chunks + one "
                "deterministic lab-name swap, each re-answered and re-graded "
                "through the production path. A grounded label survives both. "
                "Resumable; results land inside this run's folder."
            )
            c1, c2, c3 = st.columns(3)
            n_para = c1.number_input("Paraphrases per item", 1, 10, 3,
                                     key="mm_para")
            mm_sample = c2.number_input("Sample size (0 = all items)", 0, 500, 30,
                                        key="mm_sample")
            mm_seed = c3.number_input("Sample seed", 0, 9999, 0, key="mm_seed")
            n_items = len(dfh) if not mm_sample else min(int(mm_sample), len(dfh))
            st.caption(f"≈ {n_items} item(s) × ({int(n_para)} paraphrases + 1 swap) "
                       f"≈ {n_items * (3 * int(n_para) + 2)} chat calls.")
            if st.button("Run metamorphic eval", type="primary",
                         disabled=not api_key_present() or not hal_run,
                         key="mm_go"):
                args = [PYTHON, "scripts/03_run_metamorphic_eval.py",
                        "--run", hal_run,
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
            st.caption(f"Evaluated {s['items']} item(s) "
                       f"({s['paraphrases_per_item']} paraphrases + 1 swap each"
                       + (f", sample={s['sample']}" if s.get("sample") else "")
                       + ") — self-referential: the same model paraphrases and "
                         "classifies, so audit a few variants by hand.")
            m1, m2, m3, m4 = st.columns(4)
            ms = s.get("mean_label_stability")
            m1.metric("Mean label stability",
                      "—" if ms is None else f"{ms:.2f}",
                      help="fraction of paraphrase variants keeping the "
                           "original label, averaged over items")
            pf = s.get("pct_fully_stable")
            m2.metric("Fully stable items",
                      "—" if pf is None else f"{pf:.0f}%")
            m3.metric("🎲 Unstable items", s.get("n_unstable", 0),
                      delta=None if not s.get("n_unstable") else "detection",
                      delta_color="inverse")
            m4.metric("🏷️ Lab-swap flips",
                      f"{s.get('n_swap_label_changed', 0)}"
                      f"/{s.get('n_swap_evaluated', 0)}",
                      help="label changed although only the lab's name changed "
                           "— suggests prior-keyed, not text-grounded")

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

            flagged = items[(items.get("unstable") == True)  # noqa: E712
                            | (items.get("swap_label_changed") == True)]  # noqa: E712
            if flagged.empty:
                st.success("✅ Every evaluated item kept its label under all "
                           "paraphrases and the lab swap.")
            else:
                st.markdown(f"**⚠️ {len(flagged)} flagged item(s)** — the "
                            "detection firing, item by item:")
                vdf = load_variants(hal_run)
                for _, it in flagged.iterrows():
                    badges = []
                    if it.get("unstable"):
                        badges.append("🎲 unstable")
                    if it.get("swap_label_changed"):
                        badges.append("🏷️ swap flip")
                    ls = it.get("label_stability")
                    title = (f"{' + '.join(badges)} · {it['org']} · "
                             f"{it['source_type']} · {it['qid']} — original "
                             f"label: {it['original_label']}"
                             + (f", stability {ls:.2f}" if ls is not None else ""))
                    with st.expander(title):
                        if it.get("swap_label_changed"):
                            st.markdown(
                                f"**Lab swap:** text renamed to "
                                f"**{it.get('swap_to', '?')}** → label flipped "
                                f"**{it['original_label']} → "
                                f"{it.get('swap_label', '?')}**")
                        if vdf is not None:
                            sub = vdf[(vdf["org"] == it["org"])
                                      & (vdf["source_type"] == it["source_type"])
                                      & (vdf["qid"] == it["qid"])]
                            if not sub.empty:
                                disp = sub[["variant_kind", "variant_idx",
                                            "label", "label_matches_original"]
                                           ].copy() if "label" in sub.columns else None
                                if disp is not None:
                                    disp["label_matches_original"] = disp[
                                        "label_matches_original"].map(
                                        {True: "✅ kept", False: "❌ flipped"})
                                    st.dataframe(
                                        disp.rename(columns={
                                            "variant_kind": "variant",
                                            "variant_idx": "#",
                                            "label_matches_original": "vs original",
                                        }), hide_index=True, width="stretch")
                                flips = sub[(sub.get("label_matches_original")
                                             == False)]  # noqa: E712
                                for _, v in flips.iterrows():
                                    st.markdown(
                                        f"**{v['variant_kind']} #"
                                        f"{v['variant_idx']} → "
                                        f"{v.get('label', '?')}** — variant "
                                        f"answer:")
                                    st.caption(v.get("answer") or "(no answer)")

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
                            a_ref = a_block.get("reference_answers", {})
                            b_ref = b_block.get("reference_answers", {})
                            for logic in LOGICS:
                                ra_, rb_ = a_ref.get(logic, ""), b_ref.get(logic, "")
                                if ra_.strip() != rb_.strip():
                                    changed += 1
                                    with st.expander(f"📐 {cat} · reference[{logic}]"):
                                        st.markdown("**A:** " + (ra_ or "_(none)_"))
                                        st.markdown("**B:** " + (rb_ or "_(none)_"))
                                        st.markdown("**Diff:** " + word_diff_md(ra_, rb_))
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
