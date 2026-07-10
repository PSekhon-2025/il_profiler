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

from il_rag import runs
from il_rag.config import CHROMA_DIR, COLLECTION_NAME, ORGS, SOURCE_TYPES
from il_rag.questionnaire import CATEGORIES, LOGICS

PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
ENV_PATH = PROJECT_ROOT / ".env"

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

tab_run, tab_results, tab_audit, tab_compare = st.tabs(
    ["▶️ Run", "📊 Results", "🔍 Audit", "🆚 Compare runs"])

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
                st.caption("retrieved: " + ", ".join(row["retrieved_ids"][:5]))

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
