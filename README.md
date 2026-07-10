# Institutional-Logics RAG Profiler

Profiles three AI labs — **OpenAI, DeepMind, Anthropic** — against Thornton &
Ocasio's seven institutional logics (State, Profession, Market, Corporation,
Family, Religion, Community), producing a percentage alignment profile per lab
and per source type. Built for the *structural transparency* argument: AI
alignment should be evaluated through a company-wide institutional lens, not a
purely technical one.

## Design A in one paragraph

Each lab is probed with a **fixed 27-question questionnaire** (9 elemental
categories × 3 phrasings, identical for every lab). Each question is answered by
a RAG chain over the lab's corpus — the answering model sees only retrieved
excerpts, never the logics taxonomy. Each free-form answer is then **answer-
matched** (after Chandak et al. 2025, generalized from binary to graded): a
matcher LLM compares the answer against the 7 logics' reference answers for that
question's category and distributes a weight of 1.0 across the logics. Profiles
are the mean weight per logic across answered questions, as percentages.
Questions with no usable corpus evidence **abstain** and are excluded from the
denominator — silence never shifts a profile.

Two source types are profiled **separately** per lab:

| source_type | corpus | what it captures |
|---|---|---|
| `published` | the lab's own documents (`* PDF's/pdf_corpus.txt`) | self-presentation |
| `thirdparty` | press articles by outsiders (`* Articles/*.RTF`) | external perception |

3 labs × 2 source types = **6 independent profiles**.

### The Family/Religion sanity check

All 7 logics are scored, including Family and Religion, which have no natural
place in an AI lab's institutional environment. They are expected to land **near
0%**. If they don't, the method is misfiring — this is a deliberate built-in
falsification check, not an oversight.

> **Note:** the questionnaire and reference answers in
> `il_rag/questionnaire.py` are **placeholders** that the researcher will
> rewrite. Only their structure (9 × 3 questions; 7 reference answers per
> category) is load-bearing.

## Setup

```bash
cd il_profiler
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your TOGETHER_API_KEY
```

## Run — GUI (recommended)

Double-click **`Launch IL Profiler.command`** in Finder (or run
`.venv/bin/streamlit run app.py`). The app opens in your browser with four tabs:

- **Run** — paste/save your API key, build the vector index, and run the
  questionnaire for any subset of labs/sources, with live logs. Stages run as
  the same resumable subprocesses as the CLI. Each run is saved as its own
  snapshot (optionally labelled), so a re-run never overwrites an earlier one.
- **Results** — pick any saved run, then view the six profiles as grouped bar
  charts (published vs thirdparty per lab), dominant-logic metrics, an automatic
  Family/Religion sanity-check banner, a per-category heatmap, and downloads.
- **Audit** — pick a run, then filter and read every question's RAG answer,
  graded weights, and matcher reasoning.
- **Compare runs** — diff two snapshots: per-logic profile deltas (B − A),
  a question-wording diff, and a per-question answer/weight diff. This is how you
  see what a rewritten questionnaire changed.

## Run — CLI

```bash
.venv/bin/python scripts/01_ingest.py --fresh     # build the vector index (once)
.venv/bin/python scripts/02_run_profiles.py       # run all 6 profiles (162 questions)

# iterate on a subset first (recommended before a full run):
.venv/bin/python scripts/02_run_profiles.py --orgs OpenAI --sources published

# start a fresh, labelled run snapshot (e.g. after rewriting the questionnaire):
.venv/bin/python scripts/02_run_profiles.py --fresh --label "questionnaire v2"
```

Both stages are **resumable**: rerunning skips completed work. `--fresh` starts
a **new run snapshot** rather than overwriting the previous one. All LLM calls
run at temperature 0.

## Outputs — run snapshots (`data/profiles/runs/<run_id>/`)

Every run is archived immutably under its own timestamped folder, so old and new
runs can be compared in the app's **Compare** tab. `data/profiles/CURRENT` names
the active run (used for resumption). Each snapshot contains:

- `company_profiles.json` — `lab -> source_type -> {logic_pct, answered, abstained, by_category}`
- `profiles_matrix.csv` — wide table: one row per (lab, source_type), one column per logic
- `per_question.jsonl` — full audit trail: every question's RAG answer, retrieved
  chunk ids, graded weights, and matcher reasoning
- `questionnaire.json` — the exact questionnaire (questions + reference answers)
  that produced this run, so wording changes can be diffed
- `meta.json` — label, timestamps, params, answered/abstained counts, status

Pre-snapshot results (flat files from before this layout) are migrated into a
`legacy` run automatically on first launch. A console report still prints each
profile as a ranked bar list.

## Layout

```
il_rag/
  config.py           paths, models, hyperparameters, study design
  questionnaire.py    27 questions + 63 reference answers (PLACEHOLDER data)
  llm.py              Together chat/embeddings with transient-error retry
  ingest.py           parse corpora -> chunk -> embed -> Chroma
  retriever.py        (org, source_type)-scoped semantic retrieval
  rag_qa.py           retrieve -> grounded free-form answer
  graded_matcher.py   answer -> weight distribution over the 7 logics
  profile_harness.py  orchestration, aggregation, outputs, report
  runs.py             run snapshots: archive/list/compare, legacy migration
scripts/
  01_ingest.py        stage 1: build the index
  02_run_profiles.py  stage 2: produce the profiles
app.py                Streamlit GUI (Run / Results / Audit / Compare runs)
Launch IL Profiler.command   double-clickable launcher (macOS)
```

## Cost notes

Ingestion embeds the full corpora (thousands of embedding calls). A full
profile run makes 162 generation calls + 162 matching calls + 162 query
embeddings. Stages are resumable, so an interrupted run loses nothing.
