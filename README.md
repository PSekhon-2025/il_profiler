# Institutional-Logics RAG Profiler

Profiles three AI labs — **OpenAI, DeepMind, Anthropic** — against Thornton &
Ocasio's seven institutional logics (State, Profession, Market, Corporation,
Family, Religion, Community), producing a percentage alignment profile per lab
and per source type. Built for the *structural transparency* argument: AI
alignment should be evaluated through a company-wide institutional lens, not a
purely technical one.

> **Full documentation:** see [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the
> whole system works — theory, pipeline, scoring, validation checks, confidence
> intervals, determinism, and the file map.

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

macOS / Linux:

```bash
cd il_profiler
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your TOGETHER_API_KEY
```

Windows (PowerShell or cmd):

```bat
cd il_profiler
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      REM then paste your TOGETHER_API_KEY
```

## Run — GUI (recommended)

macOS: double-click **`Launch IL Profiler.command`** in Finder (or run
`.venv/bin/streamlit run app.py`).
Windows: double-click **`Launch IL Profiler.bat`** in Explorer (or run
`.venv\Scripts\streamlit run app.py`).

The app opens in your browser with four tabs:

- **Run** — paste/save your API key, build the vector index, and run the
  questionnaire for any subset of labs/sources, with live logs. Stages run as
  the same resumable subprocesses as the CLI. Each run is saved as its own
  snapshot (optionally labelled), so a re-run never overwrites an earlier one.
- **Results** — pick any saved run, then view the six profiles as grouped bar
  charts (published vs thirdparty per lab), dominant-logic metrics, an automatic
  Family/Religion sanity-check banner, a per-category heatmap, and downloads.
- **Audit** — pick a run, then filter and read every question's RAG answer,
  graded weights, and matcher reasoning (plus supporting quotes and grounding
  bucket when those checks were enabled).
- **Hallucination** — the three opt-in checks for any saved run: alert banners
  when a detection fires, retrieval-grounding buckets with a score histogram,
  unverified-quote listings, and the metamorphic label-stability eval — which
  can be launched from this tab against any existing run, with flagged items
  shown variant-by-variant.
- **Compare runs** — diff two snapshots: per-logic profile deltas (B − A),
  a question-wording diff, and a per-question answer/weight diff. This is how you
  see what a rewritten questionnaire changed.

## Run — CLI

(On Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.)

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

## Hallucination & grounding checks (opt-in)

Three additive checks, all black-box / API-only (no logits, weights, or
attention — nothing the Together API doesn't expose). **All are off by
default; a default run produces byte-identical output to before.**

### 1. Retrieval-grounding pre-check — `--grounding`

```bash
.venv/bin/python scripts/02_run_profiles.py --grounding
```

Costs **zero extra API calls**: each row gains `retrieval_grounding_score`
(max lexical content-token recall of the question against its retrieved
chunks), `retrieval_cosine_top` (the retriever's best cosine score), and a
three-way `grounding_bucket`:

| bucket | meaning |
|---|---|
| `retrieval_missed` | grounding score below `GROUNDING_LOW_THRESHOLD` (config): retrieval likely never surfaced relevant text — the failure is retrieval's, whatever the model did next |
| `abstained` | retrieval looked plausible but the matcher abstained |
| `committed` | retrieval looked plausible and the answer was graded into logic weights |

The report adds a per-bucket breakdown (size, abstention rate, mean top-logic
weight). There are no gold labels in this pipeline, so buckets separate
*failure modes*, not accuracy.

### 2. Quote-grounded answers — `--quotes`

```bash
.venv/bin/python scripts/02_run_profiles.py --quotes
```

The answer model must return, alongside its answer, the verbatim excerpt spans
its conclusion rests on. Each quote is **verified in code** (whitespace-
normalized substring check against the actual chunks) and persisted on the row
as `quotes` (each with its own `verified` flag) and `quotes_verified`, so
grounding is auditable per question. The answering model still never sees the
logics taxonomy — quotes support the answer, never a logic choice. The
free-form path is untouched when the flag is off.

### 3. Metamorphic label-stability eval — `scripts/03_run_metamorphic_eval.py`

```bash
# after a profile run exists; start with a sample — the full run is ~1,800 calls
.venv/bin/python scripts/03_run_metamorphic_eval.py --sample 30
.venv/bin/python scripts/03_run_metamorphic_eval.py --run 2026-07-01_120000 --paraphrases 5
```

For each item of an existing run (after MetaQA, Yang et al. FSE 2025): the
exact retrieved chunks are refetched by id and perturbed into *k*
meaning-preserving **paraphrases** (LLM-generated) plus one **lab-name swap**
(deterministic regex — e.g. every "OpenAI" becomes "DeepMind" — the described
decisions stay intact). Each variant flows through the production
answer → match path, and its predicted label (argmax logic, or abstain) is
compared with the original run's:

- `label_stability` — fraction of paraphrase variants whose label matches the
  original. A grounded label should survive paraphrase; items below
  `METAMORPHIC_STABILITY_THRESHOLD` are flagged **unstable**.
- `swap_label_changed` — the swap leaves the decision text intact, so a
  grounded label should survive it too; a **flip** suggests the label was
  keyed on the model's prior about the named lab rather than on the text.

Outputs land inside the evaluated run's snapshot
(`data/profiles/runs/<run_id>/metamorphic/`): `variants.jsonl` (resumable
audit trail) and `stability.json` (per-item records + aggregate summary,
including stability by category and — if the run used `--grounding` — by
bucket). The console report prints stability alongside the run's profiles.

> **Self-referential caveat:** the same model (gpt-oss-120b) generates the
> paraphrases and classifies them, so "meaning-preserving" is only as good as
> the model's own judgment, and stability numbers inherit that circularity.
> Anchor them by human-auditing a small sample of `variants.jsonl` — check
> that paraphrases genuinely preserve the decision content, and that flagged
> flips aren't artifacts of a drifted paraphrase — before reading the
> aggregate numbers as evidence.

### Tests

```bash
.venv/bin/python -m pytest tests/   # offline: no API key, no index needed
```

The suite pins the invariant that default runs keep the exact original row
schema, and covers grounding scores/buckets, quote verification, lab-swap
substitution, and the stability math.

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
  json_utils.py       shared JSON extraction from LLM replies
  ingest.py           parse corpora -> chunk -> embed -> Chroma
  retriever.py        (org, source_type)-scoped semantic retrieval
  rag_qa.py           retrieve -> grounded free-form answer (opt-in: quotes)
  graded_matcher.py   answer -> weight distribution over the 7 logics
  grounding.py        opt-in: no-LLM retrieval-grounding score + buckets
  metamorphic.py      opt-in: paraphrase / lab-swap label-stability eval
  profile_harness.py  orchestration, aggregation, outputs, report
  runs.py             run snapshots: archive/list/compare, legacy migration
scripts/
  01_ingest.py                stage 1: build the index
  02_run_profiles.py          stage 2: produce the profiles
  03_run_metamorphic_eval.py  stage 3 (optional): label-stability eval
tests/                offline unit tests (pytest; all API calls stubbed)
app.py                Streamlit GUI (Run / Results / Audit / Compare runs)
Launch IL Profiler.command   double-clickable launcher (macOS)
Launch IL Profiler.bat       double-clickable launcher (Windows)
```

## Cost notes

Ingestion embeds the full corpora (thousands of embedding calls). A full
profile run makes 162 generation calls + 162 matching calls + 162 query
embeddings (`--grounding` adds nothing; `--quotes` changes the answer prompt
but not the call count). A full metamorphic eval at the default 3 paraphrases
is ~1,800 chat calls (paraphrase + answer + match per variant) — use
`--sample` first. Stages are resumable, so an interrupted run loses nothing.
