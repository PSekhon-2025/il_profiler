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

The app opens in your browser with five tabs:

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
- **Hallucination** — the five opt-in checks for any saved run: alert banners
  when a detection fires, retrieval-grounding buckets with a score histogram,
  unverified-quote listings, quote provenance (why each failed quote failed,
  and whether its content holds up anyway — launchable from this tab), the
  metamorphic eval (launchable from this tab, flagged items shown
  variant-by-variant), and the embedding-agreement check (binary + graded
  closeness metrics, launchable from this tab).
- **Compare runs** — diff two snapshots: per-logic profile deltas (B − A),
  a question-wording and reference-answer diff (including per-question
  overrides), and a per-question answer/weight diff. This is how you see what
  a rewritten questionnaire changed.

### Source links in the audit trail (optional, local only)

By default the Audit tab shows a supporting quote as `[excerpt 3] "…"`, where
the number is a dead end. Point the app at the raw dataset and the number
becomes a link that opens the PDF the answer cited:

```bash
# in .env — an absolute path to the dataset tree (<root>/<Org>/<Category>/*.pdf)
IL_PROFILER_DATASET_ROOT=C:/path/to/dataset
```

The chunk metadata records only a bare filename, so the app builds a
filename → path index over that tree and publishes each cited PDF into
`static/`, which Streamlit serves at `/app/static/...` (enabled in
`.streamlit/config.toml`). Only documents actually cited are ever published,
and the link opens in a new tab.

Two things to know:

- **Verification is not attribution.** A quote verifies against *any* retrieved
  excerpt, so a ✅ next to a link does not mean the span is in *that* PDF. When
  the span is really in a different document, the row says so and links there
  too.
- **Scope.** Published (PDF) evidence only. Third-party chunks come from RTF
  press dumps where one file bundles many articles, so a file-level link would
  point at the bundle rather than the article; those render unchanged.

Unset the variable — or run with `IL_PROFILER_CLOUD=1` — and every citation
falls back to the plain text above. The cloud deployment ships no corpus, so
the links never appear there.

**Page anchors** (`#page=N`) are an extra step, mirroring the topic layer: a
heavy dependency and the raw PDFs stay local, and only a small JSON is read at
runtime.

```bash
.venv/Scripts/python -m pip install -r requirements-pdf.txt
.venv/Scripts/python scripts/09_build_pdf_pages.py --dry-run   # report hit rates
.venv/Scripts/python scripts/09_build_pdf_pages.py             # write the map
```

This writes `data/pdf_pages.json`. On the current corpus it places 89% of
chunks; the rest open at page 1. The gap is honest rather than incidental — 21
documents are recorded in the dataset's own manifests as `method: ocr`, i.e.
web pages saved as images with no text layer to align against, and the builder
declines to guess a page for those. Chrome, Edge and Firefox honour `#page=`;
Safari ignores it and opens at page 1.

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

Five additive checks, all black-box / API-only (no logits, weights, or
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

**How the score is computed** (full derivation in ARCHITECTURE.md §9.1; the
GUI's Hallucination tab has the same math in an expander):

```
T(x)          = content tokens of x: lowercased alphanumeric tokens,
                minus stopwords, minus tokens ≤ 2 chars
overlap(q, c) = |T(q) ∩ T(c)| / |T(q)|          per-chunk, in [0, 1]
g(q)          = max over retrieved chunks c of overlap(q, c)

bucket(q)     = retrieval_missed   if g(q) < τ
                abstained          if g(q) ≥ τ and the matcher abstained
                committed          otherwise
```

`τ = GROUNDING_LOW_THRESHOLD` (currently **0.2**, set in `il_rag/config.py` —
a per-corpus heuristic, not a learned parameter). The max (not mean) over
chunks means one genuinely relevant chunk is enough to ground an answer.

*Worked example:* for the question "How does the lab describe its safety
mission?", stopword/length filtering leaves the content tokens
`{lab, describe, safety, mission}` (4 tokens; "how", "does", "the", "its" are
stopwords). If the best retrieved chunk contains `lab`, `safety`, and
`mission` but not `describe`, then `overlap = 3/4 = 0.75 ≥ 0.2` → the row is
`committed` (or `abstained` if the matcher abstained). A chunk containing
none of the four tokens would score `0/4 = 0 < 0.2` → `retrieval_missed`.

The report adds a per-bucket breakdown (size, abstention rate, mean top-logic
weight). There are no gold labels in this pipeline, so buckets separate
*failure modes*, not accuracy.

Literature basis: the score is a set-based variant of ROUGE-1 recall
([Lin, 2004](https://aclanthology.org/W04-1013/)); its use as a groundedness
signal follows Knowledge F1
([Shuster et al., 2021](https://aclanthology.org/2021.findings-emnlp.320/));
the bucketing mirrors the "Missing Content" failure point of
[Barnett et al., 2024](https://arxiv.org/abs/2401.05856). Full reference list
in ARCHITECTURE.md §9.1.

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

**How verification works** (full derivation in ARCHITECTURE.md §9.2; the
GUI's Hallucination tab has the same logic in an expander):

```
norm(s)         = lowercase(collapse_ws(s))       whitespace-tolerant, case-
                                                  insensitive, punctuation verbatim
verified(q)     = norm(q) ≠ "" ∧ ∃ chunk c : norm(q) ⊑ norm(c)   (⊑ = substring)
quotes_verified = |Q| > 0 ∧ ∀ q ∈ Q : verified(q)
```

The cited excerpt *number* is displayed but not used for matching — the
auditable claim is "this text is in the sources," not the model's index
bookkeeping. The `|Q| > 0` guard makes an empty quote list unverified by
definition (a conjunction over an empty set would be vacuously true). There
is no tunable threshold: a span either occurs verbatim after normalization
or it doesn't.

*Worked example:* if a chunk contains `"The  charter\ncommits to Broadly
distributed benefits"`, the quoted span `"charter commits to broadly
distributed benefits"` **verifies** (double space, newline, and capital B
are absorbed by normalization), while `"the charter guarantees benefits"`
**fails** — a paraphrase, not a verbatim span. (This is exactly the case
pinned by `tests/test_rag_qa.py::test_quotes_verified_verbatim`.)

The report distinguishes two failure shapes — abstaining is not fabrication:

| report bucket | meaning |
|---|---|
| ❌ unverified quotes ("fabricated") | the model **did** cite quotes and at least one span is not in the sources — possible fabricated support |
| ∅ no quotes returned | empty quote list — typically an honest abstention, or a JSON parse fallback (retried once at doubled tokens, then degraded with `quotes_verified = False`) |

Literature basis: verbatim-quote support with mechanical verification
follows GopherCite ([Menick et al., 2022](https://arxiv.org/abs/2203.11147));
the audited property is Attribution to Identified Sources
([Rashkin et al., 2023](https://aclanthology.org/2023.cl-4.2/);
[Bohnet et al., 2022](https://arxiv.org/abs/2212.08037)); citation-quality
evaluation follows ALCE
([Gao et al., 2023](https://aclanthology.org/2023.emnlp-main.398/)). Full
reference list in ARCHITECTURE.md §9.2.

### 2b. Quote provenance & paraphrase grounding — `scripts/06_run_quote_provenance.py`

```bash
# works on any saved run, including ones answered WITHOUT --quotes
.venv/bin/python scripts/06_run_quote_provenance.py                 # CURRENT run
.venv/bin/python scripts/06_run_quote_provenance.py --run 2026-07-01_120000
```

Check 2 answers one question with one bit, so a single ❌ conflates four
different failures: a **copy that drifted** (curly quote, em-dash, an elided
`…`), a **faithful paraphrase**, a **figure of speech** (scare quotes, terms of
art, hypotheticals — quotation marks that never claimed anything about a
source), and an actual **fabrication**. Only the last is a hallucination.

This stage grades them, and adds the axis check 2 lacks entirely:

```
PROVENANCE   does this TEXT exist in the sources?     (no LLM)
VERACITY     is what it ASSERTS supported by them?    (one LLM call, flagged spans only)
```

They are independent, so the verdict is a 2×2:

| | content supported | content not supported |
|---|---|---|
| **text in sources** | `attributed` | `misattributed` |
| **text not in sources** | `paraphrase_grounded` · `misquote_but_true` | `fabricated` |

`misquote_but_true` is the interesting cell: the model manufactured a
quotation, but the proposition underneath it *does* hold up against the
evidence — a citation-integrity failure, not a factual one. Check 2 cannot tell
that apart from outright invention.

Spans are also triaged for **intent** before being graded, by deterministic cue
rules (`the charter states …` → attributive; `so-called …` → scare quote;
`a critic might say …` → hypothetical). Only attributive spans are graded, so
figures of speech stop being counted as fabrications. Candidates come from the
model's structured `quotes` *and* from quotation marks in the answer prose —
which is why this works on runs that never used `--quotes`, and where the
figures of speech actually live.

Post-hoc over a saved run (it replays each row's evidence via `retrieved_ids`),
so it needs no re-answering, **never rewrites check 2's `quotes_verified`**, and
can be re-run freely after retuning thresholds. Cheapest-first and
short-circuiting: a run whose quotes are all verbatim costs **zero** generation
and zero embedding calls.

> **`unsupported` ≠ false.** The evidence is scoped by (lab, source type) and
> this pipeline has no world-knowledge oracle by design. `unsupported` means
> *unsupported by this corpus*; a true claim the corpus is silent on lands
> there too. Only `contradicted` is evidence **against** a span.

Literature basis: the provenance/veracity split is the attribution-vs-correctness
distinction from AIS ([Rashkin et al., 2023](https://aclanthology.org/2023.cl-4.2/);
[Bohnet et al., 2022](https://arxiv.org/abs/2212.08037)); scoring a claim
against retrieved evidence rather than its surface string follows FActScore
([Min et al., 2023](https://aclanthology.org/2023.emnlp-main.741/)). Full
derivation and reference list in ARCHITECTURE.md §9.5.

### 3. Metamorphic stability & evidence sensitivity — `scripts/03_run_metamorphic_eval.py`

```bash
# after a profile run exists; start with a sample — the full run is ~2,400 calls
.venv/bin/python scripts/03_run_metamorphic_eval.py --sample 30
.venv/bin/python scripts/03_run_metamorphic_eval.py --run 2026-07-01_120000 --paraphrases 5
.venv/bin/python scripts/03_run_metamorphic_eval.py --probes control paraphrase
```

There are no gold labels here, so instead of checking whether an answer is
right, this check **changes the evidence it was built from and watches what the
label does**. For each item of an existing run the exact retrieved chunks are
refetched by id, and four probes run — each re-answered and re-graded through
the production answer → match path, so a changed label is down to the
perturbation alone. Each probe is toggleable with `--probes`.

| Probe | In plain terms | What a grounded label should do |
|---|---|---|
| `control` | run it again, change nothing | keep the same label |
| `paraphrase` | say the same thing in different words | keep the same label |
| `ablation` | remove the excerpt the answer quoted | weaken, toward *abstain* |
| `distractor` | ask the same question over real same-lab text that doesn't answer it | say the excerpts don't address this |

For the first two a **changed** label is suspicious. For the last two it is the
reverse: a label that **survives** is the warning sign, because the answer
plainly did not need the evidence it cited.

**The control comes first.** It re-runs an item on unchanged evidence, so
anything that flips there flipped on its own. `control_flip_rate` is the noise
floor every other number is read against — and an item whose own control
flipped is never counted as unstable, because its paraphrase result can't be
interpreted.

**Paraphrases must pass three gates, checked in code.** A rewrite that quietly
changed meaning would show up as a label flip and be misread as a
hallucination, so fidelity is verified rather than trusted to the prompt:

1. **Facts kept** — every number, date and name in the original survives.
2. **Actually reworded** — word overlap stays at or below
   `PARAPHRASE_MAX_TOKEN_OVERLAP` (0.95), so a copy can't pass as a paraphrase.
3. **Still means the same** — the rewrite must embed closer to its own excerpt
   than to any sibling excerpt, with a floor of `PARAPHRASE_MIN_COSINE` (0.85).

A rewrite failing any gate is **thrown out and retried, never scored**, and the
rejection is reported with its reason.

**How the scores are computed** (full derivation in ARCHITECTURE.md §9.3; the
GUI's Hallucination tab has the same math in an expander):

```
label(v)           = "abstain" if the matcher abstained, else argmax_k w_k
                     (ties break by the fixed LOGICS order; abstain is a label)

control_flip_rate  = |{ i : label(control_i) ≠ label₀(i) }| / n_control

label_stability    = |{ v ∈ P_ok : label(v) = label₀ }| / |P_ok|
                     P_ok = rewrites that ran AND passed all three gates
unstable           = label_stability < θ  ∧  the item's control did NOT flip

ablation_survival_rate = |{ i : label(abl_i) = label₀ ≠ "abstain" }| / n_abl
prior_leak_rate        = |{ i : label(dis_i) = label₀ ≠ "abstain" }| / n_dis
```

Failed and rejected variants are excluded from denominators — never counted as
flips. Config: `k` = `METAMORPHIC_PARAPHRASES` (3), `θ` =
`METAMORPHIC_STABILITY_THRESHOLD` (1.0 — any single flip flags the item; relax
if paraphrase noise is high), paraphrase temperature =
`METAMORPHIC_PARAPHRASE_TEMPERATURE` (0.7, so the k variants differ; answering
and matching stay at temperature 0).

*Worked example:* an item labeled `State` gets 3 rewrites; one is discarded for
dropping a figure, of the remaining 2 both keep the label → `label_stability =
1.0`, stable. Its distractor variant — the same question over excerpts
retrieved for an unrelated category of the same lab — comes back `State`
again → **prior-keyed**: the label reappeared without the evidence that was
supposed to produce it.

`prior_leak_rate` is the headline detection metric, and it replaces the
lab-name swap that earlier versions used. That arm renamed the lab throughout
the evidence and read a flip as prior-keying, but the rename changes what the
question asks *and* left product names and people untouched, so the model was
being shown a self-contradictory document and a flip could not be pinned on the
name. The distractor probe tests the same worry with text that is real and
correctly named. See ARCHITECTURE.md §9.3 for the full argument.

Outputs land inside the evaluated run's snapshot
(`data/profiles/runs/<run_id>/metamorphic/`): `variants.jsonl` (resumable audit
trail — paraphrase rows carry the original excerpts next to their rewrites, so
a flagged item can be checked by hand) and `stability.json` (per-item records +
aggregate summary, including stability by category and — if the run used
`--grounding` — by bucket). The console report prints the noise floor first,
then everything else.

> **What is still taken on trust:** the gates verify facts, wording and topic,
> but not shifts of emphasis subtle enough to clear all three. The distractor
> probe also uses *unrelated* evidence, a harder test than the merely thin
> evidence a real run meets, so `prior_leak_rate` is an upper bound on the
> tendency rather than its production rate. Skim a few flagged variants before
> reading the aggregates as evidence.

Literature basis: metamorphic testing for the no-oracle setting
([Chen et al., 1998](https://arxiv.org/abs/2002.12543); survey:
[Segura et al., 2016](https://doi.org/10.1109/TSE.2016.2532875));
invariance tests under label-preserving perturbations
([Ribeiro et al., 2020](https://aclanthology.org/2020.acl-main.442/));
consistency under paraphrase
([Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.60/)); MetaQA
([Yang et al., FSE 2025](https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/48/Hallucination-Detection-in-Large-Language-Models-with-Metamorphic-Relations));
abstention on unsupported questions
([Rajpurkar et al., 2018](https://aclanthology.org/P18-2124/)); parametric
knowledge overriding the context
([Longpre et al., 2021](https://aclanthology.org/2021.emnlp-main.565/));
sampling-based consistency as a baseline
([Manakul et al., 2023](https://aclanthology.org/2023.emnlp-main.557/)).
Full reference list in ARCHITECTURE.md §9.3.

### 4. Embedding agreement — `scripts/04_run_embedding_agreement.py`

```bash
.venv/bin/python scripts/04_run_embedding_agreement.py          # CURRENT run
.venv/bin/python scripts/04_run_embedding_agreement.py --run 2026-07-01_120000
```

A **non-LLM second judge** over any saved run: embeds every committed answer
and the run's own seven reference answers per question (respecting per-question
overrides), ranks the references by cosine similarity, and reports both a
**binary** agreement (nearest reference's logic == the matcher's top logic) and
**graded** metrics — min-shifted closeness *shares* per logic compared against
the matcher's weight distribution (mean share on the matcher's pick, chance
1/7; distribution overlap vs a uniform-judge baseline). Deterministic; costs a
few hundred embedding calls. Outputs land in the run's `embedding_agreement/`.
Interpretation caveat: e5 compresses cosines into a narrow band, so only
rankings/shares/margins are meaningful — and whole-answer embeddings track
*topic* more than institutional stance, so treat this as triangulation, not
ground truth.

### Bootstrap confidence intervals — `scripts/05_run_bootstrap_ci.py`

```bash
.venv/bin/python scripts/05_run_bootstrap_ci.py                 # CURRENT run
```

Each profile % is a mean over the answered questions, so its error bar comes
from **resampling those questions with replacement** (default 2000×, 95% CI,
seeded — fully deterministic, zero API calls). Chosen over "re-run N times"
because the temperature-0 pipeline makes repeats near-identical, which would
understate the real uncertainty. Results land in the run's `bootstrap_ci/`
(`ci.json`, `ci.csv`) and appear as error-bar whiskers on the Results tab.

### Tests

```bash
.venv/bin/python -m pytest tests/   # offline: no API key, no index needed
```

The suite pins the invariant that default runs keep the exact original row
schema, and covers grounding scores/buckets, quote verification, quote
provenance (each ladder tier, each intent rule, and every cell of the 2×2
verdict), the paraphrase fidelity gates, distractor selection, and the
stability math.

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

Optional checks write into subdirectories of the same snapshot:
`metamorphic/`, `embedding_agreement/`, `bootstrap_ci/`, and
`quote_provenance/` (`spans.jsonl` — one graded record per quoted span;
`summary.json` — tier/intent/verdict histograms and rates, overall and per
category and lab/source).

Pre-snapshot results (flat files from before this layout) are migrated into a
`legacy` run automatically on first launch. A console report still prints each
profile as a ranked bar list.

## Layout

```
il_rag/
  config.py           paths, models, hyperparameters, study design
  questionnaire.py    27 questions + per-question 7-logic reference answers
                      (the researcher's finalized set, from New Question Set.docx)
  llm.py              Together chat/embeddings with transient-error retry
  json_utils.py       shared JSON extraction from LLM replies
  ingest.py           parse corpora -> chunk -> embed -> Chroma
  retriever.py        (org, source_type)-scoped semantic retrieval
  rag_qa.py           retrieve -> grounded free-form answer (opt-in: quotes)
  graded_matcher.py   answer -> weight distribution over the 7 logics
  grounding.py        opt-in: no-LLM retrieval-grounding score + buckets
  metamorphic.py      opt-in: control / paraphrase / ablation / distractor eval
  embedding_agreement.py  opt-in: non-LLM second judge (binary + graded)
  quote_provenance.py opt-in: graded quote provenance x veracity (2x2 verdict)
  pdf_sources.py      opt-in: resolve a chunk back to its source PDF, so the
                      audit trail's [excerpt N] citations become links
  bootstrap_ci.py     confidence intervals over the profiles (seeded, no API)
  profile_harness.py  orchestration, aggregation, outputs, report
  runs.py             run snapshots: archive/list/compare, legacy migration
scripts/
  01_ingest.py                stage 1: build the index
  02_run_profiles.py          stage 2: produce the profiles
  03_run_metamorphic_eval.py  stage 3 (optional): metamorphic eval
  04_run_embedding_agreement.py  stage 4 (optional): embedding second judge
  05_run_bootstrap_ci.py      stage 5 (optional): bootstrap error bars
  06_run_quote_provenance.py  stage 6 (optional): graded quote provenance
  09_build_pdf_pages.py       stage 9 (optional, local): chunk -> PDF page map
tests/                offline unit tests (pytest; all API calls stubbed)
app.py                Streamlit GUI (Run / Results / Audit / Hallucination / Compare)
Launch IL Profiler.command   double-clickable launcher (macOS)
Launch IL Profiler.bat       double-clickable launcher (Windows)
```

## Cost notes

Ingestion embeds the full corpora (thousands of embedding calls). A full
profile run makes 162 generation calls + 162 matching calls + 162 query
embeddings (`--grounding` adds nothing; `--quotes` changes the answer prompt
but not the call count). A full metamorphic eval with all four probes at the
default 3 paraphrases is ~2,400 chat calls (answer + match per variant, plus one
generation call per paraphrase) — use `--sample`, or `--probes`, first. Quote
provenance is cheapest-first and short-circuits: a run whose quotes are all
verbatim costs nothing, and otherwise it is one chat call per non-verbatim span
plus a handful of cached embeddings. Stages are resumable, so an interrupted run
loses nothing.
