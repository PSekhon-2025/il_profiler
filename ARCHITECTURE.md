# IL Profiler — How It Works

Detailed documentation of the Institutional-Logics Profiler: what it does, the
theory behind it, how the pipeline is built, and how to run and trust it.

- [1. What the app does](#1-what-the-app-does)
- [2. Theoretical basis](#2-theoretical-basis)
- [3. The data](#3-the-data)
- [4. Pipeline overview](#4-pipeline-overview)
- [5. Stage by stage](#5-stage-by-stage)
- [6. The questionnaire and reference answers](#6-the-questionnaire-and-reference-answers)
- [7. Scoring: from answers to percentages](#7-scoring-from-answers-to-percentages)
- [8. Runs, snapshots, and resumability](#8-runs-snapshots-and-resumability)
- [9. Validation & hallucination checks](#9-validation--hallucination-checks)
- [10. Confidence intervals](#10-confidence-intervals)
- [11. Determinism & reproducibility](#11-determinism--reproducibility)
- [12. The GUI](#12-the-gui)
- [13. Running it](#13-running-it)
- [14. Deployment](#14-deployment)
- [15. File map](#15-file-map)
- [16. Known limitations](#16-known-limitations)

---

## 1. What the app does

The IL Profiler reads a corpus of documents about three AI labs — **OpenAI,
DeepMind, Anthropic** — and produces, for each lab, a **percentage profile over
seven institutional logics** (State, Profession, Market, Corporation, Family,
Religion, Community). It does this **separately** for two kinds of source:

- **`published`** — the lab's own documents (its self-presentation)
- **`thirdparty`** — press articles written about the lab (external perception)

So the output is **six profiles** (3 labs × 2 source types). The research claim
is that AI "alignment" should be examined through a *company-wide institutional*
lens, not a purely technical one — and these profiles operationalize that lens.

The method is **RAG + answer matching**: for a fixed questionnaire, the system
retrieves evidence from a lab's corpus, has an LLM write a grounded answer, then
a second LLM step grades that answer against reference answers representing each
logic. The grades are aggregated into the percentage profile.

---

## 2. Theoretical basis

Three sources anchor the design:

1. **Structural Transparency of Societal AI Alignment through Institutional
   Logics** (the researcher's paper). Argues that organizational and
   institutional forces shaping alignment decisions should be made analyzable.
2. **Thornton & Ocasio's inter-institutional system** (via the MISQ paper, Faik,
   Barrett & Oborn). Supplies the typology: **7 institutional orders** ×
   **9 elemental categories** (Basis of Norms, Sources of Legitimacy, Sources of
   Authority, Technology Affordances, Sources of Identity, Basis of Attention,
   Basis of Strategy, Informal Control, Economic System).
3. **Answer Matching Outperforms Multiple Choice** (Chandak et al., 2025). The
   evaluation methodology: ask open-ended questions, get free-form answers, and
   grade them against reference answers with an LLM — more faithful than
   multiple choice, which can be gamed by option-elimination.

**Family and Religion are retained on purpose.** Real AI labs should score ≈0%
on them; if they don't, the method is misfiring. This is a built-in sanity
check, not an oversight.

---

## 3. The data

Per lab, two corpora (paths in `il_rag/config.py`):

| Lab | Published documents | Third-party articles |
|-----|---------------------|----------------------|
| OpenAI | `OpenAI/OpenAI PDF's/pdf_corpus.txt` | `OpenAI/OpenAI Articles/*.RTF` |
| DeepMind | `DM/Deepmind PDF's/pdf_corpus.txt` | `DM/Deepmind Articles/*.RTF` |
| Anthropic | `Anthropic/Anthropic PDF's/pdf_corpus.txt` | `Anthropic/Anthropic Articles/*.RTF` |

- **Published**: the lab's PDFs converted to one delimited text file; individual
  documents are separated by `FILE:` header blocks.
- **Third-party**: large RTF exports (Nexis / BuySellSignals / News Bites) that
  bundle many press clippings, plus machine-generated boilerplate.

The raw corpus lives outside this repo and is **not** committed (copyrighted).
The deployed cloud instance ships only the *derived vector index*, never the
source files.

---

## 4. Pipeline overview

```
        ┌─────────── ingest (once) ───────────┐
raw corpus → parse → strip boilerplate → chunk → embed → Chroma vector index
        └──────────────────────────────────────┘

        ┌─────────── per question (×27 per lab×source) ───────────┐
question → retrieve (scoped, deduped) → RAG answer → graded match → weights
        └──────────────────────────────────────────────────────────┘

weights → aggregate (mean per logic) → percentage profile per (lab, source)
        → bootstrap CI, embedding-agreement, grounding, metamorphic (optional)
```

- **LLM**: TogetherAI `openai/gpt-oss-120b` (both the answer and the grading).
- **Embeddings**: TogetherAI `intfloat/multilingual-e5-large-instruct` (1024-dim).
- **Vector store**: local **ChromaDB**, collection `il_corpus`, cosine space.

---

## 5. Stage by stage

### Ingest (`il_rag/ingest.py`)

1. **Parse.** Published text is split on `FILE:` headers so each chunk keeps its
   source filename; RTF dumps are split into individual articles on delimiters
   (`End of Document`, `Title:`, `Length: N words`, …).
2. **Strip boilerplate** (`strip_boilerplate`). Removes Nexis/BuySellSignals
   junk — metadata fields (`PermID:`, `Load-Date:`, `Length:`…), company-profile
   scaffolding (`SECTION 2 …`, `Top Management`), and pipe-table rows. Editorial
   prose is kept. Applied per article, *after* article splitting (the split
   markers are themselves boilerplate).
3. **Chunk** (`chunk_text`). Sliding window, ~1400 chars with ~150 overlap,
   breaking on sentence/newline boundaries where possible.
4. **De-duplicate** (`dedup_corpus`). The third-party corpus is ~50% duplicates
   (syndicated articles + repeated profile blocks). Near-duplicate chunks (by a
   240-char normalized signature, scoped per lab+source) are dropped so copies
   are never embedded and don't dominate retrieval.
5. **Embed & store.** Batches to the embedding API (resilient: oversize batches
   bisect so only a single too-long chunk is dropped), upserts into Chroma with
   metadata `{org, source_type, doc_type, filename}`.

### Retrieve (`il_rag/retriever.py`)

- Embeds the question, then queries Chroma **filtered to one `(org, source_type)`**
  — a compound `$and` filter. This scoping is what keeps the six profiles
  independent: an OpenAI-published question can only match OpenAI-published
  chunks.
- **Over-fetches `k×6` and de-duplicates** at query time as a second guard,
  returning the top `k=5` distinct chunks.

### Answer (`il_rag/rag_qa.py`)

- The answering model sees **only the retrieved excerpts** — never the logics
  matrix or the reference answers. That separation is what makes the subsequent
  matching meaningful: the answer reflects the *corpus*, not the *taxonomy*.
- The prompt asks it to state a conclusion first, then justify from the
  excerpts, and to **say explicitly when the excerpts don't answer** (so the
  matcher can abstain rather than grade a hallucination).
- `temperature=0`. `max_tokens=2048` (gpt-oss is a reasoning model — the budget
  must cover hidden reasoning *plus* the visible answer, or the answer comes back
  empty).

### Match (`il_rag/graded_matcher.py`)

- Given the answer and the **7 reference answers for that question's category**,
  the grader assigns a **weight in [0,1] per logic**, summing to 1 — a graded,
  multi-logic verdict rather than a single pick (institutional logics co-exist).
- Guarantees enforced **in code**, never trusted to the LLM: weights are clamped
  non-negative and renormalized to sum to 1; an all-zero distribution becomes an
  **abstention**; on abstention all weights are zeroed so "no evidence" can't
  leak weight into any logic.
- `temperature=0`, with a parse-failure retry at a larger token budget.

---

## 6. The questionnaire and reference answers

Defined in `il_rag/questionnaire.py` — the researcher's finalized set,
transcribed verbatim from `New Question Set.docx` (kept in the repo as the
source of record).

**Structure (load-bearing):** 9 categories × 3 questions = **27 questions**, each
with a full **7-logic reference set**.

- Each category has a base `reference_answers` block (7 logics).
- Each category also has `reference_overrides`: `{variant: {logic: text}}`,
  giving the **per-question** exemplar where the document provides one.
- A `(variant, logic)` cell with no override falls back to the base text. A few
  cells fall back deliberately (e.g. Basis of Strategy Q2; Economic System Q2/Q3
  for Family and Community).

`reference_answers(category, variant)` resolves the base + override for a
specific question. **Both** the LLM matcher and the embedding-agreement check use
this same resolver, so the two judges grade against identical references.

Question-writing principles (in the module header): never enumerate the logics
inside a question (that leads the model); ask about concrete, observable things
(so the question also works as a retrieval query); the three variants triangulate
(self-description / observable behavior / a contested trade-off).

---

## 7. Scoring: from answers to percentages

In `il_rag/profile_harness.py`:

- Every **answered** (non-abstained) question contributes a weight vector summing
  to 1.
- A lab/source profile is the **mean weight per logic across its answered
  questions**, reported as percentages summing to ~100.
- **Abstentions are excluded from the denominator.** A silent corpus lowers
  *confidence* (fewer answered questions) but never *shifts* the distribution.
- A **per-category breakdown** is also produced (mean within each category).

Outputs per run: `company_profiles.json`, `profiles_matrix.csv`, and the audit
trail `per_question.jsonl` (one row per lab×source×question with the answer,
retrieved chunk ids, weights, and matcher reasoning).

---

## 8. Runs, snapshots, and resumability

`il_rag/runs.py` makes every profiling run an **immutable snapshot** under
`data/profiles/runs/<run_id>/` (`run_id` = `YYYY-MM-DD_HHMMSS`). Each snapshot
holds the per-question rows, the aggregated profiles, **a copy of the
questionnaire that produced it**, and a `meta.json`.

- `--fresh` mints a **new** snapshot; the previous one is untouched. This is what
  lets you change the questionnaire and diff old vs. new instead of overwriting.
- Without `--fresh`, the **CURRENT** run is resumed: questions already in its
  `per_question.jsonl` are skipped, so an interrupted run continues where it
  stopped. Every completed row is flushed to disk immediately (crash-safe).
- `migrate_legacy()` folds any pre-snapshot flat files into a run on first use.

---

## 9. Validation & hallucination checks

All four are **opt-in and post-hoc** — a default run is byte-identical without
them, and they operate on a saved run. They live on the GUI's **Hallucination**
tab and as `scripts/03`–`04`.

### 9.1 Retrieval grounding (`il_rag/grounding.py`, `--grounding` on a run)

No LLM — pure computation over the already-retrieved chunks, so enabling it
adds zero API cost. It separates "the model hallucinated over good evidence"
from "retrieval never found evidence." The notation below (`T(·)`, `overlap`,
`g`, `τ`) is the same one used in the GUI's "How this score is computed"
expander and in `grounding.py`'s docstrings.

**Content tokens.** Lowercased alphanumeric tokens, minus a ~70-word English
stopword list, minus tokens of ≤ 2 characters (so function words cannot
inflate overlap):

```
T(x) = { t ∈ tokens(lower(x)) : t ∉ stopwords, |t| > 2 }
```

**Per-chunk lexical overlap** — ROUGE-1-recall-style set overlap: the fraction
of the question's content tokens present in the chunk, in [0, 1] (defined as 0
when `T(q)` is empty):

```
overlap(q, c) = |T(q) ∩ T(c)| / |T(q)|
```

**Grounding score and cosine subscore** over the retrieved set `R(q)`:

```
g(q)          = max_{c ∈ R(q)} overlap(q, c)        → retrieval_grounding_score
cosine_top(q) = max_{c ∈ R(q)} clip(score_c, 0, 1)  → retrieval_cosine_top
```

Max, not mean: one genuinely relevant chunk is enough to ground an answer, so
a strong hit shouldn't be diluted by weak siblings.

**Bucket rule**, with `τ = GROUNDING_LOW_THRESHOLD` (0.2 in `config.py`):

```
bucket(q) = retrieval_missed   if g(q) < τ
            abstained          if g(q) ≥ τ and the matcher abstained
            committed          otherwise
```

`retrieval_missed` takes precedence over `abstained`: when retrieval failed,
abstaining was the *right* response, so the item's failure belongs to
retrieval, not to the model's grounding.

**Per-bucket summary.** There are no gold labels in this pipeline, so instead
of accuracy each bucket `b` reports:

```
n_b             = |{rows in b}|
abstain_rate_b  = n_abstained_in_b / n_b
mean_top_weight = mean over committed rows in b of max_k weights[k]
```

`mean_top_weight` is a proxy for how decisively the matcher graded the
bucket's committed answers. The buckets separate *failure modes*, not
correctness.

**Design rationale.** The thresholded signal is *lexical*, not cosine: e5
embeddings compress cosine into a narrow high band even for weak matches, so
token overlap is the discriminative, interpretable signal; `cosine_top` is
kept as an unthresholded diagnostic subscore. `τ` is a per-corpus heuristic,
not a learned or label-calibrated parameter — tune it against the score
histogram on the Hallucination tab.

**Limitations.** Pure lexical overlap is blind to synonymy and paraphrase: a
chunk that answers the question in different words can score low. That is
exactly why `cosine_top` is retained (a low-`g`, high-cosine row hints at a
paraphrased rather than missing match). And because `τ` is uncalibrated, the
`retrieval_missed` bucket is a *flag for review*, not a verdict.

**Basis in the literature.** The score is a composition of established
metrics rather than a novel one; each design element has a direct precedent:

- *The overlap formula* is a set-based variant of **ROUGE-1 recall**
  (Lin, 2004): unigram recall of the question's content tokens against a
  chunk, computed over deduplicated token *sets* rather than token counts.
  Token-level overlap is likewise the standard lexical-match measure in
  extractive QA evaluation (Rajpurkar et al., 2016).
- *Using lexical overlap with retrieved text as a groundedness signal*
  follows **Knowledge F1** (Shuster et al., 2021), which measures word
  overlap between generated text and the retrieved knowledge to quantify
  grounding vs. hallucination. (We use recall of the *question*, not F1 of
  the answer, because the object audited here is retrieval coverage.)
- *Stopword removal and bag-of-words lexical matching* are standard IR
  preprocessing (Manning, Raghavan & Schütze, 2008).
- *The `retrieval_missed` vs. model-failure split* mirrors the RAG
  failure-point taxonomy of Barnett et al. (2024), whose first failure point
  — **"Missing Content"** — is exactly this case: the retrieved documents do
  not contain the answer, so whatever the model generates next is
  unsupported. It also reflects the source-error vs. generation-error
  distinction in the hallucination survey of Ji et al. (2023).
- *Not thresholding the cosine* is motivated by embedding **anisotropy**:
  contextual embedding vectors occupy a narrow cone, so even unrelated texts
  receive high cosine similarity (Ethayarajh, 2019). The retriever's e5
  model (Wang et al., 2022) shows this band compression on this corpus,
  which is why cosine is retained only as a diagnostic subscore.

**References**

- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M.
  (2024). Seven Failure Points When Engineering a Retrieval Augmented
  Generation System. *CAIN 2024*. <https://arxiv.org/abs/2401.05856>
- Ethayarajh, K. (2019). How Contextual are Contextualized Word
  Representations? Comparing the Geometry of BERT, ELMo, and GPT-2
  Embeddings. *EMNLP 2019*. <https://aclanthology.org/D19-1006/>
- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.,
  Madotto, A., & Fung, P. (2023). Survey of Hallucination in Natural
  Language Generation. *ACM Computing Surveys*, 55(12).
  <https://arxiv.org/abs/2202.03629>
- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
  *Text Summarization Branches Out* (ACL Workshop).
  <https://aclanthology.org/W04-1013/>
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to
  Information Retrieval*. Cambridge University Press.
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+
  Questions for Machine Comprehension of Text. *EMNLP 2016*.
  <https://aclanthology.org/D16-1264/>
- Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval
  Augmentation Reduces Hallucination in Conversation. *Findings of EMNLP
  2021*. <https://aclanthology.org/2021.findings-emnlp.320/>
- Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder,
  R., & Wei, F. (2022). Text Embeddings by Weakly-Supervised Contrastive
  Pre-training. *arXiv:2212.03533*. <https://arxiv.org/abs/2212.03533>

**Output fields** added to each per-question row: `retrieval_grounding_score`,
`retrieval_cosine_top`, `grounding_bucket`.

### 9.2 Quote verification (`il_rag/rag_qa.py`, `--quotes` on a run)

The answer model must return, alongside its answer, the verbatim excerpt
spans its conclusion rests on; each span is then re-checked **in code**. The
model attests, the code audits — nothing the model says about its own quotes
is trusted, and verification adds zero API calls beyond the answer call
itself. The notation below (`norm(·)`, `verified(q)`, `Q`) is the same one
used in the GUI's "How quotes are verified" expander and in `rag_qa.py`'s
docstrings.

**The contract.** The quote-mode prompt requires strict JSON:

```
{"answer": "...", "quotes": [{"excerpt": <i>, "quote": "<span>"}]}

- 1 to 3 entries: the specific spans the conclusion rests on
- each span "copied character-for-character from the numbered excerpt
  it cites; never paraphrase inside a quote"
- if the excerpts cannot answer, say so and return an empty quotes list
```

**Normalization.** Both the quote and every retrieved chunk are normalized
before comparison — whitespace-tolerant and case-insensitive, but otherwise
verbatim (punctuation is *not* stripped):

```
norm(s) = lowercase(collapse_ws(s))     collapse_ws: every whitespace run → " ", ends stripped
```

**Per-quote check.** A quote verifies iff its normalized form is non-empty
and occurs as a substring (⊑) of *any* normalized retrieved chunk:

```
verified(q) = norm(q) ≠ "" ∧ ∃ c ∈ R : norm(q) ⊑ norm(c)
```

The cited excerpt index is persisted for display but deliberately **not**
used for matching: the auditable claim is "this text is in the sources," not
the model's index bookkeeping — a correct span with a wrong number is sloppy
citing, not fabrication.

**Row verdict.** The row's `quotes_verified` is a guarded conjunction over
its quote set `Q`:

```
quotes_verified = |Q| > 0 ∧ ∀ q ∈ Q : verified(q)
```

The `|Q| > 0` guard exists because a conjunction over an empty set is
vacuously true — an empty or unusable quote list is unverified *by
definition*.

**Edge cases.** If the model's JSON does not parse, the call is retried once
at a doubled token budget (3072 → 6144, temperature 0); if it still fails,
the raw text is kept as the answer with `quotes = []` and
`quotes_verified = False` — the run degrades gracefully instead of dying,
and the row stays auditable. Non-list `quotes` payloads are treated as
empty; non-dict entries are skipped.

**Design rationale.** *Any-excerpt* matching beats index-strict matching
because the failure being hunted is fabricated text, not miscounted
excerpts. Whitespace/case normalization absorbs the copying artifacts models
actually produce, while keeping punctuation verbatim keeps the check strict
on content. And the report separates **fabricated** (non-empty quote list,
at least one span not in the sources) from **no quotes returned** (empty
list — typically an honest abstention or a parse fallback): abstaining is
not fabrication.

**Limitations.** Verbatim substring matching cannot credit a
paraphrased-but-faithful quote, so a failed quote means "not verbatim in the
sources," which is not identical to "the answer is wrong" — the check errs
toward false alarms, never toward missed fabrications. Conversely, a
verified quote proves the span exists in the sources, not that the
conclusion follows from it (this is attribution, not entailment). Unlike
grounding's `τ`, there is no tunable constant: a span either occurs verbatim
after normalization or it doesn't.

**Basis in the literature.**

- *Answering with verbatim quotes that are then mechanically verified
  against the sources* follows **GopherCite** (Menick et al., 2022), where
  supporting evidence is a quote checked to appear verbatim in the source
  document. (GopherCite verifies exact strings; this check adds only
  whitespace/case normalization.)
- *The property being audited* is **Attributable to Identified Sources
  (AIS)** as formalized by Rashkin et al. (2023) and operationalized for QA
  by Bohnet et al. (2022): can the produced text be supported by the cited
  source?
- *Evaluating citation quality of LLM output* follows **ALCE** (Gao et al.,
  2023), which measures whether generated citations actually support the
  generated statements.
- *Fabricated supporting evidence* is a recognized hallucination mode in the
  survey of Ji et al. (2023) — see the reference in §9.1.

**References**

- Bohnet, B., Tran, V. Q., Verga, P., Aharoni, R., Andor, D., Baldini
  Soares, L., Ciaramita, M., Eisenstein, J., Ganchev, K., Herzig, J., Hui,
  K., Kwiatkowski, T., Ma, J., Ni, J., Sestorain Saralegui, L., Schuster,
  T., Cohen, W. W., Collins, M., Das, D., Metzler, D., Petrov, S., &
  Webster, K. (2022). Attributed Question Answering: Evaluation and
  Modeling for Attributed Large Language Models. *arXiv:2212.08037*.
  <https://arxiv.org/abs/2212.08037>
- Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling Large Language
  Models to Generate Text with Citations. *EMNLP 2023*.
  <https://aclanthology.org/2023.emnlp-main.398/>
- Menick, J., Trebacz, M., Mikulik, V., Aslanides, J., Song, F., Chadwick,
  M., Glaese, M., Young, S., Campbell-Gillingham, L., Irving, G., &
  McAleese, N. (2022). Teaching Language Models to Support Answers with
  Verified Quotes. *arXiv:2203.11147*. <https://arxiv.org/abs/2203.11147>
- Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., Collins, M., Das, D.,
  Petrov, S., Tomar, G. S., Turc, I., & Reitter, D. (2023). Measuring
  Attribution in Natural Language Generation Models. *Computational
  Linguistics*, 49(4). <https://aclanthology.org/2023.cl-4.2/>

**Output fields** added to each per-question row: `quotes` (each entry
`{"excerpt": int, "quote": str, "verified": bool}`) and `quotes_verified`
(their guarded conjunction).

The remaining checks:

3. **Metamorphic label stability** (`il_rag/metamorphic.py`). For each answered
   item it makes *k* meaning-preserving paraphrases of the evidence (LLM) and one
   **lab-name swap** (deterministic regex), re-runs the production answer→match
   path, and checks whether the label survives. A paraphrase flip = unstable; a
   swap flip suggests the label was keyed on the lab's *name*, not the text.
4. **Embedding agreement** (`il_rag/embedding_agreement.py`). A **non-LLM second
   judge**: embed each committed answer and the run's 7 reference answers for its
   category, rank the references by cosine similarity, and check whether the
   nearest one's logic matches the LLM matcher's top logic. Deterministic.
   *Interpretation:* absolute cosine values are not meaningful (e5 compresses
   them into a narrow band); only the ranking and the agreement rate are. Low
   agreement is a known property of whole-answer embeddings (topical vocabulary
   dominates institutional stance), not evidence the matcher is wrong.

---

## 10. Confidence intervals

`il_rag/bootstrap_ci.py` (`scripts/05`, and the Results-tab error bars).

A profile % is a **mean over the answered questions**, so its error bar comes
from **bootstrapping those questions**: resample them with replacement (default
2000×, 95%, seeded), recompute the profile each time, and take the 2.5/97.5
percentiles per logic. Zero API cost, fully deterministic (seeded).

This is chosen over "re-run the pipeline N times" because the pipeline is
temperature-0 — repeats barely move, so a repeat-based CI would be spuriously
~0 and would **understate** the real uncertainty. The bootstrap answers the
meaningful question: *how much does the profile depend on which questions were
asked?* With ~27 questions per profile the bars are wide (often ±15 points),
which is honest: the dominant-logic ranking is robust, the exact percentages are
not tightly pinned.

---

## 11. Determinism & reproducibility

- **Analysis layers (bootstrap CI, embedding agreement)** are provably
  deterministic: seeded RNG and pure arithmetic over saved data. Identical every
  run.
- **LLM layers (answer, match)** run at **temperature 0** — greedy decoding, no
  sampling. Effectively deterministic, but *not* guaranteed bit-identical across
  calls: shared-GPU batching, floating-point ordering, and mixture-of-experts
  routing can occasionally flip a near-tie token. In practice a full re-run moves
  a profile by at most a point or two, and dominant logics stay put — far below
  the bootstrap CI width.

Net: results carry no injected noise, and error bars come from a reproducible
resampling of the questions rather than from re-rolling the model.

---

## 12. The GUI

`app.py` (Streamlit). Five tabs:

- **Run** — save the API key, build the index (hidden in cloud mode), and run
  the questionnaire for any subset of labs/sources, with live logs. Stages run
  as resumable subprocesses.
- **Results** — the six profiles as charts (published vs. thirdparty per lab),
  the Family/Religion sanity banner, per-category breakdown, bootstrap-CI error
  bars, and downloads.
- **Audit** — every question's RAG answer, weights, matcher reasoning, and (when
  enabled) quotes + grounding bucket.
- **Hallucination** — the four checks from §9, with alert banners when a
  detection fires.
- **Compare** — diff two run snapshots: profile deltas, question-wording diff,
  and per-question label changes.

---

## 13. Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # paste TOGETHER_API_KEY

# GUI (recommended)
.venv/bin/streamlit run app.py     # or double-click Launch IL Profiler.command / .bat

# CLI
.venv/bin/python scripts/01_ingest.py --fresh        # build the index (once)
.venv/bin/python scripts/02_run_profiles.py --fresh  # run all six profiles
.venv/bin/python scripts/05_run_bootstrap_ci.py      # error bars
.venv/bin/python scripts/04_run_embedding_agreement.py
.venv/bin/python scripts/03_run_metamorphic_eval.py --sample 30
```

Tests are offline (all API calls monkeypatched): `.venv/bin/python -m pytest tests/`.

---

## 14. Deployment

Deployed to **Fly.io** as a single always-on (scale-to-zero) container with a
persistent volume holding the prebuilt index. Full runbook in `DEPLOY.md`. Key
points:

- `IL_PROFILER_CLOUD=1` hides the ingest UI (no raw corpus in the cloud).
- `APP_PASSWORD` gates the app (constant-time comparison). Cloudflare Access is
  the upgrade path for per-reviewer identity.
- Code changes ship with `fly deploy` (deploys **local** files — `git pull`
  first). The **index** is separate: rebuilt locally and re-seeded onto the
  volume; a code deploy never touches it.

---

## 15. File map

```
il_rag/
  config.py              paths, models, thresholds, study design
  questionnaire.py       27 questions + per-question 7-logic references
  ingest.py              parse → strip boilerplate → chunk → dedup → embed → Chroma
  retriever.py           scoped, deduped semantic retrieval
  rag_qa.py              retrieve → grounded answer (+ optional quotes)
  graded_matcher.py      answer → weight distribution over 7 logics
  profile_harness.py     run the questionnaire, aggregate to % profiles
  runs.py                immutable per-run snapshots + resumability
  grounding.py           (check) retrieval-grounding buckets
  metamorphic.py         (check) paraphrase + lab-swap label stability
  embedding_agreement.py (check) non-LLM second judge
  bootstrap_ci.py        confidence intervals over the profiles
  json_utils.py, llm.py  shared JSON extraction; Together chat/embed wrappers
scripts/
  01_ingest.py  02_run_profiles.py  03_run_metamorphic_eval.py
  04_run_embedding_agreement.py  05_run_bootstrap_ci.py
app.py                   Streamlit GUI (Run / Results / Audit / Hallucination / Compare)
tests/                   offline unit tests
Dockerfile, fly.toml, DEPLOY.md   deployment
```

---

## 16. Known limitations

- **Small instrument.** ~27 questions per profile → wide confidence intervals.
  Dominant-logic rankings are trustworthy; exact percentages are not.
- **Self-referential checks.** The metamorphic paraphraser and the answer model
  are the same model; stability numbers should be anchored by hand-reviewing a
  few variants.
- **Embedding agreement is weak here** by design of the medium — whole-answer
  embeddings track topic more than institutional stance. It is a triangulation
  signal, not ground truth.
- **Third-party coverage is uneven.** Press rarely discusses internal authority
  or informal control, so some categories legitimately abstain on the
  `thirdparty` side — a finding, not a bug.
- **LLM grading is the classifier.** There is no gold-labeled ground truth; the
  reference answers *are* the standard, so results are only as good as they are.
- **Copyright.** The third-party corpus is licensed news content — keep any
  deployment private/gated.
```
