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

All six are **opt-in and post-hoc** — a default run is byte-identical without
them, and they operate on a saved run. §9.1–§9.5 live on the GUI's
**Hallucination** tab and as `scripts/03`–`04` and `scripts/06`; §9.6 lives on
the **Topics** tab as `scripts/13`, because what it measures is a property of
the inductive topic layer rather than of a quote.

§9.5 is a refinement of §9.2 rather than an independent signal: §9.2 asks
whether a quoted span is verbatim in the sources, §9.5 grades what a "no"
actually means and adds an entailment axis. §9.5 reads §9.2's verdict and never
rewrites it, so both can be reported side by side.

§9.6 stands in the same relation to `il_rag/keyword_agreement.py`: that judge
asks whether an answer contains a keyword, §9.6 grades what a "no" means by
placing the keyword on a distance ladder. It likewise never rewrites the other's
verdict — they use different keyword sets to answer different questions, and both
are reported.

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

Both limitations are what **§9.5** exists to address: it grades the failed
spans (drifted copy / paraphrase / figure of speech / fabrication) and adds
the entailment axis this check lacks. §9.5 reads this check's verdict and
never rewrites it, so the strict number above stays comparable across runs.

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

### 9.3 Metamorphic stability & evidence sensitivity (`il_rag/metamorphic.py`, `scripts/03_run_metamorphic_eval.py`)

There are no gold labels in this pipeline, so correctness cannot be checked
directly. Instead this check **changes the evidence an answer was built from
and watches what the label does** — the metamorphic-testing move for the
no-oracle setting. Four probes run against each answered item of a saved run,
each re-answered and re-graded through the same `answer_question` →
`match_graded` path at temperature 0, so a change in the label is attributable
to the perturbation and nothing else. The notation below (`label(v)`, `label₀`,
`P_ok`, `θ`) is the same one used in the GUI's "How these checks work" expander
and in `metamorphic.py`'s docstrings.

**The label.** Each item's label is the matcher's abstention, or else the
top-weight logic; ties break deterministically by the fixed `LOGICS` order.
Abstain is a first-class label — a variant that abstains matches iff the
original also abstained:

```
label(v) = "abstain"        if the matcher abstained
           argmax_k w_k     otherwise   (ties → LOGICS order)
```

**The four probes.** Two ask whether the label *survives* something harmless;
two ask whether it *stops* when it should. That difference is the whole design:

| Probe | Perturbation | Relation | Suspicious outcome |
|---|---|---|---|
| `control` | none — the same excerpts, run again | — | any change (it is noise) |
| `paraphrase` | the excerpts rewritten, meaning preserved | invariance | the label **changes** |
| `ablation` | the excerpt the answer quoted is removed | directional | the label **survives** |
| `distractor` | real same-lab excerpts that don't address the question | directional | the label **survives** |

Each is independently toggleable (`METAMORPHIC_PROBES`, `--probes`), and the
eval is resumable per `(item, probe, index)`, so enabling a probe later runs
only the variants that were added.

#### The control, and why it comes first

The control re-runs an item on its unchanged evidence. Anything that flips here
flipped on its own — matcher nondeterminism, or an argmax tie between two
near-equal weights. It is the floor every other number is read against:

```
control_flipped(i) = label(control_i) ≠ label₀(i)
control_flip_rate  = |{ i : control_flipped(i) }| / n_control
```

This is not only reported: it **gates the verdict**. An item whose own control
flipped is never counted as unstable, because its paraphrase result cannot be
interpreted.

#### Paraphrases, and the three gates they must pass

`k = METAMORPHIC_PARAPHRASES` (3) rewrites of the item's excerpts are generated
in one LLM call per variant, under a strict preservation rule ("every fact,
name, number, date, and claim is preserved exactly while the wording and
sentence structure change substantially"), sampled at
`METAMORPHIC_PARAPHRASE_TEMPERATURE` (0.7) so the k variants differ.

A rewrite that quietly changed meaning would surface as a label flip and be
read as a hallucination, so **fidelity is verified in code, never attested by
the prompt**. For each original excerpt `c` and its rewrite `p`:

```
gate 1  "facts kept"            numbers(c) ⊆ numbers(p)  ∧  entities(c) ⊆ entities(p)
gate 2  "actually reworded"     overlap(T(c), T(p)) ≤ ρ            ρ = PARAPHRASE_MAX_TOKEN_OVERLAP (0.95)
gate 3  "still means the same"  argmax_j cos(c_j, p) = c's own index
                                ∧ cos(c, p) ≥ PARAPHRASE_MIN_COSINE (0.85)
```

- `numbers` are `\d[\d,.:%/-]*` tokens; `entities` are capitalised tokens, minus
  stopwords and minus any word that also appears lowercased in the same text
  (which is what a common noun at a sentence start looks like). Both are
  compared as sets, and the entity pattern excludes apostrophes and hyphens so
  that "OpenAI's" and "OpenAI" normalize to the same token — otherwise a
  faithful rewrite that merely dropped a possessive would be rejected.
- `overlap` and `T(·)` are exactly Feature 1's `lexical_overlap` and
  `content_tokens`, reused rather than reimplemented.
- Gate 3 is **rank-based on purpose**: e5 compresses absolute cosines into a
  narrow band (the same property that makes §9.4 read only rankings), so the
  load is carried by "nearest to its own excerpt", with `0.85` as a coarse
  sanity floor rather than a calibrated threshold. It costs one batched
  embedding call, and runs only after the two free text gates have passed.

A variant failing any gate is **discarded and retried once**, then recorded with
its reason and excluded from the denominator. With `P_ok` the rewrites that ran
*and* passed:

```
label_stability = |{ v ∈ P_ok : label(v) = label₀ }| / |P_ok|    (None if P_ok empty)
unstable        = label_stability < θ  ∧  ¬control_flipped(i)    θ = METAMORPHIC_STABILITY_THRESHOLD (1.0)
```

#### The two directional probes

**Ablation** removes the excerpt the answer actually rested on — the one cited
by the first quote §9.2 verified, else the top-ranked excerpt (retrieval returns
nearest-first). Quote records index the run's *original* retrieved list, so they
are only trusted when every id was refetched; a shortened set falls back to the
top-ranked excerpt. Items with fewer than two excerpts are skipped
(`ablation_no_remainder`).

**Distractor** keeps the question but supplies excerpts retrieved for a
*different* question of the same lab and source type — a fixed half-list
rotation over `CATEGORIES`, so names and framing stay correct and only the topic
is wrong. Any excerpt the item was actually answered from is removed, and the
resulting set is re-scored against the **original** question with §9.1's
grounding score: a "distractor" that turns out to be relevant would make the
probe meaningless, so it is rejected (`distractor_relevant`) rather than scored.
The checks cross-validate each other.

For both, the correct behaviour is to weaken toward abstention, so a surviving
label is the finding:

```
label_survived_ablation = label(abl) = label₀ ≠ "abstain"
ablation_survival_rate  = n_survived / n_ablation_evaluated

distractor_committed    = label(dis) ≠ "abstain"
prior_keyed             = label(dis) = label₀ ≠ "abstain"
prior_leak_rate         = n_prior_keyed / n_distractor_evaluated
```

`prior_leak_rate` is the headline detection metric: the model was shown text
that does not answer the question and returned the original logic anyway.

**Summary statistics** over the evaluated items:

```
control_flip_rate           the noise floor (read everything against it)
mean_label_stability        mean of per-item stabilities (scored items only)
pct_fully_stable            share of scored items with label_stability ≥ 1.0
n_paraphrases_rejected      rewrites discarded, with rejected_by_reason
mean_paraphrase_divergence  1 − mean overlap with the source wording
mean_paraphrase_cosine      mean similarity of a rewrite to its own excerpt
ablation_survival_rate      labels that held without their evidence
distractor_commit_rate      answered instead of abstaining
prior_leak_rate             answered with the ORIGINAL label
by_category                 mean stability per question category
by_grounding_bucket         mean stability per Feature-1 bucket (only when the
                            source run carried grounding scores)
```

**Edge cases.** A paraphrase whose JSON does not parse is retried at a larger
token budget, then recorded as `error: "paraphrase_failed"`; one that parses but
fails a gate is `"paraphrase_infidelity"` with the gate's reason. Rows whose
chunks cannot be refetched are `"chunks_not_found"`. **Failed and rejected
variants are excluded from denominators, never counted as flips** — a parse
failure or a broken rewrite is evidence of nothing about stability, and their
counts are reported separately so each probe's health stays visible. Item
sampling is deterministic per seed (`random.Random(seed).sample`).

**Design rationale.** Paraphrases are sampled at nonzero temperature so the k
variants actually differ, while answering and matching stay at temperature 0
like the production path — the *system under test* is unchanged, only the input
moves. 0.7 rather than a higher setting because the gates now guarantee the
variants differ from their source, so extra sampling temperature buys only
drift. And `θ = 1.0` is the strictest default — any single paraphrase flip flags
the item; the config comment says to relax it if paraphrase noise is high.

**Interpretation.** The four numbers say different things. A **control flip**
says nothing about the model's grounding — it is the measurement error of the
instrument, and every other rate should be read as a distance above it. A
**paraphrase flip** says the label is not robust to rewording. A **surviving
ablation** says the answer did not need the evidence it quoted. A **prior leak**
says the label came back from text that does not support it — the closest thing
this pipeline has to a positive hallucination detection.

**Why the lab-name swap was removed.** An earlier version of this check
included a fifth arm: a deterministic regex rewriting every alias of the lab
throughout the excerpts and the question (OpenAI → DeepMind → Anthropic →
OpenAI), on the premise that a grounded label should survive a change of name.
That premise does not hold, for two independent reasons.

1. **The perturbation is not meaning-preserving.** These three labs are the
   subject of the question, not incidental strings in it. Asking what
   "DeepMind" says about its obligations, over text describing OpenAI's
   decisions, is a different question — not the same question in different
   words. An invariance relation requires that the perturbation leave the
   intended output unchanged, and this one does not.
2. **The variant context was self-contradictory.** The regex renamed only the
   lab aliases; product names, people, and events stayed put, so a swapped
   excerpt still read "Anthropic's GPT-4 launch". A label change under that
   perturbation is confounded between the effect of the name and the model
   reacting to an incoherent document, and the two cannot be separated after
   the fact.

The worry the swap was built to test — *is the label coming from the model's
prior about this lab rather than from the text?* — is real, and it is
unanswerable by trying to talk the model out of a prior it certainly has about
three of the best-documented organizations in its training data. The
`distractor` probe tests it the other way round: hold the lab's identity fixed
and correct, take the *supporting evidence* away, and see whether the label
comes back regardless. Everything it shows the model is real corpus text with
real names, so its number needs no caveat about coherence.

**Limitations.** The fidelity gates verify facts, wording and topic; they do not
catch subtle shifts of emphasis, so a rewrite can pass and still read a little
differently — flagged items are worth a human look, and the GUI shows each
original next to its rewrite for exactly that. The distractor probe measures
behaviour on *unrelated* evidence, which is a harder test than the merely thin
evidence a real run meets, so `prior_leak_rate` is an upper bound on the
tendency rather than its rate in production. At `k = 3`, per-item stability is
coarse-grained (steps of 1/3), so the aggregates are the readable numbers.

**Basis in the literature.**

- *Testing output relations under input transformations when no oracle exists*
  is **metamorphic testing** (Chen, Cheung & Yiu, 1998; surveyed by Segura et
  al., 2016). This pipeline has no gold labels — exactly the no-oracle setting
  the technique was built for. The `paraphrase` probe is an *invariance*
  relation; `ablation` and `distractor` are *directional* relations, where the
  expected effect on the output has a known sign.
- *Label-preserving perturbations* follow the **invariance tests** of CheckList
  (Ribeiro et al., 2020): surface forms that should not matter must not change
  the prediction.
- *Prediction consistency under paraphrase* as a model-quality probe follows
  Elazar et al. (2021).
- *Metamorphic relations for LLM hallucination detection* follow **MetaQA**
  (Yang et al., FSE 2025), which detects hallucinations via paraphrase-based
  metamorphic relations without external resources.
- *Abstaining when the retrieved evidence does not support an answer* is the
  behaviour SQuAD 2.0 was built to measure (Rajpurkar, Jia & Liang, 2018); the
  `distractor` probe is that test applied post-hoc to a RAG pipeline.
- *Reading a surviving label as the model preferring its parametric knowledge
  over the context it was given* follows the knowledge-conflict setting of
  Longpre et al. (2021).
- *Re-running an unchanged input to establish a consistency baseline* follows
  sampling-based consistency checking, **SelfCheckGPT** (Manakul, Liusie &
  Gales, 2023).

**References**

- Chen, T. Y., Cheung, S. C., & Yiu, S. M. (1998). Metamorphic Testing: A
  New Approach for Generating Next Test Cases. *Technical Report
  HKUST-CS98-01*, Hong Kong University of Science and Technology.
  <https://arxiv.org/abs/2002.12543>
- Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E.,
  Schütze, H., & Goldberg, Y. (2021). Measuring and Improving Consistency
  in Pretrained Language Models. *Transactions of the Association for
  Computational Linguistics*, 9, 1012–1031.
  <https://aclanthology.org/2021.tacl-1.60/>
- Longpre, S., Perisetla, K., Chen, A., Ramesh, N., DuBois, C., & Singh, S.
  (2021). Entity-Based Knowledge Conflicts in Question Answering.
  *EMNLP 2021*. <https://aclanthology.org/2021.emnlp-main.565/>
- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT:
  Zero-Resource Black-Box Hallucination Detection for Generative Large
  Language Models. *EMNLP 2023*.
  <https://aclanthology.org/2023.emnlp-main.557/>
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don't Know:
  Unanswerable Questions for SQuAD. *ACL 2018*.
  <https://aclanthology.org/P18-2124/>
- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond
  Accuracy: Behavioral Testing of NLP Models with CheckList. *ACL 2020*.
  <https://aclanthology.org/2020.acl-main.442/>
- Segura, S., Fraser, G., Sanchez, A. B., & Ruiz-Cortés, A. (2016). A
  Survey on Metamorphic Testing. *IEEE Transactions on Software
  Engineering*, 42(9), 805–824. <https://doi.org/10.1109/TSE.2016.2532875>
- Yang, B., et al. (2025). Hallucination Detection in Large Language Models
  with Metamorphic Relations (MetaQA). *FSE 2025*.
  <https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/48/Hallucination-Detection-in-Large-Language-Models-with-Metamorphic-Relations>

**Output files**, inside the evaluated run's snapshot
(`<run_dir>/metamorphic/`):

- `variants.jsonl` — one row per variant (resumable audit trail):
  `org, source_type, qid, category, variant, variant_kind
  ("control" | "paraphrase" | "ablation" | "distractor"), variant_idx,
  original_label`, plus on success `question, answer, abstain, weights,
  reasoning, label, label_matches_original`, or on failure an `error` field
  (`"paraphrase_failed" | "paraphrase_infidelity" | "ablation_no_remainder" |
  "distractor_relevant" | "distractor_empty" | "chunks_not_found"`). Per-probe
  extras: paraphrases carry a `fidelity` block (`ok, reason, attempts,
  divergence, missing_numbers, missing_entities, min_cosine`) plus
  `source_context` and `context` — the originals and the rewrites, so a flagged
  item can be audited without refetching; ablations carry `ablation_basis` and
  `ablation_removed_id`; distractors carry `distractor_category`,
  `distractor_grounding` and `distractor_ids`.
- `stability.json` — `{"summary": {...}, "per_item": [...]}`; each per-item
  record carries `org, source_type, qid, category, original_label,
  control_label, control_flipped, n_paraphrases, n_paraphrases_ok,
  n_paraphrases_rejected, label_stability, unstable, paraphrase_divergence,
  paraphrase_cosine, ablation_label, ablation_basis, label_survived_ablation,
  distractor_label, distractor_category, distractor_committed, prior_keyed`
  (+ `grounding_bucket` when present).

### 9.4 Embedding agreement (`il_rag/embedding_agreement.py`, `scripts/04_run_embedding_agreement.py`)

A **non-LLM second judge**: embed each committed answer and the run's 7
reference answers for its category, rank the references by cosine similarity,
and check whether the nearest one's logic matches the LLM matcher's top logic.
Deterministic. *Interpretation:* absolute cosine values are not meaningful (e5
compresses them into a narrow band); only the ranking and the agreement rate
are. Low agreement is a known property of whole-answer embeddings (topical
vocabulary dominates institutional stance), not evidence the matcher is wrong.

### 9.5 Quote provenance & paraphrase grounding (`il_rag/quote_provenance.py`, `scripts/06_run_quote_provenance.py`)

§9.2 answers one question with one bit: is this span verbatim in the sources?
Everything that is not a verbatim hit collapses into a single ❌, which
conflates four different failures — a **copy that drifted** (a curly quote, an
em-dash, an elided `…`), a **faithful paraphrase**, a **figure of speech**
(scare quotes, terms of art, hypotheticals — quotation marks that never claimed
anything about a source), and an actual **fabrication**. Only the last is a
hallucination. This check separates them, and then asks a second question §9.2
cannot ask at all:

```
PROVENANCE   does this TEXT exist in the sources?     (stages A-C, no LLM)
VERACITY     is what it ASSERTS supported by them?    (stage D, one LLM call)
```

These are independent, so the verdict is a 2×2 rather than a line. Its most
useful cell is `misquote_but_true`: the model manufactured a quotation, but the
proposition underneath it does hold up against the evidence.

This is a **post-hoc** stage over a saved run, like §9.3 and §9.4:
`Retriever.get_by_ids` replays the exact evidence a row was answered from, so
nothing in the answering path changes and thresholds can be retuned without
re-running the profile pass. It reads §9.2's `quotes` and each row's `answer`,
and **never rewrites §9.2's verdict**.

**Stage A — extraction.** Candidates come from the model's structured `quotes`
entries *and* from quotation marks in the answer prose. The prose spans matter
for two reasons: §9.2 never audits them, and they are where the figures of
speech live. Straight single quotes are deliberately not matched (apostrophes
make them unparseable without a tokenizer); spans under
`QUOTE_MIN_SPAN_TOKENS` content tokens are dropped, since one- and two-word
quotations are overwhelmingly terms of art. A prose span duplicating a
structured entry is counted once.

**Stage B — intent triage** (no LLM). Deterministic cue rules over the ~70
characters before the opening quote, in a fixed precedence:

```
counterfactual  ("a critic might say ...")   -> hypothetical
reporting verb  ("the charter states ...")   -> attributive
mention         ("so-called ...")            -> scare_quote
example         ("for example, ...")         -> hypothetical
shape           (< 4 content tokens, unpunctuated) -> term_of_art
default                                      -> attributive, low confidence
```

The precedence is load-bearing: `might say` *contains* a reporting verb but
attributes nothing, so the counterfactual rule must be tested first. The rule
that fired is stored with the label, so every classification is inspectable
rather than a black box. An unmatched span defaults to **attributive at low
confidence** — wrongly auditing a scare quote is a false alarm a reviewer
dismisses; wrongly excusing a fabricated quotation hides exactly what the check
exists to find. Only attributive spans are graded.

**Stage C — the provenance ladder** (no LLM). Cheapest-first; the first tier to
clear its bar wins:

```
exact          norm(s) ⊑ norm(c)                                for some chunk c
near_verbatim  strip(norm(s)) ⊑ strip(norm(c))
               ∨ elided fragments occur in order
               ∨ max_w ratio(s, w) ≥ τ_near
paraphrase     overlap(s, c) ≥ τ_lex  ∨  max_w cos(s, w) ≥ τ_cos
unsupported    nothing cleared a bar
```

where `strip` additionally removes punctuation (the one normalization §9.2
deliberately refuses — it is what makes a smart-quote artifact stop looking
like a rewrite), `ratio` is a character-similarity ratio over sliding windows,
`overlap` is §9.1's `lexical_overlap`, and `w` ranges over 2-sentence windows
of a chunk. The `exact` tier is **bit-identical to §9.2's predicate**, so every
span §9.2 verifies lands in `exact`: this check only adds resolution below that
line, never reinterprets above it.

Ordering an elided quote's fragments is part of the check because `"A ... B"`
claims that A precedes B in the source. Lexical overlap is the primary
paraphrase signal rather than cosine, for §9.1's reason: e5 compresses cosine
into a narrow high band. Measured on this stack against one passage, a faithful
reword scored **0.849** while a wholly unrelated claim scored **0.807** — 0.04
apart, far too tight to gate on, so `τ_cos` is set above both and fires only on
near-identity.

**Stage D — veracity** (LLM, flagged spans only). Spans reaching `paraphrase`
or `unsupported` get one entailment call returning strict JSON:

```
{"support": "supported|partial|contradicted|not_addressed",
 "evidence_sentence": "...", "grounded_fragment": "...", "reason": "..."}
```

The evidence window widens with the tier, because the question changes. A
**paraphrase** is judged against the passage it aligned to — *did the model
reword this faithfully?* An **unsupported** span aligned to nothing, so it is
judged against the row's **entire** retrieved set — *the text is not there, but
is the claim?* That widening is what makes `misquote_but_true` detectable at
all. The adjudicator never sees the institutional-logics taxonomy, the same
separation §9.2 and `rag_qa.py` insist on. A run whose quotes are all verbatim
triggers **zero** generation calls and zero embedding calls.

**Stage D′ — the verdict.** Derived by a pure function of
`(tier, support, intent)` — no LLM in the derivation:

| | content supported | content not supported |
|---|---|---|
| **text in sources** | `attributed` | `misattributed` |
| **text not in sources** | `paraphrase_grounded` · `misquote_but_true` | `fabricated` |

The bottom-left cell subdivides by how far the text drifted:
`paraphrase_grounded` (the model reworded a real passage) is a much milder
failure than `misquote_but_true` (the model manufactured a quotation whose
content happens to hold), and collapsing them would hide that difference.
`misattributed` requires entailment-checking spans that *did* match, which is
opt-in (`--adjudicate-verbatim`) because it costs one call per verified span;
by default a located span needs no adjudication and is `attributed`.

**Row verdict.** Over a row's attributive spans `A`:

```
quotes_grounded    = |A| > 0 ∧ ∀ s ∈ A : verdict(s) ∈ {attributed, paraphrase_grounded}
quotes_fabricated  = ∃ s ∈ A : verdict(s) = fabricated
quotes_misquoted   = ∃ s ∈ A : verdict(s) ∈ {misquote_but_true, misattributed}
```

`quotes_grounded` carries the same `|A| > 0` guard as §9.2's
`quotes_verified`, and for the same reason: a conjunction over an empty set is
vacuously true, and a row that cited nothing has grounded nothing. Fabricated
and misquoted are not mutually exclusive across a multi-span row — a row can
both invent one span and half-rescue another.

**Edge cases.** A row whose `retrieved_ids` are no longer in the index (e.g.
after a `--fresh` reingest) cannot be audited at all; it is counted in
`n_rows_without_evidence` and skipped rather than silently scored zero. An
unrecognized or missing support label degrades to `not_addressed` — the neutral
option — so a malformed reply can never manufacture a `supported` or a
`contradicted` verdict. Adjudicator JSON that does not parse retries once at a
doubled budget (1024 → 2048), then degrades to `not_addressed`.

**Design rationale.** *§9.2 is read, never rewritten*, so runs from before this
check stay directly comparable with runs after it and both numbers can be
reported side by side. *Post-hoc* means being wrong is cheap: thresholds retune
for the cost of the flagged spans alone. *A conservative tier boundary is
nearly free*, because a span that misses the paraphrase bar is not lost — it
falls through to `unsupported` and is then adjudicated against the full
evidence set. The threshold decides which label a grounded span earns, never
whether it gets checked. Unlike §9.2, this check necessarily *has* tunable
constants: "close enough to be a copy" and "close enough to be a paraphrase"
are matters of degree.

**Limitations.** The evidence is scoped by `(org, source_type)` and this
pipeline has **no world-knowledge oracle** by design — the answerer only ever
sees the corpus. So `unsupported` means *unsupported by this lab's scoped
corpus*, **not** *false in the world*. A claim can be true and still land there
because the corpus is silent on it, which is precisely why `not_addressed` is a
separate label from `contradicted`: **only `contradicted` is evidence against a
span**. Beyond that, the intent rules are English cue patterns, not a parser,
and will miss unusual phrasings (they fail toward auditing, so the cost is
false alarms rather than missed fabrications); and the entailment judge is the
same model family being audited, so it is a consistency check, not an
independent oracle. The paraphrase thresholds still want calibrating against a
real corpus run.

**Basis in the literature.**

- *Separating attribution from correctness* — that a span can be attributable
  without being correct, and correct without being attributable — is the
  distinction formalized as **AIS** by Rashkin et al. (2023) and
  operationalized for QA by Bohnet et al. (2022) (both cited in §9.2). The 2×2
  above is that distinction made into a verdict.
- *Scoring a claim against retrieved evidence rather than against its surface
  string* follows **FActScore** (Min et al., 2023), which decomposes generated
  text into atomic claims and scores each for support against a knowledge
  source.
- *Verified quoting* remains **GopherCite** (Menick et al., 2022) and
  *citation-quality evaluation* remains **ALCE** (Gao et al., 2023), both cited
  in §9.2; this check grades the failures those methods reject wholesale.

**References** (see §9.2 for Bohnet et al., Gao et al., Menick et al., and
Rashkin et al.)

- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
  *Text Summarization Branches Out*. <https://aclanthology.org/W04-1013/>
  (the lexical-overlap signal reused from §9.1)
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M.,
  Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained Atomic
  Evaluation of Factual Precision in Long Form Text Generation. *EMNLP 2023*.
  <https://aclanthology.org/2023.emnlp-main.741/>

**Output files**, inside the evaluated run's snapshot
(`<run_dir>/quote_provenance/`):

- `spans.jsonl` — one record per candidate span: `org, source_type, qid,
  category, quote, source ("quotes_field" | "answer_prose"), excerpt,
  verbatim_verified` (§9.2's bit for the same span, carried through
  unchanged), `intent, intent_rule, intent_confidence, match_tier, match_rule,
  match_score, best_chunk_id, best_span, support, evidence_sentence,
  grounded_fragment, support_reason, verdict`.
- `summary.json` — `overall` / `by_category` / `by_org_source` slices, each
  with tier and verdict histograms plus `paraphrase_rescue_rate`,
  `true_fabrication_rate`, `misquote_but_true_rate`, `contradicted_rate`,
  `non_attributive_rate`; also an `intents` histogram, per-row verdicts, and
  `n_rows_without_evidence`.

**Config** (`il_rag/config.py`): `QUOTE_NEAR_VERBATIM_THRESHOLD`,
`QUOTE_PARAPHRASE_LEX_THRESHOLD`, `QUOTE_PARAPHRASE_COS_THRESHOLD`,
`QUOTE_MIN_SPAN_TOKENS`.

---

### 9.6 Topic-keyword semantic matching (`il_rag/topic_keywords.py`, `scripts/13_run_topic_keywords.py`)

The odd one out in this section: it lives on the **Topics** tab, not the
Hallucination tab, because what it measures is a property of the *inductive*
layer (§ the topic model) rather than of a quote. It is included here because it
is the same kind of instrument — an opt-in, post-hoc, graded check over a saved
run.

**The question.** BERTopic names each topic by its ten most distinctive
c-TF-IDF terms. Those terms were, until this stage, descriptive labels that
nothing scored. This asks: when the pipeline answered a question from evidence
belonging to topic *k*, how much of topic *k*'s vocabulary actually reached the
answer — verbatim, as an inflection, as a semantic neighbour, or not at all?

**Relation to §9.2/§9.5, and to the keyword judge.** §9.5 is to §9.2 what this
is to `il_rag/keyword_agreement.py`: a yes/no predicate replaced by a graded
ladder. `keyword_agreement.py`'s own docstring names the limitation —

> it cannot see synonymy — an answer saying "government" earns no credit for a
> reference that says "state" — so its miss rate is structurally high

— and this closes it. It does **not** replace that judge, and the two are not
interchangeable: `keyword_agreement` matches *per-logic reference* keywords to
classify an answer; this matches *topic c-TF-IDF* keywords to measure vocabulary
survival. Different keywords, different target, different question.

**The ladder** (`locate_span`'s shape — cheapest first, first bar cleared wins):

```
exact          κ occurs as a whole token, or (bigram) as two ADJACENT tokens
                                                                    -> 1.0
morphological  some answer word shares κ's stem: shared leading prefix
               >= p* AND difflib ratio >= τ_morph                   -> ratio
semantic       some answer word w* has π(cos(κ, w*)) >= τ_sem       -> min(π, σ_max)
absent         nothing cleared a bar                                -> 0
```

Every verdict carries `{tier, score, rule, matched, cosine, near_miss}`.
`matched` is the answer word that produced the match — the analogue of §9.5's
`best_span`, and the reason a reviewer can read the evidence rather than take
the tier on faith.

Three details are load-bearing:

- **`absent` scores exactly 0**, not its near-miss percentile. A keyword that
  cleared no bar must contribute nothing to retention, or `dropped_share` and
  `retention` tell contradictory stories. How close it came is kept separately
  in `near_miss`, which is diagnostic only: it is what distinguishes "nowhere
  near" from "a hair under the bar" when retuning τ_sem.
- **Bigram adjacency.** "export controls" is `exact` only if it occurs
  adjacently, because adjacency is what makes it the phrase. Otherwise each
  part is scored and the phrase takes the **minimum**, capped at
  `morphological`. Minimum, not mean: the phrase is present only to the extent
  that *both* concepts are, and a mean would let "controls" alone carry it.
- **The exact rung tokenizes UNFILTERED** (`_words`, not
  `grounding.content_tokens`). Stopwords and two-character tokens must stay
  matchable, or a legitimate topic keyword like `ai` would be structurally
  unmatchable. The semantic rung *does* use `content_tokens` — there a cosine
  against "the" is noise.

**Why the semantic rung reports a percentile, not a cosine.** This is the
design's one hard constraint, and it follows directly from the measurement
recorded at `QUOTE_PARAPHRASE_COS_THRESHOLD` (§9.5): e5 put a faithful reword at
0.849 and a wholly unrelated claim at 0.807 — 0.04 apart. Single *words* are
worse; this corpus's whole vocabulary sits in a band roughly 0.74–0.82 wide. No
absolute cutoff discriminates in that band, and a number from it cannot be shown
to a reader as a "% match" without misleading them.

So the score is the cosine's **percentile in the distribution of cosines between
random pairs of this corpus's own vocabulary**:

```
π(x) = |{ g ∈ G : g ≤ x }| / |G|     G = { cos(u,v) : {u,v} ⊂ V, u ≠ v }
```

That is also what makes the bar selective rather than a rubber stamp. The corpus
is entirely about AI labs, so `manager`/`sales` genuinely *is* fairly close in
absolute cosine — but it is a *typical* pair here and lands near the background
median, while `manager`/`hierarchy` lands in the tail. Measured on a synthetic
fixture the difference is 95th percentile against 18th, from raw cosines that
differ by less than the eye would trust.

`G` is computed **exhaustively, not sampled**: all C(|V|, 2) pairs, swept in row
blocks of the cosine matrix and accumulated into a fixed histogram, so ~8M pairs
cost bounded memory. Computing every pair removes a random seed from the design
entirely — same corpus and same model give the same file, byte for byte. Only
`CALIBRATION_GRID_POINTS` quantiles are stored, not the pairs.

**When the calibration is missing the semantic rung is disabled outright.**
There is deliberately no fallback to raw cosine: an 0.80 rendered as "80%
match" between two unrelated words would misrepresent the entire measure. The
run still completes on the two lexical rungs and records
`summary["calibrated"] = false`.

**Rollup.** A row's keywords are the union over the topics of its retrieved
chunks (outlier topic −1 excluded; ids missing from the map skipped, as in
§topics). Retention is the mean score; the per-topic figure weights each
answered row equally, so a heavily-retrieved topic cannot dominate its own
average:

```
V_i = (1/|K(i)|) Σ_{κ ∈ K(i)} S(κ, a_i)
V_k = (1/|A_k|) Σ_{i ∈ A_k} (1/|K_k|) Σ_{κ ∈ K_k} S(κ, a_i)
L_i = V_i − σ_exact_i          "semantic lift" — the headline
```

`L_i` is this check's `paraphrase_rescue_rate`: exactly the part of retention an
exact-match judge is blind to.

**Two reference arms**, because a retention figure with no referent is not a
finding:

- **Ceiling `C_i`** — the same keywords, exact rung only, against the retrieved
  *chunks*. Keywords are derived *from* those chunks by c-TF-IDF, so it runs
  high by construction. This is the circularity objection stated as a displayed
  number rather than caveated away. Free (no embeddings); degrades to `null`
  when Chroma is absent.
- **Floor `N_i`** — the full ladder against `KEYWORD_NULL_DRAWS` topics the row
  never retrieved, seeded from the row's own identifiers so the draw is stable
  regardless of row order, resumption, or parallelism. Same role the metamorphic
  control arm (§9.3) plays for flip rates.

Read as `N ≤ V ≤ C`. If `V` sits at `N`, the answers carry no more of the
topic's vocabulary than a random topic's — and that is the finding.

**Cost.** Word vectors are cached to disk, append-only and unit-normalized
before a float16 cast (cosine error under 1e-3, an order of magnitude below the
grid's resolution). A first run embeds a few hundred new words; a rerun embeds
zero. Cosines are never cached — they are a matmul over cached vectors, so
retuning a threshold costs nothing. `--no-embeddings` disables the semantic rung
and makes the whole ladder pure computation, which is what the offline tests
use.

**Output files.** Corpus-level, built locally and shipped like `data/topics/`:

- `data/lexicon/word_vectors.npz` — `words` (fixed-width `U32`, so `np.load`
  needs no pickle) and `vectors` (float16 unit vectors).
- `data/lexicon/calibration.json` — the quantile grid, the vocabulary and its
  document frequencies (so the neighborhood explorer needs no Chroma), and full
  provenance: `built_at`, `embedding_model`, `vocab_size`, `n_pairs`, and the
  cosine band.

Run-level, inside `<run_dir>/topic_keywords/`:

- `rows.jsonl` — per answered row: `topics, n_keywords, skipped, retention,
  verbatim_share, morph_share, semantic_share, dropped_share, semantic_lift,
  verbatim_ceiling, retention_null, null_topics`, plus an **uncapped**
  `keywords` list of per-keyword verdicts. Uncapped because that list *is* the
  audit trail (`keyword_agreement` caps its matched words at 12 because there
  they are a by-product; here they are the product).
- `terms.jsonl` — per (topic, keyword): `retention`, the tier histogram,
  `verbatim_ceiling`, and the three most frequent matched words.
- `summary.json` — `overall` / `by_topic` / `by_category` / `by_org_source`
  slices, the calibration provenance, the thresholds in force, and the embedding
  cost actually incurred.

**Config** (`il_rag/config.py`): `KEYWORD_MORPH_MIN_PREFIX`,
`KEYWORD_MORPH_MIN_RATIO`, `KEYWORD_MIN_WORD_CHARS`,
`KEYWORD_SEMANTIC_MIN_PERCENTILE`, `KEYWORD_SEMANTIC_MAX_SCORE`,
`KEYWORD_MAX_CANDIDATES`, `CALIBRATION_VOCAB_SIZE`, `CALIBRATION_MIN_DF`,
`CALIBRATION_BINS`, `CALIBRATION_GRID_POINTS`, `KEYWORD_NULL_DRAWS`,
`KEYWORD_NULL_SEED`.

**Known limitations**, restated verbatim in the GUI expander:

1. **Circularity** — topic keywords are derived *from* the corpus chunks, so
   scoring them against those chunks is near-tautological. That is what the
   ceiling arm measures. The non-trivial comparison is against the answers.
2. **These vectors capture topical relatedness, not synonymy.** `manager` and
   `hierarchy` are related, not synonyms — the motivating example is itself a
   relatedness judgment, and that is the honest claim this rung supports.
   Antonyms embed close (`permit`/`prohibit`), so a high semantic score can mean
   the answer discusses the concept *and takes the opposite position*. Read the
   matched word, never the score alone.
3. **Percentiles are readable, not absolute** — a rank within *this* corpus
   under *this* model. Both are recorded in `calibration.json`.
4. **The keyword sets are vectorizer artifacts**: ten c-TF-IDF terms under one
   particular `min_df` / `ngram_range` / `n_keywords`. Changing any of those
   changes every retention figure.
5. **The morphological rung is a prefix heuristic, not a stemmer.** It
   over-fires on shared-prefix pairs (`policy`/`police` at ratio 0.833) and
   under-fires on suppletive forms (`good`/`better`). It never fabricates a
   claim: the matched surface form is always stored and shown.
6. **Per-row retention is noisy** — ten keywords against one answer. Read the
   per-topic rollup.
7. **Attribution inherits the cross-tab's imprecision**: a row's topics are the
   topics of *all* its retrieved chunks, including ones the answer legitimately
   ignored (the `1/k` argument in `topics.py` applies here too).
8. **Every keyword counts equally** — no IDF re-weighting within a topic.
   c-TF-IDF already filtered for distinctiveness once; weighting again would
   double-count it.

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

### 12.1 Source links in the audit trail (`il_rag/pdf_sources.py`)

A supporting quote is displayed as `[excerpt 3] "…"`. The number alone is not
auditable — checking it meant opening the evidence list, counting down it, and
finding the file by hand. When `IL_PROFILER_DATASET_ROOT` points at the raw
dataset, the number becomes a link to the PDF.

**Resolution.** A chunk's metadata carries a bare basename, never a path
(`ingest.py` reads it from the corpus text file's `FILE:` header), so the join
is by normalized basename — lowercase, alphanumerics only — over one walk of
the dataset tree. Across the 98 published documents that resolves all of them:
97 match exactly and one needs the normalization (the corpus writes
`Inside Google_s AGI Strategy.pdf` where the disk has an apostrophe). Two
basenames occur twice in the tree; preferring a path under the chunk's own
`org` decides both, and is exhaustively correct here rather than a heuristic,
since every document does sit under its own lab.

**Delivery.** Streamlit only serves files beneath `<app dir>/static`, and its
path check resolves symlinks before testing containment, so a junction into the
dataset is rejected — the bytes must be reachable through a real path. Cited
PDFs are therefore hard-linked (falling back to a copy) into `static/` on first
reference, so only documents actually cited are ever published.

**What the link claims.** It names the document the answer *cited*, which is
what `[excerpt N]` says. It is not evidence the span is there: §9.2 verifies a
quote against **any** retrieved chunk, deliberately, so a ✅ beside a link
certifies "this text is in the sources", not "in this PDF". Where the span is
really in a different retrieved document, the row names and links that one too
— the same normalized-substring test §9.2 uses, over the row's own ≤5 chunks,
so nothing expensive runs while rendering.

**Page anchors.** `scripts/09_build_pdf_pages.py` writes
`data/pdf_pages.json` = `{chunk_id: page}`. It reconstructs ids by re-running
ingestion's own splitting and chunking over `pdf_corpus.txt`, so no vector
index is needed, then aligns each PDF's pages against the document body:
proportional cumulative page length as a first estimate, refined by searching
for real page text nearby. Anchoring on a page's *opening* text alone fails
here — pypdf emits the printed page number first, where the corpus converter
did not — which is why the estimate exists. Two refusals keep it honest: a
document whose PDF yields no text at all (21 are recorded in the dataset's own
manifests as `method: ocr` — web pages saved as images) and one where fewer
than 60% of pages could be confirmed by real text are both skipped entirely,
and their links open at page 1. Coverage is 89% of chunks; on a 335-chunk
audit, 99% of the anchors placed land on the correct page.

**Safety.** Every unresolvable case falls back to the exact plain text the UI
showed before, so the feature is invisible without a dataset. It is off in the
cloud by two independent mechanisms; see DEPLOY.md.

### 12.2 Press records (`il_rag/article_pdfs.py`, `scripts/10_build_article_pdfs.py`)

Third-party chunks name only `O1.RTF` — a 45 MB Nexis export holding 500
records — so a file-level link identifies nothing. Stage 10 splits each bundle
into one PDF per record ahead of time, and the citation resolves through the map
it leaves behind (`data/article_sources.json`).

**Which record a chunk belongs to.** `ARTICLE_SPLIT_RE` lists seven markers but
five can never fire: `Title:`/`Headline:`/`Byline:`/`Publication-Date:` are each
followed by `:` then U+00A0, and the pattern ends in `\b`, which needs a word
boundary that is not there (`Byline:` occurs 140 times in O1.RTF and causes zero
splits). The accident is load-bearing — it means every record splits into
exactly two fragments in strict alternation: an identity block (headline,
publication, date) and a body. The body is what gets indexed, so **a chunk's
identity lives in the preceding fragment**, which is why these citations looked
anonymous. Articles are recovered by walking the fragment list, not by parity:
13 identity fragments in A1 survive the 200-character filter and parity would
mis-assign them. Two shapes are special — a Nexis job header at fragment 0
(whose own chunks are a search-results index belonging to no article, and stay
unlinked) and, in the Anthropic exports, no job header at all.

**Identity parsing.** The date line must match *in full*: News Bites headlines
begin with a date (`May 06, 2026: EVE Online and…`), and prefix-matching there
mistakes the headline for the date and shifts every field. Measured, that
distinction is worth 93.45% → 99.67%, and five date-format variants close the
rest. Headline is the first line; publication is everything between it and the
date, which handles two-line publisher blocks.

**Rendering.** A PDF core font under cp1252, embedding no font at all. Measured
across all six dumps, after a six-character normalization only 67 characters in
15.8 million fall outside cp1252 (runes, three CJK ideographs, arrows) — and
those are folded or replaced *and counted*. Embedding a subsetted Unicode TTF
would add 12-25 KB to each of ~3,000 files; core fonts keep the whole corpus at
21 MB. Page anchors are exact rather than estimated: pages are recorded as each
visual line is drawn, and a chunk is located in *its own source fragment* (not
in the rendered text, of which it is only a line-subsequence), so the match is
exact. Measured 639/649 sampled chunks on the stated page, zero wrong.

**What the link claims.** It opens *our rendering* of the press record the index
was built from — not the publisher's page. The exports contain no URL field
anywhere, so there is nothing else to point at, and the UI says so.

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
  metamorphic.py         (check) control / paraphrase / ablation / distractor
  embedding_agreement.py (check) non-LLM second judge
  quote_provenance.py    (check) graded quote provenance x veracity
  topics.py              (local) inductive BERTopic layer + topic x logic cross-tab
  keyword_agreement.py   (check) exact-match lexical third judge
  topic_keywords.py      (check) graded topic-keyword distance ladder (§9.6)
  pdf_sources.py         chunk → source PDF, for the audit trail's links (§12.1)
  article_pdfs.py        (local) RTF press dumps → one PDF per record (§12.2)
  bootstrap_ci.py        confidence intervals over the profiles
  json_utils.py, llm.py  shared JSON extraction; Together chat/embed wrappers
scripts/
  01_ingest.py  02_run_profiles.py  03_run_metamorphic_eval.py
  04_run_embedding_agreement.py  05_run_bootstrap_ci.py
  06_run_quote_provenance.py  07_run_topics.py  08_run_replicates.py
  09_build_pdf_pages.py  10_build_article_pdfs.py  11_bundle_cloud_sources.py
  12_run_keyword_agreement.py  13_run_topic_keywords.py
data/
  chroma/                the vector index
  topics/                (local, shipped) fitted topic model: topic_info, chunk_topics
  lexicon/               (local, shipped) word vectors + word-pair calibration (§9.6)
  profiles/runs/         immutable per-run snapshots and every check's output
app.py                   Streamlit GUI (Run / Results / Audit / Hallucination /
                         Topics / Analyse a document / Compare runs)
tests/                   offline unit tests
Dockerfile, fly.toml, DEPLOY.md   deployment
```

---

## 16. Known limitations

- **Small instrument.** ~27 questions per profile → wide confidence intervals.
  Dominant-logic rankings are trustworthy; exact percentages are not.
- **Self-referential checks.** The metamorphic paraphraser and the answer model
  are the same model. Fidelity is no longer taken on trust — §9.3's three gates
  verify in code that facts, wording and topic survived a rewrite, and rewrites
  that fail are discarded rather than scored — but shifts of emphasis subtle
  enough to clear all three gates would still go unnoticed, so hand-reviewing a
  few flagged variants remains worthwhile.
- **Embedding agreement is weak here** by design of the medium — whole-answer
  embeddings track topic more than institutional stance. It is a triangulation
  signal, not ground truth.
- **Third-party coverage is uneven.** Press rarely discusses internal authority
  or informal control, so some categories legitimately abstain on the
  `thirdparty` side — a finding, not a bug.
- **LLM grading is the classifier.** There is no gold-labeled ground truth; the
  reference answers *are* the standard, so results are only as good as they are.
- **"Unsupported" is not "false."** Every check reasons only over the
  `(lab, source_type)`-scoped corpus; there is no world-knowledge oracle here by
  design. So §9.5's `unsupported` means *this corpus does not support it*, not
  *it is untrue* — a true claim the corpus is simply silent on lands there. Only
  the `contradicted` label is evidence against a span.
- **Word embeddings measure relatedness, not synonymy.** §9.6's semantic rung
  can say that *manager* and *hierarchy* sit close in this corpus; it cannot say
  they mean the same thing, and antonyms sit close too. Its percentiles are also
  a rank within *this* corpus under *this* embedding model — refit either and
  every number moves. Read the matched word, never the score alone.
- **The topic layer's keywords describe, they do not define.** §9.6 scores an
  answer against c-TF-IDF terms that were themselves derived from the chunks
  being answered from, which is why it reports a verbatim ceiling and a null
  floor alongside the measurement rather than the measurement alone.
- **Copyright.** The third-party corpus is licensed news content — keep any
  deployment private/gated.
```
