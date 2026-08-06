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

The remaining check:

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
  metamorphic.py         (check) control / paraphrase / ablation / distractor
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
- **Copyright.** The third-party corpus is licensed news content — keep any
  deployment private/gated.
```
