"""Embedding-agreement check: a second, non-LLM judge over a saved run.

For every committed (non-abstained) row of a run, embed the RAG answer and the
seven reference answers for that row's category, and rank the references by
cosine similarity to the answer. The nearest reference's logic is the
"embedding verdict"; comparing it with the LLM matcher's top-weighted logic
gives an agreement rate — independent, deterministic evidence that the graded
matcher's classifications track the reference space rather than judge whim.

Design notes:
  - References come from the RUN'S OWN questionnaire.json snapshot, not the
    current questionnaire.py — so the check compares against the exact
    reference wording that produced the run, even after later rewrites.
  - Both sides are embedded as raw text (no e5 query prefix): answer-vs-
    reference is a symmetric text-similarity comparison, unlike retrieval's
    asymmetric query-vs-passage case.
  - e5 compresses cosine into a narrow high band, so ABSOLUTE similarity values
    are not meaningful; only the RANKING among the 7 references is used. The
    margin (top1 - top2) is reported as a confidence signal.
  - Abstained rows are skipped: there is no committed answer to compare.
  - Cost: one embedding call per committed row plus 63 reference embeddings,
    batched — a few hundred texts total, fractions of a cent, fully
    deterministic. Rerunning simply recomputes and overwrites.

Outputs (in <run>/embedding_agreement/):
  similarities.jsonl  per-row: 7 sims, nearest logic, matcher top, agree, margin
  summary.json        agreement rate overall / per category / per (org, source)
"""
import json
import math

from tqdm import tqdm

from . import runs
from .llm import embed

OUT_DIR_NAME = "embedding_agreement"
EMBED_BATCH = 32

# e5's context window is 512 tokens; RAG answers can run ~4,000 chars and get
# rejected (400: maximum context length). Answers are truncated to this many
# characters before embedding. Deliberately acceptable: the answer template
# demands "conclusion first", so the head of the answer carries the
# classification-relevant content; the tail is supporting quotes/citations.
MAX_EMBED_CHARS = 1400


def _truncate(text: str, limit: int = MAX_EMBED_CHARS) -> str:
    """Head-truncate on a word boundary so no embedding call can overflow."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > limit // 2 else limit]


# ---------------------------------------------------------------------------
# Small vector helpers (avoid a numpy dependency in the hot path; the vectors
# are short lists and the row count is small).
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _embed_batched(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="embed", unit="batch"):
        out.extend(embed(texts[i:i + EMBED_BATCH]))
    return out


def _matcher_top(weights: dict) -> tuple[str, float]:
    """The LLM matcher's top-weighted logic for a row."""
    logic, w = max(weights.items(), key=lambda kv: kv[1])
    return logic, float(w)


def soft_weights(sims: dict[str, float]) -> dict[str, float]:
    """Turn the 7 raw cosines into proportional 'closeness shares'.

    Min-shifted normalization: subtract the farthest reference's similarity,
    then normalize to sum 1. Chosen over softmax because it is parameter-free
    and scale-invariant — e5 compresses all cosines into a narrow high band
    (~0.78-0.87), so what carries information is the relative spread WITHIN a
    row, not the absolute values. The farthest logic gets exactly 0 by
    construction; a row with no spread at all degrades to uniform 1/7.
    """
    m = min(sims.values())
    shifted = {logic: s - m for logic, s in sims.items()}
    total = sum(shifted.values())
    if total <= 0.0:
        return {logic: 1.0 / len(sims) for logic in sims}
    return {logic: v / total for logic, v in shifted.items()}


def distribution_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    """Overlap coefficient of two weight distributions: sum of per-logic mins.

    1.0 = identical distributions; 0.0 = disjoint support. Continuous
    counterpart of the binary argmax agreement.
    """
    return sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b))


def _load_rows(run_id: str) -> list[dict]:
    path = runs.run_paths(run_id)["per_question"]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def run_embedding_agreement(run_id: str | None = None) -> dict:
    """Compute embedding agreement for a run. Returns the summary dict."""
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run profiles first")

    snapshot = json.loads(
        runs.run_paths(run_id)["questionnaire"].read_text(encoding="utf-8"))
    logics: list[str] = snapshot["logics"]
    questionnaire: dict = snapshot["questionnaire"]

    rows = [r for r in _load_rows(run_id) if not r.get("abstain")]
    if not rows:
        raise SystemExit(f"run {run_id} has no committed rows to check")

    # Resolve the reference set per (category, variant), applying any
    # reference_overrides — the same resolution the matcher uses, so both
    # judges grade against identical references. Snapshot caveat: JSON turns
    # the override keys {2: ...} into {"2": ...}, so look up both forms.
    ref_text_for: dict[tuple, dict[str, str]] = {}
    for category, block in questionnaire.items():
        base = block["reference_answers"]
        overrides = block.get("reference_overrides", {})
        for variant in (1, 2, 3):
            refs = dict(base)
            refs.update(overrides.get(variant) or overrides.get(str(variant)) or {})
            ref_text_for[(category, variant)] = refs

    # Embed each distinct reference text once, then every committed answer.
    unique_refs = sorted({t for refs in ref_text_for.values()
                          for t in refs.values()})
    print(f"embedding {len(unique_refs)} distinct references + {len(rows)} "
          f"answers for run {run_id}")
    ref_vec_by_text = dict(zip(
        unique_refs, _embed_batched([_truncate(t) for t in unique_refs])))
    ans_vecs = _embed_batched([_truncate(r["answer"]) for r in rows])

    out_dir = runs.run_dir(run_id) / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row = []
    for row, avec in zip(rows, ans_vecs):
        cat = row["category"]
        # Old rows always carry a 1-3 variant; fall back to the base set (v1)
        # if a hand-edited row lacks one.
        refs = ref_text_for.get((cat, row.get("variant") or 1),
                                ref_text_for[(cat, 1)])
        sims = {logic: round(_cosine(avec, ref_vec_by_text[refs[logic]]), 4)
                for logic in logics}
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        nearest, top_sim = ranked[0]
        margin = round(top_sim - ranked[1][1], 4)
        m_logic, m_weight = _matcher_top(row["weights"])
        # Graded comparison: closeness SHARES vs the matcher's weight
        # distribution, alongside the binary argmax agreement.
        shares = soft_weights(sims)
        matcher_w = {logic: float(row["weights"].get(logic, 0.0))
                     for logic in logics}
        per_row.append({
            "org": row["org"], "source_type": row["source_type"],
            "qid": row["qid"], "category": cat,
            "similarities": sims,
            "embedding_shares": {logic: round(v, 4) for logic, v in shares.items()},
            "embedding_nearest": nearest,
            "matcher_top": m_logic, "matcher_top_weight": round(m_weight, 4),
            "agree": nearest == m_logic,
            "margin": margin,
            # share of embedding closeness on the matcher's pick (chance = 1/7)
            "share_on_matcher_top": round(shares[m_logic], 4),
            # overlap of the two distributions (1 = identical)
            "overlap": round(distribution_overlap(shares, matcher_w), 4),
            # what overlap a totally uninformative (uniform) embedding judge
            # would score against THIS matcher distribution — the row's baseline
            "overlap_uniform_baseline": round(distribution_overlap(
                {logic: 1.0 / len(logics) for logic in logics}, matcher_w), 4),
        })

    with open(out_dir / "similarities.jsonl", "w", encoding="utf-8") as f:
        for r in per_row:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary ----
    def _rate(items: list[dict]) -> dict:
        n = len(items)
        agree = sum(r["agree"] for r in items)
        return {"n": n, "agree": agree,
                "rate": round(agree / n, 4) if n else None}

    by_category = {c: _rate([r for r in per_row if r["category"] == c])
                   for c in sorted({r["category"] for r in per_row})}
    by_pair = {f'{r["org"]}|{r["source_type"]}': None for r in per_row}
    for key in by_pair:
        org, st = key.split("|")
        by_pair[key] = _rate([r for r in per_row
                              if r["org"] == org and r["source_type"] == st])
    n = len(per_row)
    summary = {
        "run_id": run_id,
        "overall": _rate(per_row),
        "by_category": by_category,
        "by_org_source": by_pair,
        "mean_margin": round(sum(r["margin"] for r in per_row) / n, 4),
        # Graded (ratio-of-closeness) metrics — continuous counterparts of the
        # binary agreement above.
        "mean_share_on_matcher_top": round(
            sum(r["share_on_matcher_top"] for r in per_row) / n, 4),
        "share_chance_baseline": round(1.0 / len(logics), 4),
        "mean_overlap": round(sum(r["overlap"] for r in per_row) / n, 4),
        "mean_overlap_uniform_baseline": round(
            sum(r["overlap_uniform_baseline"] for r in per_row) / n, 4),
        "note": ("Agreement = embedding-nearest reference logic equals the LLM "
                 "matcher's top-weighted logic. Graded metrics compare the "
                 "min-shifted closeness shares against the matcher's weight "
                 "distribution. Absolute cosine values are not interpretable "
                 "with e5; only rankings, margins, and shares are."),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    o = summary["overall"]
    print(f"\n=== Embedding agreement (run {run_id}) ===")
    print(f"overall (binary argmax): {o['agree']}/{o['n']}  ({o['rate']:.1%})")
    print(f"mean closeness share on matcher's pick: "
          f"{summary['mean_share_on_matcher_top']:.3f} "
          f"(chance {summary['share_chance_baseline']:.3f})")
    print(f"mean distribution overlap: {summary['mean_overlap']:.3f} "
          f"(uniform baseline {summary['mean_overlap_uniform_baseline']:.3f})")
    print("by category (binary):")
    for cat, s in by_category.items():
        print(f"  {cat:<24} {s['agree']:>3}/{s['n']:<3} ({s['rate']:.0%})")
    print(f"mean top1-top2 margin: {summary['mean_margin']}")
    print(f"outputs: {out_dir}")
    return summary
