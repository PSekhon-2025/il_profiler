"""Keyword agreement: the transparent third judge, now graded and curated.

The pipeline has three independent graders of the same question — answer
pairs, each blind to the others' reasoning:

  1. the LLM matcher      semantic, evidence-weighing   (the primary instrument)
  2. embedding agreement  distributional similarity     (no LLM)
  3. THIS MODULE          keyword vocabulary            (no LLM)

v1 derived its keywords automatically from each question's seven reference
answers and matched them by bare set intersection. Both halves proved to be
the weak point, and both are replaced here:

  - DERIVED -> CURATED. Distinctiveness-within-one-question let generic words
    through whenever only one reference happened to use them ("made", "fill",
    "need", "first", "place"), so the sets read as noise. The lexicon below is
    hand-curated instead: one list per logic, drawn from Thornton & Ocasio's
    ideal types and the questionnaire's own reference vocabulary, holding only
    words that carry institutional signal on their own.
  - EXACT -> LADDER. Set intersection could not see morphology or synonymy —
    an answer saying "government" earned nothing from a reference saying
    "state". Scoring now reuses topic_keywords' graded ladder, so the two
    keyword features share one matching methodology:

      exact          the keyword occurs as a token / adjacent phrase   -> 1.00
      morphological  an answer word shares its stem                    -> ratio
      semantic       an answer word is closer than the calibrated
                     percentile bar of random corpus word pairs        -> pctile (cap 0.99)
      absent         nothing cleared a bar                             -> 0

The semantic rung needs the local lexicon files (data/lexicon/, built once by
`scripts/13_run_topic_keywords.py calibrate`). Without them the ladder simply
stops at the morphological rung and the whole check stays pure computation —
deterministic, offline, zero API — exactly as v1 was. With them, the only cost
is embedding answer words not already in the append-only word-vector cache; a
rerun costs nothing. There is still no LLM call anywhere in this module.

Honest scope: this stays the most transparent judge, not the most accurate.
Every verdict is still a visible list of matched words — now annotated with
the rung that matched them — so where it dissents from the matcher, the lists
show exactly why. e5 word vectors capture topical relatedness rather than true
synonymy (antonyms embed close), so read the matched word, never the score
alone.

Outputs (in <run>/keyword_agreement/):
  keywords.json   the curated lexicon this run was scored against
  rows.jsonl      per answered row: per-logic scores, matched words, verdicts
  summary.json    binary + graded rates, overall and per slice
"""
from collections import Counter

import json

from . import runs
from .questionnaire import LOGICS
from .topic_keywords import (
    Calibration,
    TIER_ABSENT,
    WordVectors,
    _Ctx,
    _score_one,
)

OUT_DIR_NAME = "keyword_agreement"

# Per-row cap on stored matched words (audit readability, not scoring).
MAX_MATCHED_STORED = 12

# ---------------------------------------------------------------------------
# The lexicon (v2): one hand-curated keyword list per logic.
#
# Sources, in order: Thornton & Ocasio's inter-institutional system (the ideal
# types behind ARCHITECTURE.md §2), filtered through the vocabulary the
# questionnaire's reference answers actually use. Curation rules:
#   - institutional signal only — a word must point at the logic on its own,
#     with no question context ("welfare", not "made"); generic corpus words
#     ("company", "model", "research") are excluded even when frequent.
#   - one surface form per stem: the morphological rung credits inflections
#     ("regulation" covers regulators/regulatory at ~0.8-0.95), so listing
#     each variant would only pad the denominator.
#   - no token appears in two logics' lists — cross-logic credit must come
#     from the answer, never from the lexicon.
#   - phrases are allowed ("peer review"); the ladder scores them adjacent-
#     first, then by their weakest part capped at morphological.
# Known accepted noise, kept because the audit trail exposes it: "leadership"
# (Corporation) morph-matches "leaders" in any context, including religious
# ones; "faith" (Religion) fires on "good faith".
# ---------------------------------------------------------------------------
LOGIC_KEYWORDS: dict[str, list[str]] = {
    "State": [
        "government", "regulation", "legislation", "laws", "legal",
        "policymakers", "ministries", "officials", "oversight", "compliance",
        "bureaucratic", "mandate", "sovereignty", "welfare", "citizens",
        "national security", "public interest",
    ],
    "Profession": [
        "researchers", "scientists", "scientific", "expertise", "engineers",
        "academic", "credentials", "methodology", "rigor", "esteem",
        "reputation", "publications", "benchmarks", "peer review",
        "professional judgment",
    ],
    "Market": [
        "market", "customers", "investors", "competition", "pricing",
        "profit", "revenue", "demand", "valuation", "commercial", "sales",
        "transactions", "monetization", "growth", "market share",
        "shareholder value",
    ],
    "Corporation": [
        "corporate", "firm", "hierarchy", "executives", "management",
        "board", "leadership", "employees", "headcount", "divisions",
        "organizational", "centralized", "procedures", "restructuring",
        "chain of command",
    ],
    "Family": [
        "family", "founder", "loyalty", "kinship", "household", "dynasty",
        "nepotism", "patriarch", "lineage", "favoritism", "paternalistic",
        "inner circle",
    ],
    "Religion": [
        "sacred", "faith", "god", "divine", "religious", "transcendent",
        "calling", "believers", "devotion", "worship", "prophet", "doctrine",
        "dogma", "salvation",
    ],
    "Community": [
        "community", "contributors", "volunteers", "grassroots", "collective",
        "commons", "transparency", "openness", "participatory", "forums",
        "reciprocity", "solidarity", "belonging", "movement", "stewardship",
        "open source", "common good",
    ],
}


# ---------------------------------------------------------------------------
# Row scoring (pure; semantic rung only when vectors AND calibration exist)
# ---------------------------------------------------------------------------
def _annotate(kw: str, verdict: dict) -> str:
    """One matched keyword for the audit trail, tagged with how it matched.

    exact -> the keyword itself; morphological -> "keyword~word";
    semantic -> "keyword≈word". The notation is what lets a reader see at a
    glance which credit is literal and which is inferred.
    """
    matched = verdict.get("matched")
    if verdict["tier"] == "morphological" and matched and matched != kw:
        return f"{kw}~{matched}"
    if verdict["tier"] == "semantic" and matched:
        return f"{kw}≈{matched}"
    return kw


def score_row(answer: str, lexicon: dict[str, list[str]] | None = None, *,
              vectors=None, calibration=None) -> dict:
    """Score one answer against every logic's keyword list.

    Per logic, graded recall: each keyword earns its ladder score in [0, 1]
    (exact 1.0, morphological ratio, semantic percentile, absent 0), and

        raw_l = sum of keyword scores / |K_l|

    normalized into shares summing to 1 across logics (all-zero rows are
    flagged no_overlap instead of being forced uniform). Matched words are
    kept, annotated with their rung — the point of a transparent judge.
    """
    lexicon = lexicon or LOGIC_KEYWORDS
    all_kws = [k for kws in lexicon.values() for k in kws]
    ctx = _Ctx(answer, vectors=vectors, calibration=calibration,
               keywords=all_kws)

    raw: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    tiers: dict[str, dict[str, int]] = {}
    for logic in LOGICS:
        kws = lexicon.get(logic, [])
        total = 0.0
        hits: list[str] = []
        fired: Counter = Counter()
        for kw in kws:
            v = _score_one(kw, ctx)
            total += v["score"]
            if v["tier"] != TIER_ABSENT and v["score"] > 0.0:
                hits.append(_annotate(kw, v))
                fired[v["tier"]] += 1
        raw[logic] = (total / len(kws)) if kws else 0.0
        matched[logic] = hits[:MAX_MATCHED_STORED]
        tiers[logic] = dict(fired)

    total = sum(raw.values())
    if total <= 0.0:
        return {"no_overlap": True, "raw": raw,
                "shares": {logic: 0.0 for logic in LOGICS},
                "keyword_top": None, "matched": matched, "tiers": tiers}
    shares = {logic: v / total for logic, v in raw.items()}
    # Ties break by LOGICS order, so the verdict is deterministic.
    top = max(LOGICS, key=lambda logic: raw[logic])
    return {"no_overlap": False, "raw": raw, "shares": shares,
            "keyword_top": top, "matched": matched, "tiers": tiers}


def distribution_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    """Overlap coefficient (sum of per-logic minima); 1 = identical."""
    return sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b))


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------
def _load_committed(run_id: str) -> list[dict]:
    path = runs.run_paths(run_id)["per_question"]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("abstain"):
                rows.append(r)
    return rows


def run_keyword_agreement(run_id: str | None = None, *,
                          semantic: bool = True) -> dict:
    """Compute keyword agreement for a saved run. Returns the summary dict.

    `semantic=True` enables the semantic rung IF the local lexicon files
    exist (data/lexicon/, built by scripts/13_run_topic_keywords.py
    calibrate); without them the ladder stops at the morphological rung and
    the run costs zero API calls, like v1. The same wiring rule as
    topic_keywords: no calibration, no vectors — a raw cosine must never be
    scored.
    """
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run profiles first")

    calibration = Calibration.load() if semantic else None
    vectors = WordVectors() if calibration is not None else None
    semantic_on = calibration is not None

    rows = _load_committed(run_id)
    if not rows:
        raise SystemExit(f"run {run_id} has no committed rows to check")

    out_dir = runs.run_dir(run_id) / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row = []
    tier_totals: Counter = Counter()
    for r in rows:
        s = score_row(r["answer"], vectors=vectors, calibration=calibration)
        for fired in s["tiers"].values():
            tier_totals.update(fired)
        m_logic = max(r["weights"], key=r["weights"].get)
        matcher_w = {logic: float(r["weights"].get(logic, 0.0))
                     for logic in LOGICS}
        per_row.append({
            "org": r["org"], "source_type": r["source_type"],
            "qid": r["qid"], "category": r["category"],
            "variant": r.get("variant") or 1,
            "no_overlap": s["no_overlap"],
            "keyword_scores": {k: round(v, 4) for k, v in s["raw"].items()},
            "keyword_shares": {k: round(v, 4) for k, v in s["shares"].items()},
            "keyword_top": s["keyword_top"],
            "matched_words": {k: v for k, v in s["matched"].items() if v},
            "keyword_tiers": {k: v for k, v in s["tiers"].items() if v},
            "matcher_top": m_logic,
            "matcher_top_weight": round(float(r["weights"][m_logic]), 4),
            "agree": (s["keyword_top"] == m_logic
                      if s["keyword_top"] is not None else None),
            "share_on_matcher_top": round(s["shares"].get(m_logic, 0.0), 4),
            "overlap": round(distribution_overlap(s["shares"], matcher_w), 4),
        })

    if vectors is not None:
        vectors.save()

    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in per_row:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "keywords.json").write_text(json.dumps(
        {"semantic_enabled": semantic_on, "lexicon": LOGIC_KEYWORDS},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- summary ----
    def _slice(items: list[dict]) -> dict:
        scored = [r for r in items if r["agree"] is not None]
        n, ns = len(items), len(scored)
        agree = sum(1 for r in scored if r["agree"])
        out = {"n": n, "n_scored": ns,
               "n_no_overlap": sum(1 for r in items if r["no_overlap"]),
               "agree": agree,
               "rate": round(agree / ns, 4) if ns else None}
        if ns:
            out["mean_share_on_matcher_top"] = round(
                sum(r["share_on_matcher_top"] for r in scored) / ns, 4)
            out["mean_overlap"] = round(
                sum(r["overlap"] for r in scored) / ns, 4)
        return out

    by_category = {c: _slice([r for r in per_row if r["category"] == c])
                   for c in sorted({r["category"] for r in per_row})}
    by_pair = {}
    for key in sorted({(r["org"], r["source_type"]) for r in per_row}):
        by_pair["|".join(key)] = _slice(
            [r for r in per_row
             if (r["org"], r["source_type"]) == key])

    summary = {
        "run_id": run_id,
        "semantic_enabled": semantic_on,
        "lexicon_sizes": {k: len(v) for k, v in LOGIC_KEYWORDS.items()},
        "tier_totals": dict(tier_totals),
        "overall": _slice(per_row),
        "by_category": by_category,
        "by_org_source": by_pair,
        "share_chance_baseline": round(1.0 / len(LOGICS), 4),
        "note": ("The keyword judge scores a hand-curated lexicon per logic "
                 "on the exact/morphological/semantic ladder shared with the "
                 "topic-keyword feature. Its value is transparency (every "
                 "verdict is a visible, rung-annotated word list) and "
                 "triangulation with the LLM matcher and the embedding "
                 "judge. Rows where no keyword of any logic cleared any rung "
                 "are reported as no_overlap and excluded from the rates."),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    o = summary["overall"]
    print(f"\n=== Keyword agreement (run {run_id}) ===")
    print("semantic rung: " + (
        "ON (calibrated percentile)" if semantic_on else
        "off — exact + morphological only (build data/lexicon with "
        "scripts/13_run_topic_keywords.py calibrate to enable)"))
    if tier_totals:
        print("matches by rung: " + ", ".join(
            f"{t}={n}" for t, n in sorted(tier_totals.items())))
    print(f"rows: {o['n']}  scored: {o['n_scored']}  "
          f"no keyword overlap at all: {o['n_no_overlap']}")
    if o["rate"] is not None:
        print(f"binary agreement with matcher: {o['agree']}/{o['n_scored']} "
              f"({o['rate']:.1%})   chance ~{1 / len(LOGICS):.0%}")
        print(f"mean share on matcher's pick: "
              f"{o['mean_share_on_matcher_top']:.3f} (chance 0.143)")
        print(f"mean distribution overlap:    {o['mean_overlap']:.3f}")
    print("by category:")
    for cat, s in by_category.items():
        rate = f"{s['rate']:.0%}" if s["rate"] is not None else "—"
        print(f"  {cat:<24} {rate:>5}  (scored {s['n_scored']}/{s['n']})")
    print(f"outputs: {out_dir}")
    return summary
