"""Keyword agreement: a third, fully lexical judge over a saved run.

The pipeline now has three independent graders of the same question — answer
pairs, each blind to the others' reasoning:

  1. the LLM matcher      semantic, evidence-weighing   (the primary instrument)
  2. embedding agreement  distributional similarity     (no LLM)
  3. THIS MODULE          bare lexical overlap          (no LLM, no embeddings)

Instead of comparing whole texts, each (category, question, logic) reference
answer is reduced to a small set of DISTINCTIVE KEYWORDS, the RAG answer is
reduced to its content tokens, and the judge is simply: which logic's keywords
does the answer actually contain? Everything is set arithmetic over visible
words — the one judge whose every verdict can be checked by eye, which is the
point of adding it.

Reference keywords are DERIVED, not hand-written (v1): the content tokens of
each reference answer, minus tokens that appear in too many of that question's
seven references. The subtraction is the load-bearing step — within a category
the seven references share framing vocabulary ("lab", "conduct", "appropriate")
that says nothing about WHICH logic; a token may appear in at most
KEYWORD_MAX_LOGIC_DF of the seven keyword sets to count as distinctive.
Because keywords come from the run's own questionnaire snapshot, the same
resolution rules apply as everywhere else (base + per-variant overrides,
including JSON's stringified override keys).

Honest scope: this is the bluntest of the three judges. It cannot see
synonymy — an answer saying "government" earns no credit for a reference that
says "state" — so its miss rate is structurally high and its value is
TRANSPARENCY and triangulation, not accuracy. Where all three judges agree,
the classification is very hard to dismiss; where this one dissents, the
matched-word lists show exactly why.

Zero API cost, fully deterministic. Outputs (in <run>/keyword_agreement/):
  keywords.json   the derived per-(category, variant, logic) keyword sets
  rows.jsonl      per answered row: per-logic scores, matched words, verdicts
  summary.json    binary + graded rates, overall and per slice
"""
import json
from collections import Counter

from . import runs
from .grounding import content_tokens
from .questionnaire import LOGICS

OUT_DIR_NAME = "keyword_agreement"

# A reference token may appear in at most this many of a question's seven
# keyword sets and still count as distinctive for those logics. 1 = strictly
# unique tokens only; 2 tolerates natural pairwise sharing (e.g. "capitalism"
# variants) without letting category-wide framing vocabulary through.
KEYWORD_MAX_LOGIC_DF = 2

# Per-row cap on stored matched words (audit readability, not scoring).
MAX_MATCHED_STORED = 12


# ---------------------------------------------------------------------------
# Reference keyword derivation (pure)
# ---------------------------------------------------------------------------
def derive_keywords(refs: dict[str, str],
                    max_df: int = KEYWORD_MAX_LOGIC_DF) -> dict[str, list[str]]:
    """Distinctive keyword set per logic for ONE question's seven references.

    keywords(l) = content_tokens(ref_l) minus tokens present in more than
    `max_df` of the seven references' token sets.
    """
    toks = {logic: content_tokens(text) for logic, text in refs.items()}
    df = Counter(t for s in toks.values() for t in set(s))
    return {logic: sorted(t for t in s if df[t] <= max_df)
            for logic, s in toks.items()}


def resolve_reference_texts(questionnaire: dict) -> dict[tuple, dict[str, str]]:
    """(category, variant) -> {logic: reference text}, honoring overrides.

    Same resolution as the matcher and the embedding judge; JSON snapshots
    stringify override keys, so both forms are looked up.
    """
    out: dict[tuple, dict[str, str]] = {}
    for category, block in questionnaire.items():
        base = block.get("reference_answers", {})
        overrides = block.get("reference_overrides", {})
        for variant in (1, 2, 3):
            ov = overrides.get(variant) or overrides.get(str(variant)) or {}
            out[(category, variant)] = {**base, **ov}
    return out


# ---------------------------------------------------------------------------
# Row scoring (pure)
# ---------------------------------------------------------------------------
def score_row(answer: str, keywords: dict[str, list[str]]) -> dict:
    """Score one answer against one question's per-logic keyword sets.

    Per logic: recall of its keywords in the answer's content tokens,
        raw_l = |tokens(answer) ∩ keywords_l| / |keywords_l|
    normalized into shares summing to 1 across logics (all-zero rows are
    flagged no_overlap instead of being forced uniform). Matched words are
    kept for the audit trail — the whole point of a lexical judge.
    """
    ans = content_tokens(answer)
    raw: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for logic in LOGICS:
        kws = keywords.get(logic, [])
        hit = sorted(ans.intersection(kws))
        matched[logic] = hit[:MAX_MATCHED_STORED]
        raw[logic] = (len(hit) / len(kws)) if kws else 0.0
    total = sum(raw.values())
    if total <= 0.0:
        return {"no_overlap": True, "raw": raw,
                "shares": {logic: 0.0 for logic in LOGICS},
                "keyword_top": None, "matched": matched}
    shares = {logic: v / total for logic, v in raw.items()}
    # Ties break by LOGICS order, so the verdict is deterministic.
    top = max(LOGICS, key=lambda logic: raw[logic])
    return {"no_overlap": False, "raw": raw, "shares": shares,
            "keyword_top": top, "matched": matched}


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


def run_keyword_agreement(run_id: str | None = None,
                          max_df: int = KEYWORD_MAX_LOGIC_DF) -> dict:
    """Compute keyword agreement for a saved run. Returns the summary dict."""
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run profiles first")
    snapshot = json.loads(
        runs.run_paths(run_id)["questionnaire"].read_text(encoding="utf-8"))
    ref_texts = resolve_reference_texts(snapshot["questionnaire"])
    keywords = {key: derive_keywords(refs, max_df=max_df)
                for key, refs in ref_texts.items()}

    rows = _load_committed(run_id)
    if not rows:
        raise SystemExit(f"run {run_id} has no committed rows to check")

    out_dir = runs.run_dir(run_id) / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row = []
    for r in rows:
        key = (r["category"], r.get("variant") or 1)
        kws = keywords.get(key) or keywords[(r["category"], 1)]
        s = score_row(r["answer"], kws)
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
            "matcher_top": m_logic,
            "matcher_top_weight": round(float(r["weights"][m_logic]), 4),
            "agree": (s["keyword_top"] == m_logic
                      if s["keyword_top"] is not None else None),
            "share_on_matcher_top": round(s["shares"].get(m_logic, 0.0), 4),
            "overlap": round(distribution_overlap(s["shares"], matcher_w), 4),
        })

    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in per_row:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "keywords.json").write_text(json.dumps(
        {f"{c}#{v}": kws for (c, v), kws in keywords.items()},
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
        "max_logic_df": max_df,
        "overall": _slice(per_row),
        "by_category": by_category,
        "by_org_source": by_pair,
        "share_chance_baseline": round(1.0 / len(LOGICS), 4),
        "note": ("Keyword agreement is the bluntest of the three judges: it "
                 "cannot see synonymy, so misses are structurally common. Its "
                 "value is transparency (every verdict is a visible word list) "
                 "and triangulation with the LLM matcher and the embedding "
                 "judge. Rows whose answer shares no keyword with ANY logic "
                 "are reported as no_overlap and excluded from the rates."),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    o = summary["overall"]
    print(f"\n=== Keyword agreement (run {run_id}) ===")
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
