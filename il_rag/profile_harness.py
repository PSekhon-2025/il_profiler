"""Profile harness: run the questionnaire for every (lab, source_type), grade
each answer, and aggregate into percentage alignment profiles.

Aggregation rule: every answered (non-abstained) question contributes a weight
vector summing to 1, so the profile is simply the mean weight per logic across
answered questions, reported as percentages summing to ~100. Abstentions are
counted separately and excluded from the denominator — a silent corpus reduces
confidence (fewer answered questions), it never shifts the distribution.

Outputs (data/profiles/):
  per_question.jsonl    audit trail, one row per (org, source_type, question)
  company_profiles.json org -> source_type -> {logic_pct, answered, abstained, by_category}
  profiles_matrix.csv   wide table: one row per (org, source_type)

Resumable: rows already present in per_question.jsonl are skipped on rerun, so
an interrupted run continues where it stopped. --fresh starts over.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import ORGS, PROFILES_DIR, SOURCE_TYPES, TOP_K
from .graded_matcher import match_graded
from .questionnaire import LOGICS, build_questionnaire
from .rag_qa import answer_question
from .retriever import Retriever

PER_QUESTION_PATH = PROFILES_DIR / "per_question.jsonl"
PROFILES_JSON_PATH = PROFILES_DIR / "company_profiles.json"
PROFILES_CSV_PATH = PROFILES_DIR / "profiles_matrix.csv"


def _load_existing(path: Path) -> tuple[set[tuple], list[dict]]:
    """Read prior results; returns (completed keys, rows) for resumption."""
    if not path.exists():
        return set(), []
    done, rows = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            done.add((row["org"], row["source_type"], row["qid"]))
            rows.append(row)
    return done, rows


def run_profiles(orgs: list[str] | None = None,
                 source_types: list[str] | None = None,
                 k: int = TOP_K, fresh: bool = False) -> dict:
    """Run the full Design-A evaluation and write all outputs.

    Returns the nested profiles dict (also written to company_profiles.json).
    """
    orgs = orgs or ORGS
    source_types = source_types or SOURCE_TYPES

    done, rows = (set(), []) if fresh else _load_existing(PER_QUESTION_PATH)
    if done:
        print(f"resuming: {len(done)} questions already answered on disk")

    todo = [
        (org, st, q)
        for org in orgs
        for st in source_types
        for q in build_questionnaire(org)
        if (org, st, q["qid"]) not in done
    ]

    retriever = Retriever()
    with open(PER_QUESTION_PATH, "w" if fresh else "a", encoding="utf-8") as f:
        for org, st, q in tqdm(todo, desc="profile", unit="q"):
            rag = answer_question(retriever, q["question"], org=org, source_type=st, k=k)
            verdict = match_graded(
                question=q["question"], candidate=rag.answer, category=q["category"]
            )
            row = {
                "org": org,
                "source_type": st,
                "qid": q["qid"],
                "category": q["category"],
                "variant": q["variant"],
                "question": q["question"],
                "answer": rag.answer,
                "retrieved_ids": [c.id for c in rag.chunks],
                "abstain": verdict["abstain"],
                "weights": verdict["weights"],
                "reasoning": verdict["reasoning"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # crash-safe: every completed row is durable immediately
            rows.append(row)

    profiles = aggregate(rows, orgs, source_types)
    _write_outputs(profiles)
    _print_report(profiles)
    return profiles


def aggregate(rows: list[dict], orgs: list[str], source_types: list[str]) -> dict:
    """Fold per-question weight vectors into per-(org, source_type) profiles."""
    sums = defaultdict(lambda: {l: 0.0 for l in LOGICS})       # noqa: E741
    n_answered = defaultdict(int)
    n_abstained = defaultdict(int)
    cat_sums = defaultdict(lambda: defaultdict(lambda: {l: 0.0 for l in LOGICS}))  # noqa: E741
    cat_n = defaultdict(lambda: defaultdict(int))

    for r in rows:
        key = (r["org"], r["source_type"])
        if r["abstain"]:
            n_abstained[key] += 1
            continue
        n_answered[key] += 1
        cat_n[key][r["category"]] += 1
        for logic in LOGICS:
            w = float(r["weights"].get(logic, 0.0))
            sums[key][logic] += w
            cat_sums[key][r["category"]][logic] += w

    profiles: dict = {}
    for org in orgs:
        profiles[org] = {}
        for st in source_types:
            key = (org, st)
            n = n_answered[key]
            if n == 0:
                profiles[org][st] = {
                    "logic_pct": {l: 0.0 for l in LOGICS},  # noqa: E741
                    "answered": 0,
                    "abstained": n_abstained[key],
                    "by_category": {},
                }
                continue
            profiles[org][st] = {
                "logic_pct": {l: round(100.0 * sums[key][l] / n, 2) for l in LOGICS},  # noqa: E741
                "answered": n,
                "abstained": n_abstained[key],
                "by_category": {
                    cat: {l: round(100.0 * s[l] / cat_n[key][cat], 2) for l in LOGICS}  # noqa: E741
                    for cat, s in cat_sums[key].items()
                },
            }
    return profiles


def _write_outputs(profiles: dict) -> None:
    with open(PROFILES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    records = []
    for org, by_st in profiles.items():
        for st, p in by_st.items():
            records.append({
                "org": org, "source_type": st,
                "answered": p["answered"], "abstained": p["abstained"],
                **p["logic_pct"],
            })
    pd.DataFrame(
        records, columns=["org", "source_type", "answered", "abstained", *LOGICS]
    ).to_csv(PROFILES_CSV_PATH, index=False)


def _print_report(profiles: dict) -> None:
    print("\n=== Institutional-logic alignment profiles (% per logic) ===")
    for org, by_st in profiles.items():
        for st, p in by_st.items():
            print(f"\n{org} [{st}]  answered={p['answered']}  abstained={p['abstained']}")
            for logic, pct in sorted(p["logic_pct"].items(), key=lambda kv: -kv[1]):
                print(f"  {logic:<12} {pct:5.1f}%  {'#' * round(pct / 2)}")
    print(f"\noutputs: {PROFILES_DIR}")
