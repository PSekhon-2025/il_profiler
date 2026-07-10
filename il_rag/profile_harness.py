"""Profile harness: run the questionnaire for every (lab, source_type), grade
each answer, and aggregate into percentage alignment profiles.

Aggregation rule: every answered (non-abstained) question contributes a weight
vector summing to 1, so the profile is simply the mean weight per logic across
answered questions, reported as percentages summing to ~100. Abstentions are
counted separately and excluded from the denominator — a silent corpus reduces
confidence (fewer answered questions), it never shifts the distribution.

Outputs live in a per-run snapshot (data/profiles/runs/<run_id>/), managed by
runs.py, so a run never overwrites an earlier one:
  per_question.jsonl    audit trail, one row per (org, source_type, question)
  company_profiles.json org -> source_type -> {logic_pct, answered, abstained, by_category}
  profiles_matrix.csv   wide table: one row per (org, source_type)
  questionnaire.json    the questionnaire that produced this run (for diffing)
  meta.json             run params, counts, status

Resumable: rows already present in the active run's per_question.jsonl are
skipped on rerun, so an interrupted run continues where it stopped. --fresh
starts a NEW snapshot (the previous one is kept untouched).
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import runs
from .config import ORGS, SOURCE_TYPES, TOP_K
from .graded_matcher import match_graded
from .questionnaire import LOGICS, build_questionnaire
from .rag_qa import answer_question
from .retriever import Retriever


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
                 k: int = TOP_K, fresh: bool = False,
                 label: str | None = None) -> dict:
    """Run the full Design-A evaluation into a run snapshot and write all outputs.

    fresh=True mints a NEW snapshot; otherwise the active (CURRENT) run is
    resumed, or a new one is created if none exists. Returns the nested profiles
    dict (also written to the run's company_profiles.json).
    """
    orgs = orgs or ORGS
    source_types = source_types or SOURCE_TYPES

    runs.migrate_legacy()  # one-time: fold pre-snapshot flat files into a run

    if fresh or runs.get_current() is None:
        run_id = runs.new_run(label=label, orgs=orgs, source_types=source_types, k=k)
        print(f"new run: {run_id}")
    else:
        run_id = runs.get_current()
        if label:
            runs.update_meta(run_id, label=label)
        print(f"resuming run: {run_id}")
    paths = runs.run_paths(run_id)

    # A fresh run's folder is empty, so resumption is a no-op there; for a
    # resumed run we skip questions already on disk in THIS run.
    done, rows = _load_existing(paths["per_question"])
    if done:
        print(f"resuming: {len(done)} questions already answered in this run")

    todo = [
        (org, st, q)
        for org in orgs
        for st in source_types
        for q in build_questionnaire(org)
        if (org, st, q["qid"]) not in done
    ]

    retriever = Retriever()
    with open(paths["per_question"], "a", encoding="utf-8") as f:
        for org, st, q in tqdm(todo, desc="profile", unit="q"):
            rag = answer_question(retriever, q["question"], org=org, source_type=st, k=k)
            verdict = match_graded(
                question=q["question"], candidate=rag.answer,
                category=q["category"], variant=q["variant"],
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
    _write_outputs(profiles, paths)
    runs.finalize_meta(run_id, rows, orgs, source_types)
    _print_report(profiles, run_id)
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


def _write_outputs(profiles: dict, paths: dict) -> None:
    with open(paths["profiles_json"], "w", encoding="utf-8") as f:
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
    ).to_csv(paths["profiles_csv"], index=False)


def _print_report(profiles: dict, run_id: str) -> None:
    print(f"\n=== Institutional-logic alignment profiles (% per logic) "
          f"[run {run_id}] ===")
    for org, by_st in profiles.items():
        for st, p in by_st.items():
            print(f"\n{org} [{st}]  answered={p['answered']}  abstained={p['abstained']}")
            for logic, pct in sorted(p["logic_pct"].items(), key=lambda kv: -kv[1]):
                print(f"  {logic:<12} {pct:5.1f}%  {'#' * round(pct / 2)}")
    print(f"\noutputs: {runs.run_dir(run_id)}")
