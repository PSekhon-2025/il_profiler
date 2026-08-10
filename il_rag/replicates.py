"""Replicate runs: averaging out decoding noise, and measuring how big it is.

Every LLM call in this pipeline runs at temperature 0, which is greedy decoding
— but greedy is not bit-reproducible on shared GPU infrastructure. Batching and
floating-point ordering can flip a near-tie token, the answer is then reworded,
and a borderline row can land on a different logic or abstain. Measured on two
runs of an identical questionnaire: 145/162 answers were reworded, 33 rows
flipped abstention, 22 flipped label, and 2 of 6 dominant logics changed.

That variance is real and previously invisible in the outputs. This module does
two things about it:

  1. AVERAGES it down. The mean of R replicates has a standard error 1/sqrt(R)
     of a single run's, so 3-4 replicates roughly halve the jitter.
  2. REPORTS it. The between-run standard deviation is the size of the decoding
     noise, and belongs next to the bootstrap interval — they measure different
     things and neither substitutes for the other:

       bootstrap CI      "how much would this move if we had asked a
                          different sample of questions?"     (instrument)
       between-run SD    "how much does it move if we ask the SAME questions
                          again?"                              (decoding)

Correctness guard: only runs produced by the SAME questionnaire may be averaged.
Reference wording changes what the matcher grades against, so mixing question
sets would silently average two different measurements. Every run snapshot
stores the questionnaire that produced it, so this module fingerprints it and
refuses to combine runs that disagree.

Outputs (data/profiles/replicates/<group_id>/):
  report.json   per (lab, source, logic): mean, SD, SEM, min/max, per-run values
  report.csv    the same as a flat table
"""
import csv
import hashlib
import json
import statistics
from datetime import datetime

from . import runs
from .config import ORGS, PROFILES_DIR, SOURCE_TYPES
from .questionnaire import LOGICS

REPLICATES_DIR = PROFILES_DIR / "replicates"
REPORT_JSON = "report.json"
REPORT_CSV = "report.csv"

# Below this many replicates the spread estimate is too thin to report as a
# standard deviation (2 runs give it a single degree of freedom).
MIN_REPLICATES_FOR_SD = 3


# ---------------------------------------------------------------------------
# Questionnaire fingerprinting
# ---------------------------------------------------------------------------
def questionnaire_fingerprint(run_id: str) -> str | None:
    """Stable hash of the questionnaire a run used, or None if not snapshotted.

    Canonical JSON (sorted keys) so formatting differences never masquerade as
    a content change.
    """
    path = runs.run_paths(run_id)["questionnaire"]
    if not path.exists():
        return None
    blob = json.dumps(json.loads(path.read_text(encoding="utf-8")),
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def group_by_questionnaire(run_ids: list[str]) -> dict[str | None, list[str]]:
    """Bucket run ids by questionnaire fingerprint — only same-bucket runs are
    comparable."""
    groups: dict[str | None, list[str]] = {}
    for rid in run_ids:
        groups.setdefault(questionnaire_fingerprint(rid), []).append(rid)
    return groups


# ---------------------------------------------------------------------------
# Aggregation math (pure — no files, no API)
# ---------------------------------------------------------------------------
def aggregate_profiles(profiles_by_run: dict[str, dict],
                       orgs: list[str] | None = None,
                       source_types: list[str] | None = None) -> dict:
    """Mean and spread of each logic percentage across replicate runs.

    For a (lab, source, logic) observed as P_1..P_R across R runs:

        mean = (1/R) * sum(P_r)
        sd   = sample standard deviation of P_1..P_R   (R >= 3)
        sem  = sd / sqrt(R)        the mean's own standard error

    `dominant_agreement` counts how many runs put the same logic on top — a
    label-level stability figure that survives even when the percentages wobble.
    """
    orgs = orgs or ORGS
    source_types = source_types or SOURCE_TYPES
    run_ids = sorted(profiles_by_run)
    R = len(run_ids)

    out: dict = {}
    for org in orgs:
        out[org] = {}
        for st in source_types:
            per_logic: dict = {}
            dominants: list[str] = []
            answered: list[int] = []
            present = 0
            for rid in run_ids:
                p = (profiles_by_run[rid].get(org, {}) or {}).get(st)
                if not p or not p.get("answered"):
                    continue
                present += 1
                answered.append(p["answered"])
                pct = p["logic_pct"]
                dominants.append(max(pct, key=pct.get))
                for logic in LOGICS:
                    per_logic.setdefault(logic, []).append(float(pct.get(logic, 0.0)))
            if present == 0:
                continue

            stats = {}
            for logic in LOGICS:
                vals = per_logic.get(logic, [])
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                sd = (statistics.stdev(vals)
                      if len(vals) >= MIN_REPLICATES_FOR_SD else None)
                stats[logic] = {
                    "mean": round(mean, 2),
                    "sd": round(sd, 2) if sd is not None else None,
                    "sem": (round(sd / (len(vals) ** 0.5), 2)
                            if sd is not None else None),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "range": round(max(vals) - min(vals), 2),
                    "per_run": {rid: round(v, 2)
                                for rid, v in zip(run_ids, vals)},
                }
            top = max(stats, key=lambda k: stats[k]["mean"]) if stats else None
            out[org][st] = {
                "replicates": present,
                "logics": stats,
                "mean_dominant": top,
                # How often the single-run dominant logic matched the dominant
                # of the AVERAGED profile: label stability under re-running.
                "dominant_agreement": (
                    round(sum(1 for d in dominants if d == top) / len(dominants), 3)
                    if dominants and top else None),
                "dominant_per_run": dominants,
                "answered_per_run": answered,
            }
    return {"run_ids": run_ids, "n_replicates": R, "profiles": out}


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------
def build_report(run_ids: list[str], group_id: str | None = None,
                 force: bool = False) -> dict:
    """Load the given runs' profiles, verify they share a questionnaire, and
    write the averaged report."""
    if len(run_ids) < 2:
        raise SystemExit("need at least 2 runs to average; pass --runs a b [c ...]")

    groups = group_by_questionnaire(run_ids)
    if len(groups) > 1 and not force:
        detail = "; ".join(
            f"{fp or 'no-snapshot'}: {', '.join(rs)}" for fp, rs in groups.items())
        raise SystemExit(
            "these runs did NOT use the same questionnaire, so averaging them "
            f"would mix two different measurements ({detail}). Re-run with "
            "--force only if you know why that is acceptable.")

    profiles_by_run = {}
    for rid in run_ids:
        path = runs.run_paths(rid)["profiles_json"]
        if not path.exists():
            raise SystemExit(f"run {rid} has no company_profiles.json")
        profiles_by_run[rid] = json.loads(path.read_text(encoding="utf-8"))

    agg = aggregate_profiles(profiles_by_run)
    agg["questionnaire_fingerprint"] = next(iter(groups))
    agg["mixed_questionnaires"] = len(groups) > 1
    agg["created_at"] = datetime.now().isoformat(timespec="seconds")
    agg["note"] = (
        "Between-run SD is DECODING noise (same questions, re-asked). It is a "
        "different quantity from the bootstrap CI, which is question-sampling "
        "uncertainty. Report both; combining them in quadrature assumes they "
        "are independent."
    )

    group_id = group_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = REPLICATES_DIR / group_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_JSON).write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["org", "source_type", "logic", "mean_pct", "sd", "sem",
                    "min", "max", "range", "replicates"])
        for org, by_st in agg["profiles"].items():
            for st, d in by_st.items():
                for logic, s in d["logics"].items():
                    w.writerow([org, st, logic, s["mean"], s["sd"], s["sem"],
                                s["min"], s["max"], s["range"], d["replicates"]])

    _print_report(agg, out_dir)
    return agg


def _print_report(agg: dict, out_dir) -> None:
    print(f"\n=== Replicate average over {agg['n_replicates']} runs ===")
    print("runs:", ", ".join(agg["run_ids"]))
    for org, by_st in agg["profiles"].items():
        for st, d in by_st.items():
            top = d["mean_dominant"]
            s = d["logics"].get(top, {})
            sd = s.get("sd")
            sd_txt = f" ± {sd:.1f} sd" if sd is not None else " (sd needs 3+ runs)"
            print(f"  {org:<10} {st:<11} {top:<12} "
                  f"{s.get('mean', 0):.1f}%{sd_txt}   "
                  f"dominant agreed in {d['dominant_agreement']:.0%} of runs"
                  if d.get("dominant_agreement") is not None else "")
    print(f"outputs: {out_dir}")
