"""Bootstrap confidence intervals for the alignment profiles.

A profile percentage is the MEAN weight a logic receives across a lab/source's
answered questions (see profile_harness). That makes it a sample statistic over
a finite questionnaire, so the honest error bar is a bootstrap CI: resample the
answered questions with replacement many times, recompute the mean each time,
and read the percentiles of the resulting distribution.

What the CI means: "given that we asked THESE questions, how much would each
logic's percentage wobble if we'd sampled a slightly different set of questions
of the same kind." It is NOT model stochasticity (the pipeline is temperature 0
and deterministic) — repeating the run would reproduce the same answers, so a
repeat-based CI would be spuriously ~0. Bootstrapping the questions is the
statistically meaningful width.

Zero API cost and fully deterministic given the seed: pure resampling over the
weights already stored in the run's per_question.jsonl.

Output (in <run>/bootstrap_ci/):
  ci.json  org -> source_type -> logic -> {mean, lo, hi, std, n}
  ci.csv   flat table for the paper
"""
import csv
import json
from collections import defaultdict

import numpy as np

from . import runs
from .questionnaire import LOGICS

OUT_DIR_NAME = "bootstrap_ci"
DEFAULT_ITERATIONS = 2000
DEFAULT_CI = 0.95


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


def run_bootstrap_ci(run_id: str | None = None,
                     iterations: int = DEFAULT_ITERATIONS,
                     ci: float = DEFAULT_CI, seed: int = 0) -> dict:
    """Compute bootstrap CIs per (org, source_type, logic). Returns the dict."""
    run_id = run_id or runs.get_current()
    if not run_id:
        raise SystemExit("no run found — run profiles first")

    rows = _load_committed(run_id)
    if not rows:
        raise SystemExit(f"run {run_id} has no committed rows to bootstrap")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["org"], r["source_type"])].append(r)

    rng = np.random.default_rng(seed)
    lo_pct, hi_pct = 100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)

    result: dict = defaultdict(dict)
    for (org, stype), grp in groups.items():
        # (n_questions, n_logics) matrix of weight vectors (each row sums to ~1).
        W = np.array([[float(g["weights"].get(l, 0.0)) for l in LOGICS]  # noqa: E741
                      for g in grp])
        n = W.shape[0]
        point = W.mean(axis=0) * 100.0  # matches profile_harness logic_pct
        if n >= 2:
            # iterations resamples of n questions with replacement.
            idx = rng.integers(0, n, size=(iterations, n))
            boot = W[idx].mean(axis=1) * 100.0        # (iterations, n_logics)
            lo = np.percentile(boot, lo_pct, axis=0)
            hi = np.percentile(boot, hi_pct, axis=0)
            sd = boot.std(axis=0, ddof=1)
        else:
            # A single question: CI is undefined; report zero width and flag n.
            lo = hi = point
            sd = np.zeros_like(point)
        result[org][stype] = {
            LOGICS[i]: {
                "mean": round(float(point[i]), 2),
                "lo": round(float(lo[i]), 2),
                "hi": round(float(hi[i]), 2),
                "std": round(float(sd[i]), 2),
                "n": int(n),
            }
            for i in range(len(LOGICS))
        }

    out_dir = runs.run_dir(run_id) / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "iterations": iterations, "ci": ci, "seed": seed,
            "profiles": result}
    (out_dir / "ci.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / "ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["org", "source_type", "logic", "mean_pct",
                    "ci_low", "ci_high", "std", "n_questions"])
        for org, by_st in result.items():
            for stype, by_logic in by_st.items():
                for logic, s in by_logic.items():
                    w.writerow([org, stype, logic, s["mean"], s["lo"],
                                s["hi"], s["std"], s["n"]])

    # Console summary: dominant logic per group with its CI.
    print(f"\n=== Bootstrap {int(ci*100)}% CIs (run {run_id}, "
          f"{iterations} iters) ===")
    for org, by_st in result.items():
        for stype, by_logic in by_st.items():
            top = max(by_logic, key=lambda l: by_logic[l]["mean"])  # noqa: E741
            s = by_logic[top]
            print(f"  {org:<10} {stype:<10} n={s['n']:<3} "
                  f"dominant {top}: {s['mean']:.1f}% "
                  f"[{s['lo']:.1f}, {s['hi']:.1f}]")
    print(f"outputs: {out_dir}")
    return meta
