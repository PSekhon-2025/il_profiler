"""Stage 5 (optional): bootstrap confidence intervals over an existing run.

Usage:
    python scripts/05_run_bootstrap_ci.py                      # CURRENT run
    python scripts/05_run_bootstrap_ci.py --run 2026-07-01_120000
    python scripts/05_run_bootstrap_ci.py --iterations 5000 --ci 0.9

Puts a bootstrap error bar on every logic percentage by resampling the
answered questions with replacement — see il_rag/bootstrap_ci.py. Zero API
cost, deterministic given --seed. Outputs land in the run's bootstrap_ci/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.bootstrap_ci import DEFAULT_CI, DEFAULT_ITERATIONS, run_bootstrap_ci

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to bootstrap (default: the CURRENT run)")
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                    help="bootstrap resamples (default 2000)")
    ap.add_argument("--ci", type=float, default=DEFAULT_CI,
                    help="confidence level, e.g. 0.95")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (keep fixed for reproducibility)")
    args = ap.parse_args()
    run_bootstrap_ci(run_id=args.run, iterations=args.iterations,
                     ci=args.ci, seed=args.seed)
