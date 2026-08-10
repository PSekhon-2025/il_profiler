"""Stage 8 (optional): replicate runs — average out decoding noise and size it.

Temperature 0 is greedy but not bit-reproducible on shared GPUs, so re-asking
the identical questionnaire moves the profiles. This stage runs the
questionnaire R times and reports the mean plus the between-run spread.

    # run 3 fresh replicates, then aggregate them automatically
    python scripts/08_run_replicates.py run --n 3
    python scripts/08_run_replicates.py run --n 3 --orgs OpenAI --sources published

    # or aggregate runs you already have (must share a questionnaire)
    python scripts/08_run_replicates.py aggregate --runs 2026-07-13_210346 2026-07-14_213712

Cost: each replicate is a full run (162 answer + 162 matcher calls at the
default scope), so --n 3 costs three times a normal run. Scope it down with
--orgs/--sources while iterating.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.config import ORGS, SOURCE_TYPES, TOP_K
from il_rag.profile_harness import run_profiles
from il_rag.replicates import build_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute R fresh replicates and aggregate")
    p_run.add_argument("--n", type=int, default=3,
                       help="number of replicate runs (3+ needed for an SD)")
    p_run.add_argument("--orgs", nargs="+", choices=ORGS, default=None)
    p_run.add_argument("--sources", nargs="+", choices=SOURCE_TYPES, default=None)
    p_run.add_argument("--k", type=int, default=TOP_K)
    p_run.add_argument("--label", default=None,
                       help="base label; the replicate number is appended")

    p_agg = sub.add_parser("aggregate", help="average existing run snapshots")
    p_agg.add_argument("--runs", nargs="+", required=True,
                       help="run ids to average (must share a questionnaire)")
    p_agg.add_argument("--group-id", default=None,
                       help="output folder name (default: a timestamp)")
    p_agg.add_argument("--force", action="store_true",
                       help="average even if the questionnaires differ — this "
                            "mixes two different measurements, so only use it "
                            "deliberately")

    args = ap.parse_args()

    if args.cmd == "run":
        base = args.label or "replicate"
        run_ids = []
        for i in range(1, args.n + 1):
            print(f"\n===== replicate {i}/{args.n} =====")
            # fresh=True mints a NEW snapshot each time, so the replicates are
            # independent runs rather than a resumed one.
            run_profiles(orgs=args.orgs, source_types=args.sources, k=args.k,
                         fresh=True, label=f"{base} {i}/{args.n}")
            from il_rag import runs as _runs
            run_ids.append(_runs.get_current())
        print(f"\nreplicates: {', '.join(run_ids)}")
        build_report(run_ids)
    else:
        build_report(args.runs, group_id=args.group_id, force=args.force)
