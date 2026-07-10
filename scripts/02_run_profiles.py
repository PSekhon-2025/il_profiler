"""Stage 2: run the questionnaire and produce the six alignment profiles.

Usage:
    python scripts/02_run_profiles.py
    python scripts/02_run_profiles.py --orgs OpenAI --sources published   # subset
    python scripts/02_run_profiles.py --fresh                             # new snapshot
    python scripts/02_run_profiles.py --fresh --label "questionnaire v2"  # named run

Each --fresh run is archived as its own snapshot under data/profiles/runs/ (the
previous run is preserved), so runs can be compared in the app.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.config import ORGS, SOURCE_TYPES, TOP_K
from il_rag.profile_harness import run_profiles

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orgs", nargs="+", choices=ORGS, default=None,
                    help=f"labs to profile (default: {ORGS})")
    ap.add_argument("--sources", nargs="+", choices=SOURCE_TYPES, default=None,
                    help=f"source types to profile (default: {SOURCE_TYPES})")
    ap.add_argument("--k", type=int, default=TOP_K,
                    help="chunks retrieved per question")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new run snapshot (previous runs are kept)")
    ap.add_argument("--label", default=None,
                    help="optional label for this run (e.g. the questionnaire version)")
    args = ap.parse_args()
    run_profiles(orgs=args.orgs, source_types=args.sources, k=args.k,
                 fresh=args.fresh, label=args.label)
