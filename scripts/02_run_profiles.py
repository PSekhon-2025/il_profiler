"""Stage 2: run the questionnaire and produce the six alignment profiles.

Usage:
    python scripts/02_run_profiles.py
    python scripts/02_run_profiles.py --orgs OpenAI --sources published   # subset
    python scripts/02_run_profiles.py --fresh                             # start over
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
                    help="discard prior results and start over")
    args = ap.parse_args()
    run_profiles(orgs=args.orgs, source_types=args.sources, k=args.k, fresh=args.fresh)
