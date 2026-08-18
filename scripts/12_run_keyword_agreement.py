"""Stage 12 (optional): keyword agreement — the lexical third judge.

Usage:
    python scripts/12_run_keyword_agreement.py                 # CURRENT run
    python scripts/12_run_keyword_agreement.py --run 2026-07-13_210346
    python scripts/12_run_keyword_agreement.py --max-df 1      # stricter keywords

Reduces each question's seven reference answers to distinctive keyword sets,
reduces every committed RAG answer to its content tokens, and grades by set
overlap — see il_rag/keyword_agreement.py. Zero API calls, deterministic,
rerunning overwrites.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.keyword_agreement import KEYWORD_MAX_LOGIC_DF, run_keyword_agreement

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to check (default: the CURRENT run)")
    ap.add_argument("--max-df", type=int, default=KEYWORD_MAX_LOGIC_DF,
                    help="a reference token may appear in at most this many of "
                         "a question's 7 keyword sets and stay distinctive")
    args = ap.parse_args()
    run_keyword_agreement(run_id=args.run, max_df=args.max_df)
