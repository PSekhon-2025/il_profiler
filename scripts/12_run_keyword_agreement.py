"""Stage 12 (optional): keyword agreement — the transparent third judge.

Usage:
    python scripts/12_run_keyword_agreement.py                 # CURRENT run
    python scripts/12_run_keyword_agreement.py --run 2026-07-13_210346
    python scripts/12_run_keyword_agreement.py --no-semantic   # ladder stops at morph

Scores a hand-curated keyword lexicon (one list per logic, in
il_rag/keyword_agreement.py) against every committed answer on the graded
exact/morphological/semantic ladder shared with the topic-keyword feature.

The semantic rung engages only when the local lexicon files exist
(data/lexicon/, built once by `scripts/13_run_topic_keywords.py calibrate`);
without them — or with --no-semantic — the run is pure computation: zero API
calls, deterministic, like v1. With them, the only cost is embedding answer
words not already in the append-only word-vector cache. No LLM calls either
way. Rerunning overwrites.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.keyword_agreement import run_keyword_agreement

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to check (default: the CURRENT run)")
    ap.add_argument("--no-semantic", action="store_true",
                    help="skip the semantic rung even if a calibration exists "
                         "(exact + morphological only)")
    args = ap.parse_args()
    run_keyword_agreement(run_id=args.run, semantic=not args.no_semantic)
