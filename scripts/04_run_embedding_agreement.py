"""Stage 4 (optional): embedding-agreement check over an existing run.

Usage:
    python scripts/04_run_embedding_agreement.py                     # CURRENT run
    python scripts/04_run_embedding_agreement.py --run 2026-07-01_120000

For every committed answer in the chosen run, ranks the run's own seven
reference answers (per category) by cosine similarity to the answer and checks
whether the nearest reference's logic agrees with the LLM matcher's top logic.
A second, deterministic, non-LLM judge — see il_rag/embedding_agreement.py.

Cost note: ~63 + one embedding per committed row (batched) — fractions of a
cent. Rerunning recomputes and overwrites.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.embedding_agreement import run_embedding_agreement

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to check (default: the CURRENT run)")
    args = ap.parse_args()
    run_embedding_agreement(run_id=args.run)
