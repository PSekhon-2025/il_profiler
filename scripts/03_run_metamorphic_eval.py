"""Stage 3 (optional): metamorphic label-stability eval over an existing run.

Usage:
    python scripts/03_run_metamorphic_eval.py                       # CURRENT run
    python scripts/03_run_metamorphic_eval.py --run 2026-07-01_120000
    python scripts/03_run_metamorphic_eval.py --sample 30           # cost control
    python scripts/03_run_metamorphic_eval.py --paraphrases 5
    python scripts/03_run_metamorphic_eval.py --orgs OpenAI --sources published

For each per-question row of the chosen run this produces k paraphrase
variants and 1 lab-name-swap variant of the retrieved evidence, re-answers and
re-grades each through the production path, and reports how often the
predicted label survives. Outputs land in the run's own folder under
metamorphic/. Resumable like the other stages.

Cost note: a full 162-item run at the default k=3 is 162 x 4 variants, each
costing one answer call + one matching call, plus ~486 paraphrase-generation
calls (~1,800 chat calls total). Start with --sample.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.config import METAMORPHIC_PARAPHRASES, ORGS, SOURCE_TYPES
from il_rag.metamorphic import run_metamorphic

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to evaluate (default: the CURRENT run)")
    ap.add_argument("--paraphrases", type=int, default=METAMORPHIC_PARAPHRASES,
                    help="meaning-preserving paraphrase variants per item")
    ap.add_argument("--sample", type=int, default=None,
                    help="evaluate only N randomly sampled items (deterministic per seed)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed (keep fixed to resume the same sample)")
    ap.add_argument("--orgs", nargs="+", choices=ORGS, default=None,
                    help="restrict to these labs")
    ap.add_argument("--sources", nargs="+", choices=SOURCE_TYPES, default=None,
                    help="restrict to these source types")
    args = ap.parse_args()
    run_metamorphic(run_id=args.run, paraphrases=args.paraphrases,
                    sample=args.sample, seed=args.seed,
                    orgs=args.orgs, source_types=args.sources)
