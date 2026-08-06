"""Stage 3 (optional): metamorphic eval over an existing run.

Usage:
    python scripts/03_run_metamorphic_eval.py                       # CURRENT run
    python scripts/03_run_metamorphic_eval.py --run 2026-07-01_120000
    python scripts/03_run_metamorphic_eval.py --sample 30           # cost control
    python scripts/03_run_metamorphic_eval.py --paraphrases 5
    python scripts/03_run_metamorphic_eval.py --probes control paraphrase
    python scripts/03_run_metamorphic_eval.py --orgs OpenAI --sources published

For each per-question row of the chosen run this perturbs the evidence the row
was answered from and re-runs the production answer -> match path on each
variant, reporting what the predicted label does. Four probes, all optional:

    control     same evidence, run again — the NOISE FLOOR every other number
                is read against
    paraphrase  same meaning, different words — a flip is suspicious
    ablation    the excerpt the answer quoted is removed — a label that
                SURVIVES is suspicious
    distractor  real same-lab text that doesn't answer the question — the right
                response is to abstain, so repeating the original label is the
                strongest hallucination signal here

Outputs land in the run's own folder under metamorphic/. Resumable like the
other stages, so enabling another probe later only runs the new variants.

Cost note: at the defaults each item costs ~15 chat calls (1 control + 3
paraphrases + 1 ablation + 1 distractor, each an answer call plus a matching
call, plus one generation call per paraphrase) and a few embedding calls. A
full 162-item run is therefore ~2,400 chat calls. Start with --sample.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.config import (
    METAMORPHIC_CONTROLS,
    METAMORPHIC_PARAPHRASES,
    METAMORPHIC_PROBES,
    ORGS,
    SOURCE_TYPES,
)
from il_rag.metamorphic import run_metamorphic

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to evaluate (default: the CURRENT run)")
    ap.add_argument("--probes", nargs="+", choices=list(METAMORPHIC_PROBES),
                    default=list(METAMORPHIC_PROBES),
                    help="which probes to run (default: all four)")
    ap.add_argument("--controls", type=int, default=METAMORPHIC_CONTROLS,
                    help="unperturbed re-runs per item (the noise floor)")
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
                    orgs=args.orgs, source_types=args.sources,
                    probes=tuple(args.probes), controls=args.controls)
