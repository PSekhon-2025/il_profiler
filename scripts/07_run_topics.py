"""Stage 7 (optional, LOCAL ONLY): inductive topic layer over the corpus.

Two subcommands:

    python scripts/07_run_topics.py fit            # cluster the corpus (once)
    python scripts/07_run_topics.py crosstab       # topic x logic for CURRENT run
    python scripts/07_run_topics.py crosstab --run 2026-07-01_120000

`fit` needs the local-only extras (`pip install -r requirements-topics.txt`)
and reuses the embeddings already in Chroma, so it makes no API calls. It is
deliberately not runnable in the deployed container — see il_rag/topics.py.

`crosstab` needs nothing beyond a fitted model and a saved run: it attributes
each answered question's logic weights to the topics of the chunks it was
answered from, and audits which topics the questionnaire never reached.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.topics import (
    DEFAULT_MIN_TOPIC_SIZE,
    DEFAULT_SEED,
    build_crosstab,
    fit_topics,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="cluster the corpus into topics")
    p_fit.add_argument("--min-topic-size", type=int,
                       default=DEFAULT_MIN_TOPIC_SIZE,
                       help="smallest cluster HDBSCAN will call a topic; "
                            "larger = fewer, broader topics")
    p_fit.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="UMAP random_state (keep fixed for reproducibility)")
    p_fit.add_argument("--keywords", type=int, default=10,
                       help="keywords stored per topic")

    p_x = sub.add_parser("crosstab", help="topic x logic + coverage for a run")
    p_x.add_argument("--run", default=None,
                     help="run id (default: the CURRENT run)")

    args = ap.parse_args()
    if args.cmd == "fit":
        fit_topics(min_topic_size=args.min_topic_size, seed=args.seed,
                   n_keywords=args.keywords)
    else:
        build_crosstab(run_id=args.run)
