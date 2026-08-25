"""Stage 13 (optional): topic-keyword semantic matching.

Three subcommands:

    python scripts/13_run_topic_keywords.py calibrate           # LOCAL, once
    python scripts/13_run_topic_keywords.py score               # CURRENT run
    python scripts/13_run_topic_keywords.py score --run 2026-07-01_120000
    python scripts/13_run_topic_keywords.py score --no-embeddings
    python scripts/13_run_topic_keywords.py neighbors manager --top 25

`calibrate` builds the background distribution of corpus word-pair cosines that
makes a semantic score readable as a percentage. It needs the Chroma index and
costs a few hundred embedded words, ONCE — the vectors are cached to disk, so
everything afterwards is free. Its two outputs live in data/lexicon/ and are
shipped alongside data/topics/ (see DEPLOY.md).

`score` places every keyword of every topic a run's evidence came from on the
four-rung ladder against that row's answer, and rolls the result up per row,
per topic and per keyword. `--no-embeddings` restricts it to the two lexical
rungs, which makes it pure computation and zero cost.

`neighbors` prints the nearest corpus words to a query word with their
calibrated scores — the quickest check that the calibration is doing its job,
and the one that shows why a raw cosine could not be used directly.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.config import (
    CALIBRATION_MIN_DF,
    CALIBRATION_VOCAB_SIZE,
)
from il_rag.topic_keywords import (
    Calibration,
    WordVectors,
    build_calibration,
    neighbors,
    run_topic_keywords,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate",
                           help="build the background word-pair distribution")
    p_cal.add_argument("--vocab-size", type=int, default=CALIBRATION_VOCAB_SIZE,
                       help="how many of the corpus's most frequent content "
                            "words to draw the background from")
    p_cal.add_argument("--min-df", type=int, default=CALIBRATION_MIN_DF,
                       help="a word must appear in at least this many chunks")

    p_score = sub.add_parser("score", help="score a run's answers")
    p_score.add_argument("--run", default=None,
                         help="run id (default: the CURRENT run)")
    p_score.add_argument("--no-embeddings", action="store_true",
                         help="lexical rungs only — pure computation, no cost")
    p_score.add_argument("--no-ceiling", action="store_true",
                         help="skip the verbatim-ceiling arm (needs Chroma)")

    p_nb = sub.add_parser("neighbors", help="nearest corpus words to a word")
    p_nb.add_argument("word")
    p_nb.add_argument("--top", type=int, default=25)

    args = ap.parse_args()
    if args.cmd == "calibrate":
        build_calibration(vocab_size=args.vocab_size, min_df=args.min_df)
    elif args.cmd == "score":
        run_topic_keywords(run_id=args.run,
                           embeddings=not args.no_embeddings,
                           ceiling=not args.no_ceiling)
    else:
        cal = Calibration.load()
        if cal is None:
            raise SystemExit(
                "no calibration on disk — run "
                "`scripts/13_run_topic_keywords.py calibrate` first")
        rows = neighbors(args.word, vectors=WordVectors(readonly=True),
                         calibration=cal, top_n=args.top)
        if not rows:
            raise SystemExit(
                f"'{args.word}' is not in the cached lexicon "
                f"(vocabulary size {cal.meta.get('vocab_size')})")
        print(f"\nnearest to '{args.word}':")
        print(f"{'word':<24} {'cosine':>8} {'percentile':>11}  tier")
        for r in rows:
            pct = "—" if r["percentile"] is None else f"{r['percentile']:.1%}"
            print(f"{r['word']:<24} {r['cosine']:>8.4f} {pct:>11}  {r['tier']}")
        band = max(r["cosine"] for r in rows) - min(r["cosine"] for r in rows)
        print(f"\nraw cosine spans {band:.4f} across these {len(rows)} words — "
              "that narrow band is exactly why the percentile column exists.")
