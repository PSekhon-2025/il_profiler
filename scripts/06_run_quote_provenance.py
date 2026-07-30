"""Stage 6 (optional): quote provenance & paraphrase grounding over an existing run.

Usage:
    python scripts/06_run_quote_provenance.py                     # CURRENT run
    python scripts/06_run_quote_provenance.py --run 2026-07-01_120000
    python scripts/06_run_quote_provenance.py --adjudicate-verbatim

Feature 2 asks one question with one bit: is this span verbatim in the sources?
This stage grades HOW each quoted span relates to them (verbatim / drifted copy
/ paraphrase / absent), separates quotation marks that never claimed anything
about a source (scare quotes, terms of art) from real attributions, and then
asks whether a span's CONTENT holds up even when the span itself does not —
the `misquote_but_true` case. See il_rag/quote_provenance.py.

Reads `quotes` and `answer` from the run's per_question.jsonl and replays each
row's evidence via retrieved_ids, so it needs no re-answering and never
rewrites feature 2's `quotes_verified`.

Cost note: the ladder is cheapest-first and short-circuits — a run whose quotes
are all verbatim costs zero generation and zero embedding calls. Only spans
that reach the paraphrase/unsupported tiers are adjudicated, one call each.
Rerunning recomputes and overwrites, so thresholds can be retuned freely.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.quote_provenance import run_quote_provenance

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run id to check (default: the CURRENT run)")
    ap.add_argument("--adjudicate-verbatim", action="store_true",
                    help="also entailment-check spans that DID match verbatim; "
                         "the only way to populate the 'misattributed' verdict "
                         "(a real span that does not support what it was cited "
                         "for), at one LLM call per verified span")
    args = ap.parse_args()
    run_quote_provenance(run_id=args.run,
                         adjudicate_verbatim=args.adjudicate_verbatim)
