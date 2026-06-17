"""Stage 1: build the vector index from the corpora.

Usage:
    python scripts/01_ingest.py            # incremental (upserts)
    python scripts/01_ingest.py --fresh    # delete and rebuild the collection
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag.ingest import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true",
                    help="delete the existing collection and rebuild from scratch")
    args = ap.parse_args()
    build_index(fresh=args.fresh)
