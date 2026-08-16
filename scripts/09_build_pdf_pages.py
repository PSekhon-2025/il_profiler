"""Stage 9 (optional, LOCAL ONLY): page map for the audit tab's excerpt links.

    python scripts/09_build_pdf_pages.py            # build and save the map
    python scripts/09_build_pdf_pages.py --dry-run  # report hit rates only

Writes data/pdf_pages.json = {chunk_id: page number}, which turns each
[excerpt N] citation on the Audit tab from "opens the PDF" into "opens the PDF
at the page the span is on". The app reads that JSON and nothing else — it
never imports pypdf — so this mirrors the topic layer: heavy dependency and raw
corpus stay on the workstation, only the small result is used at runtime.

Needs `pip install -r requirements-pdf.txt` and IL_PROFILER_DATASET_ROOT set.
Makes no API calls and does not touch the vector index: chunk ids are
reconstructed by re-running ingestion's own splitting and chunking, so they
match whatever an ingest of the same corpus produced.

The corpus text files were converted by an external tool on another machine,
while this reads the PDFs with pypdf — the two extractions do not always agree,
so the run reports a per-lab hit rate. Chunks that cannot be placed are simply
left out (their links open at page 1); a poor overall rate is a reason not to
keep the file, not something to work around.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag import pdf_sources
from il_rag.config import PDF_DATASET_ROOT

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report hit rates without writing the map")
    args = ap.parse_args()

    if PDF_DATASET_ROOT is None:
        sys.exit("IL_PROFILER_DATASET_ROOT is not set — see .env.example.")
    if not PDF_DATASET_ROOT.is_dir():
        sys.exit(f"dataset root does not exist: {PDF_DATASET_ROOT}")

    index = pdf_sources.build_index(PDF_DATASET_ROOT)
    print(f"{len(index)} distinct PDF names under {PDF_DATASET_ROOT}")

    page_map, stats = pdf_sources.build_page_map(
        pdf_sources.published_corpora(PDF_DATASET_ROOT), index)

    print()
    for org, s in stats.items():
        print(f"  {org:<10} {s['mapped']:>5}/{s['chunks']:<5} chunks placed "
              f"({s['rate']:.0%})")
    total = sum(s["chunks"] for s in stats.values())
    mapped = sum(s["mapped"] for s in stats.values())
    print(f"  {'overall':<10} {mapped:>5}/{total:<5} "
          f"({(mapped / total if total else 0):.0%})")

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        print(f"\nwrote {len(page_map)} entries -> {pdf_sources.save_page_map(page_map)}")
