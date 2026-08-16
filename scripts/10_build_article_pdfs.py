"""Stage 10 (optional, LOCAL ONLY): one PDF per third-party press record.

    python scripts/10_build_article_pdfs.py --dry-run   # report, write nothing
    python scripts/10_build_article_pdfs.py --limit 5   # 5 records per file
    python scripts/10_build_article_pdfs.py            # the full ~3,000
    python scripts/10_build_article_pdfs.py --check-ids # compare against the index

Writes one PDF per record into data/articles/<Org>/ plus the map
data/article_sources.json, which is what turns a third-party `[excerpt N]` on
the Audit tab into a link to the article the span came from. Before this, those
citations were unlinkable: the chunk metadata names only `O1.RTF`, a 45 MB Nexis
export bundling 500 articles.

Needs `pip install -r requirements-pdf.txt` and IL_PROFILER_DATASET_ROOT set.
Makes no API calls and does not touch the vector index — ids are reconstructed
by reusing ingestion's own splitting and chunking, so they match by construction.

Read il_rag/article_pdfs.py first: the record-pairing rule it implements is
coupled to a quirk of ingest.ARTICLE_SPLIT_RE, and that coupling is documented
there rather than here.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag import article_pdfs
from il_rag.config import ARTICLE_PDF_DIR, PDF_DATASET_ROOT


def check_ids() -> None:
    """Compare the SAVED map against the third-party ids actually indexed.

    The map is only correct if a real ingest read the same six RTFs. That has
    not been exercised while config.ARTICLE_DIRS points at a stale layout, so
    this is the way to find out rather than assume.

    Reads the map from disk rather than from this run, so it checks exactly
    what the app will read — and so it still means something under --dry-run,
    which writes nothing.
    """
    import chromadb
    from chromadb.config import Settings

    from il_rag.config import CHROMA_DIR, COLLECTION_NAME
    from il_rag.pdf_sources import load_article_sources

    articles = load_article_sources()
    if not articles:
        print("\n--check-ids: no saved map yet — run without --dry-run first")
        return
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        res = client.get_collection(COLLECTION_NAME).get(
            where={"source_type": "thirdparty"}, include=[])
    except Exception as e:  # noqa: BLE001
        print(f"\n--check-ids: no usable index ({e})")
        return
    indexed = set(res["ids"])
    if not indexed:
        print("\n--check-ids: the index holds no third-party chunks yet")
        return
    missing = [cid for cid in sorted(indexed)
               if cid.rsplit("|", 1)[0] not in articles]
    covered = len(indexed) - len(missing)
    print(f"\n--check-ids: {covered}/{len(indexed)} indexed third-party chunks "
          f"have an article ({covered / len(indexed):.1%})")
    # Chunks whose fragment index is 0 are the Nexis job header — genuinely
    # article-less, so they are expected here rather than a problem.
    headers = [c for c in missing if c.split("|")[3] == "0"]
    if headers:
        print(f"    {len(headers)} of the unmapped are ai=0 job headers (expected)")
    for cid in [c for c in missing if c not in headers][:5]:
        print(f"    unmapped: {cid}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts without writing any file")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N records per RTF (for a fast sample)")
    ap.add_argument("--force", action="store_true",
                    help="re-render PDFs that already exist")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the pypdf round-trip check (faster re-runs)")
    ap.add_argument("--check-ids", action="store_true",
                    help="also compare the map against the vector index")
    args = ap.parse_args()

    if PDF_DATASET_ROOT is None:
        sys.exit("IL_PROFILER_DATASET_ROOT is not set — see .env.example.")
    if not PDF_DATASET_ROOT.is_dir():
        sys.exit(f"dataset root does not exist: {PDF_DATASET_ROOT}")

    dirs = article_pdfs.article_dirs(PDF_DATASET_ROOT)
    print(f"reading press dumps from {PDF_DATASET_ROOT}")
    print(f"writing PDFs to {ARTICLE_PDF_DIR}\n")

    articles, stats = article_pdfs.build_articles(
        dirs, ARTICLE_PDF_DIR, limit=args.limit, force=args.force,
        verify=not args.no_verify, dry_run=args.dry_run)

    for org, s in stats.items():
        if org.startswith("_"):
            continue
        print(f"  {org:<10} records={s['records']:<5} written={s['written']:<5} "
              f"skipped={s['skipped']:<4} no-identity={s['no_identity']:<4} "
              f"mapped-fragments={s['fragments']:<5} "
              f"{s['bytes'] / 1048576:.1f} MB")

    subs = stats.get("_substitution_total", 0)
    print(f"\n  characters outside cp1252 substituted: {subs}")
    for ch, n in stats.get("_substitutions", {}).items():
        print(f"      U+{ord(ch):04X} x{n}")
    if "_verify_min" in stats:
        print(f"  pypdf round-trip: mean {stats['_verify_mean']:.1%}, "
              f"worst {stats['_verify_min']:.1%}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        path = article_pdfs.save_article_sources(articles)
        print(f"\nwrote {len(articles)} fragment entries -> {path}")

    if args.check_ids:
        check_ids()
