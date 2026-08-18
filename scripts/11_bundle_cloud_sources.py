"""Stage 11 (LOCAL ONLY): bundle the source documents for the cloud volume.

    python scripts/11_bundle_cloud_sources.py            # build dist/cloud_sources.tar
    python scripts/11_bundle_cloud_sources.py --dry-run  # list contents and size

The deployed app can link an `[excerpt N]` to its source only if that source is
on the machine. The image deliberately ships neither the corpus nor the
generated article PDFs (both are .dockerignore'd), so they travel separately —
to the persistent volume mounted at /app/data, exactly the way the prebuilt
index and the topic layer already do.

The archive unpacks INTO /app/data and contains:

    corpus/<Org>/<Category>/<name>.pdf   the published PDFs the corpus cites
    articles/<Org>/<stem>__<ai>.pdf      one per third-party press record
    pdf_pages.json                       chunk -> page, for #page= anchors
    article_sources.json                 chunk -> press record + page

Only PDFs actually referenced by a pdf_corpus.txt are included — the dataset
holds others that no chunk can ever cite. The `<Org>/<Category>/` layout is
preserved because pdf_sources.resolve breaks basename ties by looking for the
chunk's org among the path parts.

Deployment steps and the access caveat are in DEPLOY.md; read them before
running this, because publishing the corpus makes it reachable at a guessable
URL for anyone who can reach the host.
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from il_rag import pdf_sources
from il_rag.config import (ARTICLE_PDF_DIR, DATA_DIR, PDF_DATASET_ROOT,
                           PROJECT_ROOT)
from il_rag.ingest import split_published_corpus


def corpus_pdfs() -> dict[str, Path]:
    """{archive path: source path} for every PDF a published chunk can cite."""
    index = pdf_sources.build_index(PDF_DATASET_ROOT)
    out: dict[str, Path] = {}
    for org, corpus in pdf_sources.published_corpora(PDF_DATASET_ROOT).items():
        if not corpus.exists():
            print(f"[warn] missing published corpus for {org}: {corpus}")
            continue
        raw = corpus.read_text(encoding="utf-8", errors="ignore")
        for filename, _ in split_published_corpus(raw):
            src = pdf_sources.resolve(filename, org, index)
            if src is None:
                print(f"[warn] {org}: no PDF on disk for {filename}")
                continue
            # Keep the path relative to the dataset root so the org-preference
            # tie-break still works on the other side.
            rel = src.relative_to(PDF_DATASET_ROOT).as_posix()
            out[f"corpus/{rel}"] = src
    return out


def article_pdfs() -> dict[str, Path]:
    """{archive path: source path} for the generated press-record PDFs."""
    if not ARTICLE_PDF_DIR.is_dir():
        return {}
    return {f"articles/{p.relative_to(ARTICLE_PDF_DIR).as_posix()}": p
            for p in sorted(ARTICLE_PDF_DIR.rglob("*.pdf"))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would go in, and how big, without writing")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "dist" / "cloud_sources.tar"),
                    help="archive path (default: dist/cloud_sources.tar)")
    args = ap.parse_args()

    if PDF_DATASET_ROOT is None or not PDF_DATASET_ROOT.is_dir():
        sys.exit("IL_PROFILER_DATASET_ROOT is not set or does not exist.")

    members = corpus_pdfs()
    members.update(article_pdfs())
    for name in ("pdf_pages.json", "article_sources.json"):
        path = DATA_DIR / name
        if path.exists():
            members[name] = path
        else:
            print(f"[warn] {name} not built yet — "
                  f"{'page anchors' if 'pages' in name else 'press links'} "
                  "will be missing in the cloud")

    groups: dict[str, list[int]] = {}
    for arc, src in members.items():
        groups.setdefault(arc.split("/")[0] if "/" in arc else "maps", []).append(
            src.stat().st_size)
    print()
    for group, sizes in sorted(groups.items()):
        print(f"  {group:<10} {len(sizes):>5} files  {sum(sizes) / 1048576:8.1f} MB")
    total = sum(sum(s) for s in groups.values())
    print(f"  {'TOTAL':<10} {len(members):>5} files  {total / 1048576:8.1f} MB")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        sys.exit()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed on purpose: PDFs are already deflate-compressed, so gzip
    # costs minutes and saves almost nothing.
    with tarfile.open(out, "w") as tar:
        for arc, src in sorted(members.items()):
            tar.add(src, arcname=arc)
    print(f"\nwrote {out}  ({out.stat().st_size / 1048576:.1f} MB)")
    print("\nNext (see DEPLOY.md):")
    print(f"  fly sftp put {out} /app/data/cloud_sources.tar")
    print("  fly ssh console -C \"sh -c 'cd /app/data && tar xf cloud_sources.tar "
          "&& rm cloud_sources.tar'\"")
    print("  fly apps restart <app>")
