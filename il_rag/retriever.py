"""Semantic retrieval over the Chroma index, scoped by (org, source_type).

The compound metadata filter is the mechanism that keeps the study's six
profiles independent: a question about OpenAI's published documents can never
be answered from DeepMind chunks or from press coverage.
"""
import re
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from .llm import embed

# The third-party article corpus (Nexis RTF exports) is heavily duplicated:
# the same article is syndicated across outlets, and every export repeats a
# LexisNexis company-profile boilerplate block (PermID / Address / Website /
# classification codes). Without dedup, all k retrieved slots collapse onto one
# repeated passage — effectively k=1 of evidence — which starves the answer and
# drives spurious abstentions. We over-fetch and keep the top-k DISTINCT chunks.
DEDUP_FETCH_MULTIPLIER = 6
# Length of the normalized text prefix used as a near-duplicate signature.
_DEDUP_SIGNATURE_CHARS = 240


def _dedup_signature(text: str) -> str:
    """Normalized prefix used to detect near-duplicate chunks."""
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return norm[:_DEDUP_SIGNATURE_CHARS]


@dataclass
class Chunk:
    """One retrieved chunk with provenance and similarity score."""
    id: str
    text: str
    org: str
    source_type: str
    filename: str
    score: float  # cosine similarity in [0, 1]


class Retriever:
    def __init__(self) -> None:
        chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = chroma.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str, *, org: str | None = None,
                 source_type: str | None = None, k: int = TOP_K) -> list[Chunk]:
        """Return the k most similar chunks, optionally scoped.

        Chroma requires an explicit $and when filtering on multiple fields.
        """
        clauses = []
        if org:
            clauses.append({"org": org})
        if source_type:
            clauses.append({"source_type": source_type})
        where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)

        # Over-fetch so we can drop near-duplicates and still return k distinct
        # chunks. Ranking is preserved (Chroma returns nearest-first).
        vector = embed([query])[0]
        res = self.collection.query(
            query_embeddings=[vector], n_results=k * DEDUP_FETCH_MULTIPLIER, where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[Chunk] = []
        seen: set[str] = set()
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            sig = _dedup_signature(doc)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(Chunk(
                id=cid, text=doc,
                org=meta.get("org", ""), source_type=meta.get("source_type", ""),
                filename=meta.get("filename", ""), score=1.0 - dist,
            ))
            if len(out) >= k:
                break
        return out

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Refetch chunks by id, preserving the given order (provenance replay).

        Used by the metamorphic eval to reconstruct the exact evidence a past
        run answered from. There is no query, so similarity is undefined —
        score is set to 0.0 and must not be interpreted. Ids missing from the
        collection (e.g. after a --fresh reingest) are silently dropped; the
        caller decides whether a shortened set is still usable.
        """
        if not ids:
            return []
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {
            cid: (doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        }
        out = []
        for cid in ids:
            if cid not in by_id:
                continue
            doc, meta = by_id[cid]
            out.append(Chunk(
                id=cid, text=doc,
                org=meta.get("org", ""), source_type=meta.get("source_type", ""),
                filename=meta.get("filename", ""), score=0.0,
            ))
        return out
