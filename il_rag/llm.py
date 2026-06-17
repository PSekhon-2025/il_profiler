"""Together AI client wrappers: chat and embeddings, with transient-error retry.

Why this exists: long ingestion and evaluation runs make hundreds of API calls;
a single 502 or rate-limit must not kill a run. Transient errors are retried
with exponential backoff (~3 minutes total). Anything non-transient is raised
immediately so real bugs surface instead of being retried forever.
"""
import time

from together import Together

from .config import EMBEDDING_MODEL, GENERATION_MODEL, require_api_key

# Exponential backoff schedule (seconds). Generous because the eval is
# resumable — better to wait out a blip than abort a half-finished run.
BACKOFF = [2, 5, 10, 20, 40, 60, 60]

_client: Together | None = None


def client() -> Together:
    """Singleton Together client (reuses HTTP connections across calls)."""
    global _client
    if _client is None:
        _client = Together(api_key=require_api_key())
    return _client


def _is_transient(err: Exception) -> bool:
    """Heuristic: is this error worth retrying (server-side / network blip)?"""
    msg = str(err)
    low = msg.lower()
    return (
        any(code in msg for code in ("429", "500", "502", "503", "504"))
        or "connection" in low
        or "timeout" in low
        or "temporarily unavailable" in low
    )


def chat(messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Chat completion with retry. Temperature defaults to 0 for determinism."""
    last = None
    for wait in BACKOFF:
        try:
            resp = client().chat.completions.create(
                model=GENERATION_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — classify and re-raise below
            if not _is_transient(e):
                raise
            last = e
            print(f"  [retry] transient chat error, sleeping {wait}s: {type(e).__name__}")
            time.sleep(wait)
    raise RuntimeError(f"Together chat API still failing after retries: {last}")


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with retry. Raises on persistent failure.

    Oversize (400) errors are NOT handled here — the caller (ingest) bisects
    the batch, because only it knows how to drop a single bad chunk.
    """
    last = None
    for wait in BACKOFF:
        try:
            resp = client().embeddings.create(input=texts, model=EMBEDDING_MODEL)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            if not _is_transient(e):
                raise
            last = e
            print(f"  [retry] transient embed error, sleeping {wait}s: {type(e).__name__}")
            time.sleep(wait)
    raise RuntimeError(f"Together embeddings API still failing after retries: {last}")
