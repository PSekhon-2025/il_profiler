"""Metamorphic label-stability eval (after MetaQA, Yang et al. FSE 2025).

Hypothesis: a label that is genuinely grounded in the retrieved text should
survive meaning-preserving perturbations of that text. For each answered item
of an existing run, this module:

  1. refetches the exact chunks the item was answered from (by stored ids),
  2. generates k meaning-preserving PARAPHRASES of those chunks (LLM, nonzero
     temperature so variants differ) and 1 LAB-NAME SWAP (deterministic regex:
     the source org's aliases are replaced by a different lab, the described
     decision stays intact — no LLM, so the swap cannot drift the meaning),
  3. pushes each variant through the production answer -> match path
     (answer_question with injected chunks, then match_graded), and
  4. compares each variant's predicted label (argmax logic, or "abstain")
     against the original run's label.

Metrics per item:
  label_stability     fraction of paraphrase variants whose label matches the
                      original — low values flag the item as unstable;
  swap_label_changed  whether the lab-swap variant's label differs from the
                      original. Since the decision text is unchanged, a
                      grounded label should SURVIVE the swap; a flip suggests
                      the model keyed on its prior about the named lab rather
                      than on the text.

Self-referential caveat: the same model generates the paraphrases and
classifies them, so a "meaning-preserving" paraphrase is only as trustworthy
as the model's own judgment. Stability numbers should be anchored by human
review of a small sample of variants (see README).

Everything is black-box: chat + the existing retriever, no logits or weights.
Outputs live under the source run's folder (data/profiles/runs/<run_id>/
metamorphic/) and the variant loop is resumable in the same append-and-skip
style as the profile harness.
"""
import dataclasses
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from . import runs
from .config import (
    LAB_ALIASES,
    LAB_SWAP,
    METAMORPHIC_PARAPHRASES,
    METAMORPHIC_PARAPHRASE_TEMPERATURE,
    METAMORPHIC_STABILITY_THRESHOLD,
)
from .graded_matcher import match_graded
from .json_utils import extract_json
from .llm import chat
from .questionnaire import LOGICS
from .rag_qa import answer_question
from .retriever import Chunk, Retriever

VARIANTS_NAME = "variants.jsonl"
STABILITY_NAME = "stability.json"

PARAPHRASE_SYSTEM = (
    "You rewrite text. Paraphrase each numbered excerpt so that every fact, "
    "name, number, date, and claim is preserved exactly while the wording and "
    "sentence structure change substantially. Never add, drop, soften, or "
    "strengthen information. Never change organization, product, or person "
    "names."
)

PARAPHRASE_TEMPLATE = """Paraphrase each of the {n} numbered excerpts below. Preserve all facts and names exactly; change only the wording.

{context}

Return strictly this JSON object, nothing else:
{{"paraphrases": ["<paraphrase of excerpt 1>", "<paraphrase of excerpt 2>", ...]}}
The list must contain exactly {n} strings, in the same order as the excerpts."""


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------
def swap_lab_text(text: str, from_org: str, to_org: str) -> str:
    """Replace every alias of from_org with to_org's canonical name.

    Case-insensitive with word boundaries, longest alias first so
    "Google DeepMind" doesn't decay into "Google Anthropic... DeepMind".
    """
    aliases = sorted(LAB_ALIASES.get(from_org, [from_org]), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
    return re.sub(pattern, to_org, text, flags=re.IGNORECASE)


def paraphrase_texts(texts: list[str]) -> list[str] | None:
    """Paraphrase all excerpts of one item in a single chat call.

    One call per variant (not per chunk) keeps the cost at k calls per item.
    Returns None when the reply doesn't parse into exactly len(texts) strings
    even after a retry — the caller records the variant as failed rather than
    grading a half-paraphrased context.
    """
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM},
        {"role": "user", "content": PARAPHRASE_TEMPLATE.format(
            n=len(texts), context=context)},
    ]
    # Reasoning model: the budget must cover hidden reasoning plus a full
    # rewrite of up to TOP_K chunks (~7000 chars); a truncated reply parses as
    # nothing, so parse failure retries once with a larger budget.
    for max_tokens in (8192, 12288):
        raw = chat(messages, temperature=METAMORPHIC_PARAPHRASE_TEMPERATURE,
                   max_tokens=max_tokens)
        parsed = extract_json(raw)
        if parsed is None:
            continue
        out = parsed.get("paraphrases")
        if (isinstance(out, list) and len(out) == len(texts)
                and all(isinstance(t, str) and t.strip() for t in out)):
            return [t.strip() for t in out]
    return None


# ---------------------------------------------------------------------------
# Labels and stability math (pure — no API)
# ---------------------------------------------------------------------------
def predicted_label(abstain: bool, weights: dict) -> str:
    """Collapse a matcher verdict to one label: "abstain" or the argmax logic.

    Ties break by LOGICS order, so the label is deterministic for a given
    weight vector.
    """
    if abstain:
        return "abstain"
    return max(LOGICS, key=lambda logic: float(weights.get(logic, 0.0)))


def row_label(row: dict) -> str:
    return predicted_label(bool(row["abstain"]), row.get("weights", {}))


def compute_stability(src_rows: list[dict], variant_rows: list[dict],
                      threshold: float = METAMORPHIC_STABILITY_THRESHOLD,
                      ) -> tuple[list[dict], dict]:
    """Fold variant rows into per-item stability records and an aggregate.

    Failed variants (rows carrying "error") are excluded from denominators:
    stability measures label behavior on variants that actually ran, and the
    failure counts are reported separately.
    """
    by_item: dict[tuple, list[dict]] = {}
    for v in variant_rows:
        by_item.setdefault((v["org"], v["source_type"], v["qid"]), []).append(v)

    per_item = []
    for r in src_rows:
        key = (r["org"], r["source_type"], r["qid"])
        vs = by_item.get(key, [])
        paras = [v for v in vs if v["variant_kind"] == "paraphrase"]
        paras_ok = [v for v in paras if not v.get("error")]
        matches = sum(1 for v in paras_ok if v["label_matches_original"])
        stability = round(matches / len(paras_ok), 4) if paras_ok else None
        swap = next((v for v in vs if v["variant_kind"] == "lab_swap"
                     and not v.get("error")), None)
        item = {
            "org": r["org"],
            "source_type": r["source_type"],
            "qid": r["qid"],
            "category": r["category"],
            "original_label": row_label(r),
            "n_paraphrases": len(paras),
            "n_paraphrases_ok": len(paras_ok),
            "label_stability": stability,
            "unstable": stability is not None and stability < threshold,
            "swap_to": swap["swap_to"] if swap else None,
            "swap_label": swap["label"] if swap else None,
            "swap_label_changed": (not swap["label_matches_original"]) if swap else None,
        }
        if "grounding_bucket" in r:
            item["grounding_bucket"] = r["grounding_bucket"]
        per_item.append(item)

    scored = [i for i in per_item if i["label_stability"] is not None]
    swapped = [i for i in per_item if i["swap_label_changed"] is not None]

    def _mean_stability(items):
        vals = [i["label_stability"] for i in items if i["label_stability"] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_category = {}
    for i in per_item:
        by_category.setdefault(i["category"], []).append(i)
    by_bucket = {}
    for i in per_item:
        if "grounding_bucket" in i:
            by_bucket.setdefault(i["grounding_bucket"], []).append(i)

    summary = {
        "items": len(per_item),
        "items_scored": len(scored),
        "mean_label_stability": _mean_stability(scored),
        "pct_fully_stable": (
            round(100.0 * sum(1 for i in scored if i["label_stability"] >= 1.0)
                  / len(scored), 1) if scored else None),
        "n_unstable": sum(1 for i in per_item if i["unstable"]),
        "stability_threshold": threshold,
        "n_swap_evaluated": len(swapped),
        "n_swap_label_changed": sum(1 for i in swapped if i["swap_label_changed"]),
        "swap_flip_rate": (
            round(sum(1 for i in swapped if i["swap_label_changed"]) / len(swapped), 4)
            if swapped else None),
        "by_category": {c: _mean_stability(items) for c, items in sorted(by_category.items())},
    }
    if by_bucket:
        summary["by_grounding_bucket"] = {
            b: {"n": len(items), "mean_label_stability": _mean_stability(items)}
            for b, items in sorted(by_bucket.items())
        }
    return per_item, summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
    return rows


def _variant_key(v: dict) -> tuple:
    return (v["org"], v["source_type"], v["qid"], v["variant_kind"], v["variant_idx"])


def _run_variant(row: dict, kind: str, idx: int, chunks: list[Chunk],
                 original_label: str) -> dict:
    """Generate one variant context, answer from it, grade it, label it."""
    out = {
        "org": row["org"], "source_type": row["source_type"], "qid": row["qid"],
        "category": row["category"], "variant": row["variant"],
        "variant_kind": kind, "variant_idx": idx,
        "original_label": original_label,
    }
    if kind == "paraphrase":
        texts = paraphrase_texts([c.text for c in chunks])
        if texts is None:
            out["error"] = "paraphrase_failed"
            return out
        vchunks = [dataclasses.replace(c, text=t) for c, t in zip(chunks, texts)]
        question = row["question"]
    else:  # lab_swap
        to_org = LAB_SWAP.get(row["org"])
        if not to_org:
            out["error"] = "no_swap_target"
            return out
        vchunks = [dataclasses.replace(c, org=to_org,
                                       text=swap_lab_text(c.text, row["org"], to_org))
                   for c in chunks]
        question = swap_lab_text(row["question"], row["org"], to_org)
        out["swap_to"] = to_org

    rag = answer_question(None, question, org=row["org"],
                          source_type=row["source_type"], chunks=vchunks)
    verdict = match_graded(question=question, candidate=rag.answer,
                           category=row["category"], variant=row["variant"])
    label = predicted_label(verdict["abstain"], verdict["weights"])
    out.update({
        "question": question,
        "answer": rag.answer,
        "abstain": verdict["abstain"],
        "weights": verdict["weights"],
        "reasoning": verdict["reasoning"],
        "label": label,
        "label_matches_original": label == original_label,
    })
    return out


def run_metamorphic(run_id: str | None = None,
                    paraphrases: int = METAMORPHIC_PARAPHRASES,
                    sample: int | None = None, seed: int = 0,
                    orgs: list[str] | None = None,
                    source_types: list[str] | None = None) -> dict:
    """Run the metamorphic eval against an existing run snapshot.

    Reads the run's per_question.jsonl, produces `paraphrases` + 1 variants per
    (sampled) item, and writes variants.jsonl + stability.json into
    <run_dir>/metamorphic/. Resumable: completed variants are skipped on rerun;
    failed ones (error rows) are retried. Sampling is deterministic for a given
    seed and filter set, so a resumed sampled run continues the same sample.
    Returns the aggregate summary dict.
    """
    run_id = run_id or runs.get_current()
    if run_id is None:
        raise RuntimeError("no run snapshot found — run scripts/02_run_profiles.py first")
    paths = runs.run_paths(run_id)
    src_rows = _load_jsonl(paths["per_question"])
    if not src_rows:
        raise RuntimeError(f"run {run_id} has no per-question rows to evaluate")

    if orgs:
        src_rows = [r for r in src_rows if r["org"] in orgs]
    if source_types:
        src_rows = [r for r in src_rows if r["source_type"] in source_types]
    if sample and sample < len(src_rows):
        src_rows = random.Random(seed).sample(src_rows, sample)

    out_dir = paths["dir"] / "metamorphic"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_path = out_dir / VARIANTS_NAME

    # Resume: keep the LAST row per key (a retried failure supersedes the old
    # error row); only error-free variants count as done.
    existing: dict[tuple, dict] = {}
    for v in _load_jsonl(variants_path):
        existing[_variant_key(v)] = v
    done = {k for k, v in existing.items() if not v.get("error")}
    if done:
        print(f"resuming: {len(done)} variants already evaluated for run {run_id}")

    kinds = [("paraphrase", i) for i in range(1, paraphrases + 1)] + [("lab_swap", 0)]
    todo = [
        (row, kind, idx)
        for row in src_rows
        for kind, idx in kinds
        if (row["org"], row["source_type"], row["qid"], kind, idx) not in done
    ]

    retriever = Retriever() if todo else None
    chunk_cache: dict[tuple, list[Chunk]] = {}
    with open(variants_path, "a", encoding="utf-8") as f:
        for row, kind, idx in tqdm(todo, desc="metamorphic", unit="variant"):
            item_key = (row["org"], row["source_type"], row["qid"])
            if item_key not in chunk_cache:
                chunk_cache[item_key] = retriever.get_by_ids(row.get("retrieved_ids", []))
            chunks = chunk_cache[item_key]
            if not chunks:
                v = {
                    "org": row["org"], "source_type": row["source_type"],
                    "qid": row["qid"], "category": row["category"],
                    "variant": row["variant"], "variant_kind": kind,
                    "variant_idx": idx, "original_label": row_label(row),
                    "error": "chunks_not_found",
                }
            else:
                v = _run_variant(row, kind, idx, chunks, row_label(row))
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
            f.flush()  # crash-safe, same contract as the profile harness
            existing[_variant_key(v)] = v

    per_item, summary = compute_stability(src_rows, list(existing.values()))
    summary = {
        "run_id": run_id,
        "paraphrases_per_item": paraphrases,
        "sample": sample,
        "seed": seed,
        **summary,
    }
    with open(out_dir / STABILITY_NAME, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_item": per_item}, f,
                  ensure_ascii=False, indent=2)

    _print_report(summary, out_dir)
    return summary


def _print_report(summary: dict, out_dir: Path) -> None:
    print(f"\n=== Metamorphic label stability [run {summary['run_id']}] ===")
    print(f"items: {summary['items']} ({summary['paraphrases_per_item']} paraphrases "
          f"+ 1 lab swap each), scored: {summary['items_scored']}")
    if summary["mean_label_stability"] is not None:
        print(f"mean label stability: {summary['mean_label_stability']:.3f}")
        print(f"fully stable items:   {summary['pct_fully_stable']:.1f}%")
    print(f"unstable items (stability < {summary['stability_threshold']}): "
          f"{summary['n_unstable']}")
    if summary["n_swap_evaluated"]:
        print(f"lab-swap label flips: {summary['n_swap_label_changed']}"
              f"/{summary['n_swap_evaluated']} "
              f"(flip rate {summary['swap_flip_rate']:.3f}) — a flip suggests the "
              f"label was keyed on the lab's name, not the text")
    print("stability by category:")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<24} {'—' if s is None else f'{s:.3f}'}")
    if "by_grounding_bucket" in summary:
        print("stability by grounding bucket:")
        for b, s in summary["by_grounding_bucket"].items():
            ms = s["mean_label_stability"]
            print(f"  {b:<17} n={s['n']:<4} {'—' if ms is None else f'{ms:.3f}'}")
    print(f"\noutputs: {out_dir}")
