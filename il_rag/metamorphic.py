"""Metamorphic eval: label stability + evidence sensitivity.

There are no gold labels in this pipeline, so correctness cannot be checked
directly. Instead this check perturbs the evidence an answer was built from and
watches what the label does — the metamorphic-testing move for the no-oracle
setting (Chen, Cheung & Yiu, 1998). Four probes run against an existing run's
saved items, each answering the same question through the SAME production path
(`answer_question` -> `match_graded`, temperature 0) so any change in the label
is attributable to the perturbation alone:

  control      run it again, change nothing. Any flip here is pipeline noise,
               not a finding — this is the FLOOR the other numbers are read
               against, and an item whose control flipped cannot be called
               unstable.
  paraphrase   say the same thing differently. A label grounded in the text
               should survive a meaning-preserving rewrite. INVARIANCE: a flip
               is suspicious.
  ablation     take away the excerpt the answer quoted. A grounded label should
               weaken toward abstention. DIRECTIONAL: a label that SURVIVES is
               suspicious — the answer did not need that evidence.
  distractor   ask the same question over real text from the same lab that does
               not address it. The right answer is "the excerpts don't say".
               DIRECTIONAL: reproducing the ORIGINAL label from irrelevant
               evidence is the strongest hallucination signal available here.

Paraphrase fidelity is verified IN CODE, never trusted to the model, because a
rewrite that quietly changed meaning would be scored as a label flip and read
as a hallucination. Three gates (see `check_paraphrase_text` /
`check_paraphrase_semantics`): facts kept, actually reworded, still means the
same. A rewrite failing any gate is DISCARDED, not counted — a bad paraphrase
is evidence of nothing about stability.

Metrics per item:
  label_stability          fraction of gate-passing paraphrases keeping the
                           original label — low values flag the item unstable;
  control_flipped          the unperturbed re-run changed the label (noise);
  label_survived_ablation  the label held without the evidence it quoted;
  prior_keyed              the same label came back from irrelevant evidence.

A note on what this check no longer does: earlier versions included a
lab-name swap (rewriting "OpenAI" to "DeepMind" throughout the evidence). It
was removed because the perturbation is not meaning-preserving — renaming the
lab changes what the question is asking, and the regex left lab-specific
details (product names, people) untouched, so the swapped context was
self-contradictory and a flip could not be attributed to the name. The
distractor probe measures the same underlying worry — is the label coming from
the model's prior about the lab rather than the text? — using real, correctly
named corpus text. See ARCHITECTURE.md §9.3.

Everything is black-box: chat + embeddings + the existing retriever, no logits
or weights. Outputs live under the source run's folder
(data/profiles/runs/<run_id>/metamorphic/) and the variant loop is resumable in
the same append-and-skip style as the profile harness.
"""
import dataclasses
import json
import math
import random
import re
from pathlib import Path

from tqdm import tqdm

from . import runs
from .config import (
    GROUNDING_LOW_THRESHOLD,
    METAMORPHIC_CONTROLS,
    METAMORPHIC_PARAPHRASES,
    METAMORPHIC_PARAPHRASE_TEMPERATURE,
    METAMORPHIC_PROBES,
    METAMORPHIC_STABILITY_THRESHOLD,
    PARAPHRASE_MAX_TOKEN_OVERLAP,
    PARAPHRASE_MIN_COSINE,
    TOP_K,
)
from .graded_matcher import match_graded
from .grounding import STOPWORDS, grounding_scores, lexical_overlap
from .json_utils import extract_json
from .llm import chat, embed
from .questionnaire import CATEGORIES, LOGICS, QUESTIONNAIRE
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
# Paraphrase fidelity gates (pure — no API)
#
# The model is asked to preserve meaning, but nothing in a prompt can enforce
# that. These gates enforce what CAN be checked mechanically, so that a scored
# "flip" always means the classifier moved, never that the text did.
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\d[\d,.:%/-]*")
# Deliberately excludes apostrophes, dots and hyphens so that "OpenAI's" and
# "OpenAI" yield the same token and "GPT-4" splits into the name "GPT" plus a
# number the number gate already covers — otherwise a faithful rewrite that
# merely dropped a possessive would be rejected.
_CAPITALISED_RE = re.compile(r"\b[A-Z][A-Za-z0-9&]*")
_LOWERCASE_RE = re.compile(r"\b[a-z][a-z0-9&]*")


def numbers_in(text: str) -> set[str]:
    """Numeric tokens (counts, dates, percentages), trailing punctuation stripped."""
    return {m.rstrip(".,:-/") for m in _NUMBER_RE.findall(text) if m.rstrip(".,:-/")}


def entities_in(text: str) -> set[str]:
    """Capitalised tokens that look like names rather than sentence openers.

    Two filters keep ordinary words out, so the gate rejects genuinely dropped
    names instead of firing on every re-worded sentence start:
      - the token is not a stopword ("The", "It", "When", ...);
      - the same word does not also appear lowercased in the text, which is
        what happens when a common noun merely starts a sentence.
    """
    lowered = set(_LOWERCASE_RE.findall(text))
    out = set()
    for tok in _CAPITALISED_RE.findall(text):
        low = tok.lower()
        if len(tok) < 2 or low in STOPWORDS or low in lowered:
            continue
        out.add(tok)
    return out


def check_paraphrase_text(original: str, paraphrase: str) -> dict:
    """Gates 1 and 2 for one excerpt: facts kept, and actually reworded.

        missing_numbers  = numbers(c) \\ numbers(p)
        missing_entities = entities(c) \\ entities(p)
        overlap          = |T(c) ∩ T(p)| / |T(c)|        (grounding.lexical_overlap)

    A paraphrase passes when nothing is missing and overlap <= rho
    (PARAPHRASE_MAX_TOKEN_OVERLAP): near-total overlap means the model returned
    a copy, which would inflate stability without testing anything.
    """
    missing_numbers = sorted(numbers_in(original) - numbers_in(paraphrase))
    missing_entities = sorted(entities_in(original) - entities_in(paraphrase))
    overlap = round(lexical_overlap(original, paraphrase), 4)
    reason = None
    if missing_numbers:
        reason = "dropped_numbers"
    elif missing_entities:
        reason = "dropped_entities"
    elif overlap > PARAPHRASE_MAX_TOKEN_OVERLAP:
        reason = "not_reworded"
    return {
        "ok": reason is None,
        "reason": reason,
        "missing_numbers": missing_numbers,
        "missing_entities": missing_entities,
        "overlap": overlap,
    }


def cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity; 0.0 if either vector is degenerate."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def check_paraphrase_semantics(orig_vecs: list[list[float]],
                               para_vecs: list[list[float]],
                               min_cosine: float = PARAPHRASE_MIN_COSINE) -> dict:
    """Gate 3: did the rewrite stay on the same subject?

    Primarily RANK-based, because e5 compresses absolute cosines into a narrow
    band (the same reason check 4 only reads its rankings): each paraphrase
    must be nearer to its OWN source excerpt than to any sibling excerpt of the
    item. `min_cosine` is a coarse sanity floor on top of that, not a
    calibrated threshold.

        nearest(i) = argmax_j cos(orig_j, para_i)     must equal i
        cos(orig_i, para_i) >= min_cosine
    """
    self_cosines = []
    reason = None
    for i, pv in enumerate(para_vecs):
        sims = [cosine(ov, pv) for ov in orig_vecs]
        self_cosines.append(round(sims[i], 4))
        if reason is None and max(range(len(sims)), key=lambda j: sims[j]) != i:
            reason = "drifted_to_other_excerpt"
        if reason is None and sims[i] < min_cosine:
            reason = "low_similarity"
    return {
        "ok": reason is None,
        "reason": reason,
        "cosines": self_cosines,
        "min_cosine": min(self_cosines) if self_cosines else None,
    }


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------
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


def build_paraphrase(texts: list[str], attempts: int = 2
                     ) -> tuple[list[str] | None, dict]:
    """Generate a paraphrase set and put it through all three fidelity gates.

    Retries once on failure (the rewrite is sampled, so a second draw often
    passes). Returns (paraphrases or None, fidelity record). The record is
    stored on the variant row either way, so a rejection is auditable rather
    than silent.
    """
    fidelity: dict = {"ok": False, "reason": "paraphrase_unparseable", "attempts": 0}
    for attempt in range(1, attempts + 1):
        out = paraphrase_texts(texts)
        if out is None:
            fidelity = {"ok": False, "reason": "paraphrase_unparseable",
                        "attempts": attempt}
            continue

        per_excerpt = [check_paraphrase_text(o, p) for o, p in zip(texts, out)]
        overlaps = [c["overlap"] for c in per_excerpt]
        failed = next((c for c in per_excerpt if not c["ok"]), None)
        fidelity = {
            "ok": False,
            "reason": None,
            "attempts": attempt,
            "divergence": round(1.0 - sum(overlaps) / len(overlaps), 4) if overlaps else None,
            "missing_numbers": sorted({n for c in per_excerpt for n in c["missing_numbers"]}),
            "missing_entities": sorted({e for c in per_excerpt for e in c["missing_entities"]}),
            "min_cosine": None,
        }
        if failed is not None:
            fidelity["reason"] = failed["reason"]
            continue

        # Gate 3 costs one batched embedding call, so it runs only after the
        # free text gates have passed.
        vecs = embed(list(texts) + list(out))
        sem = check_paraphrase_semantics(vecs[:len(texts)], vecs[len(texts):])
        fidelity["min_cosine"] = sem["min_cosine"]
        if not sem["ok"]:
            fidelity["reason"] = sem["reason"]
            continue

        fidelity["ok"] = True
        return out, fidelity
    return None, fidelity


def ablation_target(row: dict, chunks: list[Chunk]) -> tuple[int | None, str]:
    """Which excerpt to remove, and why that one.

    Preference is the excerpt the answer actually cited and whose quote check 2
    verified — that is the evidence the label demonstrably rested on. The quote
    records index into the run's ORIGINAL retrieved list, so they are only
    trusted when every id was refetched; otherwise, and when there are no
    verified quotes, the top-ranked excerpt is removed instead (retrieval
    returns nearest-first).
    """
    if not chunks:
        return None, ""
    ids = row.get("retrieved_ids") or []
    if len(chunks) == len(ids):
        for q in row.get("quotes") or []:
            if not isinstance(q, dict) or not q.get("verified"):
                continue
            try:
                idx = int(q.get("excerpt", 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(chunks):
                return idx, "verified_quote"
    return 0, "top_retrieved"


def donor_category(category: str) -> str:
    """The category whose question supplies distractor evidence.

    A fixed half-list rotation over CATEGORIES: deterministic, never returns
    its own input, and puts maximum distance between the two topics.
    """
    idx = CATEGORIES.index(category)
    return CATEGORIES[(idx + len(CATEGORIES) // 2) % len(CATEGORIES)]


def build_distractor_chunks(retriever: Retriever, row: dict, cache: dict
                            ) -> tuple[list[Chunk] | None, str | None, dict]:
    """Retrieve real same-lab evidence that does NOT address the item's question.

    Same org and source type, so names and context stay correct — only the
    topic is wrong. Any excerpt the item was actually answered from is removed,
    and the set is then re-checked against the ORIGINAL question with the
    Feature-1 grounding score: a "distractor" that happens to be relevant would
    make the probe meaningless, so it is rejected rather than scored.
    """
    donor = donor_category(row["category"])
    meta = {"distractor_category": donor}
    key = (row["org"], row["source_type"], donor)
    if key not in cache:
        question = QUESTIONNAIRE[donor]["questions"][0].format(org=row["org"])
        cache[key] = retriever.retrieve(question, org=row["org"],
                                        source_type=row["source_type"], k=TOP_K)
    exclude = set(row.get("retrieved_ids") or [])
    chunks = [c for c in cache[key] if c.id not in exclude]
    if not chunks:
        return None, "distractor_empty", meta
    score = grounding_scores(row["question"], chunks)["score"]
    meta["distractor_grounding"] = score
    if score >= GROUNDING_LOW_THRESHOLD:
        return None, "distractor_relevant", meta
    return chunks, None, meta


# ---------------------------------------------------------------------------
# Labels and stability math (pure — no API)
# ---------------------------------------------------------------------------
def predicted_label(abstain: bool, weights: dict) -> str:
    """Collapse a matcher verdict to one label: "abstain" or the argmax logic.

        label(v) = "abstain" if the matcher abstained, else argmax_k w_k

    Ties break by LOGICS order, so the label is deterministic for a given
    weight vector.
    """
    if abstain:
        return "abstain"
    return max(LOGICS, key=lambda logic: float(weights.get(logic, 0.0)))


def row_label(row: dict) -> str:
    return predicted_label(bool(row["abstain"]), row.get("weights", {}))


def _rejection_reason(v: dict) -> str:
    """Why a variant was discarded — the fidelity gate if one fired, else the error."""
    return (v.get("fidelity") or {}).get("reason") or v.get("error") or "unknown"


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_stability(src_rows: list[dict], variant_rows: list[dict],
                      threshold: float = METAMORPHIC_STABILITY_THRESHOLD,
                      ) -> tuple[list[dict], dict]:
    """Fold variant rows into per-item records and an aggregate summary.

        control_flipped         label(control) != label0     — pipeline noise
        label_stability         |{v in P_ok : label(v) = label0}| / |P_ok|
                                (P_ok = paraphrases that ran AND passed the
                                 fidelity gates; None if empty)
        unstable                label_stability < threshold AND NOT control_flipped
        label_survived_ablation label(ablation) = label0 != "abstain"
        distractor_committed    label(distractor) != "abstain"
        prior_keyed             label(distractor) = label0 != "abstain"

    Failed and gate-rejected variants are excluded from every denominator and
    are never counted as flips: a rewrite that broke the rules, or a call that
    failed, is evidence of nothing. Their counts are reported separately so the
    health of each probe stays visible.
    """
    by_item: dict[tuple, list[dict]] = {}
    for v in variant_rows:
        by_item.setdefault((v["org"], v["source_type"], v["qid"]), []).append(v)

    per_item = []
    for r in src_rows:
        key = (r["org"], r["source_type"], r["qid"])
        vs = by_item.get(key, [])
        label0 = row_label(r)

        def _ok(kind):
            return [v for v in vs if v["variant_kind"] == kind and not v.get("error")]

        # --- control: the noise floor ---------------------------------------
        controls = _ok("control")
        control_flipped = (any(not v["label_matches_original"] for v in controls)
                           if controls else None)

        # --- paraphrase: invariance -----------------------------------------
        paras = [v for v in vs if v["variant_kind"] == "paraphrase"]
        paras_ok = [v for v in paras if not v.get("error")]
        rejected = [v for v in paras if v.get("error")]
        matches = sum(1 for v in paras_ok if v["label_matches_original"])
        stability = round(matches / len(paras_ok), 4) if paras_ok else None

        # --- ablation / distractor: directional ------------------------------
        abl = next(iter(_ok("ablation")), None)
        dis = next(iter(_ok("distractor")), None)

        item = {
            "org": r["org"],
            "source_type": r["source_type"],
            "qid": r["qid"],
            "category": r["category"],
            "original_label": label0,
            "control_label": controls[0]["label"] if controls else None,
            "control_flipped": control_flipped,
            "n_paraphrases": len(paras),
            "n_paraphrases_ok": len(paras_ok),
            "n_paraphrases_rejected": len(rejected),
            "label_stability": stability,
            "unstable": (stability is not None and stability < threshold
                         and not control_flipped),
            "paraphrase_divergence": _mean(
                (v.get("fidelity") or {}).get("divergence") for v in paras_ok),
            "paraphrase_cosine": _mean(
                (v.get("fidelity") or {}).get("min_cosine") for v in paras_ok),
            "ablation_label": abl["label"] if abl else None,
            "ablation_basis": abl.get("ablation_basis") if abl else None,
            "label_survived_ablation": (
                abl["label"] == label0 and label0 != "abstain") if abl else None,
            "distractor_label": dis["label"] if dis else None,
            "distractor_category": dis.get("distractor_category") if dis else None,
            "distractor_committed": (dis["label"] != "abstain") if dis else None,
            "prior_keyed": (
                dis["label"] == label0 and label0 != "abstain") if dis else None,
        }
        if "grounding_bucket" in r:
            item["grounding_bucket"] = r["grounding_bucket"]
        per_item.append(item)

    scored = [i for i in per_item if i["label_stability"] is not None]
    controlled = [i for i in per_item if i["control_flipped"] is not None]
    ablated = [i for i in per_item if i["label_survived_ablation"] is not None]
    distracted = [i for i in per_item if i["prior_keyed"] is not None]

    rejected_by_reason: dict[str, int] = {}
    for v in variant_rows:
        if v["variant_kind"] == "paraphrase" and v.get("error"):
            reason = _rejection_reason(v)
            rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1

    def _mean_stability(items):
        return _mean(i["label_stability"] for i in items)

    by_category: dict[str, list] = {}
    by_bucket: dict[str, list] = {}
    for i in per_item:
        by_category.setdefault(i["category"], []).append(i)
        if "grounding_bucket" in i:
            by_bucket.setdefault(i["grounding_bucket"], []).append(i)

    n_control_flipped = sum(1 for i in controlled if i["control_flipped"])
    n_survived_ablation = sum(1 for i in ablated if i["label_survived_ablation"])
    n_prior_keyed = sum(1 for i in distracted if i["prior_keyed"])
    n_distractor_committed = sum(1 for i in distracted if i["distractor_committed"])

    summary = {
        "items": len(per_item),
        "items_scored": len(scored),
        # control (noise floor)
        "n_control_evaluated": len(controlled),
        "n_control_flipped": n_control_flipped,
        "control_flip_rate": (round(n_control_flipped / len(controlled), 4)
                              if controlled else None),
        # paraphrase (invariance)
        "mean_label_stability": _mean_stability(scored),
        "pct_fully_stable": (
            round(100.0 * sum(1 for i in scored if i["label_stability"] >= 1.0)
                  / len(scored), 1) if scored else None),
        "n_unstable": sum(1 for i in per_item if i["unstable"]),
        "stability_threshold": threshold,
        "n_paraphrases_rejected": sum(i["n_paraphrases_rejected"] for i in per_item),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "mean_paraphrase_divergence": _mean(
            i["paraphrase_divergence"] for i in per_item),
        "mean_paraphrase_cosine": _mean(i["paraphrase_cosine"] for i in per_item),
        # ablation (directional)
        "n_ablation_evaluated": len(ablated),
        "n_label_survived_ablation": n_survived_ablation,
        "ablation_survival_rate": (round(n_survived_ablation / len(ablated), 4)
                                   if ablated else None),
        # distractor (directional)
        "n_distractor_evaluated": len(distracted),
        "n_distractor_committed": n_distractor_committed,
        "distractor_commit_rate": (round(n_distractor_committed / len(distracted), 4)
                                   if distracted else None),
        "n_prior_keyed": n_prior_keyed,
        "prior_leak_rate": (round(n_prior_keyed / len(distracted), 4)
                            if distracted else None),
        "by_category": {c: _mean_stability(items)
                        for c, items in sorted(by_category.items())},
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
                 original_label: str, *, retriever: Retriever | None = None,
                 distractor_cache: dict | None = None) -> dict:
    """Build one variant context, answer from it, grade it, label it."""
    out = {
        "org": row["org"], "source_type": row["source_type"], "qid": row["qid"],
        "category": row["category"], "variant": row["variant"],
        "variant_kind": kind, "variant_idx": idx,
        "original_label": original_label,
    }
    question = row["question"]

    if kind == "control":
        vchunks = chunks

    elif kind == "paraphrase":
        texts, fidelity = build_paraphrase([c.text for c in chunks])
        out["fidelity"] = fidelity
        if texts is None:
            # Distinguish "the call/JSON failed" from "the rewrite broke a rule":
            # only the latter says anything about the paraphraser's quality.
            out["error"] = ("paraphrase_failed"
                            if fidelity.get("reason") == "paraphrase_unparseable"
                            else "paraphrase_infidelity")
            return out
        vchunks = [dataclasses.replace(c, text=t) for c, t in zip(chunks, texts)]
        # The only probe that rewrites the evidence is the only one that stores
        # it: keeping both sides lets a flagged item be audited (original next
        # to rewrite) without refetching from the index. The other probes are
        # reconstructible from the ids they record, so they store no text.
        out["source_context"] = [c.text for c in chunks]
        out["context"] = texts

    elif kind == "ablation":
        if len(chunks) < 2:
            out["error"] = "ablation_no_remainder"
            return out
        target, basis = ablation_target(row, chunks)
        out["ablation_basis"] = basis
        out["ablation_removed_id"] = chunks[target].id
        vchunks = [c for i, c in enumerate(chunks) if i != target]

    elif kind == "distractor":
        vchunks, error, meta = build_distractor_chunks(
            retriever, row, distractor_cache if distractor_cache is not None else {})
        out.update(meta)
        if error:
            out["error"] = error
            return out
        out["distractor_ids"] = [c.id for c in vchunks]

    else:
        out["error"] = f"unknown_probe:{kind}"
        return out

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


def _build_kinds(probes, paraphrases: int, controls: int) -> list[tuple[str, int]]:
    """The (variant_kind, variant_idx) pairs to run, in a stable order."""
    kinds: list[tuple[str, int]] = []
    if "control" in probes:
        kinds += [("control", i) for i in range(1, controls + 1)]
    if "paraphrase" in probes:
        kinds += [("paraphrase", i) for i in range(1, paraphrases + 1)]
    if "ablation" in probes:
        kinds.append(("ablation", 0))
    if "distractor" in probes:
        kinds.append(("distractor", 0))
    return kinds


def run_metamorphic(run_id: str | None = None,
                    paraphrases: int = METAMORPHIC_PARAPHRASES,
                    sample: int | None = None, seed: int = 0,
                    orgs: list[str] | None = None,
                    source_types: list[str] | None = None,
                    probes: tuple[str, ...] | list[str] = METAMORPHIC_PROBES,
                    controls: int = METAMORPHIC_CONTROLS) -> dict:
    """Run the metamorphic eval against an existing run snapshot.

    Reads the run's per_question.jsonl, produces one variant per enabled probe
    per (sampled) item, and writes variants.jsonl + stability.json into
    <run_dir>/metamorphic/. Resumable: completed variants are skipped on rerun;
    failed ones (error rows) are retried, so enabling a probe later only runs
    the newly requested variants. Sampling is deterministic for a given seed and
    filter set, so a resumed sampled run continues the same sample. Returns the
    aggregate summary dict.
    """
    probes = tuple(probes)
    unknown = [p for p in probes if p not in METAMORPHIC_PROBES]
    if unknown:
        raise ValueError(f"unknown probe(s): {', '.join(unknown)}; "
                         f"choose from {', '.join(METAMORPHIC_PROBES)}")

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

    kinds = _build_kinds(probes, paraphrases, controls)
    todo = [
        (row, kind, idx)
        for row in src_rows
        for kind, idx in kinds
        if (row["org"], row["source_type"], row["qid"], kind, idx) not in done
    ]

    retriever = Retriever() if todo else None
    chunk_cache: dict[tuple, list[Chunk]] = {}
    distractor_cache: dict[tuple, list[Chunk]] = {}
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
                v = _run_variant(row, kind, idx, chunks, row_label(row),
                                 retriever=retriever,
                                 distractor_cache=distractor_cache)
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
            f.flush()  # crash-safe, same contract as the profile harness
            existing[_variant_key(v)] = v

    # Score only the variants belonging to the probes that were requested, so a
    # narrowed rerun isn't silently graded against stale rows of other probes.
    scored_kinds = {k for k, _ in kinds}
    rows_for_scoring = [v for v in existing.values()
                        if v["variant_kind"] in scored_kinds]
    per_item, summary = compute_stability(src_rows, rows_for_scoring)
    summary = {
        "run_id": run_id,
        "probes": list(probes),
        "controls_per_item": controls if "control" in probes else 0,
        "paraphrases_per_item": paraphrases if "paraphrase" in probes else 0,
        "sample": sample,
        "seed": seed,
        **summary,
    }
    with open(out_dir / STABILITY_NAME, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_item": per_item}, f,
                  ensure_ascii=False, indent=2)

    _print_report(summary, out_dir)
    return summary


def _pct(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _print_report(summary: dict, out_dir: Path) -> None:
    print(f"\n=== Metamorphic stability & evidence sensitivity "
          f"[run {summary['run_id']}] ===")
    print(f"items: {summary['items']}   probes: {', '.join(summary['probes'])}")

    if summary["n_control_evaluated"]:
        print("\n-- control (noise floor: same evidence, run again) --")
        print(f"label flips: {summary['n_control_flipped']}"
              f"/{summary['n_control_evaluated']} "
              f"(rate {_pct(summary['control_flip_rate'])}) — read every number "
              f"below against this")

    if summary["items_scored"]:
        print("\n-- paraphrase (same meaning, different words) --")
        print(f"mean label stability: {_pct(summary['mean_label_stability'])}")
        print(f"fully stable items:   {summary['pct_fully_stable']:.1f}%")
        print(f"unstable items (stability < {summary['stability_threshold']}, "
              f"excluding control flips): {summary['n_unstable']}")
    if summary["n_paraphrases_rejected"]:
        reasons = ", ".join(f"{r}={n}" for r, n
                            in summary["rejected_by_reason"].items())
        print(f"rewrites rejected by the fidelity gates: "
              f"{summary['n_paraphrases_rejected']} ({reasons}) — discarded, "
              f"never counted as flips")
    if summary["mean_paraphrase_divergence"] is not None:
        print(f"paraphrase quality: divergence "
              f"{_pct(summary['mean_paraphrase_divergence'])}, "
              f"similarity {_pct(summary['mean_paraphrase_cosine'])}")

    if summary["n_ablation_evaluated"]:
        print("\n-- ablation (its quoted excerpt removed) --")
        print(f"label survived anyway: {summary['n_label_survived_ablation']}"
              f"/{summary['n_ablation_evaluated']} "
              f"(rate {_pct(summary['ablation_survival_rate'])}) — survival "
              f"means the answer didn't need that evidence")

    if summary["n_distractor_evaluated"]:
        print("\n-- distractor (real same-lab text that doesn't answer it) --")
        print(f"answered instead of abstaining: "
              f"{summary['n_distractor_committed']}"
              f"/{summary['n_distractor_evaluated']} "
              f"(rate {_pct(summary['distractor_commit_rate'])})")
        print(f"same label as the original:     {summary['n_prior_keyed']}"
              f"/{summary['n_distractor_evaluated']} "
              f"(prior leak {_pct(summary['prior_leak_rate'])}) — the label came "
              f"back without the evidence that supposedly produced it")

    print("\nstability by category:")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<24} {_pct(s)}")
    if "by_grounding_bucket" in summary:
        print("stability by grounding bucket:")
        for b, s in summary["by_grounding_bucket"].items():
            print(f"  {b:<17} n={s['n']:<4} {_pct(s['mean_label_stability'])}")
    print(f"\noutputs: {out_dir}")
