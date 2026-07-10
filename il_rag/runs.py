"""Run snapshots: every profiling run is archived immutably so results survive.

Before this module a profiling run overwrote three flat files in data/profiles/,
so re-running with a changed questionnaire destroyed the previous results and
there was nothing to compare against. Now each run lives in its own timestamped
folder together with a COPY of the questionnaire that produced it, so an old run
and a new run can be diffed question-by-question and logic-by-logic.

Layout:
  data/profiles/runs/<run_id>/
      per_question.jsonl     audit rows (one per org x source x question)
      company_profiles.json  aggregated profiles
      profiles_matrix.csv    wide table
      questionnaire.json     the questionnaire that produced THIS run
      meta.json              run_id, label, timestamps, params, counts, status
  data/profiles/CURRENT      text file naming the active run_id (used for resume)

run_id is "YYYY-MM-DD_HHMMSS" — sortable and filesystem-safe. A run is the unit
of comparison; --fresh always mints a new one, a resumed run reuses CURRENT.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

from .config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    PROFILES_DIR,
)
from .questionnaire import CATEGORIES, LOGICS, QUESTIONNAIRE

RUNS_DIR = PROFILES_DIR / "runs"
CURRENT_PTR = PROFILES_DIR / "CURRENT"

# Per-run filenames (same names the old flat layout used, now scoped to a run).
PER_QUESTION_NAME = "per_question.jsonl"
PROFILES_JSON_NAME = "company_profiles.json"
PROFILES_CSV_NAME = "profiles_matrix.csv"
QUESTIONNAIRE_NAME = "questionnaire.json"
META_NAME = "meta.json"

QUESTIONS_PER_ORG = len(CATEGORIES) * 3  # structural invariant (9 categories x 3)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def run_paths(run_id: str) -> dict[str, Path]:
    d = run_dir(run_id)
    return {
        "dir": d,
        "per_question": d / PER_QUESTION_NAME,
        "profiles_json": d / PROFILES_JSON_NAME,
        "profiles_csv": d / PROFILES_CSV_NAME,
        "questionnaire": d / QUESTIONNAIRE_NAME,
        "meta": d / META_NAME,
    }


# ---------------------------------------------------------------------------
# Questionnaire snapshot
# ---------------------------------------------------------------------------
def snapshot_questionnaire() -> dict:
    """Serializable copy of the questionnaire currently in code.

    Stores raw question templates (with the {org} placeholder) and the per-logic
    reference answers, so a later run can diff exactly what changed.
    """
    return {
        "logics": list(LOGICS),
        "categories": list(CATEGORIES),
        "questionnaire": QUESTIONNAIRE,
    }


# ---------------------------------------------------------------------------
# CURRENT pointer
# ---------------------------------------------------------------------------
def get_current() -> str | None:
    if not CURRENT_PTR.exists():
        return None
    run_id = CURRENT_PTR.read_text(encoding="utf-8").strip()
    return run_id if run_id and run_dir(run_id).exists() else None


def set_current(run_id: str) -> None:
    CURRENT_PTR.write_text(run_id + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
def read_meta(run_id: str) -> dict:
    path = run_paths(run_id)["meta"]
    if not path.exists():
        return {"run_id": run_id}
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(run_id: str, meta: dict) -> None:
    run_paths(run_id)["meta"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_meta(run_id: str, **fields) -> dict:
    meta = read_meta(run_id)
    meta.update(fields)
    write_meta(run_id, meta)
    return meta


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------
def _mint_run_id() -> str:
    """Timestamp-based id; bump by a second on the rare same-second collision."""
    rid = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    while run_dir(rid).exists():
        # extremely unlikely; append a disambiguating suffix
        i = 2
        while run_dir(f"{rid}-{i}").exists():
            i += 1
        rid = f"{rid}-{i}"
    return rid


def new_run(label: str | None, orgs: list[str], source_types: list[str],
            k: int) -> str:
    """Create a fresh run folder, snapshot the questionnaire, set it CURRENT."""
    run_id = _mint_run_id()
    paths = run_paths(run_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    paths["questionnaire"].write_text(
        json.dumps(snapshot_questionnaire(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    now = datetime.now().isoformat(timespec="seconds")
    write_meta(run_id, {
        "run_id": run_id,
        "label": label or "",
        "created_at": now,
        "updated_at": now,
        "orgs": list(orgs),
        "source_types": list(source_types),
        "k": k,
        "generation_model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "status": "partial",
        "answered": 0,
        "abstained": 0,
        "questions": 0,
    })
    set_current(run_id)
    return run_id


def finalize_meta(run_id: str, rows: list[dict], orgs: list[str],
                  source_types: list[str]) -> None:
    """Record counts/status after a run finishes (or is interrupted)."""
    answered = sum(1 for r in rows if not r["abstain"])
    abstained = sum(1 for r in rows if r["abstain"])
    expected = len(orgs) * len(source_types) * QUESTIONS_PER_ORG
    update_meta(
        run_id,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        answered=answered,
        abstained=abstained,
        questions=len(rows),
        status="complete" if len(rows) >= expected else "partial",
    )


def list_runs() -> list[dict]:
    """All run metas, newest first (by run_id, which is chronological)."""
    if not RUNS_DIR.exists():
        return []
    metas = []
    for d in RUNS_DIR.iterdir():
        if d.is_dir() and (d / META_NAME).exists():
            metas.append(read_meta(d.name))
    return sorted(metas, key=lambda m: m.get("run_id", ""), reverse=True)


def display_name(meta: dict) -> str:
    """Human label for a run in dropdowns: id, optional label, status."""
    run_id = meta.get("run_id", "?")
    label = meta.get("label") or ""
    flag = "" if meta.get("status") == "complete" else " · partial"
    return f"{run_id}" + (f" — {label}" if label else "") + flag


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------
def migrate_legacy() -> str | None:
    """Move pre-snapshot flat outputs (data/profiles/*.json[l]) into a run.

    No-op once any run exists. Returns the new run_id if a migration happened.
    The legacy run is stamped with the old files' mtime so it sorts correctly,
    and snapshots the CURRENT questionnaire (unchanged since that run produced it).
    """
    if RUNS_DIR.exists() and any(RUNS_DIR.iterdir()):
        return None  # already migrated / runs exist

    legacy_pq = PROFILES_DIR / PER_QUESTION_NAME
    legacy_json = PROFILES_DIR / PROFILES_JSON_NAME
    legacy_csv = PROFILES_DIR / PROFILES_CSV_NAME
    if not legacy_pq.exists() and not legacy_json.exists():
        return None  # nothing to migrate

    stamp = datetime.fromtimestamp(
        (legacy_json if legacy_json.exists() else legacy_pq).stat().st_mtime
    )
    run_id = stamp.strftime("%Y-%m-%d_%H%M%S")
    paths = run_paths(run_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    for src, key in ((legacy_pq, "per_question"),
                     (legacy_json, "profiles_json"),
                     (legacy_csv, "profiles_csv")):
        if src.exists():
            shutil.move(str(src), str(paths[key]))

    paths["questionnaire"].write_text(
        json.dumps(snapshot_questionnaire(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    if paths["per_question"].exists():
        for line in paths["per_question"].read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    orgs = sorted({r["org"] for r in rows}) or None
    sources = sorted({r["source_type"] for r in rows}) or None
    now = datetime.now().isoformat(timespec="seconds")
    write_meta(run_id, {
        "run_id": run_id,
        "label": "legacy (pre-snapshot)",
        "created_at": stamp.isoformat(timespec="seconds"),
        "updated_at": now,
        "orgs": orgs or [],
        "source_types": sources or [],
        "k": None,
        "generation_model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "status": "complete",
        "answered": sum(1 for r in rows if not r.get("abstain")),
        "abstained": sum(1 for r in rows if r.get("abstain")),
        "questions": len(rows),
    })
    set_current(run_id)
    return run_id
