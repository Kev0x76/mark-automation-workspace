"""Single-worker product-region jobs with durable, per-SKU progress.

Contract extraction is intentionally synchronous and small.  The expensive
image-analysis stage runs here so the upload request can return as soon as the
contract fields and original images are available.  A single worker avoids
CPU oversubscription, while the JSON state survives normal API requests and
makes interrupted work explicit after a process restart.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .product_regions import (
    generate_product_regions_for_project,
    load_normalized,
    resolve_project_dir,
    utc_now,
)


JOB_SCHEMA_VERSION = "1.0.0"
ACTIVE_STATUSES = {"queued", "running"}
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="product-regions")
_LOCK = threading.RLock()
_FUTURES: dict[str, Future[None]] = {}


def start_product_region_job(
    workspace_root: Path | str,
    project_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Start one idempotent full-project job and return immediately."""

    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    key = job_key(project_dir)

    with _LOCK:
        future = _FUTURES.get(key)
        if future is not None and not future.done():
            state = read_job_state(project_dir)
            state["already_running"] = True
            return state

        state = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": uuid.uuid4().hex,
            "project_id": normalized["project_id"],
            "status": "queued",
            "phase": "product_regions",
            "total_items": len(normalized.get("items", [])),
            "completed_items": 0,
            "current_item_no": None,
            "force": bool(force),
            "created_at": utc_now(),
            "started_at": None,
            "updated_at": utc_now(),
            "finished_at": None,
            "error": None,
            "summary": None,
            "progress_url": f"/api/projects/{project_id}/product-regions/progress",
            "already_running": False,
        }
        write_job_state(project_dir, state)
        future = _EXECUTOR.submit(_run_product_region_job, project_dir, state, force)
        _FUTURES[key] = future
        future.add_done_callback(lambda completed, job_key=key: _forget_future(job_key, completed))
        return state


def get_product_region_job(
    workspace_root: Path | str,
    project_id: str,
) -> dict[str, Any]:
    """Read persisted progress without starting or resuming any work."""

    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    path = job_state_path(project_dir)
    if not path.exists():
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": None,
            "project_id": normalized["project_id"],
            "status": "idle",
            "phase": "product_regions",
            "total_items": len(normalized.get("items", [])),
            "completed_items": 0,
            "current_item_no": None,
            "force": False,
            "created_at": None,
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
            "error": None,
            "summary": None,
            "progress_url": f"/api/projects/{project_id}/product-regions/progress",
            "already_running": False,
        }

    with _LOCK:
        state = read_job_state(project_dir)
        if state.get("status") in ACTIVE_STATUSES:
            future = _FUTURES.get(job_key(project_dir))
            if future is None or future.done():
                state.update(
                    {
                        "status": "interrupted",
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "current_item_no": None,
                        "error": {
                            "type": "WorkerInterrupted",
                            "message": "The API process stopped before this job completed; start it again to resume processing.",
                        },
                    }
                )
                write_job_state(project_dir, state)
        state["already_running"] = False
        return state


def _run_product_region_job(project_dir: Path, initial_state: dict[str, Any], force: bool) -> None:
    state = dict(initial_state)
    state.update(
        {
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    write_job_state(project_dir, state)

    def report_progress(completed: int, total: int, item_no: str | None) -> None:
        state.update(
            {
                "completed_items": completed,
                "total_items": total,
                "current_item_no": item_no,
                "updated_at": utc_now(),
            }
        )
        write_job_state(project_dir, state)

    try:
        result = generate_product_regions_for_project(
            project_dir,
            force=force,
            progress_callback=report_progress,
        )
        state.update(
            {
                "status": "completed",
                "completed_items": state.get("total_items", 0),
                "current_item_no": None,
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "summary": result.get("summary"),
                "error": None,
            }
        )
    except Exception as exc:  # keep the worker alive for later projects
        state.update(
            {
                "status": "failed",
                "current_item_no": None,
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    write_job_state(project_dir, state)


def _forget_future(key: str, future: Future[None]) -> None:
    with _LOCK:
        if _FUTURES.get(key) is future:
            _FUTURES.pop(key, None)


def job_key(project_dir: Path) -> str:
    return str(project_dir.resolve())


def job_state_path(project_dir: Path) -> Path:
    return project_dir / "reports" / "product_region_job.json"


def read_job_state(project_dir: Path) -> dict[str, Any]:
    path = job_state_path(project_dir)
    with _LOCK:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("reports/product_region_job.json is unreadable.") from exc


def write_job_state(project_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically replace state so polling never observes partial JSON."""

    path = job_state_path(project_dir)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
