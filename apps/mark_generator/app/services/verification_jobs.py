"""Single-worker project verification jobs with durable progress.

Verification is intentionally expensive (PDF render, barcode, OCR, crop
re-evidence).  HTTP requests return immediately; the UI polls
reports/verification_job.json.  Gates and evidence rules stay unchanged.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .lineart import resolve_project_dir
from .manifest import utc_now
from .verification import VerificationError, verify_project

JOB_SCHEMA_VERSION = "1.0.0"
ACTIVE_STATUSES = {"queued", "running"}
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verification")
_LOCK = threading.RLock()
_FUTURES: dict[str, Future[None]] = {}


def start_verification_job(
    workspace_root: Path | str,
    project_id: str,
    *,
    template_mode: str = "kmt_style_v1",
) -> dict[str, Any]:
    """Queue one verification job and return immediately."""

    project_dir = resolve_project_dir(workspace_root, project_id)
    # Fail closed before queueing when inputs are obviously missing.
    if not (project_dir / "generated" / "standard_mark.pdf").exists():
        raise VerificationError("generated/standard_mark.pdf not found; generate the standard PDF first.")
    contract_ok = any(
        (project_dir / "input" / name).exists()
        for name in ("contract.xlsx", "contract.xls")
    )
    if not contract_ok:
        raise VerificationError("input/contract.xlsx or input/contract.xls not found; extract the contract first.")

    key = job_key(project_dir)
    with _LOCK:
        future = _FUTURES.get(key)
        if future is not None and not future.done():
            state = read_job_state(project_dir)
            state["already_running"] = True
            return state

        total_items = 0
        items_path = project_dir / "data" / "items.normalized.json"
        if items_path.exists():
            try:
                payload = json.loads(items_path.read_text(encoding="utf-8"))
                items = payload.get("items") if isinstance(payload, dict) else payload
                if isinstance(items, list):
                    total_items = len(items)
            except (OSError, json.JSONDecodeError, TypeError, AttributeError):
                total_items = 0

        state = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": uuid.uuid4().hex,
            "project_id": project_id,
            "status": "queued",
            "phase": "queued",
            "phase_label": "排队等待核对",
            "total_items": total_items,
            "completed_items": 0,
            "current_item_no": None,
            "created_at": utc_now(),
            "started_at": None,
            "updated_at": utc_now(),
            "finished_at": None,
            "error": None,
            "result_status": None,
            "verification": None,
            "progress_url": f"/api/projects/{project_id}/verify/progress",
            "already_running": False,
        }
        write_job_state(project_dir, state)
        future = _EXECUTOR.submit(_run_verification_job, str(workspace_root), project_id, template_mode, state)
        _FUTURES[key] = future
        future.add_done_callback(lambda completed, job_key=key: _forget_future(job_key, completed))
        return state


def get_verification_job(workspace_root: Path | str, project_id: str) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    path = job_state_path(project_dir)
    if not path.exists():
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": None,
            "project_id": project_id,
            "status": "idle",
            "phase": "idle",
            "phase_label": "未开始",
            "total_items": 0,
            "completed_items": 0,
            "current_item_no": None,
            "created_at": None,
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
            "error": None,
            "result_status": None,
            "verification": None,
            "progress_url": f"/api/projects/{project_id}/verify/progress",
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
                        "phase": "interrupted",
                        "phase_label": "进程中断",
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "error": {
                            "type": "WorkerInterrupted",
                            "message": "The API process stopped before verification completed; start it again.",
                        },
                    }
                )
                write_job_state(project_dir, state)
        state["already_running"] = False
        return state


def _run_verification_job(
    workspace_root: str,
    project_id: str,
    template_mode: str,
    initial_state: dict[str, Any],
) -> None:
    project_dir = resolve_project_dir(workspace_root, project_id)
    state = dict(initial_state)
    state.update(
        {
            "status": "running",
            "phase": "starting",
            "phase_label": "开始独立核对",
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    write_job_state(project_dir, state)

    def on_progress(phase: str, label: str, *, completed: int | None = None, total: int | None = None) -> None:
        if completed is not None:
            state["completed_items"] = completed
        if total is not None:
            state["total_items"] = total
        state.update(
            {
                "phase": phase,
                "phase_label": label,
                "updated_at": utc_now(),
            }
        )
        write_job_state(project_dir, state)

    try:
        outcome = verify_project(
            workspace_root,
            project_id,
            template_mode=template_mode,
            progress_callback=on_progress,
        )
        verification = outcome.get("verification") if isinstance(outcome, dict) else None
        state.update(
            {
                "status": "completed",
                "phase": "completed",
                "phase_label": "核对完成",
                "completed_items": state.get("total_items") or 0,
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "error": None,
                "result_status": outcome.get("status") if isinstance(outcome, dict) else None,
                "verification": verification,
            }
        )
    except Exception as exc:  # keep worker alive
        state.update(
            {
                "status": "failed",
                "phase": "failed",
                "phase_label": "核对失败",
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "result_status": None,
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
    return project_dir / "reports" / "verification_job.json"


def read_job_state(project_dir: Path) -> dict[str, Any]:
    path = job_state_path(project_dir)
    with _LOCK:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("reports/verification_job.json is unreadable.") from exc


def write_job_state(project_dir: Path, payload: dict[str, Any]) -> None:
    path = job_state_path(project_dir)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
