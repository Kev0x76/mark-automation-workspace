"""Background lineart generation jobs with durable progress and bounded concurrency.

First principles:
- Wall-clock cost is dominated by external image APIs, not local canvas QC.
- A project may only own one active lineart job (idempotent start).
- Multiple SKUs may call the provider concurrently, but file writes stay serialized.
- Progress is polled from reports/lineart_job.json, same shape family as product-region jobs.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .lineart import (
    LineartError,
    generate_lineart_candidate,
    load_normalized,
    resolve_project_dir,
    utc_now,
)
from ..settings import (
    get_active_lineart_provider,
    get_openai_api_keys,
    get_xai_api_keys,
)

JOB_SCHEMA_VERSION = "1.0.0"
ACTIVE_STATUSES = {"queued", "running"}
DEFAULT_CONCURRENCY = 2
MAX_CONCURRENCY = 8

# One orchestrator worker starts jobs; item-level concurrency is inside each job.
_ORCHESTRATOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lineart-jobs")
_LOCK = threading.RLock()
_FUTURES: dict[str, Future[None]] = {}


def default_concurrency(provider_name: str | None = None) -> int:
    raw = os.environ.get("LINEART_JOB_CONCURRENCY", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_CONCURRENCY
        return max(1, min(MAX_CONCURRENCY, value))
    provider = (provider_name or get_active_lineart_provider() or "").strip()
    if provider in {"grok_image", "grok", "xai"}:
        keys = len(get_xai_api_keys())
    elif provider in {"openai_image", "openai", "gpt"}:
        keys = len(get_openai_api_keys())
    else:
        keys = 1
    # Match workers to independent keys when possible; hard ceiling is MAX_CONCURRENCY.
    value = max(DEFAULT_CONCURRENCY, keys) if keys else DEFAULT_CONCURRENCY
    return max(1, min(MAX_CONCURRENCY, value))


def start_lineart_job(
    workspace_root: Path | str,
    project_id: str,
    *,
    provider_name: str = "openai_image",
    concurrency: int | None = None,
    item_nos: list[str] | None = None,
    strategy: str = "group_representatives",
) -> dict[str, Any]:
    """Start one idempotent missing-lineart job and return immediately.

    strategy:
      - all_missing: every eligible SKU
      - group_representatives: one SKU per multi-member category group + ungrouped singles
    """

    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    strategy_name = (strategy or "group_representatives").strip().lower()
    if strategy_name not in {"all_missing", "group_representatives"}:
        raise ValueError("strategy must be all_missing or group_representatives")
    plan = _build_generation_plan(project_dir, normalized, item_nos, strategy_name)
    eligible = plan["generate_items"]
    key = job_key(project_dir)
    workers = max(1, min(MAX_CONCURRENCY, int(concurrency or default_concurrency(provider_name))))

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
            "phase": "linearts",
            "provider": provider_name,
            "strategy": strategy_name,
            "concurrency": workers,
            "total_items": len(eligible),
            "pending_shares": plan.get("pending_shares") or [],
            "completed_items": 0,
            "succeeded_items": 0,
            "failed_items": 0,
            "skipped_items": plan.get("skipped_items") or [],
            "current_item_no": None,
            "item_nos": [str(item.get("item_no")) for item in eligible],
            "representative_of": plan.get("representative_of") or {},
            "created_at": utc_now(),
            "started_at": None,
            "updated_at": utc_now(),
            "finished_at": None,
            "error": None,
            "summary": None,
            "failures": [],
            "progress_url": f"/api/projects/{project_id}/linearts/generate-progress",
            "already_running": False,
        }
        write_job_state(project_dir, state)
        if not eligible:
            state.update(
                {
                    "status": "completed",
                    "finished_at": utc_now(),
                    "summary": {
                        "generated": 0,
                        "failed": 0,
                        "skipped": len(state["skipped_items"]),
                        "message": (
                            "Representative linearts already ready; remaining SKUs are share targets only. "
                            "Share is optional — switch strategy to all_missing or generate each SKU to continue without sharing."
                            if (plan.get("pending_shares") and strategy_name == "group_representatives")
                            else "No missing linearts to generate for the selected strategy."
                        ),
                        "pending_shares": plan.get("pending_shares") or [],
                    },
                }
            )
            write_job_state(project_dir, state)
            return state

        future = _ORCHESTRATOR.submit(
            _run_lineart_job,
            str(Path(workspace_root).resolve()),
            project_id,
            provider_name,
            workers,
            [str(item.get("item_no")) for item in eligible],
            state["job_id"],
        )
        _FUTURES[key] = future
        future.add_done_callback(lambda completed, job_key=key: _forget_future(job_key, completed))
        return state


def get_lineart_job(
    workspace_root: Path | str,
    project_id: str,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    path = job_state_path(project_dir)
    if not path.exists():
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": None,
            "project_id": normalized["project_id"],
            "status": "idle",
            "phase": "linearts",
            "provider": None,
            "concurrency": default_concurrency(None),
            "total_items": 0,
            "completed_items": 0,
            "succeeded_items": 0,
            "failed_items": 0,
            "skipped_items": [],
            "current_item_no": None,
            "item_nos": [],
            "created_at": None,
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
            "error": None,
            "summary": None,
            "failures": [],
            "progress_url": f"/api/projects/{project_id}/linearts/generate-progress",
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
                            "message": "The API process stopped before this lineart job completed; start it again to resume missing items.",
                        },
                    }
                )
                write_job_state(project_dir, state)
        state["already_running"] = bool(
            state.get("status") in ACTIVE_STATUSES
            and (_FUTURES.get(job_key(project_dir)) is not None)
            and not _FUTURES[job_key(project_dir)].done()
        )
        return state


def _run_lineart_job(
    workspace_root: str,
    project_id: str,
    provider_name: str,
    concurrency: int,
    item_nos: list[str],
    job_id: str,
) -> None:
    project_dir = resolve_project_dir(workspace_root, project_id)
    state = read_job_state(project_dir)
    if state.get("job_id") != job_id:
        return
    state.update(
        {
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "current_item_no": item_nos[0] if item_nos else None,
        }
    )
    write_job_state(project_dir, state)

    succeeded = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    completed = 0
    progress_lock = threading.Lock()

    def one(item_no: str) -> tuple[str, bool, str | None]:
        try:
            generate_lineart_candidate(
                workspace_root,
                project_id,
                item_no,
                provider_name,
            )
            return item_no, True, None
        except Exception as exc:  # isolate per-SKU failure
            return item_no, False, f"{type(exc).__name__}: {exc}"

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="lineart-sku") as pool:
            futures = {pool.submit(one, item_no): item_no for item_no in item_nos}
            # Mark in-flight for UI honesty
            with progress_lock:
                state["current_item_no"] = item_nos[0]
                state["updated_at"] = utc_now()
                write_job_state(project_dir, state)

            for future in as_completed(futures):
                item_no, ok, error = future.result()
                with progress_lock:
                    completed += 1
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                        failures.append({"item_no": item_no, "message": error or "unknown_error"})
                    remaining = [futures[f] for f in futures if not f.done()]
                    state.update(
                        {
                            "completed_items": completed,
                            "succeeded_items": succeeded,
                            "failed_items": failed,
                            "current_item_no": remaining[0] if remaining else item_no,
                            "failures": failures[-50:],
                            "updated_at": utc_now(),
                        }
                    )
                    write_job_state(project_dir, state)

        state.update(
            {
                "status": "completed",
                "completed_items": completed,
                "succeeded_items": succeeded,
                "failed_items": failed,
                "current_item_no": None,
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "failures": failures,
                "summary": {
                    "generated": succeeded,
                    "failed": failed,
                    "skipped": len(state.get("skipped_items") or []),
                    "concurrency": concurrency,
                    "provider": provider_name,
                },
                "error": None,
            }
        )
    except Exception as exc:
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



def _build_generation_plan(
    project_dir: Path,
    normalized: dict[str, Any],
    item_nos: list[str] | None,
    strategy: str,
) -> dict[str, Any]:
    """Decide which SKUs to generate and which groups can later share."""

    wanted = {str(value).strip() for value in (item_nos or []) if str(value).strip()}
    items = {
        str(item.get("item_no") or "").strip(): item
        for item in normalized.get("items", [])
        if str(item.get("item_no") or "").strip()
    }
    if wanted:
        items = {key: value for key, value in items.items() if key in wanted}

    def needs_generation(item: dict[str, Any]) -> bool:
        if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
            return False
        if item.get("product_region_status") != "selected" or not item.get("selected_product_region"):
            return False
        return True

    skipped: list[dict[str, str]] = []
    for item_no, item in items.items():
        if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
            skipped.append({"item_no": item_no, "reason": "already_approved"})
        elif item.get("product_region_status") != "selected" or not item.get("selected_product_region"):
            skipped.append({"item_no": item_no, "reason": "product_region_not_selected"})

    groups_payload = {}
    groups_path = project_dir / "reports" / "lineart_groups.json"
    if groups_path.is_file():
        try:
            groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            groups_payload = {}

    grouped_members: dict[str, list[str]] = {}
    for collection_name in ("groups", "manual_groups"):
        for group in (groups_payload.get(collection_name) or groups_payload.get("suggestions", {}).get(collection_name) or []):
            # support both top-level and nested suggestions shapes
            pass
    # Normalize group list from either shape used in reports.
    group_list: list[dict[str, Any]] = []
    if isinstance(groups_payload.get("groups"), list):
        group_list.extend(groups_payload.get("groups") or [])
    if isinstance(groups_payload.get("manual_groups"), list):
        group_list.extend(groups_payload.get("manual_groups") or [])
    suggestions = groups_payload.get("suggestions")
    if isinstance(suggestions, dict):
        if isinstance(suggestions.get("groups"), list):
            group_list.extend(suggestions.get("groups") or [])
        if isinstance(suggestions.get("manual_groups"), list):
            group_list.extend(suggestions.get("manual_groups") or [])

    pending_shares: list[dict[str, Any]] = []
    representative_of: dict[str, str] = {}
    covered: set[str] = set()
    generate_item_nos: list[str] = []

    if strategy == "group_representatives":
        for group in group_list:
            group_id = str(group.get("group_id") or "").strip()
            members = [
                str(member.get("item_no") if isinstance(member, dict) else member).strip()
                for member in (group.get("members") or [])
            ]
            members = [item_no for item_no in members if item_no in items]
            if len(members) < 2:
                continue
            needing = [item_no for item_no in members if needs_generation(items[item_no])]
            # If some already approved, prefer an approved non-shared rep for sharing path.
            approved_members = [
                item_no
                for item_no in members
                if items[item_no].get("lineart_status") == "approved"
                and items[item_no].get("lineart_asset")
                and (items[item_no].get("lineart_provenance") or {}).get("binding_scope") != "category_group"
            ]
            if not needing and not approved_members:
                continue
            if approved_members:
                rep = sorted(approved_members)[0]
            else:
                # generate one representative among needing members
                rep = sorted(needing)[0]
                if rep not in generate_item_nos:
                    generate_item_nos.append(rep)
            for item_no in members:
                covered.add(item_no)
                representative_of[item_no] = rep
            share_targets = [item_no for item_no in members if item_no != rep]
            if share_targets:
                pending_shares.append(
                    {
                        "group_id": group_id,
                        "normalized_cn_name": group.get("normalized_cn_name") or group.get("group_name") or group_id,
                        "representative_item_no": rep,
                        "member_item_nos": members,
                        "share_target_item_nos": share_targets,
                        "needs_representative_generation": rep in needing and rep not in approved_members,
                    }
                )
        # ungrouped singles still need individual generation
        for item_no, item in items.items():
            if item_no in covered:
                continue
            if needs_generation(item):
                generate_item_nos.append(item_no)
    else:
        for item_no, item in items.items():
            if needs_generation(item):
                generate_item_nos.append(item_no)

    # stable unique order
    ordered = []
    seen = set()
    for item_no in generate_item_nos:
        if item_no in seen:
            continue
        seen.add(item_no)
        ordered.append(item_no)

    return {
        "generate_items": [items[item_no] for item_no in ordered if item_no in items],
        "skipped_items": skipped,
        "pending_shares": pending_shares,
        "representative_of": representative_of,
        "strategy": strategy,
    }


def _eligible_items(normalized: dict[str, Any], item_nos: list[str] | None) -> list[dict[str, Any]]:
    wanted = {str(value).strip() for value in (item_nos or []) if str(value).strip()}
    items = []
    for item in normalized.get("items", []):
        item_no = str(item.get("item_no") or "").strip()
        if not item_no:
            continue
        if wanted and item_no not in wanted:
            continue
        if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
            continue
        if item.get("product_region_status") != "selected" or not item.get("selected_product_region"):
            continue
        items.append(item)
    return items


def _skipped_summary(normalized: dict[str, Any], item_nos: list[str] | None) -> list[dict[str, str]]:
    wanted = {str(value).strip() for value in (item_nos or []) if str(value).strip()}
    skipped: list[dict[str, str]] = []
    for item in normalized.get("items", []):
        item_no = str(item.get("item_no") or "").strip()
        if not item_no:
            continue
        if wanted and item_no not in wanted:
            continue
        if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
            skipped.append({"item_no": item_no, "reason": "already_approved"})
        elif item.get("product_region_status") != "selected" or not item.get("selected_product_region"):
            skipped.append({"item_no": item_no, "reason": "product_region_not_selected"})
    return skipped


def _forget_future(key: str, future: Future[None]) -> None:
    with _LOCK:
        if _FUTURES.get(key) is future:
            _FUTURES.pop(key, None)


def job_key(project_dir: Path) -> str:
    return str(project_dir.resolve())


def job_state_path(project_dir: Path) -> Path:
    return project_dir / "reports" / "lineart_job.json"


def read_job_state(project_dir: Path) -> dict[str, Any]:
    path = job_state_path(project_dir)
    with _LOCK:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("reports/lineart_job.json is unreadable.") from exc


def write_job_state(project_dir: Path, payload: dict[str, Any]) -> None:
    path = job_state_path(project_dir)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
