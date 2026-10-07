"""Audited human release overrides for failed project verification runs.

The automated verifier remains the source of the machine decision.  A human
release is a separate, narrowly scoped record that can only unlock the exact
PDF and exact verification inputs that were reviewed.  Any regeneration,
re-verification, input change, or changed issue set makes the record inactive.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .contract_snapshot import file_sha256
from .lineart import resolve_project_dir
from .manifest import utc_now
from .schema_validation import write_standard_json


MANUAL_RELEASE_SCHEMA_VERSION = "1.1.0"
MANUAL_RELEASE_RELATIVE_PATH = "reports/verification_manual_release.json"
PASS_FIELD_STATUSES = {"PASS", "OK", "NOT_APPLICABLE"}
PASS_REVIEW_STATUSES = {"PASS", "OK"}
ACTIVE_VERIFICATION_JOB_STATUSES = {"queued", "running"}


class ManualReleaseError(ValueError):
    """Invalid manual release request."""


class ManualReleaseConflict(ManualReleaseError):
    """The verification state changed or is not currently releasable."""


def collect_manual_release_issues(
    report: dict[str, Any] | None,
    verification: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return the current actionable facts that require explicit confirmation."""

    report = report if isinstance(report, dict) else {}
    verification = verification if isinstance(verification, dict) else {}
    results = report.get("results")
    if not isinstance(results, list):
        results = report.get("items") if isinstance(report.get("items"), list) else []

    issues: dict[str, dict[str, str]] = {}

    def add_issue(
        *,
        item_number: Any,
        field: Any,
        status: Any,
        contract_value: Any = "",
        mark_value: Any = "",
        note: Any = "",
    ) -> None:
        item = str(item_number or "").strip()
        field_name = str(field or "__item__").strip() or "__item__"
        normalized_status = str(status or "NEEDS_MANUAL_REVIEW").strip().upper()
        issue_id = _issue_id(item, field_name, normalized_status, contract_value, mark_value)
        issues[issue_id] = {
            "issue_id": issue_id,
            "item_number": item,
            "field": field_name,
            "status": normalized_status,
            "contract_value": _display_value(contract_value),
            "mark_value": _display_value(mark_value),
            "note": str(note or "").strip(),
        }

    for result in results:
        if not isinstance(result, dict):
            continue
        item_number = result.get("item_number") or result.get("item_no")
        review = result.get("review") if isinstance(result.get("review"), dict) else {}
        fields = result.get("fields") if isinstance(result.get("fields"), list) else []
        field_issue_count = 0
        for field in fields:
            if not isinstance(field, dict):
                continue
            status = str(field.get("status") or "").strip().upper()
            if not status or status in PASS_FIELD_STATUSES:
                continue
            field_issue_count += 1
            add_issue(
                item_number=item_number,
                field=field.get("field"),
                status=status,
                contract_value=field.get("contract_value", field.get("expected", "")),
                mark_value=field.get("mark_value", field.get("actual", "")),
                note=field.get("note") or review.get("reason_summary") or review.get("reason_cn"),
            )

        reviewer_status = str(review.get("reviewer_status") or "").strip().upper()
        result_error = str(result.get("error") or "").strip()
        if result_error:
            add_issue(
                item_number=item_number,
                field="__item__",
                status="ERROR",
                note=result_error,
            )
        elif reviewer_status and reviewer_status not in PASS_REVIEW_STATUSES and field_issue_count == 0:
            add_issue(
                item_number=item_number,
                field="__item__",
                status=reviewer_status,
                note=review.get("reason_summary") or review.get("reason_cn") or review.get("suggested_action"),
            )

    if verification.get("coverage_complete") is False:
        add_issue(
            item_number="",
            field="__coverage__",
            status="COVERAGE_INCOMPLETE",
            contract_value=verification.get("total_items", ""),
            mark_value=verification.get("reported_items", ""),
            note="核对覆盖范围与合同货号数量不一致。",
        )
    if verification.get("evidence_archive_complete") is False:
        add_issue(
            item_number="",
            field="__evidence_archive__",
            status="EVIDENCE_ARCHIVE_INCOMPLETE",
            note="核对证据归档不完整。",
        )
    weak_evidence_items = _non_negative_int(verification.get("weak_evidence_items"))
    if weak_evidence_items:
        add_issue(
            item_number="",
            field="__weak_evidence__",
            status="WEAK_EVIDENCE",
            contract_value=verification.get("total_items", ""),
            mark_value=weak_evidence_items,
            note=f"{weak_evidence_items} 个货号使用了弱证据，机器规则因此未自动放行。",
        )
    if verification.get("status") == "verified_failed" and not issues:
        add_issue(
            item_number="",
            field="__verification__",
            status="VERIFICATION_FAILED",
            contract_value=verification.get("total_items", ""),
            mark_value=verification.get("passed_items", ""),
            note="机器核对未通过，但报告没有提供更细的阻断字段；必须人工查看完整报告后决定。",
        )
    return sorted(issues.values(), key=lambda value: (value["item_number"], value["field"], value["issue_id"]))


def current_verification_input_hashes(project_dir: Path) -> dict[str, Any]:
    """Hash the current contract, snapshot, generator inputs, and PDF."""

    from .production_readiness import get_production_input_hashes
    from .verification import find_project_contract

    result: dict[str, Any] = {}
    contract = find_project_contract(project_dir)
    snapshot = project_dir / "data" / "contract.snapshot.json"
    if contract is not None and contract.is_file():
        result["contract"] = file_sha256(contract)
    if snapshot.is_file():
        result["contract_snapshot"] = file_sha256(snapshot)
    result.update(get_production_input_hashes(project_dir))
    pdf = project_dir / "generated" / "standard_mark.pdf"
    result["marks"] = [file_sha256(pdf)] if pdf.is_file() else []
    return result


def verification_inputs_match(expected: dict[str, Any], current: dict[str, Any]) -> bool:
    if not isinstance(expected, dict) or not expected or not isinstance(current, dict) or not current:
        return False
    if set(expected) != set(current):
        return False
    for key, expected_value in expected.items():
        if key == "marks":
            if (
                not isinstance(expected_value, list)
                or not expected_value
                or not all(isinstance(value, str) and value for value in expected_value)
                or current.get(key) != expected_value
            ):
                return False
            continue
        if current.get(key) != expected_value or not expected_value:
            return False
    return True


def current_verification_evidence_hashes(
    project_dir: Path,
    verification: dict[str, Any] | None,
) -> dict[str, str]:
    """Hash the report and archived run reviewed by the human operator."""

    verification = verification if isinstance(verification, dict) else {}
    result: dict[str, str] = {}
    report_path = project_dir / "reports" / "verification_report.json"
    if report_path.is_file():
        result["verification_report"] = file_sha256(report_path)

    run_path_value = str(verification.get("run_path") or "").strip()
    if run_path_value:
        run_path = _resolve_project_path(project_dir, run_path_value)
        if run_path is not None and run_path.is_dir():
            digest = _hash_directory(run_path)
            if digest:
                result["verification_run_tree"] = digest
    return result


def get_manual_release_state(
    project_dir: Path,
    verification: dict[str, Any] | None,
    current_inputs: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    """Read and validate a release record against the current project state."""

    verification = verification if isinstance(verification, dict) else {}
    current_issue_ids = [issue["issue_id"] for issue in issues]
    issue_fingerprint = fingerprint_issues(issues)
    record = _read_json(project_dir / MANUAL_RELEASE_RELATIVE_PATH)
    expected_inputs = (
        verification.get("input_hashes")
        if isinstance(verification.get("input_hashes"), dict)
        else {}
    )
    inputs_match = verification_inputs_match(expected_inputs, current_inputs)
    job = _read_json(project_dir / "reports" / "verification_job.json")
    job_status = str(job.get("status") or "idle")
    job_active = job_status in ACTIVE_VERIFICATION_JOB_STATUSES
    evidence_ready = _verification_evidence_ready(project_dir, verification)
    current_evidence_hashes = (
        current_verification_evidence_hashes(project_dir, verification)
        if record
        else {}
    )
    active = bool(
        record
        and record.get("schema_version") == MANUAL_RELEASE_SCHEMA_VERSION
        and record.get("status") == "released"
        and record.get("verification_run_id") == verification.get("run_id")
        and record.get("issue_fingerprint") == issue_fingerprint
        and record.get("confirmed_issue_ids") == sorted(current_issue_ids)
        and record.get("input_hashes") == current_inputs
        and record.get("evidence_hashes") == current_evidence_hashes
        and inputs_match
        and not job_active
        and verification.get("status") == "verified_failed"
        and bool(issues)
    )
    can_release = bool(
        not active
        and verification.get("status") == "verified_failed"
        and bool(issues)
        and inputs_match
        and evidence_ready
        and not job_active
    )
    blocked_reason = None
    blocked_message = None
    if active:
        blocked_reason = "already_released"
        blocked_message = "当前失败核对已经完成有效人工放行。"
    elif job_active:
        blocked_reason = "verification_in_progress"
        blocked_message = "新的核对正在进行，完成后才能人工审核放行。"
    elif verification.get("status") != "verified_failed":
        blocked_reason = "verification_not_failed"
        blocked_message = "当前没有需要人工放行的失败核对结果。"
    elif not issues:
        blocked_reason = "no_reviewable_issues"
        blocked_message = "失败核对没有可供逐项确认的问题清单，请重新核对。"
    elif not inputs_match:
        blocked_reason = "verification_inputs_changed"
        blocked_message = "核对结果不属于当前完整输入，请重新生成并核对。"
    elif not evidence_ready:
        blocked_reason = "verification_evidence_incomplete"
        blocked_message = "当前核对报告或证据归档不完整，不能进行人工放行。"
    return {
        "active": active,
        "status": "released" if active else "not_released",
        "can_release": can_release,
        "blocked_reason": blocked_reason,
        "blocked_message": blocked_message,
        "verification_job_status": job_status,
        "verification_inputs_match": inputs_match,
        "evidence_ready": evidence_ready,
        "issue_count": len(issues),
        "issue_ids": current_issue_ids,
        "issue_preview": issues,
        "issue_fingerprint": issue_fingerprint,
        "release": record if active else None,
    }


def create_manual_release(
    workspace_root: Path | str,
    project_id: str,
    *,
    reviewer: str,
    reason: str,
    confirm_all_issues: bool,
    confirmed_issue_ids_json: str,
) -> dict[str, Any]:
    """Record an explicit human release for the current failed verification."""

    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if not reviewer:
        raise ManualReleaseError("人工放行必须填写操作人。")
    if not reason:
        raise ManualReleaseError("人工放行必须填写确认理由。")
    if len(reviewer) > 120 or len(reason) > 2000:
        raise ManualReleaseError("操作人或确认理由超过长度限制。")
    if not confirm_all_issues:
        raise ManualReleaseError("必须勾选确认当前所有异常事实后才能放行。")
    try:
        confirmed_issue_ids = json.loads(str(confirmed_issue_ids_json or "[]"))
    except json.JSONDecodeError as exc:
        raise ManualReleaseError("异常确认清单格式无效。") from exc
    if not isinstance(confirmed_issue_ids, list) or not all(isinstance(value, str) for value in confirmed_issue_ids):
        raise ManualReleaseError("异常确认清单格式无效。")

    project_dir = resolve_project_dir(workspace_root, project_id)
    manifest = _read_json(project_dir / "manifest.json")
    verification = manifest.get("verification") if isinstance(manifest, dict) else {}
    if not isinstance(verification, dict) or verification.get("status") != "verified_failed":
        raise ManualReleaseConflict("当前没有可人工放行的失败核对结果；请先完成一次核对。")
    job = _read_json(project_dir / "reports" / "verification_job.json")
    if job.get("status") in {"queued", "running"}:
        raise ManualReleaseConflict("核对仍在进行，完成后才能人工放行。")
    report = _read_json(project_dir / "reports" / "verification_report.json")
    current_inputs = current_verification_input_hashes(project_dir)
    expected_inputs = verification.get("input_hashes") if isinstance(verification.get("input_hashes"), dict) else {}
    if not verification_inputs_match(expected_inputs, current_inputs):
        raise ManualReleaseConflict("核对结果已过期或输入已变化，请重新生成并核对。")
    if not _verification_evidence_ready(project_dir, verification):
        raise ManualReleaseConflict("核对报告或证据归档不完整，不能人工放行。")
    evidence_hashes = current_verification_evidence_hashes(project_dir, verification)
    if not evidence_hashes.get("verification_report") or not evidence_hashes.get("verification_run_tree"):
        raise ManualReleaseConflict("核对报告或证据归档不完整，不能人工放行。")
    issues = collect_manual_release_issues(report, verification)
    if not issues:
        raise ManualReleaseConflict("当前核对没有可确认的异常事实，不需要人工放行。")
    issue_ids = sorted(issue["issue_id"] for issue in issues)
    if sorted(set(confirmed_issue_ids)) != issue_ids:
        raise ManualReleaseError("必须逐项确认当前核对列出的全部异常事实。")

    payload = {
        "schema_version": MANUAL_RELEASE_SCHEMA_VERSION,
        "project_id": project_id,
        "status": "released",
        "release_id": uuid.uuid4().hex,
        "released_at": utc_now(),
        "reviewer": reviewer,
        "reason": reason,
        "verification_run_id": str(verification.get("run_id") or ""),
        "verification_status": "verified_failed",
        "issue_fingerprint": fingerprint_issues(issues),
        "confirmed_issue_ids": issue_ids,
        "confirmed_issues": issues,
        "input_hashes": current_inputs,
        "evidence_hashes": evidence_hashes,
    }
    path = project_dir / MANUAL_RELEASE_RELATIVE_PATH
    write_standard_json(path, payload, "verification_manual_release.schema.json")
    return {
        "project_id": project_id,
        "status": "released",
        "release": payload,
        "manual_release": get_manual_release_state(project_dir, verification, current_inputs, issues),
    }


def fingerprint_issues(issues: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        sorted(issues, key=lambda value: value["issue_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _issue_id(item_number: str, field: str, status: str, contract_value: Any, mark_value: Any) -> str:
    canonical = json.dumps(
        {
            "item_number": str(item_number or ""),
            "field": field,
            "status": status,
            "contract_value": _display_value(contract_value),
            "mark_value": _display_value(mark_value),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _non_negative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _resolve_project_path(project_dir: Path, relative_path: str) -> Path | None:
    try:
        candidate = (project_dir / relative_path).resolve()
        candidate.relative_to(project_dir.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _verification_evidence_ready(
    project_dir: Path,
    verification: dict[str, Any],
) -> bool:
    report_path = project_dir / "reports" / "verification_report.json"
    run_id = str(verification.get("run_id") or "").strip()
    run_path_value = str(verification.get("run_path") or "").strip()
    run_path = _resolve_project_path(project_dir, run_path_value) if run_path_value else None
    run_manifest = (
        _read_json(run_path / "run_manifest.json")
        if run_path is not None and run_path.is_dir()
        else {}
    )
    return bool(
        run_id
        and report_path.is_file()
        and run_path is not None
        and run_path.is_dir()
        and run_manifest.get("run_id") == run_id
    )


def _hash_directory(directory: Path) -> str:
    root = directory.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        return ""
    digest = hashlib.sha256()
    for path in files:
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            return ""
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(resolved.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(resolved).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
