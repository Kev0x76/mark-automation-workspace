"""Auditable manual-lineart fallback when a contract has no product photo."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from . import lineart as lineart_service
from .contract_snapshot import file_sha256, read_contract_snapshot
from .image_postprocess import postprocess_lineart_bytes
from .lineart_target_qc import TargetSizeQcError, evaluate_target_size
from .schema_validation import validate_standard_payload


MANUAL_NO_REGION_SCOPE = "manual_item_without_product_region"


def upload_manual_lineart_without_product_region(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    filename: str,
    data: bytes,
    *,
    reviewer: str,
    reason: str,
    confirm_item_identity: bool,
) -> dict[str, Any]:
    if not confirm_item_identity:
        raise lineart_service.LineartError(
            "Manual lineart without a product photo requires explicit item-identity confirmation."
        )
    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if not reviewer or not reason:
        raise lineart_service.LineartError("Manual lineart fallback requires reviewer and reason.")
    extension = Path(filename or "").suffix.lower()
    if extension not in lineart_service.ALLOWED_LINEART_EXTENSIONS:
        raise lineart_service.LineartError("Unsupported manual lineart file type.")

    project_dir = lineart_service.resolve_project_dir(workspace_root, project_id)
    normalized = lineart_service.load_normalized(project_dir)
    item = lineart_service.find_item(normalized, item_no)
    if item.get("source_image_status") == "matched" or item.get("product_region_status") == "selected":
        raise lineart_service.LineartError(
            "This fallback is only for an item without a usable product image/region; use the normal lineart upload route."
        )
    snapshot_path = project_dir / "data" / "contract.snapshot.json"
    if not snapshot_path.is_file():
        raise lineart_service.LineartError("A canonical contract snapshot is required for manual lineart fallback.")
    contract_paths = sorted((project_dir / "input").glob("contract.xls*"))
    if len(contract_paths) != 1:
        raise lineart_service.LineartError("Exactly one archived source contract is required for manual lineart fallback.")
    try:
        snapshot_payload = read_contract_snapshot(snapshot_path, contract_path=contract_paths[0])
        validate_standard_payload(snapshot_payload, "contract.snapshot.schema.json")
    except ValueError as exc:
        raise lineart_service.LineartError(f"Contract snapshot is not production-valid: {exc}") from exc
    matching_snapshot_items = [
        entry
        for entry in snapshot_payload.get("items", [])
        if str(entry.get("item_number") or "") == str(item_no)
    ]
    if len(matching_snapshot_items) != 1:
        raise lineart_service.LineartError(
            "Manual lineart item must occur exactly once in the canonical contract snapshot."
        )

    try:
        processed, warnings = postprocess_lineart_bytes(data)
    except (OSError, ValueError) as exc:
        raise lineart_service.LineartError(f"Manual lineart file could not be decoded: {exc}") from exc
    try:
        target_qc, _ = evaluate_target_size(processed)
    except TargetSizeQcError as exc:
        target_qc = {"technical_passed": False, "failure_reasons": ["target_size_qc_unavailable"], "error": str(exc)}
        warnings = list(warnings) + [
            {
                "severity": "warning",
                "code": "target_size_qc_unavailable",
                "message": str(exc),
            }
        ]
    failures = list(target_qc.get("failure_reasons") or [])

    lineart_service.ensure_lineart_dirs(project_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_item = lineart_service.safe_name(item_no)
    safe_stem = lineart_service.safe_name(Path(filename).stem or "manual-lineart")
    uploaded_dir = project_dir / "linearts" / "uploaded" / safe_item
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    uploaded_path = lineart_service.unique_path(uploaded_dir / f"{timestamp}-{safe_stem}.png")
    uploaded_path.write_bytes(processed)
    approved_path = lineart_service.unique_path(
        project_dir / "linearts" / "approved" / f"{safe_item}.png"
    )
    approved_path.write_bytes(processed)

    approved_at = lineart_service.utc_now()
    if item.get("lineart_asset"):
        item["previous_lineart_asset"] = item["lineart_asset"]
    item["lineart_status"] = "approved"
    item["lineart_asset"] = lineart_service.project_relative(project_dir, approved_path)
    item["lineart_qc_status"] = "accepted"
    item["lineart_stale_reason"] = None
    item["lineart_provenance"] = {
        "binding_scope": MANUAL_NO_REGION_SCOPE,
        "item_no": str(item_no),
        "provider": "manual_upload_without_product_region",
        "uploaded_path": lineart_service.project_relative(project_dir, uploaded_path),
        "approved_path": lineart_service.project_relative(project_dir, approved_path),
        "source_upload_sha256": hashlib.sha256(data).hexdigest(),
        "approved_sha256": lineart_service.sha256_file(approved_path),
        "contract_snapshot_path": "data/contract.snapshot.json",
        "contract_snapshot_sha256": file_sha256(snapshot_path),
        "quality": {
            "technical_qc_passed": bool(target_qc.get("technical_passed")),
            "quality_gate_applied": False,
            "target_size_qc": target_qc,
            "failure_reasons": failures,
            "warnings": warnings,
        },
        "approval": {
            "mode": "manual_item_without_product_region",
            "reviewer": reviewer,
            "reason": reason,
            "confirmed_item_identity": True,
            "approved_at": approved_at,
        },
        "approved_at": approved_at,
    }
    lineart_service.save_normalized(project_dir, normalized)
    report = lineart_service.write_lineart_report(
        project_dir,
        normalized,
        action="manual_lineart_without_product_region",
        item_no=str(item_no),
        candidate_path=lineart_service.project_relative(project_dir, uploaded_path),
        approved_path=lineart_service.project_relative(project_dir, approved_path),
        provider="manual_upload_without_product_region",
        warnings=warnings,
    )
    return {
        "project_id": project_id,
        "item_no": str(item_no),
        "item": item,
        "approved": lineart_service.file_state(project_dir, approved_path),
        "approval": item["lineart_provenance"]["approval"],
        "quality": item["lineart_provenance"]["quality"],
        "summary": report["summary"],
    }


def validate_manual_no_region_provenance(
    project_dir: Path,
    item: dict[str, Any],
) -> dict[str, str] | None:
    provenance = item.get("lineart_provenance") or {}
    approval = provenance.get("approval") or {}
    quality = provenance.get("quality") or {}
    if approval.get("mode") != MANUAL_NO_REGION_SCOPE or not approval.get("confirmed_item_identity"):
        return _issue("Manual lineart is missing explicit item-identity confirmation.")
    if not str(approval.get("reviewer") or "").strip() or not str(approval.get("reason") or "").strip():
        return _issue("Manual lineart has incomplete reviewer evidence.")
    if item.get("lineart_qc_status") != "accepted" or quality.get("quality_gate_applied") is not False:
        return _issue("Manual lineart has no integrity-only acceptance record.")
    snapshot_path = project_dir / str(provenance.get("contract_snapshot_path") or "")
    contract_paths = sorted((project_dir / "input").glob("contract.xls*"))
    if not snapshot_path.is_file() or file_sha256(snapshot_path) != provenance.get("contract_snapshot_sha256"):
        return _issue("Manual lineart contract snapshot no longer matches.")
    if len(contract_paths) != 1:
        return _issue("Manual lineart source contract is missing or ambiguous.")
    try:
        snapshot_payload = read_contract_snapshot(snapshot_path, contract_path=contract_paths[0])
        validate_standard_payload(snapshot_payload, "contract.snapshot.schema.json")
    except ValueError as exc:
        return _issue(f"Manual lineart contract snapshot is invalid: {exc}")
    snapshot_matches = [
        entry
        for entry in snapshot_payload.get("items", [])
        if str(entry.get("item_number") or "") == str(item.get("item_no") or "")
    ]
    if len(snapshot_matches) != 1:
        return _issue("Manual lineart item no longer occurs exactly once in the contract snapshot.")
    try:
        asset_path = lineart_service.resolve_project_file(project_dir, str(item.get("lineart_asset") or ""))
    except lineart_service.LineartError as exc:
        return _issue(str(exc))
    if not asset_path.is_file() or lineart_service.sha256_file(asset_path) != provenance.get("approved_sha256"):
        return _issue("Manual lineart file hash no longer matches.")
    return None


def _issue(message: str) -> dict[str, str]:
    return {"code": "manual_lineart_binding_mismatch", "message": message}
