"""Explicit, auditable category-level lineart sharing.

Category names create suggestions.  Sharing happens only after a human states
that the visible shape is the same and differences are limited to the allowed
variant dimensions.  The representative SKU remains the true source of the
asset; member provenance never pretends that the asset came from each member's
own product region.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from . import lineart as lineart_service
from .schema_validation import write_standard_json


ALLOWED_VARIATIONS = {"size", "color", "pattern"}
GROUP_BINDING_SCOPE = "category_group"
MANUAL_GROUP_METHOD = "manual_selection_v1"


def create_manual_lineart_group(
    workspace_root: Path | str,
    project_id: str,
    *,
    item_nos: Iterable[str],
    group_name: str,
    reviewer: str,
    reason: str,
) -> dict[str, Any]:
    """Persist an operator-selected category without sharing any asset.

    Manual grouping is intentionally a separate suggestion layer.  It does not
    change the deterministic Chinese-name groups, approve a lineart, or copy a
    representative asset.  The existing explicit share action remains the
    only operation that can bind one approved lineart to the group.
    """

    group_name = str(group_name or "").strip()
    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    members = sorted({str(value).strip() for value in item_nos if str(value).strip()})
    if not group_name:
        raise lineart_service.LineartError("Manual category group requires a name.")
    if not reviewer or not reason:
        raise lineart_service.LineartError("Manual category group requires reviewer and reason.")
    if len(members) < 2:
        raise lineart_service.LineartError("Manual category group requires at least two unique items.")

    project_dir = lineart_service.resolve_project_dir(workspace_root, project_id)
    normalized = lineart_service.load_normalized(project_dir)
    items = {str(item.get("item_no") or "").strip(): item for item in normalized.get("items", [])}
    missing = [item_no for item_no in members if item_no not in items]
    if missing:
        raise lineart_service.LineartError(
            "Manual category members do not exist in items.normalized.json: " + ", ".join(missing)
        )

    payload_path = project_dir / "reports" / "lineart_groups.json"
    payload = _read_json(payload_path)
    if not payload:
        raise lineart_service.LineartError("Lineart category suggestions are missing.")
    group_id = manual_group_id(group_name, members)
    existing = next(
        (group for group in payload.get("manual_groups", []) if group.get("group_id") == group_id),
        None,
    )
    if existing:
        return {"project_id": project_id, "created": False, "group": existing}

    original_names = sorted(
        {str(items[item_no].get("cn_name") or "").strip() for item_no in members}
    )
    if "" in original_names:
        raise lineart_service.LineartError(
            "Manual category members must all have a Chinese product name."
        )
    group = {
        "group_id": group_id,
        "normalized_cn_name": group_name,
        "members": [
            {
                "item_no": item_no,
                "cn_name": str(items[item_no].get("cn_name") or "").strip(),
                "normalization_changed": False,
            }
            for item_no in members
        ],
        "evidence": {
            "method": MANUAL_GROUP_METHOD,
            "member_count": len(members),
            "original_cn_names": original_names,
            "normalization": ["operator_selected_members"],
        },
        "variant_evidence_status": "missing_from_item_contract",
        "reuse_decision": "needs_variant_evidence",
        "auto_lineart_reuse": False,
        "manual_creation": {
            "reviewer": reviewer,
            "reason": reason,
            "created_at": lineart_service.utc_now(),
        },
    }
    payload.setdefault("manual_groups", []).append(group)
    payload["manual_groups"].sort(key=lambda entry: entry["group_id"])
    write_standard_json(payload_path, payload, "lineart_groups.schema.json")
    return {"project_id": project_id, "created": True, "group": group}


def get_lineart_group_state(workspace_root: Path | str, project_id: str) -> dict[str, Any]:
    project_dir = lineart_service.resolve_project_dir(workspace_root, project_id)
    suggestions = _read_json(project_dir / "reports" / "lineart_groups.json")
    approvals = _read_json(project_dir / "reports" / "lineart_group_approvals.json")
    return {
        "project_id": project_id,
        "suggestions": suggestions,
        "approvals": approvals or {
            "schema_version": "1.0.0",
            "project_id": project_id,
            "approvals": [],
        },
    }


def share_lineart_with_group(
    workspace_root: Path | str,
    project_id: str,
    group_id: str,
    *,
    representative_item_no: str,
    reviewer: str,
    reason: str,
    allowed_variations: Iterable[str],
    confirm_same_shape: bool,
) -> dict[str, Any]:
    if not confirm_same_shape:
        raise lineart_service.LineartError(
            "Category sharing requires explicit confirmation that all members have the same visible shape."
        )
    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if not reviewer or not reason:
        raise lineart_service.LineartError("Category sharing requires reviewer and reason.")
    variations = sorted({str(value).strip().lower() for value in allowed_variations if str(value).strip()})
    unsupported = sorted(set(variations) - ALLOWED_VARIATIONS)
    if unsupported:
        raise lineart_service.LineartError(
            "Unsupported category variation: " + ", ".join(unsupported)
        )

    project_dir = lineart_service.resolve_project_dir(workspace_root, project_id)
    normalized = lineart_service.load_normalized(project_dir)
    group_report = _read_json(project_dir / "reports" / "lineart_groups.json")
    group = _find_group(group_report, group_id)
    member_item_nos = sorted(str(member["item_no"]) for member in group.get("members", []))
    if len(member_item_nos) < 2:
        raise lineart_service.LineartError("Category sharing requires at least two member items.")
    representative_item_no = str(representative_item_no or "").strip()
    if representative_item_no not in member_item_nos:
        raise lineart_service.LineartError("Representative item is not a member of this category group.")

    items = {str(item.get("item_no")): item for item in normalized.get("items", [])}
    if any(item_no not in items for item_no in member_item_nos):
        raise lineart_service.LineartError("Category group members no longer match items.normalized.json.")
    if any(items[item_no].get("product_region_status") != "selected" for item_no in member_item_nos):
        raise lineart_service.LineartError(
            "Every category member must have a selected product region before sharing lineart."
        )

    representative = items[representative_item_no]
    representative_provenance = representative.get("lineart_provenance") or {}
    if representative.get("lineart_status") != "approved":
        raise lineart_service.LineartError("Representative item must already have an approved lineart.")
    if representative_provenance.get("binding_scope") == GROUP_BINDING_SCOPE:
        raise lineart_service.LineartError("A category-shared item cannot be used as a new representative.")
    provenance_issue = lineart_service.validate_lineart_provenance(project_dir, representative)
    if provenance_issue:
        raise lineart_service.LineartError(
            "Representative lineart is not production-valid: " + provenance_issue["message"]
        )
    asset = representative.get("lineart_asset")
    asset_path = lineart_service.resolve_project_file(project_dir, str(asset or ""))
    if not asset_path.is_file():
        raise lineart_service.LineartError("Representative approved lineart file is missing.")

    members_sha256 = group_members_sha256(group["normalized_cn_name"], member_item_nos)
    approved_sha256 = lineart_service.sha256_file(asset_path)
    approved_at = lineart_service.utc_now()
    approval = {
        "group_id": group_id,
        "normalized_cn_name": group["normalized_cn_name"],
        "members": member_item_nos,
        "members_sha256": members_sha256,
        "representative_item_no": representative_item_no,
        "allowed_variations": variations,
        "confirmed_same_shape": True,
        "reviewer": reviewer,
        "reason": reason,
        "asset_path": str(asset),
        "asset_sha256": approved_sha256,
        "representative_region_artifact_id": representative_provenance.get("region_artifact_id"),
        "approved_at": approved_at,
    }

    for item_no in member_item_nos:
        if item_no == representative_item_no:
            continue
        item = items[item_no]
        if item.get("lineart_asset") and item.get("lineart_asset") != asset:
            item["previous_lineart_asset"] = item["lineart_asset"]
        item["lineart_status"] = "approved"
        item["lineart_asset"] = asset
        item["lineart_qc_status"] = "human_passed"
        item["lineart_stale_reason"] = None
        item["lineart_provenance"] = {
            "binding_scope": GROUP_BINDING_SCOPE,
            "item_no": item_no,
            "lineart_group_id": group_id,
            "normalized_cn_name": group["normalized_cn_name"],
            "group_members_sha256": members_sha256,
            "representative_item_no": representative_item_no,
            "representative_region_artifact_id": representative_provenance.get("region_artifact_id"),
            "approved_path": asset,
            "approved_sha256": approved_sha256,
            "quality": representative_provenance.get("quality") or {},
            "approval": {
                "mode": "category_group_human",
                "reviewer": reviewer,
                "reason": reason,
                "confirmed_same_shape": True,
                "allowed_variations": variations,
                "approved_at": approved_at,
            },
        }

    lineart_service.save_normalized(project_dir, normalized)
    approvals_path = project_dir / "reports" / "lineart_group_approvals.json"
    approvals_payload = _read_json(approvals_path) or {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "approvals": [],
    }
    approvals_payload["approvals"] = [
        entry
        for entry in approvals_payload.get("approvals", [])
        if entry.get("group_id") != group_id
    ] + [approval]
    approvals_payload["approvals"].sort(key=lambda entry: entry["group_id"])
    write_standard_json(
        approvals_path,
        approvals_payload,
        "lineart_group_approvals.schema.json",
    )
    report = lineart_service.write_lineart_report(
        project_dir,
        normalized,
        action="share_category_lineart",
        item_no=representative_item_no,
        approved_path=str(asset),
    )
    return {
        "project_id": project_id,
        "group": approval,
        "updated_member_items": len(member_item_nos) - 1,
        "lineart_summary": report["summary"],
    }


def validate_group_lineart_provenance(
    project_dir: Path,
    item: dict[str, Any],
) -> dict[str, str] | None:
    provenance = item.get("lineart_provenance") or {}
    approval = provenance.get("approval") or {}
    if approval.get("mode") != "category_group_human":
        return _issue("Shared lineart has no explicit category approval.")
    if not approval.get("confirmed_same_shape"):
        return _issue("Shared lineart is missing same-shape confirmation.")
    if not str(approval.get("reviewer") or "").strip() or not str(approval.get("reason") or "").strip():
        return _issue("Shared lineart has incomplete reviewer evidence.")
    variations = set(approval.get("allowed_variations") or [])
    if variations - ALLOWED_VARIATIONS:
        return _issue("Shared lineart contains unsupported variation evidence.")

    group_report = _read_json(project_dir / "reports" / "lineart_groups.json")
    try:
        group = _find_group(group_report, str(provenance.get("lineart_group_id") or ""))
    except lineart_service.LineartError as exc:
        return _issue(str(exc))
    members = sorted(str(member["item_no"]) for member in group.get("members", []))
    expected_members_hash = group_members_sha256(group["normalized_cn_name"], members)
    if provenance.get("group_members_sha256") != expected_members_hash:
        return _issue("Shared lineart category membership has changed.")
    if str(item.get("item_no")) not in members:
        return _issue("Item is no longer a member of the shared lineart category.")

    normalized = lineart_service.load_normalized(project_dir)
    representative_item_no = str(provenance.get("representative_item_no") or "")
    try:
        representative = lineart_service.find_item(normalized, representative_item_no)
    except lineart_service.LineartError as exc:
        return _issue(str(exc))
    representative_provenance = representative.get("lineart_provenance") or {}
    if representative_provenance.get("binding_scope") == GROUP_BINDING_SCOPE:
        return _issue("Shared lineart representative is no longer directly bound to a product region.")
    representative_issue = lineart_service.validate_lineart_provenance(project_dir, representative)
    if representative_issue:
        return _issue("Shared lineart representative is stale: " + representative_issue["message"])
    if representative_provenance.get("region_artifact_id") != provenance.get("representative_region_artifact_id"):
        return _issue("Shared lineart representative product region has changed.")
    if representative.get("lineart_asset") != item.get("lineart_asset"):
        return _issue("Shared lineart asset no longer matches its representative.")
    try:
        asset_path = lineart_service.resolve_project_file(project_dir, str(item.get("lineart_asset") or ""))
    except lineart_service.LineartError as exc:
        return _issue(str(exc))
    if not asset_path.is_file() or lineart_service.sha256_file(asset_path) != provenance.get("approved_sha256"):
        return _issue("Shared lineart file hash no longer matches.")
    return None


def group_members_sha256(normalized_cn_name: str, member_item_nos: Iterable[str]) -> str:
    body = json.dumps(
        {"normalized_cn_name": normalized_cn_name, "members": sorted(member_item_nos)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def manual_group_id(group_name: str, member_item_nos: Iterable[str]) -> str:
    body = json.dumps(
        {"group_name": group_name.strip(), "members": sorted(member_item_nos)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "manual-" + hashlib.sha256(body).hexdigest()[:16]


def _find_group(payload: dict[str, Any], group_id: str) -> dict[str, Any]:
    for group in [*payload.get("groups", []), *payload.get("manual_groups", [])]:
        if group.get("group_id") == group_id:
            return group
    raise lineart_service.LineartError(f"Lineart category group not found: {group_id}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise lineart_service.LineartError(f"Unreadable category lineart artifact: {path.name}") from exc
    return payload if isinstance(payload, dict) else {}


def _issue(message: str) -> dict[str, str]:
    return {"code": "lineart_group_binding_mismatch", "message": message}
