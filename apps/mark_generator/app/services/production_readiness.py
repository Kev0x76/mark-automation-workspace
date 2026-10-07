"""Pure-read production stage gates for one mark project."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from .contract_snapshot import file_sha256
from .product_regions import resolve_project_dir
from .verification_manual_release import (
    collect_manual_release_issues,
    current_verification_input_hashes,
    get_manual_release_state,
    verification_inputs_match,
)


def get_production_input_hashes(project_dir: Path) -> dict[str, str]:
    """Fingerprint every mutable generator input that can change the PDF."""

    normalized_path = project_dir / "data" / "items.normalized.json"
    normalized_hash = file_sha256(normalized_path) if normalized_path.is_file() else ""
    normalized = _read_json(normalized_path)
    linearts: list[dict[str, str]] = []
    for item in normalized.get("items", []):
        if not isinstance(item, dict) or item.get("lineart_status") != "approved":
            continue
        relative = str(item.get("lineart_asset") or "")
        path = (project_dir / relative).resolve() if relative else None
        digest = ""
        if path is not None:
            try:
                path.relative_to(project_dir.resolve())
            except ValueError:
                path = None
        if path is not None and path.is_file():
            digest = file_sha256(path)
        linearts.append(
            {
                "item_no": str(item.get("item_no") or ""),
                "path": relative.replace("\\", "/"),
                "sha256": digest,
            }
        )
    canonical = json.dumps(
        sorted(linearts, key=lambda value: (value["item_no"], value["path"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "items_normalized": normalized_hash,
        "approved_linearts": hashlib.sha256(canonical).hexdigest(),
    }


def get_production_readiness(workspace_root: Path | str, project_id: str) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = _read_json(project_dir / "data" / "items.normalized.json")
    items = normalized.get("items", []) if isinstance(normalized, dict) else []
    total = len(items)

    contract_files = list((project_dir / "input").glob("contract.xls*"))
    snapshot_exists = (project_dir / "data" / "contract.snapshot.json").is_file()
    contract_ready = bool(contract_files and snapshot_exists and total)

    job = _read_json(project_dir / "reports" / "product_region_job.json")
    job_status = str(job.get("status") or "idle") if isinstance(job, dict) else "idle"
    selected_regions = sum(1 for item in items if item.get("product_region_status") == "selected")
    manual_lineart_substitutes = sum(
        1
        for item in items
        if (item.get("lineart_provenance") or {}).get("binding_scope")
        == "manual_item_without_product_region"
        and item.get("lineart_status") == "approved"
    )
    region_attention = sum(
        1
        for item in items
        if item.get("product_region_status") in {"needs_review", "rejected"}
    )
    missing_source = sum(1 for item in items if item.get("source_image_status") != "matched")
    unresolved_missing_source = sum(
        1
        for item in items
        if item.get("source_image_status") != "matched"
        and (item.get("lineart_provenance") or {}).get("binding_scope")
        != "manual_item_without_product_region"
    )
    region_satisfied_items = sum(
        1
        for item in items
        if item.get("product_region_status") == "selected"
        or (
            (item.get("lineart_provenance") or {}).get("binding_scope")
            == "manual_item_without_product_region"
            and item.get("lineart_status") == "approved"
        )
    )
    regions_ready = total > 0 and region_satisfied_items == total
    if job_status in {"queued", "running"}:
        regions_status = "processing"
    elif regions_ready:
        regions_status = "ready"
    elif job_status in {"failed", "interrupted"} or region_attention or unresolved_missing_source:
        regions_status = "needs_attention"
    else:
        regions_status = "not_started"

    approved_linearts = sum(1 for item in items if item.get("lineart_status") == "approved")
    linearts_ready = total > 0 and approved_linearts == total
    linearts_status = "ready" if linearts_ready else (
        "blocked_by_regions" if not regions_ready else "needs_attention"
    )

    pdf_path = project_dir / "generated" / "standard_mark.pdf"
    pdf_ready = pdf_path.is_file() and pdf_path.stat().st_size > 0
    manifest = _read_json(project_dir / "manifest.json")
    verification = manifest.get("verification", {}) if isinstance(manifest, dict) else {}
    verification_status = str(verification.get("status") or "not_started")
    expected_pdf_hashes = (verification.get("input_hashes") or {}).get("marks") or []
    expected_generation_inputs = verification.get("input_hashes") or {}
    current_generation_inputs = get_production_input_hashes(project_dir)
    current_pdf_sha256 = file_sha256(pdf_path) if pdf_ready else None
    current_verification_inputs = current_verification_input_hashes(project_dir)
    verification_inputs_match_current = verification_inputs_match(
        expected_generation_inputs,
        current_verification_inputs,
    )
    verification_report = _read_json(project_dir / "reports" / "verification_report.json")
    manual_release = get_manual_release_state(
        project_dir,
        verification,
        current_verification_inputs,
        collect_manual_release_issues(verification_report, verification),
    )
    pdf_matches_verified_input = bool(
        current_pdf_sha256 and current_pdf_sha256 in expected_pdf_hashes
    )
    generation_inputs_match_verified = all(
        expected_generation_inputs.get(key) == value and bool(value)
        for key, value in current_generation_inputs.items()
    )
    automated_verified = (
        verification_status == "verified_pass"
        and bool(manifest.get("production_ready"))
        and manifest.get("needs_v5_check") is False
        and pdf_matches_verified_input
        and generation_inputs_match_verified
        and verification_inputs_match_current
    )
    verified = automated_verified or manual_release["active"]

    next_action = _next_action(
        contract_ready=contract_ready,
        job_status=job_status,
        missing_source=unresolved_missing_source,
        regions_ready=regions_ready,
        region_attention=region_attention,
        linearts_ready=linearts_ready,
        pdf_ready=pdf_ready,
        verified=verified,
        manual_release_active=manual_release["active"],
        manual_release_can_release=manual_release["can_release"],
    )
    return {
        "project_id": project_id,
        "production_ready": verified,
        "automated_production_ready": automated_verified,
        "release_mode": "manual_override" if manual_release["active"] else "automatic" if automated_verified else None,
        "manual_release": manual_release,
        "next_action": next_action,
        "stages": {
            "contract": {
                "status": "ready" if contract_ready else "incomplete",
                "items": total,
                "snapshot": "data/contract.snapshot.json" if snapshot_exists else None,
                "integrity_check": "performed_before_verification",
            },
            "product_regions": {
                "status": regions_status,
                "selected_items": selected_regions,
                "total_items": total,
                "needs_attention_items": region_attention,
                "missing_source_image_items": missing_source,
                "manual_lineart_substitute_items": manual_lineart_substitutes,
                "job": job or None,
            },
            "linearts": {
                "status": linearts_status,
                "approved_items": approved_linearts,
                "total_items": total,
                "category_suggestions": (
                    "reports/lineart_groups.json"
                    if (project_dir / "reports" / "lineart_groups.json").is_file()
                    else None
                ),
            },
            "pdf": {
                "status": "ready" if pdf_ready else "not_generated",
                "path": "generated/standard_mark.pdf" if pdf_ready else None,
                "sha256": current_pdf_sha256,
                "matches_verified_input": pdf_matches_verified_input,
                "generation_inputs_match_verified": generation_inputs_match_verified,
                "verification_inputs_match_current": verification_inputs_match_current,
            },
            "verification": {
                "status": (
                    "verified_pass" if automated_verified else "manual_release" if manual_release["active"] else "stale_or_not_releasable"
                    if verification_status == "verified_pass"
                    else verification_status
                ),
                "automated_status": verification_status,
                "contract_facts_source": verification.get("contract_facts_source"),
                "manual_release": manual_release["active"],
            },
        },
    }


def _next_action(
    *,
    contract_ready: bool,
    job_status: str,
    missing_source: int,
    regions_ready: bool,
    region_attention: int,
    linearts_ready: bool,
    pdf_ready: bool,
    verified: bool,
    manual_release_active: bool = False,
    manual_release_can_release: bool = False,
) -> dict[str, Any]:
    if not contract_ready:
        return {"code": "extract_contract", "message": "Parse the contract and create its canonical snapshot."}
    if job_status in {"queued", "running"}:
        return {"code": "wait_product_regions", "message": "Product regions are processing in the background."}
    if missing_source:
        return {
            "code": "supply_product_visual",
            "message": "Some items have no product image. Update the contract, add a sales image, or add lineart manually.",
            "affected_items": missing_source,
        }
    if not regions_ready:
        return {
            "code": "prepare_product_regions" if not region_attention else "review_product_regions",
            "message": "Start product-region preparation or resolve items that need attention.",
        }
    if not linearts_ready:
        return {"code": "prepare_linearts", "message": "Generate, bind, and approve the required lineart assets."}
    if not pdf_ready:
        return {"code": "generate_standard_pdf", "message": "Generate the delivery PDF with the standard filename."}
    if manual_release_active:
        return {
            "code": "manual_release",
            "message": "人工确认已记录，当前 PDF 可交付；如输入变化需重新核对。",
        }
    if manual_release_can_release:
        return {
            "code": "review_verification_issues",
            "message": "自动核对发现问题；请逐项人工审核后放行，或修改后重新核对。",
        }
    if not verified:
        return {"code": "verify_standard_pdf", "message": "Verify the PDF against the canonical contract snapshot."}
    return {"code": "complete", "message": "The standard PDF is verified and ready for delivery."}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
