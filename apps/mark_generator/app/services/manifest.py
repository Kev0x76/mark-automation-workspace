from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .excel_contract import contract_media_type
from .schema_validation import write_standard_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def artifact(path: str, status: str, media_type: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": path, "status": status}
    if media_type:
        payload["media_type"] = media_type
    return payload


def artifact_or_null(project_dir: Path, relative_path: str, media_type: str | None = None) -> dict[str, Any] | None:
    path = project_dir / relative_path
    if not path.exists():
        return None
    return artifact(relative_path, "generated" if path.is_file() else "present", media_type)


def expected_directory_artifact(project_dir: Path, relative_path: str) -> dict[str, Any]:
    """Describe a lazy workspace without claiming that it already exists."""

    status = "present" if (project_dir / relative_path).is_dir() else "expected"
    return artifact(relative_path, status)


def contract_artifact(project_dir: Path) -> dict[str, Any]:
    """Reference the preserved source extension in the project manifest."""

    report_path = project_dir / "reports" / "extraction_report.json"
    candidates: list[str] = []
    if report_path.exists():
        try:
            source_contract = json.loads(report_path.read_text(encoding="utf-8")).get("source_contract")
            if isinstance(source_contract, str):
                candidates.append(source_contract.replace("\\", "/"))
        except (OSError, json.JSONDecodeError):
            pass
    candidates.extend(["input/contract.xlsx", "input/contract.xls"])

    for relative_path in dict.fromkeys(candidates):
        path = (project_dir / relative_path).resolve()
        try:
            path.relative_to(project_dir.resolve())
        except ValueError:
            continue
        if path.is_file():
            return artifact(relative_path, "present", contract_media_type(path))

    # Keep the manifest schema valid for a project that has not reached the
    # extraction step yet.
    return artifact("input/contract.xlsx", "present", contract_media_type(".xlsx"))


def write_manifest(
    project_dir: Path,
    project_id: str,
    status: str,
    warnings: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    template_mode: str = "kmt_style_v1",
    verification: dict[str, Any] | None = None,
    production_ready: bool = False,
    needs_v5_check: bool = True,
) -> dict[str, Any]:
    manifest_path = project_dir / "manifest.json"
    created_at = utc_now()
    if manifest_path.exists():
        try:
            created_at = json.loads(manifest_path.read_text(encoding="utf-8")).get("created_at", created_at)
        except json.JSONDecodeError:
            pass

    data_artifacts = {
        "items_normalized": artifact("data/items.normalized.json", "present", "application/json")
    }
    contract_snapshot = artifact_or_null(
        project_dir,
        "data/contract.snapshot.json",
        "application/json",
    )
    if contract_snapshot is not None:
        data_artifacts["contract_snapshot"] = contract_snapshot

    report_artifacts = {
        "extraction_report": artifact_or_null(
            project_dir, "reports/extraction_report.json", "application/json"
        ),
        "unmatched_images": artifact_or_null(
            project_dir, "reports/unmatched_images.json", "application/json"
        ),
        "product_regions": artifact_or_null(
            project_dir, "reports/product_regions.json", "application/json"
        ),
        "generation_report": artifact_or_null(
            project_dir, "reports/generation_report.json", "application/json"
        ),
        "verification_report": artifact_or_null(
            project_dir, "reports/verification_report.json", "application/json"
        ),
        "verification_summary_csv": artifact_or_null(
            project_dir, "reports/verification_summary.csv", "text/csv"
        ),
    }
    for key, relative_path in (
        ("product_region_job", "reports/product_region_job.json"),
        ("lineart_groups", "reports/lineart_groups.json"),
        ("lineart_group_approvals", "reports/lineart_group_approvals.json"),
        ("verification_manual_release", "reports/verification_manual_release.json"),
    ):
        report_artifact = artifact_or_null(project_dir, relative_path, "application/json")
        if report_artifact is not None:
            report_artifacts[key] = report_artifact

    payload = {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "created_at": created_at,
        "updated_at": utc_now(),
        "template_mode": template_mode,
        "production_ready": production_ready,
        "needs_v5_check": needs_v5_check,
        "status": status,
        "template": {"id": "kmt_style_v1", "version": "1.0.0"},
        "tools": {"generator": {"name": "mark_generator", "version": "0.1.0"}},
        "directories": {
            "project_root": ".",
            "input": "input",
            "data": "data",
            "extracted": "extracted",
            "product_regions": "extracted/product_regions",
            "linearts": "linearts",
            "lineart_candidates": "linearts/candidates",
            "lineart_approved": "linearts/approved",
            "generated": "generated",
            "reports": "reports",
        },
        "inputs": {
            "contract": contract_artifact(project_dir)
        },
        "data": data_artifacts,
        "extracted": {
            "product_images_dir": artifact("extracted/product_images", "present"),
            "product_regions_dir": artifact_or_null(project_dir, "extracted/product_regions"),
        },
        "linearts": {
            "candidates_dir": expected_directory_artifact(project_dir, "linearts/candidates"),
            "approved_dir": expected_directory_artifact(project_dir, "linearts/approved"),
            "lineart_report": artifact_or_null(project_dir, "reports/lineart_report.json", "application/json"),
        },
        "generated": {
            "preview_pdf": artifact_or_null(project_dir, "generated/preview.pdf", "application/pdf"),
            "standard_mark_pdf": artifact_or_null(project_dir, "generated/standard_mark.pdf", "application/pdf"),
        },
        "reports": report_artifacts,
        "warnings": warnings,
        "errors": errors,
    }
    if verification is not None:
        payload["verification"] = verification
    write_standard_json(manifest_path, payload, "manifest.schema.json")
    return payload
