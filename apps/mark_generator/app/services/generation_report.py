from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema_validation import write_standard_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def artifact(path: str, status: str, media_type: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": path, "status": status}
    if media_type:
        payload["media_type"] = media_type
    return payload


def artifact_or_null(path: Path, relative_path: str, status: str = "generated", media_type: str | None = None) -> dict[str, Any] | None:
    if not (path / relative_path).exists():
        return None
    return artifact(relative_path, status, media_type)


def write_generation_report(
    project_dir: Path,
    *,
    project_id: str,
    mode: str,
    status: str,
    items: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    template_mode: str = "kmt_style_v1",
    report_relative_path: str = "reports/generation_report.json",
) -> dict[str, Any]:
    total = len(items)
    approved = sum(1 for item in items if item.get("lineart_status") == "approved")
    generated = sum(1 for item in items if item.get("status") in {"generated", "preview_only"})
    payload = {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "generated_at": utc_now(),
        "template_mode": template_mode,
        "production_ready": False,
        "needs_v5_check": True,
        "generator": {"name": "mark_generator", "version": "0.1.0"},
        "template": {"id": "kmt_style_v1", "version": "1.0.0"},
        "mode": mode,
        "status": status,
        "inputs": {
            "items_normalized": artifact("data/items.normalized.json", "present", "application/json"),
            "manifest": artifact_or_null(project_dir, "manifest.json", "present", "application/json"),
            "approved_linearts_dir": artifact("linearts/approved", "present"),
        },
        "outputs": {
            "preview_pdf": artifact_or_null(project_dir, "generated/preview.pdf", "generated", "application/pdf"),
            "standard_mark_pdf": artifact_or_null(project_dir, "generated/standard_mark.pdf", "generated", "application/pdf"),
            "manifest": artifact_or_null(project_dir, "manifest.json", "generated", "application/json"),
        },
        "summary": {
            "total_items": total,
            "generated_items": generated,
            "skipped_items": total - generated,
            "approved_linearts": approved,
            "missing_linearts": total - approved,
            "warning_count": len(warnings),
            "error_count": len(errors),
        },
        "rules": {
            "requires_approved_lineart_for_standard_pdf": True,
            "barcode_source": "programmatic_ean13",
            "layout_control": "programmatic_coordinates",
        },
        "items": items,
        "warnings": warnings,
        "errors": errors,
    }
    if report_relative_path not in {
        "reports/generation_report.json",
        "reports/preview_generation_report.json",
    }:
        raise ValueError("Unsupported generation report path.")
    report_path = project_dir / report_relative_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_standard_json(report_path, payload, "generation_report.schema.json")
    return payload
