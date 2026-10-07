"""Embedded V5 verification orchestration for generated projects.

The verifier consumes content-addressed snapshots of the original contract facts and
the final PDF.  It never accepts mutable generator-normalized fields as
verification truth.  Legacy projects without a contract snapshot still fall
back to parsing their archived workbook once during verification.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from .excel_contract import SUPPORTED_CONTRACT_EXTENSIONS
from .lineart import resolve_project_dir
from .manifest import utc_now, write_manifest
from .production_readiness import get_production_input_hashes
from ..verification_engine.service import VerificationEngineError, new_run_id, run_verification

WEAK_EVIDENCE_STATUSES = {"PAGE_FALLBACK", "VISUAL_CELL_CANDIDATE"}
PASS_STATUSES = {"OK", "PASS"}


class VerificationError(ValueError):
    pass


def issue(code: str, message: str, *, severity: str = "warning") -> dict[str, Any]:
    return {"severity": severity, "code": code, "message": message}


def verify_project(
    workspace_root: Path | str,
    project_id: str,
    *,
    template_mode: str = "kmt_style_v1",
    engine: Callable[..., dict[str, Any]] = run_verification,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    def _progress(phase: str, label: str, **kwargs: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(phase, label, **kwargs)
        except TypeError:
            progress_callback(phase, label)

    project_dir = resolve_project_dir(workspace_root, project_id)
    contract_path = find_project_contract(project_dir)
    pdf_path = project_dir / "generated" / "standard_mark.pdf"
    if contract_path is None:
        raise VerificationError("input/contract.xlsx or input/contract.xls not found; extract the contract first.")
    if not pdf_path.exists():
        raise VerificationError("generated/standard_mark.pdf not found; generate the standard PDF first.")

    warnings: list[dict[str, Any]] = []
    run_id = new_run_id()
    run_dir = project_dir / "reports" / "verification_runs" / run_id
    evidence_dir = _prepare_lineart_evidence(project_dir, run_id)
    contract_snapshot_path = project_dir / "data" / "contract.snapshot.json"
    try:
        _progress("reading_pdf", "读取正式 PDF 并取证")
        engine_options: dict[str, Any] = {
            "run_id": run_id,
            "lineart_evidence_dir": evidence_dir,
            "snapshot_inputs": True,
        }
        if contract_snapshot_path.is_file():
            engine_options["contract_snapshot_path"] = contract_snapshot_path
        payload = engine(contract_path, [pdf_path], run_dir, **engine_options)
        _progress("writing_report", "写入核对报告与交付状态")
        archived = _publish_latest_artifacts(project_dir, run_dir, warnings)
    except Exception as exc:
        if evidence_dir:
            shutil.rmtree(evidence_dir, ignore_errors=True)
        warnings.append(issue("verification_engine_failed", f"Embedded verification engine failed: {type(exc).__name__}"))
        verification = {
            "status": "unavailable",
            "engine": "embedded_v5",
            "verified_at": utc_now(),
            "error": "Embedded verification failed; see local service logs.",
        }
        manifest = write_manifest(
            project_dir,
            project_id,
            "generated_unverified",
            warnings,
            [],
            template_mode=template_mode,
            verification=verification,
            production_ready=False,
            needs_v5_check=True,
        )
        return {
            "project_id": project_id,
            "status": "generated_unverified",
            "verification": verification,
            "warnings": warnings,
            "manifest": manifest,
        }
    finally:
        if evidence_dir:
            shutil.rmtree(evidence_dir, ignore_errors=True)

    items = payload.get("items", [])
    status_counts = payload.get("status_counts", {}) if isinstance(payload.get("status_counts"), dict) else {}
    evidence_status_counts = (
        payload.get("evidence_status_counts", {})
        if isinstance(payload.get("evidence_status_counts"), dict)
        else {}
    )
    expected_items = _expected_item_count(project_dir)
    checker_summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    checker_contract_items = _non_negative_int(checker_summary.get("contract_items"))
    if expected_items == 0:
        expected_items = checker_contract_items
    reported_items = sum(_non_negative_int(value) for value in status_counts.values()) or len(items)
    passed = sum(_non_negative_int(status_counts.get(status)) for status in PASS_STATUSES)
    if not status_counts:
        passed = sum(1 for item in items if item.get("overall_status") in PASS_STATUSES)
    weak = sum(_non_negative_int(evidence_status_counts.get(status)) for status in WEAK_EVIDENCE_STATUSES)
    if not evidence_status_counts:
        weak = sum(1 for item in items if item.get("image_status") in WEAK_EVIDENCE_STATUSES)
        weak += sum(_non_negative_int(status_counts.get(status)) for status in WEAK_EVIDENCE_STATUSES)

    artifacts_complete = all(archived.values())
    coverage_complete = (
        expected_items > 0
        and checker_contract_items == expected_items
        and reported_items == expected_items
    )
    all_passed = coverage_complete and passed == expected_items and weak == 0 and artifacts_complete
    if weak:
        warnings.append(
            issue("v5_weak_evidence", f"{weak} item(s) have weak evidence (PAGE_FALLBACK/VISUAL_CELL_CANDIDATE); not counted as pass.")
        )
    if not all_passed:
        warnings.append(
            issue("v5_items_not_passed", f"{max(expected_items - passed, 0)} of {expected_items} expected item(s) did not PASS verification.")
        )
    if not coverage_complete:
        warnings.append(
            issue(
                "v5_coverage_incomplete",
                "Verification coverage mismatch: "
                f"generator expected {expected_items}, engine contract count {checker_contract_items}, "
                f"engine result count {reported_items}.",
            )
        )
    if not artifacts_complete:
        warnings.append(issue("v5_evidence_archive_incomplete", "Verification evidence artifacts were not completely archived; production readiness is blocked."))

    run_manifest = payload.get("run_manifest", {})
    input_hashes = {
        "contract": run_manifest.get("inputs", {}).get("contract", {}).get("sha256"),
        "marks": [item.get("sha256") for item in run_manifest.get("inputs", {}).get("marks", [])],
    }
    snapshot_record = run_manifest.get("inputs", {}).get("contract_snapshot")
    if isinstance(snapshot_record, dict) and snapshot_record.get("sha256"):
        input_hashes["contract_snapshot"] = snapshot_record["sha256"]
    input_hashes.update(get_production_input_hashes(project_dir))
    verification = {
        "status": "verified_pass" if all_passed else "verified_failed",
        "engine": "embedded_v5",
        "rule_version": "v5",
        "run_id": payload["run_id"],
        "run_path": f"reports/verification_runs/{payload['run_id']}",
        "verified_at": utc_now(),
        "contract_facts_source": checker_summary.get("contract_facts_source", "workbook_parse"),
        "input_hashes": input_hashes,
        "total_items": expected_items,
        "reported_items": reported_items,
        "checker_contract_items": checker_contract_items,
        "coverage_complete": coverage_complete,
        "passed_items": passed,
        "weak_evidence_items": weak,
        "evidence_status_counts": evidence_status_counts,
        "evidence_archive_complete": artifacts_complete,
        "status_counts": status_counts,
        "events": payload.get("events", []),
        "reports": archived,
    }
    manifest_status = "verified_pass" if all_passed else "verified_failed"
    manifest = write_manifest(
        project_dir,
        project_id,
        manifest_status,
        warnings,
        [],
        template_mode=template_mode,
        verification=verification,
        production_ready=all_passed,
        needs_v5_check=False,
    )
    return {
        "project_id": project_id,
        "status": manifest_status,
        "verification": verification,
        "warnings": warnings,
        "manifest": manifest,
    }


def find_project_contract(project_dir: Path) -> Path | None:
    report_path = project_dir / "reports" / "extraction_report.json"
    if report_path.exists():
        try:
            source_contract = json.loads(report_path.read_text(encoding="utf-8")).get("source_contract")
        except (OSError, json.JSONDecodeError):
            source_contract = None
        if isinstance(source_contract, str):
            candidate = (project_dir / source_contract).resolve()
            try:
                candidate.relative_to(project_dir.resolve())
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_file() and candidate.suffix.lower() in SUPPORTED_CONTRACT_EXTENSIONS:
                return candidate
    for suffix in (".xlsx", ".xls"):
        candidate = project_dir / "input" / f"contract{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _prepare_lineart_evidence(project_dir: Path, run_id: str) -> Path | None:
    normalized_path = project_dir / "data" / "items.normalized.json"
    if not normalized_path.exists():
        return None
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    evidence_dir = project_dir / "reports" / ".verification_staging" / run_id
    copied = 0
    for item in normalized.get("items", []):
        item_no = str(item.get("item_no") or "").strip()
        lineart_rel = item.get("lineart_asset") if item.get("lineart_status") == "approved" else None
        # Prefer selected product-region cutout/crop for fuzzy shape compare;
        # full sales-page backgrounds dilute the silhouette signal.
        region = item.get("selected_product_region") if isinstance(item.get("selected_product_region"), dict) else {}
        source_rel = (
            region.get("cutout_path")
            or region.get("image_path")
            or item.get("source_image")
        )
        for role, relative in (("source", source_rel), ("lineart", lineart_rel)):
            if not item_no or not relative:
                continue
            path = (project_dir / str(relative)).resolve()
            try:
                path.relative_to(project_dir.resolve())
            except ValueError:
                continue
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                continue
            evidence_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, evidence_dir / f"{item_no}__{role}{path.suffix.lower()}")
            copied += 1
    return evidence_dir if copied else None


def _publish_latest_artifacts(
    project_dir: Path, run_dir: Path, warnings: list[dict[str, Any]]
) -> dict[str, str | None]:
    reports_dir = project_dir / "reports"
    artifacts = run_dir / "artifacts"
    archived: dict[str, str | None] = {"verification_report": None, "verification_summary_csv": None}
    targets = [
        ("v5_result.json", "verification_report.json", "verification_report"),
        ("v5_item_summary.csv", "verification_summary.csv", "verification_summary_csv"),
    ]
    for source_name, target_name, key in targets:
        source = artifacts / source_name
        if not source.is_file():
            warnings.append(issue("v5_artifact_missing", f"Embedded verification did not provide {source_name}."))
            continue
        target = reports_dir / target_name
        shutil.copy2(source, target)
        archived[key] = f"reports/{target_name}"
    return archived


def _expected_item_count(project_dir: Path) -> int:
    normalized_path = project_dir / "data" / "items.normalized.json"
    if not normalized_path.is_file():
        return 0
    try:
        payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    items = payload.get("items", [])
    return len(items) if isinstance(items, list) else 0


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
