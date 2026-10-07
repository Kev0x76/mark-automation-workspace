from __future__ import annotations

import threading

import json
import hashlib
import re
import shutil
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .lineart_engine import generate_lineart_bytes
from .lineart_target_qc import TargetSizeQcError, evaluate_target_size
from .lineart_provider import (
    MAX_LINEART_REFINEMENT_CHARS,
    LineartProviderError,
    get_lineart_provider,
)
from .product_regions import ProductRegionError, get_selected_region_path, selected_region_metadata
from .schema_validation import validate_standard_payload, write_standard_json


ALLOWED_LINEART_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
APPROVABLE_PROVIDERS = {
    "opencv_v2",
    "opencv_v3",
    "openai_image",
    "grok_image",
    "manual_candidate",
    "manual_upload",
}
HUMAN_REVIEW_ONLY_FAILURES = {
    "photo_edge_route_requires_review",
    "legacy_foreground_evidence_mask_requires_review",
}
TARGET_SIZE_HUMAN_REVIEW_FAILURES = {
    "target_combined_structure_does_not_survive",
}

_PROJECT_WRITE_LOCKS: dict[str, threading.RLock] = {}
_PROJECT_WRITE_LOCKS_GUARD = threading.Lock()


def project_write_lock(project_dir: Path) -> threading.RLock:
    """Serialize normalized/lineart file mutations for one project."""

    key = str(project_dir.resolve())
    with _PROJECT_WRITE_LOCKS_GUARD:
        lock = _PROJECT_WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROJECT_WRITE_LOCKS[key] = lock
        return lock



class LineartError(ValueError):
    pass


PROJECT_LINEART_PROMPT_PATH = Path("data/lineart_prompt.json")
PROJECT_LINEART_PROMPT_SCHEMA = "lineart_prompt.schema.json"


def normalize_lineart_instruction(value: str) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) > MAX_LINEART_REFINEMENT_CHARS:
        raise LineartError(
            f"Lineart prompt instruction exceeds {MAX_LINEART_REFINEMENT_CHARS} characters."
        )
    return normalized


def combine_lineart_instructions(project_instruction: str, item_instruction: str) -> str:
    """Keep project and per-item intent visible inside the provider's fixed budget."""

    project_text = normalize_lineart_instruction(project_instruction)
    item_text = normalize_lineart_instruction(item_instruction)
    if not project_text:
        return item_text
    if not item_text:
        return project_text

    prefix_project = "Project: "
    prefix_item = " Item: "
    available = MAX_LINEART_REFINEMENT_CHARS - len(prefix_project) - len(prefix_item)
    project_budget = available // 2
    item_budget = available - project_budget
    return (
        f"{prefix_project}{project_text[:project_budget]}"
        f"{prefix_item}{item_text[:item_budget]}"
    )


def get_project_lineart_prompt(
    workspace_root: Path | str,
    project_id: str,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    prompt_path = project_dir / PROJECT_LINEART_PROMPT_PATH
    if not prompt_path.exists():
        return {
            "schema_version": "1.0.0",
            "project_id": project_id,
            "prompt_instruction": "",
            "updated_at": None,
        }
    try:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        validate_standard_payload(payload, PROJECT_LINEART_PROMPT_SCHEMA)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise LineartError(f"Project lineart prompt is invalid: {exc}") from exc
    return payload


def save_project_lineart_prompt(
    workspace_root: Path | str,
    project_id: str,
    prompt_instruction: str,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    payload = {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "prompt_instruction": normalize_lineart_instruction(prompt_instruction),
        "updated_at": utc_now(),
    }
    write_standard_json(
        project_dir / PROJECT_LINEART_PROMPT_PATH,
        payload,
        PROJECT_LINEART_PROMPT_SCHEMA,
    )
    return payload


def initialize_lineart_workspace(project_dir: Path | str) -> dict[str, Any]:
    project_dir = Path(project_dir)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    return write_lineart_report(project_dir, normalized, action="initialize")


def get_lineart_state(workspace_root: Path | str, project_id: str) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    items = build_item_states(project_dir, normalized)
    return {
        "project_id": normalized["project_id"],
        "items": items,
        "summary": build_summary(normalized, items),
        "report_path": "reports/lineart_report.json",
    }


def save_lineart_candidate(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    filename: str,
    source: BinaryIO,
    *,
    action: str = "upload_candidate",
    provider: str | None = None,
    warnings: list[dict[str, Any]] | None = None,
    product_region: dict[str, Any] | None = None,
    algorithm_version: str | None = None,
    mode: str | None = None,
    quality: dict[str, Any] | None = None,
    diagnostics: dict[str, bytes] | None = None,
    model: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    try:
        product_region = product_region or selected_region_metadata(project_dir, item)
    except ProductRegionError as exc:
        raise LineartError(str(exc)) from exc

    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_LINEART_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_LINEART_EXTENSIONS))
        raise LineartError(f"Unsupported lineart file type. Allowed: {allowed}")

    safe_item = safe_name(item_no)
    safe_stem = safe_name(Path(filename).stem or "candidate")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_dir = project_dir / "linearts" / "candidates" / safe_item
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = unique_path(candidate_dir / f"{timestamp}-{safe_stem}.png")

    raw = source.read()
    effective_provider = provider or "manual_candidate"
    process_warnings: list[dict[str, Any]] = []
    if quality is not None:
        processed = raw
    elif effective_provider != "mock":
        try:
            engine_result = generate_lineart_bytes(raw, mask_data=selected_mask_bytes(project_dir, product_region))
        except Exception as exc:
            raise LineartError(f"Could not decode or process the lineart candidate: {exc}") from exc
        processed = engine_result.data
        process_warnings = engine_result.warnings
        quality = engine_result.quality
        algorithm_version = engine_result.algorithm_version
        mode = engine_result.mode
    else:
        from .image_postprocess import postprocess_lineart_bytes

        processed, process_warnings = postprocess_lineart_bytes(raw)
    if quality is None:
        quality = {
            "auto_qc_passed": False,
            "failure_reasons": ["legacy_mock_provider_requires_review"],
            "algorithm_version": algorithm_version,
        }
    diagnostics = dict(diagnostics or {})
    quality = dict(quality)
    if diagnostics and str(algorithm_version or "").startswith("opencv-lineart-v3"):
        try:
            target_size_qc, target_previews = evaluate_target_size(
                processed,
                diagnostics,
            )
            diagnostics.update(target_previews)
            quality["target_size_qc"] = target_size_qc
            target_failures = list(target_size_qc.get("failure_reasons") or [])
            if target_failures:
                quality["failure_reasons"] = list(
                    dict.fromkeys(list(quality.get("failure_reasons") or []) + target_failures)
                )
                quality["auto_qc_passed"] = False
                quality["technical_qc_passed"] = False
        except TargetSizeQcError as exc:
            quality["failure_reasons"] = list(
                dict.fromkeys(
                    list(quality.get("failure_reasons") or [])
                    + ["target_size_qc_unavailable"]
                )
            )
            quality["auto_qc_passed"] = False
            quality["technical_qc_passed"] = False
            quality["target_size_qc"] = {"error": str(exc), "technical_passed": False}
    quality = enrich_quality(quality, product_region, mode)
    candidate_path.write_bytes(processed)
    diagnostics_manifest = write_candidate_diagnostics(
        project_dir,
        candidate_path,
        diagnostics,
    )
    warnings = (warnings or []) + process_warnings

    if item["lineart_status"] != "approved":
        item["lineart_status"] = "pending"
        item["lineart_asset"] = None
        item["lineart_qc_status"] = (
            "needs_review"
            if quality.get("visual_qc_required") or not quality.get("auto_qc_passed")
            else "passed"
        )
    save_normalized(project_dir, normalized)
    metadata = write_candidate_metadata(
        project_dir=project_dir,
        item_no=item_no,
        provider=effective_provider,
        source_image=item.get("source_image"),
        product_region=product_region,
        candidate_path=project_relative(project_dir, candidate_path),
        model=model,
        prompt=prompt,
        warnings=warnings,
        algorithm_version=algorithm_version,
        mode=mode,
        quality=quality,
        diagnostics=diagnostics_manifest,
    )
    report = write_lineart_report(
        project_dir,
        normalized,
        action=action,
        item_no=item_no,
        candidate_path=project_relative(project_dir, candidate_path),
        provider=provider,
        warnings=warnings,
    )
    return {
        "project_id": normalized["project_id"],
        "item_no": item_no,
        "candidate": file_state(project_dir, candidate_path),
        "item": item,
        "summary": report["summary"],
        "algorithm_version": algorithm_version,
        "mode": mode,
        "quality": quality,
        "metadata_path": metadata["metadata_path"],
        "diagnostics": diagnostics_manifest,
    }


def upload_lineart(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    filename: str,
    source: BinaryIO,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    try:
        product_region = selected_region_metadata(project_dir, item)
    except ProductRegionError as exc:
        raise LineartError(str(exc)) from exc
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_LINEART_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_LINEART_EXTENSIONS))
        raise LineartError(f"Unsupported lineart file type. Allowed: {allowed}")

    try:
        engine_result = generate_lineart_bytes(
            source.read(), mask_data=selected_mask_bytes(project_dir, product_region)
        )
    except Exception as exc:
        raise LineartError(f"Could not decode or process the uploaded lineart: {exc}") from exc
    processed = engine_result.data
    warnings = engine_result.warnings
    quality = enrich_quality(engine_result.quality, product_region, engine_result.mode)
    upload_dir = project_dir / "linearts" / "uploaded" / safe_name(item_no)
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uploaded_path = unique_path(upload_dir / f"{timestamp}-{safe_name(Path(filename).stem or 'uploaded')}.png")
    uploaded_path.write_bytes(processed)
    if item["lineart_status"] != "approved":
        item["lineart_status"] = "uploaded"
        item["lineart_asset"] = None
        item["lineart_qc_status"] = (
            "needs_review"
            if quality.get("visual_qc_required") or not quality.get("auto_qc_passed")
            else "passed"
        )
    save_normalized(project_dir, normalized)
    report = write_lineart_report(
        project_dir,
        normalized,
        action="upload_lineart",
        item_no=item_no,
        candidate_path=project_relative(project_dir, uploaded_path),
        provider="manual_upload",
        warnings=warnings,
    )
    metadata = write_candidate_metadata(
        project_dir=project_dir,
        item_no=item_no,
        provider="manual_upload",
        source_image=item.get("source_image"),
        product_region=product_region,
        candidate_path=project_relative(project_dir, uploaded_path),
        model=None,
        prompt=None,
        warnings=warnings,
        algorithm_version=engine_result.algorithm_version,
        mode=engine_result.mode,
        quality=quality,
    )
    result = {
        "project_id": normalized["project_id"],
        "item_no": item_no,
        "uploaded": file_state(project_dir, uploaded_path),
        "item": item,
        "summary": report["summary"],
        "algorithm_version": engine_result.algorithm_version,
        "mode": engine_result.mode,
        "quality": quality,
        "metadata_path": metadata["metadata_path"],
    }
    finalized = approve_lineart_candidate(
        workspace_root,
        project_id,
        item_no,
        project_relative(project_dir, uploaded_path),
        approval_mode="automatic_generation",
    )
    result["auto_finalized"] = True
    result["auto_used"] = True
    result["approved"] = finalized["approved"]
    result["item"] = finalized["item"]
    result["approval"] = finalized["approval"]
    return result


def generate_lineart_candidate(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    provider_name: str = "openai_image",
    prompt_instruction: str = "",
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    provider = get_lineart_provider(provider_name)

    try:
        region_path = get_selected_region_path(project_dir, item)
        region_metadata = selected_region_metadata(project_dir, item)
        mask_path = selected_mask_path(project_dir, region_metadata)
        provider_item = dict(item)
        project_instruction = get_project_lineart_prompt(
            workspace_root,
            project_id,
        ).get("prompt_instruction", "")
        normalized_instruction = combine_lineart_instructions(
            str(project_instruction or ""),
            prompt_instruction,
        )
        if normalized_instruction:
            provider_item["lineart_prompt_addition"] = normalized_instruction
        generated = provider.generate(
            project_dir=project_dir,
            item=provider_item,
            region_path=region_path,
            mask_path=mask_path,
        )
    except (LineartProviderError, ProductRegionError) as exc:
        raise LineartError(str(exc)) from exc

    # Provider I/O may run concurrently across SKUs; project file writes do not.
    with project_write_lock(project_dir):
        result = save_lineart_candidate(
            workspace_root,
            project_id,
            item_no,
            generated.filename,
            BytesIO(generated.data),
            action="generate_candidate",
            provider=provider.name,
            warnings=generated.warnings,
            product_region=region_metadata,
            algorithm_version=generated.algorithm_version,
            mode=generated.mode,
            quality=generated.quality,
            diagnostics=generated.diagnostics,
            model=generated.model,
            prompt=generated.prompt,
        )
        result["provider"] = provider.name
        result["model"] = generated.model
        quality = result.get("quality") or {}
        if provider.name in APPROVABLE_PROVIDERS:
            # Quality diagnostics remain in provenance as warnings, but the
            # project owner explicitly removed lineart quality/size gating. File
            # decode, item binding and content hashes remain integrity checks.
            finalized = approve_lineart_candidate(
                workspace_root,
                project_id,
                item_no,
                str(result["candidate"]["path"]),
                approval_mode="automatic_generation",
            )
            result["auto_finalized"] = True
            result["auto_used"] = True
            result["approved"] = finalized["approved"]
            result["item"] = finalized["item"]
        else:
            result["auto_finalized"] = False
            result["auto_used"] = False
            result["auto_use_blockers"] = list(quality.get("automation_blockers") or []) or list(
                quality.get("failure_reasons") or []
            ) or [
                "automatic_quality_checks_failed"
            ]
    return result



def resolve_lineart_image_for_refine(project_dir: Path, item: dict[str, Any]) -> Path:
    """Prefer the current approved lineart file as the refine source image."""

    asset = item.get("lineart_asset") or (item.get("approved") or {}).get("path")
    if not asset:
        raise LineartError("没有可用的当前线稿，无法二次修改生成。")
    path = resolve_project_file(project_dir, str(asset))
    if not path.is_file():
        raise LineartError("当前线稿文件缺失，无法二次修改生成。")
    return path


def refine_lineart_from_existing(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    *,
    prompt_instruction: str,
    provider_name: str = "openai_image",
) -> dict[str, Any]:
    """Regenerate lineart from the current lineart image plus a one-shot prompt.

    Uses the existing approved lineart as the model image input (not the product
    region).  On success the new asset is auto-adopted as the item-bound
    approved lineart, replacing any category-share binding.
    """

    instruction = normalize_lineart_instruction(prompt_instruction)
    if not instruction:
        raise LineartError("二次修改必须填写提示词。")

    project_dir = resolve_project_dir(workspace_root, project_id)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    provider = get_lineart_provider(provider_name)

    try:
        base_path = resolve_lineart_image_for_refine(project_dir, item)
        # Region metadata remains the production binding for provenance, but the
        # generative edit consumes the current lineart image.
        if item.get("product_region_status") == "selected" and item.get("selected_product_region"):
            region_metadata = selected_region_metadata(project_dir, item)
            mask_path = selected_mask_path(project_dir, region_metadata)
        else:
            region_metadata = None
            mask_path = None
        provider_item = dict(item)
        provider_item["lineart_prompt_override"] = (
            "Edit this existing black-and-white product line art. Keep product structure, "
            "silhouette, proportions and piece count. Do not redesign. Apply only this change: "
            f"{instruction}. Pure white background, black strokes only; no shading or photo texture."
        )
        generated = provider.generate(
            project_dir=project_dir,
            item=provider_item,
            region_path=base_path,
            mask_path=mask_path,
        )
    except (LineartProviderError, ProductRegionError) as exc:
        raise LineartError(str(exc)) from exc

    with project_write_lock(project_dir):
        # Re-load after network call.
        normalized = load_normalized(project_dir)
        item = find_item(normalized, item_no)
        result = save_lineart_candidate(
            workspace_root,
            project_id,
            item_no,
            generated.filename.replace("-lineart", "-refine-lineart"),
            BytesIO(generated.data),
            action="refine_from_lineart",
            provider=provider.name,
            warnings=list(generated.warnings or []) + [
                {"code": "refined_from_existing_lineart", "message": f"Refined from {project_relative(project_dir, base_path)}"}
            ],
            product_region=region_metadata or item.get("selected_product_region"),
            algorithm_version=generated.algorithm_version,
            mode=f"refine+{generated.mode}",
            quality=generated.quality,
            diagnostics={
                **(generated.diagnostics or {}),
                # diagnostics 的值按契约是 bytes（write_candidate_diagnostics 逐项 write_bytes），
                # 文本条目必须显式 encode，否则 refine 直接 TypeError（2026-09-19 实测）。
                "refine_from": project_relative(project_dir, base_path).encode("utf-8"),
                "refine_instruction": instruction.encode("utf-8"),
            },
            model=generated.model,
            prompt=generated.prompt,
        )
        result["provider"] = provider.name
        result["model"] = generated.model
        result["refine_from"] = project_relative(project_dir, base_path)
        result["refine_instruction"] = instruction
        if provider.name not in APPROVABLE_PROVIDERS:
            result["auto_finalized"] = False
            result["auto_used"] = False
            return result
        finalized = approve_lineart_candidate(
            workspace_root,
            project_id,
            item_no,
            str(result["candidate"]["path"]),
            # approval_mode 只接受 automatic/automatic_generation/human；
            # refine 来源已由 action="refine_from_lineart" 与 mode="refine+…" 记录
            # （"automatic_refine" 不是合法值，会让 refine 永远 400，2026-09-19 实测）。
            approval_mode="automatic",
        )
        # Force item-bound provenance even if previous was category shared:
        # approve_lineart_candidate already rebuilds from product region / file.
        result["auto_finalized"] = True
        result["auto_used"] = True
        result["approved"] = finalized["approved"]
        result["item"] = finalized["item"]
        result["approval"] = finalized.get("approval")
    return result



def generate_lineart_candidates(
    workspace_root: Path | str,
    project_id: str,
    provider_name: str = "openai_image",
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    results = []
    warnings = []
    errors = []
    skipped = []
    for item in normalized.get("items", []):
        if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
            skipped.append({"item_no": item.get("item_no"), "reason": "already_approved"})
            continue
        if item.get("product_region_status") != "selected" or not item.get("selected_product_region"):
            skipped.append({"item_no": item.get("item_no"), "reason": "product_region_not_selected"})
            continue
        try:
            results.append(generate_lineart_candidate(workspace_root, project_id, item["item_no"], provider_name))
        except Exception as exc:
            issue = {
                "severity": "warning",
                "code": "lineart_candidate_failed",
                "message": str(exc),
                "item_no": item.get("item_no"),
            }
            warnings.append(issue)
            errors.append(issue)
    return {
        "project_id": project_id,
        "provider": provider_name,
        "generated": len(results),
        "failed": len(errors),
        "skipped": len(skipped),
        "skipped_items": skipped,
        "results": results,
        "warnings": warnings,
        "errors": errors,
    }


def approve_eligible_lineart_candidates(
    workspace_root: Path | str,
    project_id: str,
) -> dict[str, Any]:
    """Accept the newest candidate per SKU after integrity validation.

    This compatibility endpoint is retained for callers that still use a
    batch action. It does not filter by quality scores; every candidate is
    rechecked by ``approve_lineart_candidate`` for provider eligibility,
    immutable metadata, item/region binding and hashes.
    """

    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    approved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in normalized.get("items", []):
        item_no = str(item.get("item_no"))
        if item.get("lineart_status") == "approved":
            skipped.append({"item_no": item_no, "reason": "already_approved"})
            continue
        candidate = next(iter(list_candidates(project_dir, item_no)), None)
        if not candidate:
            skipped.append({"item_no": item_no, "reason": "no_candidate"})
            continue
        try:
            result = approve_lineart_candidate(
                workspace_root,
                project_id,
                item_no,
                str(candidate["path"]),
                approval_mode="automatic_generation",
            )
            approved.append(
                {
                    "item_no": item_no,
                    "candidate_path": candidate["path"],
                    "approved_path": result["approved"]["path"],
                }
            )
        except (LineartError, FileNotFoundError) as exc:
            errors.append({"item_no": item_no, "reason": str(exc)})
    return {
        "project_id": project_id,
        "approved": len(approved),
        "skipped": len(skipped),
        "failed": len(errors),
        "results": approved,
        "skipped_items": skipped,
        "errors": errors,
        "approval_mode": "batch_integrity_only_acceptance",
    }


def approve_lineart_candidate(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    candidate_path: str,
    *,
    approval_mode: str = "automatic",
    reviewer: str | None = None,
    reason: str | None = None,
    confirm_target_size_legible: bool = False,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    try:
        product_region = selected_region_metadata(project_dir, item)
    except ProductRegionError as exc:
        raise LineartError(str(exc)) from exc
    current_binding = build_input_binding(
        project_dir=project_dir,
        source_image=item.get("source_image"),
        product_region=product_region,
    )
    candidate = resolve_project_file(project_dir, candidate_path)
    candidates_root = (project_dir / "linearts" / "candidates" / safe_name(item_no)).resolve()
    uploaded_root = (project_dir / "linearts" / "uploaded" / safe_name(item_no)).resolve()
    if not (is_relative_to(candidate, candidates_root) or is_relative_to(candidate, uploaded_root)):
        raise LineartError("Approved file must belong to this item's candidates or uploaded directory.")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Lineart candidate not found: {candidate_path}")

    extension = candidate.suffix.lower()
    if extension not in ALLOWED_LINEART_EXTENSIONS:
        raise LineartError("Candidate has an unsupported lineart file type.")

    metadata = load_candidate_metadata(project_dir, candidate)
    if str(metadata.get("item_no")) != str(item_no):
        raise LineartError("Lineart candidate item binding mismatch.")
    if metadata.get("provider") not in APPROVABLE_PROVIDERS:
        raise LineartError("This provider only creates review candidates and cannot be approved.")
    expected_candidate_hash = metadata.get("lineart_sha256")
    actual_candidate_hash = sha256_file(candidate)
    if not expected_candidate_hash or expected_candidate_hash != actual_candidate_hash:
        raise LineartError("Lineart candidate hash mismatch; regenerate or upload it again.")
    candidate_binding = metadata.get("input_binding") or {}
    if not bindings_match(candidate_binding, current_binding):
        raise LineartError("Lineart candidate region binding is stale or mismatched.")
    quality = metadata.get("quality") or {}
    approval_mode = str(approval_mode or "automatic").strip().lower()
    if approval_mode not in {"automatic", "automatic_generation", "human"}:
        raise LineartError("approval_mode must be automatic, automatic_generation, or human.")
    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if approval_mode == "human":
        if not reviewer:
            raise LineartError("Human approval requires a reviewer identity.")
        if not reason:
            raise LineartError("Human approval requires a review reason.")
    # The project owner removed the lineart quality/target-size gate from every
    # approval mode. Provider eligibility, file readability, immutable metadata,
    # item/region binding and hashes were already checked above. Quality results
    # remain advisory warnings regardless of whether this compatibility endpoint
    # records an automatic or human approval mode.

    approved_path = unique_path(project_dir / "linearts" / "approved" / f"{safe_name(item_no)}{extension}")
    shutil.copy2(candidate, approved_path)
    approved_at = utc_now()
    approval_record = {
        "mode": approval_mode,
        "reviewer": reviewer or None,
        "reason": reason or None,
        "confirmed_target_size_legible": bool(confirm_target_size_legible),
        "auto_qc_passed": bool(quality.get("auto_qc_passed")),
        "quality_gate_applied": False,
        "quality_warnings": list(quality.get("failure_reasons") or []),
        "semantic_review": (
            "trusted_ai_generation_no_visual_oc"
            if approval_mode == "automatic_generation" and quality.get("generative_ai")
            else "not_required"
        ),
        "overridden_failure_reasons": [],
        "approved_at": approved_at,
    }
    if item.get("lineart_asset"):
        item["previous_lineart_asset"] = item["lineart_asset"]
    item["lineart_status"] = "approved"
    item["lineart_asset"] = project_relative(project_dir, approved_path)
    # The approval mode, not the candidate's technical-QC outcome, owns the
    # provenance state. A human may review a technically passing candidate
    # when automation is blocked by region-confidence or other explicit
    # policy rules; recording that as plain ``passed`` makes the standard-PDF
    # gate reject the same human approval immediately afterwards.
    item["lineart_qc_status"] = (
        "human_passed" if approval_mode == "human" else "accepted"
    )
    item["lineart_stale_reason"] = None
    item["lineart_provenance"] = {
        "item_no": item_no,
        "candidate_id": current_binding.get("region_candidate_id"),
        "provider": metadata.get("provider"),
        "algorithm_version": metadata.get("algorithm_version"),
        "mode": metadata.get("mode"),
        "candidate_path": project_relative(project_dir, candidate),
        "approved_path": project_relative(project_dir, approved_path),
        "candidate_sha256": actual_candidate_hash,
        "approved_sha256": sha256_file(approved_path),
        "region_artifact_id": current_binding["region_artifact_id"],
        "source_sha256": current_binding["source_sha256"],
        "cutout_sha256": current_binding["cutout_sha256"],
        "mask_sha256": current_binding["mask_sha256"],
        "quality": quality,
        "approval": approval_record,
        "approved_at": approved_at,
    }
    save_normalized(project_dir, normalized)
    report = write_lineart_report(
        project_dir,
        normalized,
        action=(
            "approve_human"
            if approval_mode == "human"
            else "auto_use_generated"
            if approval_mode == "automatic_generation"
            else "approve"
        ),
        item_no=item_no,
        candidate_path=project_relative(project_dir, candidate),
        approved_path=project_relative(project_dir, approved_path),
    )
    return {
        "project_id": normalized["project_id"],
        "item_no": item_no,
        "approved": file_state(project_dir, approved_path),
        "item": item,
        "summary": report["summary"],
        "approval": approval_record,
    }


def reject_lineart(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    candidate_path: str | None = None,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    ensure_lineart_dirs(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    if candidate_path:
        candidate = resolve_project_file(project_dir, candidate_path)
        candidates_root = (project_dir / "linearts" / "candidates").resolve()
        if not is_relative_to(candidate, candidates_root):
            raise LineartError("Rejected file must come from linearts/candidates.")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Lineart candidate not found: {candidate_path}")

    item["lineart_status"] = "rejected"
    item["lineart_asset"] = None
    item["lineart_qc_status"] = "needs_review"
    item["lineart_provenance"] = None
    save_normalized(project_dir, normalized)
    report = write_lineart_report(
        project_dir,
        normalized,
        action="reject",
        item_no=item_no,
        candidate_path=candidate_path,
    )
    return {
        "project_id": normalized["project_id"],
        "item_no": item_no,
        "item": item,
        "summary": report["summary"],
    }


def resolve_project_dir(workspace_root: Path | str, project_id: str) -> Path:
    if not project_id or "/" in project_id or "\\" in project_id:
        raise LineartError("Invalid project_id.")
    projects_root = (Path(workspace_root) / "projects").resolve()
    project_dir = (projects_root / project_id).resolve()
    if not is_relative_to(project_dir, projects_root):
        raise LineartError("Invalid project path.")
    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    return project_dir


def resolve_project_file(project_dir: Path, relative_path: str) -> Path:
    if not relative_path:
        raise LineartError("Project file path is required.")
    path = (project_dir / relative_path).resolve()
    if not is_relative_to(path, project_dir.resolve()):
        raise LineartError("Project file path escapes project directory.")
    return path


def ensure_lineart_dirs(project_dir: Path) -> None:
    for directory in [
        project_dir / "linearts" / "candidates",
        project_dir / "linearts" / "approved",
        project_dir / "linearts" / "uploaded",
        project_dir / "reports",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_normalized(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "data" / "items.normalized.json"
    if not path.exists():
        raise FileNotFoundError("data/items.normalized.json not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_normalized(project_dir: Path, payload: dict[str, Any]) -> None:
    path = project_dir / "data" / "items.normalized.json"
    write_standard_json(path, payload, "items.normalized.schema.json")


def find_item(normalized: dict[str, Any], item_no: str) -> dict[str, Any]:
    for item in normalized.get("items", []):
        if item.get("item_no") == item_no:
            return item
    raise LineartError(f"Item not found: {item_no}")


def write_lineart_report(
    project_dir: Path,
    normalized: dict[str, Any],
    *,
    action: str,
    item_no: str | None = None,
    candidate_path: str | None = None,
    approved_path: str | None = None,
    provider: str | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    items = build_item_states(project_dir, normalized)
    summary = build_summary(normalized, items)
    report = {
        "schema_version": "1.0.0",
        "project_id": normalized["project_id"],
        "updated_at": utc_now(),
        "last_action": {
            "action": action,
            "item_no": item_no,
            "candidate_path": candidate_path,
            "approved_path": approved_path,
            "provider": provider,
        },
        "summary": summary,
        "items": items,
        "warnings": warnings or [],
    }
    report_path = project_dir / "reports" / "lineart_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_item_states(project_dir: Path, normalized: dict[str, Any]) -> list[dict[str, Any]]:
    states = []
    for item in normalized.get("items", []):
        approved = None
        if item.get("lineart_asset"):
            approved_path = project_dir / item["lineart_asset"]
            if approved_path.exists():
                approved = file_state(project_dir, approved_path)
        states.append(
            {
                "item_no": item["item_no"],
                "cn_name": item.get("cn_name", ""),
                "es_name": item.get("es_name", ""),
                "source_image": item.get("source_image"),
                "source_image_status": item.get("source_image_status"),
                "product_region_status": item.get("product_region_status", "missing"),
                "selected_product_region": item.get("selected_product_region"),
                "lineart_status": item.get("lineart_status", "missing"),
                "lineart_asset": item.get("lineart_asset"),
                "lineart_qc_status": item.get("lineart_qc_status", "missing"),
                "lineart_stale_reason": item.get("lineart_stale_reason"),
                "lineart_provenance": item.get("lineart_provenance"),
                "approved": approved,
                "candidates": list_candidates(project_dir, item["item_no"]),
                "uploaded": list_uploaded(project_dir, item["item_no"]),
            }
        )
    return states


def build_summary(normalized: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, int]:
    status_counts = {status: 0 for status in ["missing", "pending", "uploaded", "approved", "rejected", "stale"]}
    selected_regions = 0
    region_blocked = 0
    for item in normalized.get("items", []):
        status = item.get("lineart_status", "missing")
        status_counts[status] = status_counts.get(status, 0) + 1
        if item.get("product_region_status") == "selected":
            selected_regions += 1
        if item.get("lineart_status") == "approved" and item.get("product_region_status") != "selected":
            region_blocked += 1
    return {
        "total_items": len(normalized.get("items", [])),
        "missing": status_counts.get("missing", 0),
        "pending": status_counts.get("pending", 0),
        "uploaded": status_counts.get("uploaded", 0),
        "approved": status_counts.get("approved", 0),
        "rejected": status_counts.get("rejected", 0),
        "stale": status_counts.get("stale", 0),
        "candidate_files": sum(len(item["candidates"]) for item in items),
        "selected_product_regions": selected_regions,
        "approved_linearts_blocked_by_product_region": region_blocked,
    }


def list_candidates(project_dir: Path, item_no: str) -> list[dict[str, Any]]:
    candidate_dir = project_dir / "linearts" / "candidates" / safe_name(item_no)
    if not candidate_dir.exists():
        return []
    candidates = [
        candidate_file_state(project_dir, path)
        for path in candidate_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_LINEART_EXTENSIONS
    ]
    return sorted(candidates, key=lambda item: item["updated_at"], reverse=True)


def list_uploaded(project_dir: Path, item_no: str) -> list[dict[str, Any]]:
    uploaded_dir = project_dir / "linearts" / "uploaded" / safe_name(item_no)
    if not uploaded_dir.exists():
        return []
    uploaded = [
        candidate_file_state(project_dir, path)
        for path in uploaded_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_LINEART_EXTENSIONS
    ]
    return sorted(uploaded, key=lambda item: item["updated_at"], reverse=True)


def candidate_file_state(project_dir: Path, path: Path) -> dict[str, Any]:
    state = file_state(project_dir, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            quality = metadata.get("quality") or {}
            state.update(
                {
                    "provider": metadata.get("provider"),
                    "mode": metadata.get("mode"),
                    "algorithm_version": metadata.get("algorithm_version"),
                    "qc_passed": bool(quality.get("auto_qc_passed")),
                    "automation_eligible": bool(quality.get("automation_eligible")),
                    "human_review_eligible": not human_review_blockers(quality),
                    "human_review_blockers": human_review_blockers(quality),
                    "failure_reasons": quality.get("failure_reasons", []),
                    "metadata_path": project_relative(project_dir, sidecar),
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            state["metadata_error"] = "candidate metadata could not be read"
    return state


def human_review_blockers(
    quality: dict[str, Any],
    *,
    confirm_target_size_legible: bool = False,
) -> list[str]:
    """Return failures that a visual reviewer is not allowed to override.

    Human review is deliberately narrow: it can currently accept the otherwise
    valid photo-edge route, but cannot waive blank output, missing source
    evidence, mask violations, or other structural QC failures.
    """

    if quality.get("auto_qc_passed"):
        return []
    failures = [str(value) for value in quality.get("failure_reasons") or [] if value]
    if not failures:
        return ["automatic_quality_checks_failed_without_reason"]
    reviewable = set(HUMAN_REVIEW_ONLY_FAILURES)
    if confirm_target_size_legible:
        reviewable.update(TARGET_SIZE_HUMAN_REVIEW_FAILURES)
    return [value for value in failures if value not in reviewable]


def resolve_review_lineart_candidate(project_dir: Path, item: dict[str, Any]) -> Path | None:
    """Return the newest immutable candidate bound to the selected region.

    This resolver is for non-production previews only.  It never falls back to
    the source sales image and never changes approval state.
    """

    if item.get("product_region_status") != "selected" or not item.get("selected_product_region"):
        return None
    try:
        current_binding = build_input_binding(
            project_dir=project_dir,
            source_image=item.get("source_image"),
            product_region=item.get("selected_product_region"),
        )
    except LineartError:
        return None
    assets = list_candidates(project_dir, str(item.get("item_no"))) + list_uploaded(
        project_dir, str(item.get("item_no"))
    )
    assets.sort(key=lambda value: str(value.get("updated_at") or ""), reverse=True)
    for asset in assets:
        try:
            candidate = resolve_project_file(project_dir, str(asset.get("path") or ""))
            metadata = load_candidate_metadata(project_dir, candidate)
            if str(metadata.get("item_no")) != str(item.get("item_no")):
                continue
            if sha256_file(candidate) != metadata.get("lineart_sha256"):
                continue
            if not bindings_match(metadata.get("input_binding") or {}, current_binding):
                continue
            return candidate
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError, LineartError):
            continue
    return None


def write_candidate_metadata(
    *,
    project_dir: Path,
    item_no: str,
    provider: str,
    source_image: str | None,
    product_region: dict[str, Any] | None = None,
    candidate_path: str,
    model: str | None,
    prompt: str | None,
    warnings: list[dict[str, Any]],
    algorithm_version: str | None = None,
    mode: str | None = None,
    quality: dict[str, Any] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata_path = project_dir / "linearts" / "candidates" / f"{safe_name(item_no)}.json"
    candidate = resolve_project_file(project_dir, candidate_path)
    input_binding = build_input_binding(
        project_dir=project_dir,
        source_image=source_image,
        product_region=product_region,
    )
    metadata = {
        "schema_version": "2.0.0",
        "item_no": item_no,
        "provider": provider,
        "source_image": source_image,
        "product_region": product_region,
        "input_binding": input_binding,
        "candidate_path": candidate_path,
        "lineart_sha256": sha256_file(candidate),
        "model": model,
        "prompt": prompt,
        "algorithm_version": algorithm_version,
        "mode": mode,
        "quality": quality or {},
        "diagnostics": diagnostics or [],
        "created_at": utc_now(),
        "warnings": warnings,
    }
    sidecar_path = candidate.with_suffix(candidate.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **metadata,
        "metadata_path": project_relative(project_dir, sidecar_path),
    }


def write_candidate_diagnostics(
    project_dir: Path,
    candidate_path: Path,
    diagnostics: dict[str, bytes],
) -> list[dict[str, Any]]:
    if not diagnostics:
        return []
    directory = candidate_path.parent / f"{candidate_path.stem}-diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for name, data in sorted(diagnostics.items()):
        path = directory / f"{safe_name(name)}.png"
        path.write_bytes(data)
        manifest.append(
            {
                "name": name,
                "path": project_relative(project_dir, path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return manifest


def selected_mask_path(project_dir: Path, product_region: dict[str, Any]) -> Path | None:
    relative_path = product_region.get("mask_path")
    if not relative_path:
        return None
    path = resolve_project_file(project_dir, str(relative_path))
    if not path.exists() or not path.is_file():
        raise LineartError("Selected product-region mask is missing.")
    return path


def selected_mask_bytes(project_dir: Path, product_region: dict[str, Any]) -> bytes | None:
    path = selected_mask_path(project_dir, product_region)
    return path.read_bytes() if path else None


def enrich_quality(
    quality: dict[str, Any], product_region: dict[str, Any], mode: str | None
) -> dict[str, Any]:
    result = dict(quality)
    blockers: list[str] = []
    if not result.get("auto_qc_passed"):
        blockers.append("automatic_quality_checks_failed")
    if result.get("generative_ai"):
        # The user-approved production policy accepts the configured AI image
        # provider after integrity validation. Keep quality diagnostics in
        # provenance, but do not create a visual-review blocker or call a
        # second vision model from the checker.
        result["visual_qc_required"] = False
        result["visual_review_policy"] = "trusted_ai_generation_no_visual_oc"
    if str(result.get("algorithm_version") or "").startswith("opencv-lineart-v3"):
        blockers.append("opencv_v3_requires_explicit_human_approval")
    if mode != "existing_lineart":
        blockers.append("input_is_not_validated_existing_lineart")
    # A rectangular crop recorded with reviewer/reason is a stronger input
    # assertion than a heuristic proposal and may proceed to the same
    # deterministic output QC. Legacy manual candidate selection has no such
    # audit payload and remains ineligible for automatic finalization.
    if product_region.get("selected_by") not in {"automatic", "manual_crop"}:
        blockers.append("product_region_was_not_automatically_selected")
    minimum_region_score = 0.78 if result.get("generative_ai") else 0.86
    if (
        product_region.get("confidence") != "high"
        or float(product_region.get("score") or 0.0) < minimum_region_score
    ):
        blockers.append("product_region_confidence_is_not_high")
    result["automation_eligible"] = not blockers
    result["automation_blockers"] = blockers
    return result


def build_input_binding(
    *,
    project_dir: Path,
    source_image: str | None,
    product_region: dict[str, Any] | None,
) -> dict[str, Any]:
    region = product_region or {}
    required = ["region_artifact_id", "source_sha256", "cutout_sha256", "mask_sha256"]
    missing = [key for key in required if not region.get(key)]
    if missing:
        raise LineartError(
            "Selected product region has no complete immutable lineage; regenerate product-region candidates."
        )

    checks = [
        (source_image, "source_sha256", "source image"),
        (region.get("cutout_path"), "cutout_sha256", "selected cutout"),
        (region.get("mask_path"), "mask_sha256", "selected mask"),
    ]
    for relative_path, hash_key, label in checks:
        if not relative_path:
            raise LineartError(f"Product-region binding is missing its {label} path.")
        path = resolve_project_file(project_dir, str(relative_path))
        if not path.exists() or not path.is_file():
            raise LineartError(f"Product-region bound {label} is missing.")
        if sha256_file(path) != region.get(hash_key):
            raise LineartError(f"Product-region {label} hash mismatch; regenerate candidates.")

    text_mask_path = region.get("text_mask_path")
    text_mask_sha256 = region.get("text_mask_sha256")
    if text_mask_path:
        path = resolve_project_file(project_dir, str(text_mask_path))
        if not path.exists() or not path.is_file():
            raise LineartError("Product-region bound OCR text mask is missing.")
        actual_text_mask_sha256 = sha256_file(path)
        if text_mask_sha256 and actual_text_mask_sha256 != text_mask_sha256:
            raise LineartError("Product-region OCR text mask hash mismatch; regenerate candidates.")
        text_mask_sha256 = actual_text_mask_sha256

    return {
        "region_candidate_id": region.get("candidate_id"),
        "region_artifact_id": region.get("region_artifact_id"),
        "selected_by": region.get("selected_by"),
        "confidence": region.get("confidence"),
        "score": region.get("score"),
        "mask_semantics": region.get("mask_semantics", "foreground_evidence"),
        "source_path": source_image,
        "source_sha256": region.get("source_sha256"),
        "cutout_path": region.get("cutout_path"),
        "cutout_sha256": region.get("cutout_sha256"),
        "mask_path": region.get("mask_path"),
        "mask_sha256": region.get("mask_sha256"),
        "text_mask_path": text_mask_path,
        "text_mask_sha256": text_mask_sha256,
        "ocr_status": region.get("ocr_status"),
    }


def load_candidate_metadata(project_dir: Path, candidate: Path) -> dict[str, Any]:
    sidecar = candidate.with_suffix(candidate.suffix + ".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    # Compatibility lookup is deliberately accepted only when the summary
    # points to this exact candidate; it never guesses from an item number.
    for summary in (project_dir / "linearts" / "candidates").glob("*.json"):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("candidate_path") == project_relative(project_dir, candidate):
            return payload
    raise LineartError("Lineart candidate has no immutable metadata sidecar.")


def bindings_match(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = [
        "region_candidate_id",
        "region_artifact_id",
        "source_path",
        "source_sha256",
        "cutout_path",
        "cutout_sha256",
        "mask_path",
        "mask_sha256",
    ]
    if not all(candidate.get(key) and candidate.get(key) == current.get(key) for key in keys):
        return False
    optional_keys = ["text_mask_path", "text_mask_sha256", "ocr_status"]
    return all(
        candidate.get(key) == current.get(key)
        for key in optional_keys
        if candidate.get(key) is not None or current.get(key) is not None
    )


def validate_lineart_provenance(project_dir: Path, item: dict[str, Any]) -> dict[str, str] | None:
    """Return a stable blocking issue for a standard PDF, or ``None``."""

    if item.get("lineart_status") == "stale":
        return {"code": "stale_lineart", "message": "Lineart became stale after the product region changed."}
    if item.get("lineart_status") != "approved":
        return None
    provenance = item.get("lineart_provenance") or {}
    if not provenance:
        return {"code": "lineart_binding_mismatch", "message": "Approved lineart has no verifiable provenance."}
    if provenance.get("binding_scope") == "category_group":
        # Local import avoids making the core single-SKU lineart service depend
        # on the optional group workflow during module initialization.
        from .lineart_sharing import validate_group_lineart_provenance

        return validate_group_lineart_provenance(project_dir, item)
    if provenance.get("binding_scope") == "manual_item_without_product_region":
        from .manual_lineart import validate_manual_no_region_provenance

        return validate_manual_no_region_provenance(project_dir, item)
    quality = provenance.get("quality") or {}
    approval = provenance.get("approval") or {"mode": "automatic"}
    approval_mode = approval.get("mode") or "automatic"
    if approval_mode in {"automatic_generation", "automatic"}:
        accepted_now = item.get("lineart_qc_status") == "accepted"
        accepted_legacy = (
            item.get("lineart_qc_status") == "passed"
            and quality.get("auto_qc_passed")
            and quality.get("automation_eligible")
        )
        if not (accepted_now or accepted_legacy):
            return {
                "code": "lineart_acceptance_missing",
                "message": "Approved lineart has no integrity-only acceptance record.",
            }
    elif approval_mode == "human":
        if item.get("lineart_qc_status") != "human_passed":
            return {"code": "lineart_qc_not_passed", "message": "Human-approved lineart has no human-passed QC state."}
        if not str(approval.get("reviewer") or "").strip() or not str(approval.get("reason") or "").strip():
            return {"code": "lineart_qc_not_passed", "message": "Human approval has incomplete reviewer evidence."}
    else:
        return {"code": "lineart_qc_not_passed", "message": "Approved lineart has an unknown approval mode."}
    region = item.get("selected_product_region") or {}
    if provenance.get("region_artifact_id") != region.get("region_artifact_id"):
        return {"code": "lineart_binding_mismatch", "message": "Approved lineart belongs to a different product region."}
    try:
        current_binding = build_input_binding(
            project_dir=project_dir,
            source_image=item.get("source_image"),
            product_region=region,
        )
    except LineartError as exc:
        return {"code": "lineart_binding_mismatch", "message": str(exc)}
    if not bindings_match(
        {
            **current_binding,
            "region_artifact_id": provenance.get("region_artifact_id"),
            "source_sha256": provenance.get("source_sha256"),
            "cutout_sha256": provenance.get("cutout_sha256"),
            "mask_sha256": provenance.get("mask_sha256"),
        },
        current_binding,
    ):
        return {"code": "lineart_binding_mismatch", "message": "Approved lineart input hashes no longer match."}
    asset = item.get("lineart_asset")
    if not asset:
        return {"code": "lineart_binding_mismatch", "message": "Approved lineart asset path is missing."}
    try:
        asset_path = resolve_project_file(project_dir, str(asset))
    except LineartError as exc:
        return {"code": "lineart_binding_mismatch", "message": str(exc)}
    if not asset_path.exists() or sha256_file(asset_path) != provenance.get("approved_sha256"):
        return {"code": "lineart_binding_mismatch", "message": "Approved lineart file hash no longer matches."}
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(project_dir: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": project_relative(project_dir, path),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise LineartError("Could not allocate a unique lineart filename.")


def project_relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
