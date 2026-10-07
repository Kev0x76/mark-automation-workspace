"""Lightweight, auditable product-region proposals for sales images.

This module intentionally does not identify a candidate as approved artwork.  It
only produces crop/mask proposals and a selection state.  A line-art provider
may consume a crop only after its product region is ``selected``. Approval of
the resulting line art is governed separately by provenance and technical QC.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import layout_ocr
from .lineart_engine import generate_lineart_file_v3
from .schema_validation import write_standard_json


ALGORITHM_VERSION = "opencv-layout-v2"
LEGACY_MASK_SEMANTICS = "foreground_evidence"
MASK_SEMANTICS = "product_foreground"
AUTO_SELECT_THRESHOLD = 0.78
AUTO_SELECT_MARGIN = 0.12
MAX_AUTO_TEXT_COVERAGE = 0.08
MIN_AUTO_SUBJECT_SHARE = 0.65
MAX_COMPONENT_CANDIDATES = 4
REGION_STATUSES = {"missing", "selected", "needs_review", "rejected"}
ProgressCallback = Callable[[int, int, str | None], None]


class ProductRegionError(ValueError):
    pass


def initialize_product_region_workspace(project_dir: Path | str) -> None:
    project_dir = Path(project_dir)
    for directory in [
        project_dir / "extracted" / "product_regions",
        project_dir / "reports" / "product_region_visuals",
        project_dir / "reports",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def generate_product_regions(
    workspace_root: Path | str,
    project_id: str,
    *,
    item_no: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    return generate_product_regions_for_project(project_dir, item_no=item_no, force=force)


def generate_product_regions_for_project(
    project_dir: Path | str,
    *,
    item_no: str | None = None,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Write region proposals without changing extracted/product_images."""
    project_dir = Path(project_dir)
    initialize_product_region_workspace(project_dir)
    normalized = load_normalized(project_dir)
    requested_item = item_no
    items = normalized.get("items", [])
    if requested_item and not any(item.get("item_no") == requested_item for item in items):
        raise ProductRegionError(f"Item not found: {requested_item}")

    existing = load_region_payload(project_dir)
    existing_by_item = {entry.get("item_no"): entry for entry in existing.get("items", [])}
    results: list[dict[str, Any]] = []
    target_items = [
        item for item in items
        if not requested_item or item.get("item_no") == requested_item
    ]
    completed_items = 0

    for item in items:
        if requested_item and item.get("item_no") != requested_item:
            prior = existing_by_item.get(item.get("item_no"))
            if prior:
                results.append(prior)
            continue

        current_item_no = str(item.get("item_no") or "")
        if progress_callback:
            progress_callback(completed_items, len(target_items), current_item_no)
        prior = existing_by_item.get(item.get("item_no"))
        if prior and item.get("product_region_status") == "selected" and not force:
            ensure_entry_lineage(project_dir, item, prior)
            apply_item_region_state(item, prior)
            results.append(prior)
        else:
            result = propose_regions_for_item(project_dir, item)
            apply_item_region_state(item, result)
            results.append(result)
        completed_items += 1
        if progress_callback:
            progress_callback(completed_items, len(target_items), current_item_no)

    # Preserve other item records when regenerating one SKU.
    result_by_item = {entry.get("item_no"): entry for entry in results}
    for item in items:
        if item.get("item_no") not in result_by_item and item.get("item_no") in existing_by_item:
            results.append(existing_by_item[item.get("item_no")])

    algorithm_versions = sorted(
        {str(entry.get("algorithm_version") or "unknown") for entry in results}
    )
    payload_algorithm = algorithm_versions[0] if len(algorithm_versions) == 1 else "mixed"
    payload = {
        "schema_version": "1.0.0",
        "algorithm_version": payload_algorithm,
        "algorithm_versions": algorithm_versions,
        "project_id": normalized["project_id"],
        "updated_at": utc_now(),
        "items": sorted(results, key=lambda value: str(value.get("item_no", ""))),
        "summary": build_summary(results),
    }
    save_normalized(project_dir, normalized)
    save_region_payload(project_dir, payload)
    return payload


def get_product_region_state(workspace_root: Path | str, project_id: str) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    payload = load_region_payload(project_dir)
    item_states = []
    by_item = {entry.get("item_no"): entry for entry in payload.get("items", [])}
    for item in normalized.get("items", []):
        entry = by_item.get(item.get("item_no"), {})
        item_states.append(
            {
                "item_no": item.get("item_no"),
                "source_image": item.get("source_image"),
                "product_region_status": item.get("product_region_status", "missing"),
                "selected_product_region": item.get("selected_product_region"),
                "candidates": entry.get("candidates", []),
                "visualization_path": entry.get("visualization_path"),
                "diagnostics": entry.get("diagnostics", {}),
            }
        )
    state_summary = build_summary([
        {
            "status": item.get("product_region_status", "missing"),
            "candidates": item.get("candidates", []),
        }
        for item in item_states
    ])
    return {
        "project_id": normalized["project_id"],
        "algorithm_version": payload.get("algorithm_version") or ALGORITHM_VERSION,
        "algorithm_versions": payload.get("algorithm_versions") or [ALGORITHM_VERSION],
        "items": item_states,
        "summary": state_summary,
        "report_path": "reports/product_regions.json",
    }


def select_product_region(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    candidate_id: str,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    payload = load_region_payload(project_dir)
    entry = find_region_entry(payload, item_no)
    ensure_entry_lineage(project_dir, item, entry)
    candidate = find_candidate(entry, candidate_id)
    if candidate.get("status") == "rejected":
        raise ProductRegionError("Rejected product-region candidates cannot be selected.")
    crop_path = project_file(project_dir, candidate.get("cutout_path"))
    if not crop_path.exists():
        raise ProductRegionError("Product-region cutout file does not exist.")

    for value in entry.get("candidates", []):
        if value.get("candidate_id") == candidate_id:
            value["status"] = "selected"
        elif value.get("status") == "selected":
            value["status"] = "needs_review"
    entry["selection"] = selection_pointer(candidate, selected_by="manual")
    entry["status"] = "selected"
    entry["updated_at"] = utc_now()
    apply_item_region_state(item, entry)
    save_normalized(project_dir, normalized)
    payload["updated_at"] = utc_now()
    payload["summary"] = build_summary(payload.get("items", []))
    save_region_payload(project_dir, payload)
    return region_action_result(normalized, entry)


def create_manual_product_region(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    reviewer: str,
    reason: str,
) -> dict[str, Any]:
    """Create and select an auditable rectangular crop from the immutable source image.

    Coordinates are normalized to the displayed source image.  This deliberately
    performs no new OCR or image-model call: manual selection is the fast fallback
    when automatic layout recognition chose the wrong product region.
    """

    values = [x, y, width, height]
    if not all(np.isfinite(value) for value in values):
        raise ProductRegionError("Manual crop coordinates must be finite numbers.")
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
        raise ProductRegionError("Manual crop coordinates must stay within the source image.")
    if width < 0.02 or height < 0.02:
        raise ProductRegionError("Manual crop is too small; select the complete product.")
    reviewer = " ".join(str(reviewer or "").split())
    reason = " ".join(str(reason or "").split())
    if not reviewer:
        raise ProductRegionError("Manual crop reviewer is required.")
    if not reason:
        raise ProductRegionError("Manual crop reason is required.")

    project_dir = resolve_project_dir(workspace_root, project_id)
    initialize_product_region_workspace(project_dir)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    source_image = item.get("source_image")
    if not source_image:
        raise ProductRegionError("Manual crop requires a contract source image for this item.")
    source_path = project_file(project_dir, source_image)
    if not source_path.exists() or not source_path.is_file():
        raise ProductRegionError("Manual crop source image does not exist.")

    payload = load_region_payload(project_dir)
    entry = next(
        (value for value in payload.get("items", []) if str(value.get("item_no")) == str(item_no)),
        None,
    )
    if entry is None:
        entry = empty_region_entry(str(item_no), str(source_image), "manual_selection")
        payload.setdefault("items", []).append(entry)
    else:
        ensure_entry_lineage(project_dir, item, entry)

    image = load_rgb_image(source_path)
    image_height, image_width = image.shape[:2]
    left = max(0, min(image_width - 1, int(round(x * image_width))))
    top = max(0, min(image_height - 1, int(round(y * image_height))))
    right = max(left + 1, min(image_width, int(round((x + width) * image_width))))
    bottom = max(top + 1, min(image_height, int(round((y + height) * image_height))))
    pixel_width = right - left
    pixel_height = bottom - top
    if pixel_width < 8 or pixel_height < 8:
        raise ProductRegionError("Manual crop is too small in source pixels.")

    diagnostics = entry.get("diagnostics") or {}
    source_text_regions = list(diagnostics.get("text_regions") or [])
    source_text_mask = build_text_mask((image_height, image_width), source_text_regions)
    crop_text_mask = source_text_mask[top:bottom, left:right]
    product_mask = np.full((pixel_height, pixel_width), 255, dtype=np.uint8)
    existing_ids = {str(value.get("candidate_id")) for value in entry.get("candidates", [])}
    sequence = 1
    while f"manual-{sequence:02d}" in existing_ids:
        sequence += 1
    candidate_id = f"manual-{sequence:02d}"
    source_sha256 = sha256_file(source_path)
    candidate = write_candidate_assets(
        project_dir,
        str(item_no),
        candidate_id,
        image,
        [left, top, pixel_width, pixel_height],
        product_mask,
        crop_text_mask,
        score=1.0,
        confidence="high",
        proposal_kind="manual_rectangle",
        reasons=["manual_rectangle_selected"],
        source_sha256=source_sha256,
        proposal_bbox=[left, top, pixel_width, pixel_height],
        text_coverage=round(float(np.mean(crop_text_mask > 0)), 4) if crop_text_mask.size else 0.0,
        proposal_text_coverage=round(float(np.mean(crop_text_mask > 0)), 4) if crop_text_mask.size else 0.0,
        largest_subject_share=1.0,
        text_intersects_product_outline=False,
        text_regions=crop_text_regions(source_text_regions, [left, top, pixel_width, pixel_height]),
        score_components={"manual_selection": 1.0},
        identity_status=str(diagnostics.get("identity_status") or "unknown"),
        ocr=diagnostics.get("ocr") or {"status": diagnostics.get("ocr_status", "unavailable")},
    )
    candidate["algorithm_version"] = "manual-rectangle-v1"
    candidate["selection_audit"] = {
        "reviewer": reviewer,
        "reason": reason,
        "normalized_bbox": {
            "x": round(float(x), 6),
            "y": round(float(y), 6),
            "width": round(float(width), 6),
            "height": round(float(height), 6),
        },
        "created_at": utc_now(),
    }
    for value in entry.get("candidates", []):
        if value.get("status") == "selected":
            value["status"] = "needs_review"
    candidate["status"] = "selected"
    entry.setdefault("candidates", []).append(candidate)
    entry["selection"] = selection_pointer(candidate, selected_by="manual_crop")
    entry["status"] = "selected"
    entry_versions = {
        str(value.get("algorithm_version") or "unknown")
        for value in entry.get("candidates", [])
    }
    entry["algorithm_version"] = next(iter(entry_versions)) if len(entry_versions) == 1 else "mixed"
    entry["updated_at"] = utc_now()
    entry.setdefault("created_at", utc_now())
    entry.setdefault("source_image", source_image)
    entry.setdefault("diagnostics", diagnostics)
    entry["visualization_path"] = None
    apply_item_region_state(item, entry)
    save_normalized(project_dir, normalized)
    payload.setdefault("project_id", normalized.get("project_id") or project_id)
    payload["updated_at"] = utc_now()
    payload["summary"] = build_summary(payload.get("items", []))
    save_region_payload(project_dir, payload)
    return region_action_result(normalized, entry)


def reject_product_region(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    candidate_id: str,
) -> dict[str, Any]:
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized = load_normalized(project_dir)
    item = find_item(normalized, item_no)
    payload = load_region_payload(project_dir)
    entry = find_region_entry(payload, item_no)
    ensure_entry_lineage(project_dir, item, entry)
    candidate = find_candidate(entry, candidate_id)
    candidate["status"] = "rejected"
    if (entry.get("selection") or {}).get("candidate_id") == candidate_id:
        entry["selection"] = None
    viable = [value for value in entry.get("candidates", []) if value.get("status") != "rejected"]
    selection_id = (entry.get("selection") or {}).get("candidate_id")
    selection_is_viable = bool(
        selection_id and any(value.get("candidate_id") == selection_id for value in viable)
    )
    entry["status"] = "selected" if selection_is_viable else ("needs_review" if viable else "rejected")
    entry["updated_at"] = utc_now()
    apply_item_region_state(item, entry)
    save_normalized(project_dir, normalized)
    payload["updated_at"] = utc_now()
    payload["summary"] = build_summary(payload.get("items", []))
    save_region_payload(project_dir, payload)
    return region_action_result(normalized, entry)


def get_selected_region_path(project_dir: Path | str, item: dict[str, Any]) -> Path:
    project_dir = Path(project_dir)
    if item.get("product_region_status") != "selected":
        raise ProductRegionError(
            "Lineart generation is blocked until a product region is selected; item is needs_review."
        )
    pointer = item.get("selected_product_region") or {}
    path = project_file(project_dir, pointer.get("cutout_path"))
    if not path.exists() or not path.is_file():
        raise ProductRegionError("Selected product-region cutout is missing.")
    payload = load_region_payload(project_dir)
    entry = find_region_entry(payload, str(item.get("item_no")))
    candidate = find_candidate(entry, str(pointer.get("candidate_id")))
    if candidate.get("status") != "selected":
        raise ProductRegionError("Selected product-region state is inconsistent; regenerate or review it.")
    return path


def selected_region_metadata(project_dir: Path | str, item: dict[str, Any]) -> dict[str, Any]:
    path = get_selected_region_path(project_dir, item)
    pointer = dict(item.get("selected_product_region") or {})
    pointer["cutout_path"] = project_relative(Path(project_dir), path)
    return pointer


def propose_regions_for_item(project_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    item_no = str(item.get("item_no") or "")
    source_image = item.get("source_image")
    if not source_image:
        return empty_region_entry(item_no, source_image, "missing_source_image")
    source_path = project_file(project_dir, source_image)
    if not source_path.exists() or not source_path.is_file():
        return empty_region_entry(item_no, source_image, "source_image_not_found")

    source_sha256 = sha256_file(source_path)
    image = load_rgb_image(source_path)
    foreground, diagnostics = foreground_mask(image)
    diagnostics.update(optional_layout_diagnostics(image, item))
    source_text_mask = build_text_mask(foreground.shape, diagnostics.get("text_regions") or [])
    proposals = candidate_boxes(foreground)
    candidates = []
    for index, proposal in enumerate(proposals, start=1):
        candidate_id = f"region-{index:02d}"
        analysis = refine_candidate_region(
            proposal["bbox"],
            foreground,
            source_text_mask,
            diagnostics.get("text_regions") or [],
            proposal_kind=proposal["kind"],
            clean_single_product=bool(diagnostics.get("clean_single_product")),
        )
        score, confidence, reasons, score_components = score_candidate(
            analysis,
            foreground,
            diagnostics,
            proposal["kind"],
        )
        candidate = write_candidate_assets(
            project_dir,
            item_no,
            candidate_id,
            image,
            analysis["bbox"],
            analysis["product_mask"],
            analysis["text_mask"],
            score=score,
            confidence=confidence,
            proposal_kind=proposal["kind"],
            reasons=reasons,
            source_sha256=source_sha256,
            proposal_bbox=proposal["bbox"],
            text_coverage=analysis["text_coverage"],
            proposal_text_coverage=analysis["proposal_text_coverage"],
            largest_subject_share=analysis["largest_subject_share"],
            text_intersects_product_outline=analysis["text_intersects_product_outline"],
            text_regions=analysis["text_regions"],
            score_components=score_components,
            identity_status=str(diagnostics.get("identity_status") or "unknown"),
            ocr=diagnostics.get("ocr") or {"status": diagnostics.get("ocr_status", "failed")},
        )
        candidates.append(candidate)

    ranked = sorted(candidates, key=lambda candidate: float(candidate.get("score") or 0.0), reverse=True)
    selected = ranked[0] if ranked else None
    score_gap = round(
        float(selected.get("score") or 0.0) - float(ranked[1].get("score") or 0.0), 4
    ) if len(ranked) > 1 else (1.0 if selected else 0.0)
    automatic_selection_blockers = automatic_selection_blockers_for(
        selected,
        diagnostics=diagnostics,
    )
    automatic_selection_warnings = automatic_selection_warnings_for(
        selected,
        score_gap=score_gap,
        diagnostics=diagnostics,
    )
    diagnostics["score_gap"] = score_gap
    diagnostics["automatic_selection_blockers"] = automatic_selection_blockers
    diagnostics["automatic_selection_warnings"] = automatic_selection_warnings
    diagnostics["selected_candidate_id"] = selected.get("candidate_id") if selected and not automatic_selection_blockers else None
    if selected and not automatic_selection_blockers:
        selected["status"] = "selected"
        status = "selected"
        selection = selection_pointer(selected, selected_by="automatic")
    else:
        status = "needs_review" if candidates else "rejected"
        selection = None
        for candidate in candidates:
            candidate["status"] = "needs_review"

    entry = {
        "item_no": item_no,
        "source_image": source_image,
        "status": status,
        "selection": selection,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "diagnostics": diagnostics,
        "candidates": candidates,
        # Expensive six-panel comparison and line-art previews are diagnostics,
        # not production inputs.  They are no longer rendered during batch
        # region generation.
        "visualization_path": None,
    }
    return entry


def empty_region_entry(item_no: str, source_image: str | None, reason: str) -> dict[str, Any]:
    return {
        "item_no": item_no,
        "source_image": source_image,
        "status": "missing",
        "selection": None,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "diagnostics": {"reason": reason, "clean_single_product": False},
        "candidates": [],
        "visualization_path": None,
    }


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            canvas = Image.new("RGBA", image.size, "WHITE")
            canvas.alpha_composite(image.convert("RGBA"))
            image = canvas.convert("RGB")
        else:
            image = image.convert("RGB")
        return np.asarray(image).copy()


def foreground_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Black line art is retained by the grayscale condition; coloured products
    # are retained by saturation.  White page background remains excluded.
    foreground = ((gray < 238) | ((hsv[:, :, 1] > 32) & (hsv[:, :, 2] < 252))).astype(np.uint8) * 255
    h, w = foreground.shape
    ys, xs = np.where(foreground > 0)
    if len(xs) == 0:
        bbox = None
        union_coverage = 0.0
        touches_edge = False
    else:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox = [x1, y1, x2 - x1, y2 - y1]
        union_coverage = round((bbox[2] * bbox[3]) / float(w * h), 4)
        edge_margin = max(3, round(min(w, h) * 0.01))
        touches_edge = x1 <= edge_margin or y1 <= edge_margin or x2 >= w - edge_margin or y2 >= h - edge_margin
    foreground_ratio = round(float(np.mean(foreground > 0)), 4)
    clean_single_product = bool(
        bbox
        and foreground_ratio <= 0.24
        and union_coverage <= 0.75
        and not touches_edge
    )
    return foreground, {
        "image_size": {"width": w, "height": h},
        "foreground_ratio": foreground_ratio,
        "union_bbox": bbox,
        "union_coverage": union_coverage,
        "touches_edge": touches_edge,
        "clean_single_product": clean_single_product,
    }


def optional_layout_diagnostics(image: np.ndarray, item: dict[str, Any]) -> dict[str, Any]:
    """Collect OCR geometry and barcode identity evidence without trusting either."""
    expected_barcode = str(item.get("barcode") or "")
    detected_barcode: str | None = None
    barcode_bbox: list[int] | None = None
    barcode_detector = "opencv_barcode:not_detected"
    barcode_decode_status = "not_detected"
    try:
        detector = cv2.barcode.BarcodeDetector()
        decoded, points, _ = detector.detectAndDecode(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if points is not None:
            values = np.asarray(points).reshape(-1, 2)
            x1, y1 = values.min(axis=0)
            x2, y2 = values.max(axis=0)
            barcode_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            detected_barcode = str(decoded or "")
            barcode_detector = "opencv_barcode:detected"
            barcode_decode_status = "decoded" if detected_barcode else "not_decoded"
    except Exception as exc:
        barcode_detector = f"opencv_barcode:unavailable:{type(exc).__name__}"
        barcode_decode_status = "unavailable"

    if detected_barcode and expected_barcode:
        identity_status = "matched" if detected_barcode == expected_barcode else "mismatch"
    else:
        identity_status = "unknown"

    ocr_result = layout_ocr.detect_text_regions(image)
    text_regions = list(ocr_result.get("regions") or [])
    ocr_metadata = {key: value for key, value in ocr_result.items() if key != "regions"}
    ocr_status = str(ocr_result.get("status") or "failed")
    names_present = sum(bool(str(item.get(field) or "").strip()) for field in ("cn_name", "it_name", "es_name", "en_name"))
    return {
        "text_detector": f"rapidocr:{ocr_status}",
        "ocr_status": ocr_status,
        "ocr_engine": ocr_metadata.get("engine"),
        "ocr_version": ocr_metadata.get("version"),
        "ocr_elapsed_ms": ocr_metadata.get("elapsed_ms"),
        "ocr_model_sha256": ocr_metadata.get("model_sha256"),
        "ocr": ocr_metadata,
        "text_regions": text_regions,
        "barcode_detector": barcode_detector,
        "barcode_decode_status": barcode_decode_status,
        "barcode_bbox": barcode_bbox,
        "barcode_value": detected_barcode or None,
        "barcode_matches_contract": bool(detected_barcode and expected_barcode and detected_barcode == expected_barcode),
        "identity_status": identity_status,
        "contract_assist": {
            "item_no_present": bool(str(item.get("item_no") or "").strip()),
            "barcode_present": bool(expected_barcode),
            "name_fields_present": names_present,
        },
    }


def candidate_boxes(foreground: np.ndarray) -> list[dict[str, Any]]:
    h, w = foreground.shape
    ys, xs = np.where(foreground > 0)
    if len(xs) == 0:
        return []
    union = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    boxes = [{"bbox": pad_bbox(union, w, h), "kind": "foreground_union"}]

    kernel_size = max(3, min(w, h) // 55)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    clustered = cv2.dilate(foreground, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
    components: list[tuple[int, int, int, int, int]] = []
    min_area = w * h * 0.003
    for x, y, width, height, area in stats[1:]:
        if area >= min_area:
            components.append((int(x), int(y), int(width), int(height), int(area)))
    components.sort(key=lambda value: value[4], reverse=True)
    for x, y, width, height, _ in components:
        bbox = pad_bbox([x, y, width, height], w, h)
        if iou(bbox, boxes[0]["bbox"]) > 0.94:
            continue
        if any(iou(bbox, entry["bbox"]) > 0.88 for entry in boxes):
            continue
        boxes.append({"bbox": bbox, "kind": "foreground_component"})
        if len(boxes) >= MAX_COMPONENT_CANDIDATES + 1:
            break
    return boxes


def build_text_mask(shape: tuple[int, int], regions: list[dict[str, Any]]) -> np.ndarray:
    """Rasterize OCR geometry in source-image coordinates."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        polygon = region.get("polygon") if isinstance(region, dict) else None
        try:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            points = np.empty((0, 2), dtype=np.float32)
        if len(points) >= 3:
            points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
            points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
            continue
        bbox = text_region_bbox(region)
        if bbox is None:
            continue
        x, y, box_width, box_height = bbox
        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(x1, min(width, x + box_width))
        y2 = max(y1, min(height, y + box_height))
        mask[y1:y2, x1:x2] = 255
    return mask


def text_region_bbox(region: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if not isinstance(region, dict):
        return None
    value = region.get("bbox")
    try:
        if isinstance(value, dict):
            return (
                int(round(float(value["x"]))),
                int(round(float(value["y"]))),
                max(1, int(round(float(value["width"])))),
                max(1, int(round(float(value["height"])))),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (
                int(round(float(value[0]))),
                int(round(float(value[1]))),
                max(1, int(round(float(value[2])))),
                max(1, int(round(float(value[3])))),
            )
    except (KeyError, TypeError, ValueError):
        return None
    polygon = region.get("polygon")
    try:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 3:
        return None
    x, y, width, height = cv2.boundingRect(np.rint(points).astype(np.int32))
    return int(x), int(y), int(width), int(height)


def refine_candidate_region(
    proposal_bbox: list[int],
    foreground: np.ndarray,
    source_text_mask: np.ndarray,
    source_text_regions: list[dict[str, Any]],
    *,
    proposal_kind: str,
    clean_single_product: bool,
) -> dict[str, Any]:
    """Separate dominant non-text subject evidence from surrounding sales layout."""
    image_height, image_width = foreground.shape
    proposal = pad_bbox(proposal_bbox, image_width, image_height, padding=0)
    px, py, proposal_width, proposal_height = proposal
    proposal_fg = foreground[py : py + proposal_height, px : px + proposal_width]
    proposal_text = source_text_mask[py : py + proposal_height, px : px + proposal_width]
    text_padding = max(1, round(min(proposal_width, proposal_height) * 0.004))
    text_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (text_padding * 2 + 1, text_padding * 2 + 1)
    )
    padded_text = cv2.dilate(proposal_text, text_kernel)
    non_text = cv2.bitwise_and(proposal_fg, cv2.bitwise_not(padded_text))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (non_text > 0).astype(np.uint8), connectivity=8
    )
    minimum_area = max(8, round(non_text.size * 0.0008))
    valid_labels = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    valid_labels.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    valid_area = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in valid_labels)
    largest_area = int(stats[valid_labels[0], cv2.CC_STAT_AREA]) if valid_labels else 0
    largest_subject_share = largest_area / float(max(valid_area, 1))
    component_count = len(valid_labels)

    # A clean single-product union may legitimately consist of disconnected
    # strokes or light surfaces.  Treat the union as one subject only when the
    # page-level cleanliness gate already established that fact.
    complex_union = proposal_kind == "foreground_union" and not clean_single_product
    if not valid_labels:
        subject = proposal_fg.copy()
    elif complex_union:
        subject = non_text.copy()
    else:
        primary = (labels == valid_labels[0]).astype(np.uint8) * 255
        proximity = max(5, round(min(proposal_width, proposal_height) * 0.035))
        proximity_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (proximity * 2 + 1, proximity * 2 + 1)
        )
        near_primary = cv2.dilate(primary, proximity_kernel)
        subject = primary.copy()
        for label in valid_labels[1:]:
            component = (labels == label).astype(np.uint8) * 255
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            if component_area >= max(minimum_area, round(largest_area * 0.02)) and np.any(
                (component > 0) & (near_primary > 0)
            ):
                subject = cv2.bitwise_or(subject, component)

    if clean_single_product and proposal_kind == "foreground_union":
        largest_subject_share = 1.0
        component_count = 1

    if complex_union:
        product_in_proposal = proposal_fg.copy()
        final_bbox = proposal
    else:
        close_size = max(3, round(min(proposal_width, proposal_height) * 0.018))
        if close_size % 2 == 0:
            close_size += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        merged = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, close_kernel)
        merged = cv2.dilate(merged, np.ones((3, 3), dtype=np.uint8))
        product_in_proposal = np.zeros_like(merged)
        contours, _ = cv2.findContours(
            (merged > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        kept = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
        if kept:
            cv2.drawContours(product_in_proposal, kept, -1, 255, thickness=cv2.FILLED)
        else:
            product_in_proposal = merged

        points = cv2.findNonZero((product_in_proposal > 0).astype(np.uint8))
        if points is None:
            final_bbox = proposal
        else:
            sx, sy, subject_width, subject_height = cv2.boundingRect(points)
            pad = max(4, round(min(subject_width, subject_height) * 0.035))
            final_bbox = pad_bbox(
                [px + sx, py + sy, subject_width, subject_height],
                image_width,
                image_height,
                padding=pad,
            )

    product_full = np.zeros_like(foreground)
    product_full[py : py + proposal_height, px : px + proposal_width] = product_in_proposal
    x, y, width, height = final_bbox
    product_mask = product_full[y : y + height, x : x + width].copy()
    text_mask = source_text_mask[y : y + height, x : x + width].copy()
    boundary = cv2.morphologyEx(
        product_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    text_intersects_product_outline = bool(
        np.any((cv2.dilate(boundary, np.ones((3, 3), dtype=np.uint8)) > 0) & (text_mask > 0))
    )
    translated_regions = crop_text_regions(source_text_regions, final_bbox)
    proposal_region_count = sum(
        1 for region in source_text_regions if boxes_intersect(proposal, text_region_bbox(region))
    )
    return {
        "bbox": final_bbox,
        "proposal_bbox": proposal,
        "product_mask": product_mask,
        "text_mask": text_mask,
        "text_regions": translated_regions,
        "text_coverage": round(float(np.mean(text_mask > 0)), 4) if text_mask.size else 0.0,
        "proposal_text_coverage": round(float(np.mean(proposal_text > 0)), 4)
        if proposal_text.size
        else 0.0,
        "largest_subject_share": round(float(largest_subject_share), 4),
        "component_count": component_count,
        "proposal_text_region_count": proposal_region_count,
        "text_intersects_product_outline": text_intersects_product_outline,
    }


def boxes_intersect(
    first: list[int] | tuple[int, int, int, int],
    second: tuple[int, int, int, int] | None,
) -> bool:
    if second is None:
        return False
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + ah, by + bh)


def crop_text_regions(
    regions: list[dict[str, Any]], bbox: list[int]
) -> list[dict[str, Any]]:
    x, y, width, height = bbox
    output: list[dict[str, Any]] = []
    for region in regions:
        region_bbox = text_region_bbox(region)
        if not boxes_intersect(bbox, region_bbox) or region_bbox is None:
            continue
        rx, ry, rw, rh = region_bbox
        x1, y1 = max(x, rx), max(y, ry)
        x2, y2 = min(x + width, rx + rw), min(y + height, ry + rh)
        if x2 <= x1 or y2 <= y1:
            continue
        polygon_value = region.get("polygon") if isinstance(region, dict) else None
        polygon: list[list[int]] = []
        try:
            for point_x, point_y in np.asarray(polygon_value, dtype=np.float32).reshape(-1, 2):
                polygon.append(
                    [
                        int(np.clip(round(float(point_x)) - x, 0, max(0, width - 1))),
                        int(np.clip(round(float(point_y)) - y, 0, max(0, height - 1))),
                    ]
                )
        except (TypeError, ValueError):
            polygon = []
        output.append(
            {
                "polygon": polygon
                or [
                    [x1 - x, y1 - y],
                    [x2 - x, y1 - y],
                    [x2 - x, y2 - y],
                    [x1 - x, y2 - y],
                ],
                "bbox": {
                    "x": x1 - x,
                    "y": y1 - y,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
            }
        )
    return output


def score_candidate(
    analysis: dict[str, Any],
    foreground: np.ndarray,
    diagnostics: dict[str, Any],
    proposal_kind: str,
) -> tuple[float, str, list[str], dict[str, float]]:
    """Score product evidence, not page ink, and expose every score component."""
    height, width = foreground.shape
    x, y, box_width, box_height = analysis["bbox"]
    area_ratio = (box_width * box_height) / float(width * height)
    center_x = (x + box_width / 2.0) / width
    center_y = (y + box_height / 2.0) / height
    center_distance = min(1.0, (((center_x - 0.5) ** 2 + (center_y - 0.46) ** 2) ** 0.5) / 0.72)
    subject_area = min(1.0, area_ratio / 0.16)
    connectivity = min(1.0, float(analysis["largest_subject_share"]) / 0.80)
    text_cleanliness = max(0.0, 1.0 - float(analysis["proposal_text_coverage"]) / 0.16)
    center = max(0.0, 1.0 - center_distance)
    if area_ratio < 0.06:
        size = max(0.0, area_ratio / 0.06)
    elif area_ratio <= 0.58:
        size = 1.0
    else:
        size = max(0.0, 1.0 - (area_ratio - 0.58) / 0.42)
    edge_margin = max(3, round(min(width, height) * 0.01))
    touches_edge = (
        x <= edge_margin
        or y <= edge_margin
        or x + box_width >= width - edge_margin
        or y + box_height >= height - edge_margin
    )
    edge_safety = 0.0 if touches_edge else 1.0

    layout_penalty = 0.0
    proposal_x, proposal_y, proposal_width, proposal_height = analysis["proposal_bbox"]
    proposal_aspect = proposal_width / max(1.0, float(proposal_height))
    if proposal_y / height >= 0.58 and proposal_aspect >= 1.7:
        layout_penalty += 0.30
    if analysis["component_count"] >= 3 and analysis["largest_subject_share"] < 0.65:
        layout_penalty += min(0.20, 0.04 * (analysis["component_count"] - 1))
    if proposal_kind == "foreground_union" and diagnostics.get("union_coverage", 0.0) > 0.80:
        layout_penalty += 0.20
    if area_ratio < 0.025 and analysis["proposal_text_region_count"] >= 2:
        layout_penalty += 0.10

    barcode_penalty = 0.0
    barcode_bbox = diagnostics.get("barcode_bbox")
    if barcode_bbox and iou(analysis["proposal_bbox"], barcode_bbox) > 0.01:
        barcode_penalty = 0.22

    layout_penalty = min(0.50, layout_penalty)
    raw_score = (
        0.38 * subject_area
        + 0.20 * connectivity
        + 0.18 * text_cleanliness
        + 0.10 * center
        + 0.08 * size
        + 0.06 * edge_safety
    )
    score = round(max(0.0, min(1.0, raw_score - layout_penalty - barcode_penalty)), 4)
    components = {
        "subject_area": round(subject_area, 4),
        "connectivity": round(connectivity, 4),
        "text_cleanliness": round(text_cleanliness, 4),
        "center": round(center, 4),
        "size": round(size, 4),
        "edge_safety": round(edge_safety, 4),
        "layout_penalty": round(layout_penalty, 4),
        "barcode_penalty": round(barcode_penalty, 4),
    }
    reasons: list[str] = []
    if not diagnostics.get("clean_single_product"):
        reasons.append("source_layout_is_complex_or_covers_most_of_the_canvas")
    if touches_edge:
        reasons.append("candidate_touches_source_edge")
    if analysis["proposal_text_coverage"] > MAX_AUTO_TEXT_COVERAGE:
        reasons.append("candidate_contains_layout_text")
    if analysis["largest_subject_share"] < MIN_AUTO_SUBJECT_SHARE:
        reasons.append("multiple_or_fragmented_non_text_subjects")
    if analysis["text_intersects_product_outline"]:
        reasons.append("text_intersects_product_outline")
    if barcode_penalty:
        reasons.append("candidate_overlaps_detected_barcode")
    if diagnostics.get("identity_status") == "mismatch":
        reasons.append("source_identity_mismatch")
    confidence = "high" if (
        score >= AUTO_SELECT_THRESHOLD
        and analysis["text_coverage"] <= MAX_AUTO_TEXT_COVERAGE
        and analysis["largest_subject_share"] >= MIN_AUTO_SUBJECT_SHARE
        and not analysis["text_intersects_product_outline"]
        and diagnostics.get("ocr_status") == "success"
        and diagnostics.get("identity_status") != "mismatch"
    ) else "low"
    return score, confidence, reasons, components


def automatic_selection_blockers_for(
    candidate: dict[str, Any] | None,
    *,
    diagnostics: dict[str, Any],
) -> list[str]:
    """Return only conditions that make automatic selection factually unsafe.

    The ranked region is now the default production input.  Low confidence,
    close scores, OCR gaps and layout warnings are recorded as warnings so the
    operator can intervene when needed, but they do not turn every ordinary
    SKU into a manual step.  A decoded identity mismatch remains a hard stop.
    """

    blockers: list[str] = []
    if candidate is None:
        return ["no_product_region_candidate"]
    if diagnostics.get("identity_status") == "mismatch":
        blockers.append("source_identity_mismatch")
    return list(dict.fromkeys(blockers))


def automatic_selection_warnings_for(
    candidate: dict[str, Any] | None,
    *,
    score_gap: float,
    diagnostics: dict[str, Any],
) -> list[str]:
    """Return non-blocking reasons to expose beside the automatic selection."""

    if candidate is None:
        return []
    warnings: list[str] = []
    if diagnostics.get("ocr_status") != "success":
        warnings.append("ocr_not_available_or_failed")
    if float(candidate.get("score") or 0.0) < AUTO_SELECT_THRESHOLD:
        warnings.append("candidate_score_below_threshold")
    if score_gap < AUTO_SELECT_MARGIN:
        warnings.append("candidate_score_margin_too_small")
    if float(candidate.get("text_coverage") or 0.0) > MAX_AUTO_TEXT_COVERAGE:
        warnings.append("candidate_text_coverage_too_high")
    if float(candidate.get("largest_subject_share") or 0.0) < MIN_AUTO_SUBJECT_SHARE:
        warnings.append("candidate_subject_is_fragmented_or_ambiguous")
    if candidate.get("text_intersects_product_outline"):
        warnings.append("text_intersects_product_outline")
    if candidate.get("confidence") != "high":
        warnings.append("candidate_confidence_is_not_high")
    return list(dict.fromkeys(warnings))


def write_candidate_assets(
    project_dir: Path,
    item_no: str,
    candidate_id: str,
    image: np.ndarray,
    bbox: list[int],
    product_mask: np.ndarray,
    text_mask: np.ndarray,
    *,
    score: float,
    confidence: str,
    proposal_kind: str,
    reasons: list[str],
    source_sha256: str,
    proposal_bbox: list[int],
    text_coverage: float,
    proposal_text_coverage: float,
    largest_subject_share: float,
    text_intersects_product_outline: bool,
    text_regions: list[dict[str, Any]],
    score_components: dict[str, float],
    identity_status: str,
    ocr: dict[str, Any],
) -> dict[str, Any]:
    x, y, width, height = bbox
    region_dir = (
        project_dir
        / "extracted"
        / "product_regions"
        / safe_name(item_no)
        / safe_name(ALGORITHM_VERSION)
        / candidate_id
    )
    region_dir.mkdir(parents=True, exist_ok=True)
    crop = image[y : y + height, x : x + width]
    if product_mask.shape != crop.shape[:2] or text_mask.shape != crop.shape[:2]:
        raise ProductRegionError("Refined candidate masks do not match the candidate crop.")
    crop_mask = (product_mask > 0).astype(np.uint8) * 255
    crop_text_mask = (text_mask > 0).astype(np.uint8) * 255
    # The cutout never invents or repairs pixels.  It only replaces evidence
    # outside the selected product mask with white.
    cutout = crop.copy()
    expanded_mask = cv2.dilate(crop_mask, np.ones((3, 3), dtype=np.uint8))
    cutout[expanded_mask == 0] = (255, 255, 255)
    crop_path = region_dir / "crop.png"
    cutout_path = region_dir / "cutout.png"
    mask_path = region_dir / "mask.png"
    text_mask_path = region_dir / "text-mask.png"
    Image.fromarray(crop).save(crop_path, format="PNG", optimize=True)
    Image.fromarray(cutout).save(cutout_path, format="PNG", optimize=True)
    Image.fromarray(crop_mask).save(mask_path, format="PNG", optimize=True)
    Image.fromarray(crop_text_mask).save(text_mask_path, format="PNG", optimize=True)
    crop_sha256 = sha256_file(crop_path)
    cutout_sha256 = sha256_file(cutout_path)
    mask_sha256 = sha256_file(mask_path)
    text_mask_sha256 = sha256_file(text_mask_path)
    region_artifact_id = make_region_artifact_id(
        bbox={"x": x, "y": y, "width": width, "height": height},
        source_sha256=source_sha256,
        crop_sha256=crop_sha256,
        cutout_sha256=cutout_sha256,
        mask_sha256=mask_sha256,
        text_mask_sha256=text_mask_sha256,
    )
    return {
        "candidate_id": candidate_id,
        "status": "needs_review",
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "proposal_bbox": {
            "x": int(proposal_bbox[0]),
            "y": int(proposal_bbox[1]),
            "width": int(proposal_bbox[2]),
            "height": int(proposal_bbox[3]),
        },
        "image_path": project_relative(project_dir, crop_path),
        "cutout_path": project_relative(project_dir, cutout_path),
        "mask_path": project_relative(project_dir, mask_path),
        "text_mask_path": project_relative(project_dir, text_mask_path),
        "mask_semantics": MASK_SEMANTICS,
        "source_sha256": source_sha256,
        "crop_sha256": crop_sha256,
        "cutout_sha256": cutout_sha256,
        "mask_sha256": mask_sha256,
        "text_mask_sha256": text_mask_sha256,
        "region_artifact_id": region_artifact_id,
        "score": score,
        "confidence": confidence,
        "score_components": score_components,
        "text_coverage": text_coverage,
        "proposal_text_coverage": proposal_text_coverage,
        "largest_subject_share": largest_subject_share,
        "text_intersects_product_outline": text_intersects_product_outline,
        "text_regions": text_regions,
        "identity_status": identity_status,
        "ocr_status": str(ocr.get("status") or "failed"),
        "ocr": ocr,
        "proposal_kind": proposal_kind,
        "algorithm_version": ALGORITHM_VERSION,
        "reasons": reasons,
    }


def write_comparison_visual(project_dir: Path, item: dict[str, Any], entry: dict[str, Any]) -> None:
    source_image = item.get("source_image")
    if not source_image:
        return
    source_path = project_file(project_dir, source_image)
    if not source_path.exists():
        return
    with Image.open(source_path) as source:
        raw = ImageOps.exif_transpose(source).convert("RGB").copy()
    candidates = entry.get("candidates", [])
    selected = next((value for value in candidates if value.get("status") == "selected"), None)
    displayed = selected or (
        max(candidates, key=lambda value: float(value.get("score") or 0.0))
        if candidates
        else None
    )
    visuals_dir = (
        project_dir
        / "reports"
        / "product_region_visuals"
        / safe_name(str(item.get("item_no", "item")))
        / safe_name(str(entry.get("algorithm_version") or ALGORITHM_VERSION))
    )
    visuals_dir.mkdir(parents=True, exist_ok=True)

    layout = raw.copy()
    layout_draw = ImageDraw.Draw(layout)
    for region in (entry.get("diagnostics") or {}).get("text_regions") or []:
        polygon = region.get("polygon") if isinstance(region, dict) else None
        try:
            points = [(int(point[0]), int(point[1])) for point in polygon]
        except (TypeError, ValueError, IndexError):
            points = []
        if len(points) >= 3:
            layout_draw.line(points + [points[0]], fill="#f59e0b", width=max(2, min(raw.size) // 220))
    barcode_bbox = (entry.get("diagnostics") or {}).get("barcode_bbox")
    if isinstance(barcode_bbox, list) and len(barcode_bbox) >= 4:
        bx, by, bw, bh = [int(value) for value in barcode_bbox[:4]]
        layout_draw.rectangle((bx, by, bx + bw, by + bh), outline="#2563eb", width=max(2, min(raw.size) // 180))

    boxes = raw.copy()
    draw = ImageDraw.Draw(boxes)
    for value in candidates:
        proposal = value.get("proposal_bbox") or value.get("bbox", {})
        px, py = int(proposal.get("x", 0)), int(proposal.get("y", 0))
        px2 = px + int(proposal.get("width", 0))
        py2 = py + int(proposal.get("height", 0))
        draw.rectangle((px, py, px2, py2), outline="#dc2626", width=max(1, min(raw.size) // 300))
        bbox = value.get("bbox", {})
        x, y = int(bbox.get("x", 0)), int(bbox.get("y", 0))
        x2 = x + int(bbox.get("width", 0))
        y2 = y + int(bbox.get("height", 0))
        color = "#16a34a" if value.get("status") == "selected" else "#7c3aed"
        draw.rectangle((x, y, x2, y2), outline=color, width=max(2, min(raw.size) // 180))
        draw.text((x + 4, y + 4), f"{value.get('candidate_id')} {value.get('score')}", fill=color, stroke_width=1, stroke_fill="white")

    selected_image = open_candidate_image(project_dir, displayed, "image_path")
    cutout_image = open_candidate_image(project_dir, displayed, "cutout_path")
    mask_image = open_candidate_image(project_dir, displayed, "mask_path")
    mask_cutout_image = combine_mask_and_cutout(cutout_image, mask_image)
    if displayed:
        preview_path = visuals_dir / f"lineart_preview_{safe_name(str(displayed.get('candidate_id') or 'candidate'))}.png"
        preview = generate_lineart_file_v3(
            project_file(project_dir, displayed["cutout_path"]),
            project_file(project_dir, displayed["mask_path"]) if displayed.get("mask_path") else None,
            text_mask_path=(
                project_file(project_dir, displayed["text_mask_path"])
                if displayed.get("text_mask_path")
                else None
            ),
            text_regions=list(displayed.get("text_regions") or []),
            text_mask_status=str(displayed.get("ocr_status") or "missing"),
            mask_semantics=str(displayed.get("mask_semantics") or MASK_SEMANTICS),
        )
        preview_path.write_bytes(preview.data)
        with Image.open(preview_path) as preview_image:
            lineart_image = preview_image.convert("RGB").copy()
        qc_label = "QC passed" if preview.quality.get("auto_qc_passed") else "needs_review"
        selection_label = "selected" if selected else "not selected"
        lineart_label = f"Lineart preview ({qc_label}; {selection_label}; not approved)"
    else:
        lineart_image = None
        lineart_label = "No lineart candidate"

    labels = [
        "Original sales image",
        f"OCR / barcode ({(entry.get('diagnostics') or {}).get('ocr_status', 'missing')})",
        "Candidate boxes and scores",
        "Selected product region" if selected else "Best candidate (not selected)",
        "Product mask / cutout" if selected else "Mask / cutout (not selected)",
        lineart_label,
    ]
    images = [raw, layout, boxes, selected_image, mask_cutout_image, lineart_image]
    tile_w, tile_h = 300, 340
    canvas = Image.new("RGB", (tile_w * 6, tile_h), "#f8fafc")
    for index, (label, visual) in enumerate(zip(labels, images)):
        x = index * tile_w
        ImageDraw.Draw(canvas).text((x + 10, 10), label, fill="#111827")
        if visual is None:
            ImageDraw.Draw(canvas).multiline_text((x + 20, 150), "No selected region\nneeds_review", fill="#b91c1c", spacing=8)
            continue
        tile = contain_image(visual, tile_w - 20, tile_h - 55)
        canvas.paste(tile, (x + (tile_w - tile.width) // 2, 45 + (tile_h - 55 - tile.height) // 2))
    canvas.save(visuals_dir / "comparison.png", format="PNG", optimize=True)


def combine_mask_and_cutout(
    cutout: Image.Image | None, mask: Image.Image | None
) -> Image.Image | None:
    if cutout is None:
        return None
    left = cutout.convert("RGB")
    if mask is None:
        return left
    right = mask.convert("L").convert("RGB")
    height = max(left.height, right.height)
    result = Image.new("RGB", (left.width + right.width + 4, height), "white")
    result.paste(left, (0, (height - left.height) // 2))
    result.paste(right, (left.width + 4, (height - right.height) // 2))
    return result


def open_candidate_image(project_dir: Path, candidate: dict[str, Any] | None, key: str) -> Image.Image | None:
    if not candidate:
        return None
    path = project_file(project_dir, candidate.get(key))
    if not path.exists():
        return None
    with Image.open(path) as value:
        return value.convert("RGB").copy()


def contain_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def apply_item_region_state(item: dict[str, Any], entry: dict[str, Any]) -> None:
    status = entry.get("status", "missing")
    normalized_status = status if status in REGION_STATUSES else "needs_review"
    old_selection = item.get("selected_product_region")
    new_selection = entry.get("selection") if normalized_status == "selected" else None
    # Any non-missing lineart without a selected region has no provable input
    # binding.  This also retires legacy whole-sales-image "approved" files.
    if (new_selection is None and item.get("lineart_status") not in {None, "missing", "stale"}) or region_artifact_changed(
        old_selection, new_selection
    ):
        invalidate_lineart_for_region_change(item)
    item["product_region_status"] = normalized_status
    item["selected_product_region"] = new_selection


def selection_pointer(candidate: dict[str, Any], *, selected_by: str) -> dict[str, Any]:
    pointer = {
        "candidate_id": candidate["candidate_id"],
        "image_path": candidate["image_path"],
        "cutout_path": candidate["cutout_path"],
        "mask_path": candidate.get("mask_path"),
        "mask_semantics": candidate.get("mask_semantics") or MASK_SEMANTICS,
        "text_mask_path": candidate.get("text_mask_path"),
        "text_mask_sha256": candidate.get("text_mask_sha256"),
        "text_regions": list(candidate.get("text_regions") or []),
        "ocr_status": candidate.get("ocr_status"),
        "identity_status": candidate.get("identity_status", "unknown"),
        "source_sha256": candidate.get("source_sha256"),
        "crop_sha256": candidate.get("crop_sha256"),
        "cutout_sha256": candidate.get("cutout_sha256"),
        "mask_sha256": candidate.get("mask_sha256"),
        "region_artifact_id": candidate.get("region_artifact_id"),
        "score": candidate["score"],
        "confidence": candidate["confidence"],
        "algorithm_version": candidate["algorithm_version"],
        "selected_by": selected_by,
        "selected_at": utc_now(),
    }
    if candidate.get("selection_audit"):
        pointer["selection_audit"] = dict(candidate["selection_audit"])
    return pointer


def region_action_result(normalized: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    item = find_item(normalized, str(entry.get("item_no")))
    return {
        "project_id": normalized["project_id"],
        "item_no": entry.get("item_no"),
        "item": item,
        "region": entry,
    }


def build_summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in REGION_STATUSES}
    candidate_count = 0
    for entry in entries:
        status = entry.get("status", "missing")
        counts[status] = counts.get(status, 0) + 1
        candidate_count += len(entry.get("candidates", []))
    return {
        "total_items": len(entries),
        "selected": counts.get("selected", 0),
        "needs_review": counts.get("needs_review", 0),
        "rejected": counts.get("rejected", 0),
        "missing": counts.get("missing", 0),
        "candidate_count": candidate_count,
    }


def load_normalized(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "data" / "items.normalized.json"
    if not path.exists():
        raise FileNotFoundError("data/items.normalized.json not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_normalized(project_dir: Path, payload: dict[str, Any]) -> None:
    write_standard_json(
        project_dir / "data" / "items.normalized.json",
        payload,
        "items.normalized.schema.json",
    )


def load_region_payload(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "reports" / "product_regions.json"
    if not path.exists():
        return {"items": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductRegionError("reports/product_regions.json is invalid JSON.") from exc


def save_region_payload(project_dir: Path, payload: dict[str, Any]) -> None:
    (project_dir / "reports" / "product_regions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def find_item(normalized: dict[str, Any], item_no: str) -> dict[str, Any]:
    for item in normalized.get("items", []):
        if str(item.get("item_no")) == str(item_no):
            return item
    raise ProductRegionError(f"Item not found: {item_no}")


def find_region_entry(payload: dict[str, Any], item_no: str) -> dict[str, Any]:
    for entry in payload.get("items", []):
        if str(entry.get("item_no")) == str(item_no):
            return entry
    raise ProductRegionError(f"No product-region candidates found for item: {item_no}")


def find_candidate(entry: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in entry.get("candidates", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise ProductRegionError(f"Product-region candidate not found: {candidate_id}")


def resolve_project_dir(workspace_root: Path | str, project_id: str) -> Path:
    if not project_id or "/" in project_id or "\\" in project_id:
        raise ProductRegionError("Invalid project_id.")
    projects_root = (Path(workspace_root) / "projects").resolve()
    project_dir = (projects_root / project_id).resolve()
    if not is_relative_to(project_dir, projects_root) or not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    return project_dir


def project_file(project_dir: Path, relative_path: str | None) -> Path:
    if not relative_path:
        raise ProductRegionError("Project file path is required.")
    path = (project_dir / relative_path).resolve()
    if not is_relative_to(path, project_dir.resolve()):
        raise ProductRegionError("Project file path escapes project directory.")
    return path


def project_relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def pad_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    *,
    padding: int | None = None,
) -> list[int]:
    x, y, width, height = bbox
    if padding is None:
        padding = max(2, round(min(image_width, image_height) * 0.015))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + width + padding)
    y2 = min(image_height, y + height + padding)
    return [x1, y1, x2 - x1, y2 - y1]


def iou(first: list[int], second: list[int]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_region_artifact_id(
    *,
    bbox: dict[str, int],
    source_sha256: str,
    crop_sha256: str,
    cutout_sha256: str,
    mask_sha256: str,
    text_mask_sha256: str | None = None,
    algorithm_version: str = ALGORITHM_VERSION,
    mask_semantics: str = MASK_SEMANTICS,
) -> str:
    """Return a stable content-and-lineage identity for a region proposal.

    Candidate ordering, paths, selection timestamps and approval state are
    deliberately excluded.  A force regeneration of identical inputs therefore
    produces the same identity, while a source, geometry or derived-asset change
    produces a different identity.
    """
    identity = {
        "algorithm_version": algorithm_version,
        "bbox": bbox,
        "crop_sha256": crop_sha256,
        "cutout_sha256": cutout_sha256,
        "mask_sha256": mask_sha256,
        "mask_semantics": mask_semantics,
        "source_sha256": source_sha256,
    }
    if text_mask_sha256:
        identity["text_mask_sha256"] = text_mask_sha256
    encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def ensure_entry_lineage(project_dir: Path, item: dict[str, Any], entry: dict[str, Any]) -> None:
    """Backfill content identities for reports created before lineage support.

    This is a compatibility bridge, not a trust shortcut: every digest is
    recomputed from files currently present in the project.  If any required
    file is unavailable, the legacy selection remains unidentified and the
    stale guard fails closed when line art exists.
    """
    source_image = entry.get("source_image") or item.get("source_image")
    try:
        source_path = project_file(project_dir, source_image)
    except ProductRegionError:
        return
    if not source_path.is_file():
        return
    source_sha256 = sha256_file(source_path)

    for candidate in entry.get("candidates", []):
        try:
            crop_path = project_file(project_dir, candidate.get("image_path"))
            cutout_path = project_file(project_dir, candidate.get("cutout_path"))
            mask_path = project_file(project_dir, candidate.get("mask_path"))
        except ProductRegionError:
            continue
        if not all(path.is_file() for path in (crop_path, cutout_path, mask_path)):
            continue
        bbox_value = candidate.get("bbox") or {}
        try:
            bbox = {
                "x": int(bbox_value["x"]),
                "y": int(bbox_value["y"]),
                "width": int(bbox_value["width"]),
                "height": int(bbox_value["height"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        crop_sha256 = sha256_file(crop_path)
        cutout_sha256 = sha256_file(cutout_path)
        mask_sha256 = sha256_file(mask_path)
        text_mask_sha256: str | None = None
        text_mask_relative = candidate.get("text_mask_path")
        if text_mask_relative:
            try:
                text_mask_path = project_file(project_dir, text_mask_relative)
            except ProductRegionError:
                text_mask_path = None
            if text_mask_path is not None and text_mask_path.is_file():
                text_mask_sha256 = sha256_file(text_mask_path)
        candidate.update(
            {
                "source_sha256": source_sha256,
                "crop_sha256": crop_sha256,
                "cutout_sha256": cutout_sha256,
                "mask_sha256": mask_sha256,
                "text_mask_sha256": text_mask_sha256 or candidate.get("text_mask_sha256"),
                "mask_semantics": candidate.get("mask_semantics") or MASK_SEMANTICS,
                "region_artifact_id": make_region_artifact_id(
                    bbox=bbox,
                    source_sha256=source_sha256,
                    crop_sha256=crop_sha256,
                    cutout_sha256=cutout_sha256,
                    mask_sha256=mask_sha256,
                    text_mask_sha256=text_mask_sha256,
                    algorithm_version=str(candidate.get("algorithm_version") or ALGORITHM_VERSION),
                    mask_semantics=str(candidate.get("mask_semantics") or MASK_SEMANTICS),
                ),
            }
        )

    old_selection = entry.get("selection")
    if not isinstance(old_selection, dict):
        return
    candidate_id = old_selection.get("candidate_id")
    selected = next(
        (candidate for candidate in entry.get("candidates", []) if candidate.get("candidate_id") == candidate_id),
        None,
    )
    if not selected or not selected.get("region_artifact_id"):
        return
    pointer = selection_pointer(selected, selected_by=str(old_selection.get("selected_by") or "manual"))
    pointer["selected_at"] = old_selection.get("selected_at") or pointer["selected_at"]
    entry["selection"] = pointer


def region_artifact_changed(old_selection: Any, new_selection: Any) -> bool:
    if not old_selection and not new_selection:
        return False
    if not old_selection or not new_selection:
        return True
    old_artifact_id = old_selection.get("region_artifact_id") if isinstance(old_selection, dict) else None
    new_artifact_id = new_selection.get("region_artifact_id") if isinstance(new_selection, dict) else None
    # A legacy selection without a content identity cannot prove that existing
    # line art belongs to the current region.  Fail closed and make it stale.
    if not old_artifact_id or not new_artifact_id:
        return True
    return old_artifact_id != new_artifact_id


def invalidate_lineart_for_region_change(item: dict[str, Any]) -> None:
    lineart_status = item.get("lineart_status", "missing")
    if lineart_status in {None, "missing"}:
        return
    current_asset = item.get("lineart_asset")
    if current_asset is not None:
        item["previous_lineart_asset"] = current_asset
    elif "previous_lineart_asset" not in item:
        item["previous_lineart_asset"] = None
    item["lineart_status"] = "stale"
    item["lineart_asset"] = None
    item["lineart_provenance"] = None
    item["lineart_qc_status"] = "missing"
    item["lineart_stale_reason"] = "product_region_changed"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
