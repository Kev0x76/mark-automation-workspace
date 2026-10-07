from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from ..ocr.ocr_cli import run_ocr as run_ocr_service

from .barcode_engines import decode_barcodes
from .evidence_extractor import extract_cell_evidence
from .models import BarcodeEvidence, Cell, CellEvidence, ContractItem, TextSpan
from .normalize import normalize_barcode, normalize_item_number, normalize_qty


def write_mark_image_evidence_manifest(
    output_dir: str | Path,
    crop_manifest: dict[str, Any],
    contract_items: list[ContractItem] | None = None,
    max_ocr_images: int = 24,
    max_total_seconds: float = 120.0,
) -> dict[str, Any]:
    _evidences, manifest = extract_mark_image_evidences(
        output_dir,
        crop_manifest,
        contract_items,
        max_ocr_images=max_ocr_images,
        max_total_seconds=max_total_seconds,
    )
    return manifest


def extract_mark_image_evidences(
    output_dir: str | Path,
    crop_manifest: dict[str, Any],
    contract_items: list[ContractItem] | None = None,
    max_ocr_images: int = 24,
    max_total_seconds: float = 120.0,
) -> tuple[list[CellEvidence], dict[str, Any]]:
    """Read final crop images back into OCR/barcode evidence.

    This is an audit stage. It does not change verification results by itself:
    crop images that only have visual/residual evidence remain unverified until
    their OCR/barcode evidence is promoted by a later resolver pass.
    """

    output_dir = Path(output_dir)
    started = time.monotonic()
    events: list[str] = []
    items: list[dict[str, Any]] = []
    ocr_attempts = 0
    promoted_evidences: list[CellEvidence] = []
    contract_by_item = {
        normalize_item_number(item.item_number): item
        for item in (contract_items or [])
        if normalize_item_number(item.item_number)
    }

    for item in crop_manifest.get("items", []):
        if item.get("status") != "OK":
            continue
        image_path = _resolve_image_path(output_dir, item.get("image", ""))
        if image_path is None or not image_path.exists():
            events.append(f"MARK_IMAGE_SKIP:image_missing:{item.get('item_number', '')}")
            continue

        barcode_values: list[str] = []
        ocr_spans: list[dict[str, Any]] = []
        ocr_events: list[str] = []
        image_size = (0, 0)
        try:
            image = Image.open(image_path).convert("RGB")
            image_size = image.size
            barcode_values = _decode_crop_barcodes(image)
        except Exception as exc:
            events.append(f"MARK_IMAGE_BARCODE_ERROR:{item.get('item_number', '')}:{type(exc).__name__}:{exc}")

        if ocr_attempts < max_ocr_images and time.monotonic() - started < max_total_seconds:
            ocr_attempts += 1
            try:
                payload = run_ocr_service(image_path, "rapidocr", 0.35)
                ocr_spans = payload.get("spans", [])
                ocr_events = payload.get("events", [])
            except Exception as exc:
                ocr_events = [f"ocr.rapidocr:ERROR:{type(exc).__name__}:{exc}"]
        else:
            ocr_events = ["ocr.rapidocr:SKIPPED_BY_BUDGET"]

        item_numbers = _extract_item_numbers(ocr_spans)
        packing_qty = _extract_packing_qty(ocr_spans)
        assigned_item = normalize_item_number(str(item.get("item_number", "")))
        item_confirmed = _item_number_confirmed(assigned_item, item_numbers)
        residual_unverified = bool(item.get("crop_quality", {}).get("association_unverified"))
        recognition_status = _recognition_status(item_confirmed, item_numbers, residual_unverified)
        contract_item = contract_by_item.get(assigned_item)
        contract_barcodes = _contract_barcodes(contract_item)
        barcode_status = _barcode_match_status(contract_barcodes, barcode_values)
        qty_status = _qty_match_status(contract_item, packing_qty)
        promotion_status = _promotion_status(
            item_confirmed=item_confirmed,
            item_numbers=item_numbers,
            barcode_status=barcode_status,
            residual_unverified=residual_unverified,
            crop_method=str(item.get("crop_method", "")),
            crop_quality=item.get("crop_quality", {}),
        )

        item_out = {
            "item_number": item.get("item_number", ""),
            "image": str(image_path),
            "crop_method": item.get("crop_method", ""),
            "recognition_status": recognition_status,
            "promotion_status": promotion_status,
            "association_unverified": residual_unverified,
            "ocr_item_numbers": item_numbers,
            "item_match_status": "PASS" if item_confirmed else ("MISMATCH" if item_numbers else "MISSING"),
            "contract_barcodes": sorted(contract_barcodes),
            "barcodes": barcode_values,
            "barcode_match_status": barcode_status,
            "packing_qty": packing_qty,
            "packing_qty_match_status": qty_status,
            "ocr_span_count": len(ocr_spans),
            "ocr_events": ocr_events,
        }
        items.append(item_out)
        if promotion_status in {"PROMOTION_READY_STRONG", "PROMOTION_READY_RESIDUAL_STRONG"} and image_size != (0, 0):
            promoted_evidences.append(
                _promoted_evidence_from_crop(
                    item_out=item_out,
                    image_size=image_size,
                    ocr_spans=ocr_spans,
                    barcode_values=barcode_values,
                )
            )

    promotion_counts = dict(Counter(item["promotion_status"] for item in items))
    recognition_counts = dict(Counter(item["recognition_status"] for item in items))
    manifest = {
        "stage": "mark_image_evidence_audit",
        "item_count": len(items),
        "promoted_evidence_count": len(promoted_evidences),
        "ocr_attempts": ocr_attempts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "recognition_counts": recognition_counts,
        "promotion_counts": promotion_counts,
        "events": events,
        "items": items,
    }
    (output_dir / "mark_image_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return promoted_evidences, manifest


def _resolve_image_path(output_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [path, output_dir / path, output_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return output_dir / path


def _decode_crop_barcodes(image: Image.Image) -> list[str]:
    result = decode_barcodes(image, 0, float(image.width), float(image.height))
    values: list[str] = []
    for evidence in result.evidences:
        value = normalize_barcode(evidence.value)
        if value and value not in values:
            values.append(value)
    return values


def _promoted_evidence_from_crop(
    item_out: dict[str, Any],
    image_size: tuple[int, int],
    ocr_spans: list[dict[str, Any]],
    barcode_values: list[str],
) -> CellEvidence:
    width, height = image_size
    item_number = normalize_item_number(str(item_out.get("item_number", "")))
    image_path = str(item_out.get("image", ""))
    spans = [_span_from_payload(span) for span in ocr_spans]
    spans = [span for span in spans if span is not None]
    barcodes = [
        BarcodeEvidence(
            value=value,
            bbox=None,
            page=0,
            source="scan_mark_image",
            confidence=1.0,
        )
        for value in barcode_values
        if normalize_barcode(value)
    ]
    cell = Cell(
        cell_id=f"mark_image:{item_number}",
        file_path=image_path,
        page=0,
        bbox=(0.0, 0.0, float(width), float(height)),
        spans=spans,
        barcodes=barcodes,
        segmentation_method="mark_image_promoted_crop",
        needs_ai=True,
    )
    evidence = extract_cell_evidence(cell)
    evidence.needs_ai = True
    evidence.evidence_sources["mark_image_promotion_status"] = str(item_out.get("promotion_status", ""))
    _force_promoted_identity(evidence, item_number, barcode_values)
    _mark_promoted_sources(evidence)
    return evidence


def _span_from_payload(payload: dict[str, Any]) -> TextSpan | None:
    text = str(payload.get("text", "")).strip()
    bbox_raw = payload.get("bbox", [])
    if not text or not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        return None
    try:
        bbox = tuple(float(value) for value in bbox_raw)
    except (TypeError, ValueError):
        return None
    confidence = payload.get("confidence", "")
    return TextSpan(
        text=text,
        bbox=bbox,  # type: ignore[arg-type]
        page=0,
        size=max(1.0, bbox[3] - bbox[1]),
        source=f"mark_image_ocr:{confidence}",
    )


def _mark_promoted_sources(evidence: CellEvidence) -> None:
    for key, source in list(evidence.evidence_sources.items()):
        if source == "text_layer":
            evidence.evidence_sources[key] = "mark_image_ocr"
        elif source == "text_layer_digits":
            evidence.evidence_sources[key] = "mark_image_ocr_digits"


def _force_promoted_identity(
    evidence: CellEvidence,
    item_number: str,
    barcode_values: list[str],
) -> None:
    if item_number and item_number not in evidence.item_numbers:
        evidence.item_numbers.insert(0, item_number)
        evidence.evidence_sources["item_number"] = "mark_image_ocr"

    true_barcodes = {normalize_barcode(value) for value in barcode_values if normalize_barcode(value)}
    evidence.barcode_texts = [
        value for value in evidence.barcode_texts
        if not _looks_like_repeated_item_number_noise(value, item_number)
    ]
    evidence.barcodes = [
        value for value in evidence.barcodes
        if normalize_barcode(value) in true_barcodes
        or not _looks_like_repeated_item_number_noise(value, item_number)
    ]
    for value in barcode_values:
        norm = normalize_barcode(value)
        if norm and norm not in evidence.barcodes:
            evidence.barcodes.append(norm)
        if norm and norm not in evidence.barcode_scans:
            evidence.barcode_scans.append(norm)
    if barcode_values:
        evidence.evidence_sources["barcode"] = "scan_mark_image"
        evidence.evidence_sources["barcode_scan"] = "scan_mark_image"


def _looks_like_repeated_item_number_noise(value: str, item_number: str) -> bool:
    digits = normalize_barcode(value)
    item = normalize_item_number(item_number)
    return bool(item and len(digits) >= len(item) * 2 and digits.startswith(item))


def _extract_item_numbers(spans: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for span in spans:
        text = str(span.get("text", ""))
        for match in re.finditer(r"\d{5,8}", text):
            value = normalize_item_number(match.group(0))
            if _looks_like_item_number(value) and value not in values:
                values.append(value)
    return values


def _looks_like_item_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d{5,8}", value))


def _item_number_confirmed(expected: str, observed_values: list[str]) -> bool:
    expected = normalize_item_number(expected)
    if not expected:
        return False
    for observed in observed_values:
        observed = normalize_item_number(observed)
        if observed == expected:
            return True
        if _compatible_item_fragment(expected, observed):
            return True
    return False


def _compatible_item_fragment(expected: str, observed: str) -> bool:
    if len(expected) < 5 or len(observed) < 4:
        return False
    if observed.startswith(expected) and len(observed) <= len(expected) + 2:
        return True
    tail_min = max(4, len(expected) - 2)
    if expected.endswith(observed) and len(observed) >= tail_min:
        return True
    return False


def _extract_packing_qty(spans: list[dict[str, Any]]) -> str:
    joined = " ".join(str(span.get("text", "")) for span in spans)
    match = re.search(r"\b(\d{1,3})\s*(?:PCS|PC|PZ|PZS|UN)\b", joined, re.I)
    return match.group(1) if match else ""


def _contract_barcodes(item: ContractItem | None) -> set[str]:
    if item is None:
        return set()
    return {
        value
        for value in [
            normalize_barcode(item.barcode),
            normalize_barcode(item.outer_barcode),
        ]
        if value
    }


def _barcode_match_status(contract_barcodes: set[str], observed: list[str]) -> str:
    observed_set = {normalize_barcode(value) for value in observed if normalize_barcode(value)}
    if not contract_barcodes:
        return "NO_CONTRACT_BARCODE"
    if not observed_set:
        return "MISSING"
    if contract_barcodes & observed_set:
        extras = observed_set - contract_barcodes
        return "PASS_WITH_EXTRA" if extras else "PASS"
    return "MISMATCH"


def _qty_match_status(item: ContractItem | None, observed_qty: str) -> str:
    expected = normalize_qty(item.packing_qty if item else "")
    observed = normalize_qty(observed_qty)
    if not expected:
        return "NO_CONTRACT_QTY"
    if not observed:
        return "MISSING"
    return "PASS" if expected == observed else "MISMATCH"


def _recognition_status(
    item_confirmed: bool,
    item_numbers: list[str],
    residual_unverified: bool,
) -> str:
    if item_confirmed and not residual_unverified:
        return "OCR_ITEM_CONFIRMED"
    if item_confirmed and residual_unverified:
        return "OCR_ITEM_SEEN_BUT_ASSOCIATION_UNVERIFIED"
    if item_numbers:
        return "OCR_ITEM_MISMATCH"
    return "OCR_ITEM_NOT_FOUND"


def _promotion_status(
    item_confirmed: bool,
    item_numbers: list[str],
    barcode_status: str,
    residual_unverified: bool,
    crop_method: str,
    crop_quality: dict[str, Any],
) -> str:
    if not item_confirmed:
        return "AUDIT_ONLY_ITEM_MISMATCH" if item_numbers else "AUDIT_ONLY_ITEM_MISSING"
    if barcode_status not in {"PASS", "PASS_WITH_EXTRA"}:
        return "AUDIT_ONLY_BARCODE_NOT_CONFIRMED"
    if crop_quality.get("possible_other_items"):
        return "AUDIT_ONLY_CROP_CONTAINS_OTHER_ITEMS"
    if crop_method == "raster_residual_rect_bbox" or residual_unverified:
        return "PROMOTION_READY_RESIDUAL_STRONG"
    if crop_method in {"raster_barcode_rect_bbox", "vector_rect_anchor_bbox"}:
        return "PROMOTION_READY_STRONG"
    return "PROMOTION_CANDIDATE_NEEDS_BOUNDARY_REVIEW"
