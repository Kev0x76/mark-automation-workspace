from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .circuit_breaker import CircuitBreaker
from .models import BarcodeEvidence
from .normalize import normalize_barcode

logger = logging.getLogger(__name__)

try:
    from pyzbar import pyzbar
except Exception:  # pragma: no cover
    pyzbar = None  # type: ignore

try:
    import zxingcpp
except Exception:  # pragma: no cover
    zxingcpp = None  # type: ignore


@dataclass
class BarcodeReadResult:
    evidences: list[BarcodeEvidence] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def decode_barcodes(
    image: Any,
    page_index: int,
    page_width: float,
    page_height: float,
    breaker: CircuitBreaker | None = None,
) -> BarcodeReadResult:
    breaker = breaker or CircuitBreaker()
    result = BarcodeReadResult()
    if image is None:
        return result

    if pyzbar is None:
        breaker.events.append("barcode.pyzbar:UNAVAILABLE")
    if zxingcpp is None:
        breaker.events.append("barcode.zxingcpp:UNAVAILABLE")

    pyzbar_items = breaker.run(
        "barcode.pyzbar",
        lambda: _decode_pyzbar(image, page_index, page_width, page_height),
        [],
    )
    zxing_items = breaker.run(
        "barcode.zxingcpp",
        lambda: _decode_zxing(image, page_index, page_width, page_height),
        [],
    )
    result.evidences = _merge_engine_results(pyzbar_items + zxing_items)
    result.events = list(breaker.events)
    return result


def _decode_pyzbar(
    image: Any,
    page_index: int,
    page_width: float,
    page_height: float,
) -> list[BarcodeEvidence]:
    if pyzbar is None:
        return []
    decoded = pyzbar.decode(image)
    sx = page_width / image.width
    sy = page_height / image.height
    results: list[BarcodeEvidence] = []
    for bc in decoded:
        value = bc.data.decode("utf-8", errors="replace").strip()
        norm = normalize_barcode(value)
        if not norm:
            continue
        rect = getattr(bc, "rect", None)
        bbox = None
        if rect is not None:
            bbox = (
                float(rect.left * sx),
                float(rect.top * sy),
                float((rect.left + rect.width) * sx),
                float((rect.top + rect.height) * sy),
            )
        confidence = float(getattr(bc, "quality", 1) or 1)
        results.append(BarcodeEvidence(value=norm, bbox=bbox, page=page_index, source="scan_pyzbar", confidence=confidence))
    return results


def _decode_zxing(
    image: Any,
    page_index: int,
    page_width: float,
    page_height: float,
) -> list[BarcodeEvidence]:
    if zxingcpp is None:
        return []
    decoded = zxingcpp.read_barcodes(image)
    sx = page_width / image.width
    sy = page_height / image.height
    results: list[BarcodeEvidence] = []
    for bc in decoded:
        norm = normalize_barcode(getattr(bc, "text", ""))
        if not norm:
            continue
        bbox = _zxing_bbox(bc, sx, sy)
        results.append(BarcodeEvidence(value=norm, bbox=bbox, page=page_index, source="scan_zxingcpp", confidence=1.0))
    return results


def _zxing_bbox(bc: Any, sx: float, sy: float) -> tuple[float, float, float, float] | None:
    pos = getattr(bc, "position", None)
    if pos is None:
        return None
    points = []
    for attr in ("top_left", "top_right", "bottom_left", "bottom_right"):
        point = getattr(pos, attr, None)
        if point is not None:
            points.append(point)
    if not points:
        return None
    xs = [float(getattr(p, "x")) * sx for p in points]
    ys = [float(getattr(p, "y")) * sy for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _merge_engine_results(items: list[BarcodeEvidence]) -> list[BarcodeEvidence]:
    merged: dict[str, BarcodeEvidence] = {}
    for item in items:
        key = normalize_barcode(item.value)
        if not key:
            continue
        current = merged.get(key)
        if current is None:
            merged[key] = item
            continue
        current.source = _merge_sources(current.source, item.source)
        current.confidence = max(current.confidence, item.confidence)
        if _bbox_area(current.bbox) <= 0 and item.bbox is not None:
            current.bbox = item.bbox
    return list(merged.values())


def _merge_sources(a: str, b: str) -> str:
    parts = []
    for value in (a, b):
        for part in value.split("+"):
            if part and part not in parts:
                parts.append(part)
    return "+".join(parts)


def _bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
