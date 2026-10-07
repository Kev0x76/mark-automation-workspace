from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .barcode_engines import decode_barcodes
from .circuit_breaker import CircuitBreaker
from .models import BarcodeEvidence, PageData, TextSpan
from .normalize import normalize_barcode
from .ocr_engines import run_ocr, should_run_ocr

logger = logging.getLogger(__name__)

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def read_pdf(path: str | Path, render_image_pages: bool = True) -> list[PageData]:
    path = Path(path)
    if fitz is None:
        raise RuntimeError("PyMuPDF 未安装，无法读取 PDF")

    doc = fitz.open(str(path))
    pages: list[PageData] = []
    breaker = CircuitBreaker(max_seconds=45, max_errors=3)
    ocr_breaker = CircuitBreaker(max_seconds=60, max_errors=3)
    try:
        for page_index, page in enumerate(doc):
            spans = _extract_spans(page, page_index)
            text_lines = _extract_text_lines(page)
            has_text = _has_real_text_layer(spans, text_lines)
            page_data = PageData(
                file_path=str(path),
                page=page_index,
                width=float(page.rect.width),
                height=float(page.rect.height),
                has_text_layer=has_text,
                spans=spans,
                text_lines=text_lines,
            )

            text_barcodes = _find_barcodes_in_text(spans) if has_text else []
            if has_text:
                for value, bbox in text_barcodes:
                    page_data.barcodes.append(
                        BarcodeEvidence(value=value, bbox=bbox, page=page_index, source="text_layer_digits")
                    )

            if render_image_pages:
                # Text-layer Kmt PDFs still need a page raster for barcode scan,
                # but 2.0x (~144 DPI) is enough; keep 2.5x only for image-like pages.
                render_zoom = 2.0 if has_text else 2.5
                img = _render_page(page, zoom=render_zoom)
                page_data.render_image = img
                if should_run_ocr(has_text, spans):
                    ocr_result = run_ocr(img, page_index, page_data.width, page_data.height, ocr_breaker)
                    if ocr_result.spans:
                        page_data.spans.extend(ocr_result.spans)
                        page_data.text_lines.extend([s.text for s in ocr_result.spans])
                        page_data.has_text_layer = True
                        has_text = True
                        text_barcodes = _find_barcodes_in_text(page_data.spans)
                        for value, bbox in text_barcodes:
                            page_data.barcodes.append(
                                BarcodeEvidence(value=value, bbox=bbox, page=page_index, source="ocr_text_digits")
                            )
                    page_data.errors.extend(ocr_result.events)
                scan_result = decode_barcodes(img, page_index, page_data.width, page_data.height, breaker)
                scanned = scan_result.evidences
                _align_scanned_bboxes_to_text(scanned, text_barcodes, page_data.width, page_data.height)
                page_data.barcodes.extend(scanned)
                page_data.errors.extend(scan_result.events)

            pages.append(page_data)
    finally:
        doc.close()
    return pages


def _extract_spans(page: Any, page_index: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    try:
        data = page.get_text("dict")
    except Exception:
        return spans
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if not text:
                    continue
                bbox_raw = span.get("bbox", [0, 0, 0, 0])
                bbox = tuple(float(x) for x in bbox_raw)
                spans.append(TextSpan(text=text, bbox=bbox, page=page_index, size=float(span.get("size", 0))))
    return spans


def _extract_text_lines(page: Any) -> list[str]:
    try:
        text = page.get_text("text")
    except Exception:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _has_real_text_layer(spans: list[TextSpan], lines: list[str]) -> bool:
    alpha_or_cjk = 0
    digit = 0
    for span in spans:
        alpha_or_cjk += len(re.findall(r"[A-Za-z\u4e00-\u9fff]", span.text))
        digit += len(re.findall(r"\d", span.text))
    return len(lines) >= 3 and (alpha_or_cjk + digit) >= 20


def _find_barcodes_in_text(spans: list[TextSpan]) -> list[tuple[str, tuple[float, float, float, float] | None]]:
    results: list[tuple[str, tuple[float, float, float, float] | None]] = []
    for group in _merge_numeric_runs(spans):
        digits = _canonical_barcode_text(normalize_barcode("".join(s.text for s in group)))
        if 12 <= len(digits) <= 14:
            bbox = (
                min(s.x0 for s in group),
                min(s.y0 for s in group),
                max(s.x1 for s in group),
                max(s.y1 for s in group),
            )
            if digits not in [v for v, _ in results]:
                results.append((digits, bbox))
    return results


def _canonical_barcode_text(digits: str) -> str:
    if len(digits) == 14 and _valid_ean13(digits[:13]):
        return digits[:13]
    return digits


def _valid_ean13(digits: str) -> bool:
    digits = normalize_barcode(digits)
    if len(digits) != 13:
        return False
    total = sum(int(ch) * (1 if idx % 2 == 0 else 3) for idx, ch in enumerate(digits[:12]))
    check = (10 - (total % 10)) % 10
    return check == int(digits[-1])


def _align_scanned_bboxes_to_text(
    scanned: list[BarcodeEvidence],
    text_barcodes: list[tuple[str, tuple[float, float, float, float] | None]],
    page_width: float,
    page_height: float,
) -> None:
    by_value: dict[str, list[tuple[float, float, float, float]]] = {}
    for value, bbox in text_barcodes:
        if bbox is None:
            continue
        by_value.setdefault(value, []).append(bbox)

    page_area = max(page_width * page_height, 1)
    for scan in scanned:
        matches = by_value.get(scan.value)
        if not matches:
            continue
        if scan.bbox is None:
            scan.bbox = matches[0]
            continue
        x0, y0, x1, y1 = scan.bbox
        area_ratio = max(0, x1 - x0) * max(0, y1 - y0) / page_area
        if area_ratio > 0.02 or (y1 - y0) > page_height * 0.35 or (x1 - x0) > page_width * 0.35:
            scan.bbox = _nearest_bbox(scan.bbox, matches)


def _nearest_bbox(
    bbox: tuple[float, float, float, float],
    candidates: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return min(
        candidates,
        key=lambda b: ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2,
    )


def _merge_numeric_runs(spans: list[TextSpan]) -> list[list[TextSpan]]:
    numeric = [s for s in spans if re.fullmatch(r"[\d\s]+", s.text)]
    numeric.sort(key=lambda s: (s.page, round(s.y0 / 8), s.x0))
    rows: list[list[TextSpan]] = []
    for span in numeric:
        placed = False
        for row in rows:
            if abs(row[0].y0 - span.y0) <= max(8, span.size * 0.8):
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    groups: list[list[TextSpan]] = []
    for row in rows:
        row.sort(key=lambda s: s.x0)
        cur: list[TextSpan] = []
        for span in row:
            if not cur:
                cur = [span]
                continue
            gap = span.x0 - cur[-1].x1
            if gap <= max(28, span.size * 3.5):
                cur.append(span)
            else:
                if cur:
                    groups.append(cur)
                cur = [span]
        if cur:
            groups.append(cur)
    return groups


def _render_page(page: Any, zoom: float = 2.5) -> Any:
    if Image is None:
        return None
    mat = fitz.Matrix(zoom, zoom)  # type: ignore[union-attr]
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
