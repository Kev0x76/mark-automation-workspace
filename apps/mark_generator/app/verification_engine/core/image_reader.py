from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..ocr.ocr_cli import run_ocr as run_ocr_service

from .barcode_engines import decode_barcodes
from .circuit_breaker import CircuitBreaker
from .models import BarcodeEvidence, PageData, TextSpan
from .normalize import normalize_text

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_MAX_WORK_LONG_EDGE = 9000
_MAX_OCR_RECTS = 80
_MAX_OCR_SECONDS = 120.0


def read_image(path: str | Path, output_dir: str | Path | None = None) -> list[PageData]:
    """Read a direct image mark file as a one-page document.

    The page coordinate system is the normalized working image itself. That
    keeps downstream segmentation, cropping, barcode bboxes, and OCR bboxes in
    one coordinate space instead of pretending an image is a PDF.
    """

    source_path = Path(path)
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise RuntimeError(f"Unsupported image file type: {source_path.suffix}")

    image_dir = Path(output_dir or source_path.parent) / "image_pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    work_path, original_size, scale_note = _write_work_image(source_path, image_dir)
    image = Image.open(work_path).convert("RGB")
    rects = _detect_label_rects(work_path)
    spans, barcodes, evidence_events = _extract_rect_evidence(
        image=image,
        rects=rects,
        cell_dir=image_dir / f"{_safe_stem(source_path.stem)}_cells",
    )

    events = [
        f"IMAGE_INPUT:{source_path.name}",
        f"IMAGE_ORIGINAL_SIZE:{original_size[0]}x{original_size[1]}",
        f"IMAGE_WORK_SIZE:{image.width}x{image.height}:{scale_note}",
        f"IMAGE_RASTER_RECTS:{len(rects)}",
        *evidence_events,
    ]

    return [
        PageData(
            file_path=str(work_path),
            page=0,
            width=float(image.width),
            height=float(image.height),
            has_text_layer=bool(spans),
            spans=spans,
            text_lines=[span.text for span in spans],
            barcodes=barcodes,
            render_image=image,
            errors=events,
            visual_rects=[tuple(float(v) for v in rect) for rect in rects],
        )
    ]


def _write_work_image(source_path: Path, image_dir: Path) -> tuple[Path, tuple[int, int], str]:
    original_w, original_h = _image_size(source_path)
    work_path = image_dir / f"{_safe_stem(source_path.stem)}_page_1.png"

    if cv2 is not None and np is not None:
        data = np.fromfile(str(source_path), dtype=np.uint8)
        flag, reduction_note = _cv2_read_flag(max(original_w, original_h))
        decoded = cv2.imdecode(data, flag)
        if decoded is None:
            raise RuntimeError(f"Cannot decode image: {source_path}")
        rgb = _cv_to_rgb(decoded)
        rgb, resize_note = _resize_array_to_work_size(rgb, _MAX_WORK_LONG_EDGE)
        Image.fromarray(rgb).save(work_path)
        note = reduction_note if not resize_note else f"{reduction_note}+{resize_note}"
        return work_path, (original_w, original_h), note

    Image.MAX_IMAGE_PIXELS = None
    image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
    image.thumbnail((_MAX_WORK_LONG_EDGE, _MAX_WORK_LONG_EDGE), Image.Resampling.LANCZOS)
    image.save(work_path)
    return work_path, (original_w, original_h), "pil_thumbnail"


def _image_size(source_path: Path) -> tuple[int, int]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source_path) as image:
        return int(image.width), int(image.height)


def _cv2_read_flag(long_edge: int) -> tuple[int, str]:
    if cv2 is None:
        return 1, "pil"
    if long_edge > 36000:
        return int(cv2.IMREAD_REDUCED_COLOR_8), "cv2_reduced_8"
    if long_edge > 18000:
        return int(cv2.IMREAD_REDUCED_COLOR_4), "cv2_reduced_4"
    if long_edge > _MAX_WORK_LONG_EDGE:
        return int(cv2.IMREAD_REDUCED_COLOR_2), "cv2_reduced_2"
    return int(cv2.IMREAD_COLOR), "cv2_full"


def _cv_to_rgb(decoded: Any) -> Any:
    if cv2 is None:
        return decoded
    if len(decoded.shape) == 3 and decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    if len(decoded.shape) == 3:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)


def _resize_array_to_work_size(rgb: Any, max_long_edge: int) -> tuple[Any, str]:
    if cv2 is None:
        return rgb, ""
    height, width = rgb.shape[:2]
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return rgb, ""
    scale = max_long_edge / float(long_edge)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    return resized, f"resize_{new_size[0]}x{new_size[1]}"


def _detect_label_rects(image_path: Path) -> list[tuple[int, int, int, int]]:
    if cv2 is None or np is None:
        return []
    try:
        gray = np.array(Image.open(image_path).convert("L"))
    except Exception:
        return []

    height, width = gray.shape[:2]
    if width <= 0 or height <= 0:
        return []

    try:
        binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)[1]
        h_len = max(60, width // 80)
        v_len = max(60, height // 70)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        mask = cv2.bitwise_or(horizontal, vertical)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        contours, _hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return []

    page_area = max(width * height, 1)
    rects: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if (
            w >= width * 0.08
            and h >= height * 0.05
            and area >= page_area * 0.004
            and area <= page_area * 0.20
        ):
            rects.append((int(x), int(y), int(x + w), int(y + h)))
    return _sort_reading_order(_dedupe_pixel_bboxes(rects))


def _extract_rect_evidence(
    image: Image.Image,
    rects: list[tuple[int, int, int, int]],
    cell_dir: Path,
) -> tuple[list[TextSpan], list[BarcodeEvidence], list[str]]:
    if not rects:
        return [], [], ["IMAGE_RECT_EVIDENCE_SKIPPED:no_rects"]

    cell_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    barcode_breaker = CircuitBreaker(max_seconds=10.0, max_errors=4)
    spans: list[TextSpan] = []
    barcodes: list[BarcodeEvidence] = []
    events: list[str] = []
    ocr_attempts = 0
    ocr_success = 0
    barcode_cells = 0
    ocr_problem_events: list[str] = []

    for index, rect in enumerate(rects, start=1):
        if ocr_attempts >= _MAX_OCR_RECTS or time.monotonic() - started > _MAX_OCR_SECONDS:
            events.append(f"IMAGE_RECT_OCR_STOPPED:index_{index}:budget")
            break
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image.crop(rect).convert("RGB")
        crop_path = cell_dir / f"cell_{index:03d}.png"
        crop.save(crop_path)

        barcode_result = decode_barcodes(
            image=crop,
            page_index=0,
            page_width=float(crop.width),
            page_height=float(crop.height),
            breaker=barcode_breaker,
        )
        if barcode_result.evidences:
            barcode_cells += 1
        for evidence in barcode_result.evidences:
            barcodes.append(_offset_barcode(evidence, rect))

        ocr_attempts += 1
        try:
            payload = run_ocr_service(crop_path, "rapidocr", 0.35)
            cell_spans = [_offset_span(span, rect) for span in payload.get("spans", [])]
            cell_spans = [span for span in cell_spans if span is not None]
            spans.extend(cell_spans)
            if cell_spans:
                ocr_success += 1
            elif len(ocr_problem_events) < 10:
                ocr_problem_events.extend(str(event) for event in payload.get("events", []) if "NO_TEXT" in str(event))
        except Exception as exc:
            if len(ocr_problem_events) < 10:
                ocr_problem_events.append(f"IMAGE_RECT_OCR_ERROR:cell_{index}:{type(exc).__name__}:{exc}")

    events.append(f"IMAGE_RECT_OCR:{ocr_success}/{ocr_attempts}")
    events.append(f"IMAGE_RECT_BARCODE_CELLS:{barcode_cells}/{len(rects)}")
    events.extend(_unique_events(barcode_breaker.events)[:12])
    events.extend(ocr_problem_events[:10])
    return spans, barcodes, events


def _offset_span(payload: dict[str, Any], rect: tuple[int, int, int, int]) -> TextSpan | None:
    text = normalize_text(payload.get("text", ""))
    bbox_raw = payload.get("bbox", [])
    if not text or not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        return None
    try:
        bx0, by0, bx1, by1 = [float(value) for value in bbox_raw]
    except (TypeError, ValueError):
        return None
    x0, y0, _x1, _y1 = rect
    confidence = payload.get("confidence", "")
    source = payload.get("source", "ocr_service")
    return TextSpan(
        text=text,
        bbox=(x0 + bx0, y0 + by0, x0 + bx1, y0 + by1),
        page=0,
        size=max(1.0, by1 - by0),
        source=f"{source}:{confidence}",
    )


def _offset_barcode(
    evidence: BarcodeEvidence,
    rect: tuple[int, int, int, int],
) -> BarcodeEvidence:
    x0, y0, x1, y1 = rect
    bbox = evidence.bbox
    if bbox is None:
        mapped = (float(x0), float(y0), float(x1), float(y1))
    else:
        mapped = (
            float(x0 + bbox[0]),
            float(y0 + bbox[1]),
            float(x0 + bbox[2]),
            float(y0 + bbox[3]),
        )
    return BarcodeEvidence(
        value=evidence.value,
        bbox=mapped,
        page=0,
        source=evidence.source,
        confidence=evidence.confidence,
    )


def _dedupe_pixel_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen = set()
    deduped = []
    for bbox in sorted(bboxes, key=lambda item: (item[1], item[0], item[2], item[3])):
        key = tuple(round(value / 4) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bbox)
    return deduped


def _sort_reading_order(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not rects:
        return rects
    median_h = sorted((rect[3] - rect[1]) for rect in rects)[len(rects) // 2]
    tolerance = max(10, median_h * 0.5)
    rows: list[list[tuple[int, int, int, int]]] = []
    for rect in sorted(rects, key=lambda item: item[1]):
        if rows and abs(rows[-1][0][1] - rect[1]) <= tolerance:
            rows[-1].append(rect)
        else:
            rows.append([rect])
    ordered: list[tuple[int, int, int, int]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item[0]))
    return ordered


def _unique_events(events: list[str]) -> list[str]:
    return list(dict.fromkeys(str(event) for event in events if event))


def _safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "image"
