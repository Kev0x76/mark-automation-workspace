from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from .barcode_engines import decode_barcodes
from .circuit_breaker import CircuitBreaker
from .models import CellEvidence, VerificationResult
from .normalize import normalize_barcode

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def crop_mark_images(
    output_dir: str | Path,
    evidences: list[CellEvidence],
    results: list[VerificationResult],
    zoom: float = 2.0,
    padding_ratio: float = 0.05,
) -> dict[str, Any]:
    """Crop one full mark image for each associated item.

    Renderable pages are saved to PNG once, then cell bboxes are cropped with
    padding. PDF pages use the requested zoom; direct images use 1:1 pixels.
    """
    output_dir = Path(output_dir)
    asset_dir = output_dir / "mark_images"
    asset_dir.mkdir(parents=True, exist_ok=True)

    evidence_by_cell = {e.cell.cell_id: e for e in evidences}
    rendered_pages: dict[tuple[str, int], tuple[Path, float, float, int, int]] = {}
    ruling_rects_by_page: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    raster_rects_by_page: dict[tuple[str, int], list[tuple[int, int, int, int]]] = {}
    raster_barcode_rects_by_page: dict[tuple[str, int], dict[str, list[tuple[int, int, int, int]]]] = {}
    fallback_page = _first_pdf_page_preview(evidences, rendered_pages, asset_dir, zoom)
    item_anchors_by_page = _page_item_anchors(evidence_by_cell, results)
    items: list[dict[str, Any]] = []
    events: list[str] = []

    for result in results:
        if not result.cell_id:
            if fallback_page:
                items.append(_fallback_manifest_item(result, fallback_page))
            else:
                items.append(_manifest_item(result, "", "SKIPPED_NO_CELL", ""))
            continue
        evidence = evidence_by_cell.get(result.cell_id)
        if not evidence:
            items.append(_manifest_item(result, "", "SKIPPED_NO_EVIDENCE", ""))
            continue

        source_path = Path(evidence.cell.file_path)
        if not _is_renderable_source(source_path):
            items.append(_manifest_item(result, str(source_path), "SKIPPED_UNSUPPORTED_SOURCE", ""))
            continue

        try:
            page_key = (str(source_path), evidence.cell.page)
            if page_key not in rendered_pages:
                rendered_pages[page_key] = _render_page(source_path, evidence.cell.page, asset_dir, zoom)
            if page_key not in ruling_rects_by_page:
                ruling_rects_by_page[page_key] = _page_ruling_rects(source_path, evidence.cell.page)
            page_image, pdf_w, pdf_h, image_w, image_h = rendered_pages[page_key]
            page_zoom = _source_zoom(source_path, zoom)
            if page_key not in raster_rects_by_page:
                raster_rects_by_page[page_key] = _page_raster_rects(page_image)
            if page_key not in raster_barcode_rects_by_page:
                raster_barcode_rects_by_page[page_key] = _page_raster_barcode_rects(
                    page_image,
                    raster_rects_by_page[page_key],
                )
            crop_bbox, crop_basis, crop_meta = _evidence_crop_bbox(evidence, result.item_number)
            crop_path, box, crop_quality = _crop_cell(
                page_image=page_image,
                item_number=result.item_number,
                bbox=crop_bbox,
                pdf_w=pdf_w,
                pdf_h=pdf_h,
                image_w=image_w,
                image_h=image_h,
                zoom=page_zoom,
                padding_ratio=padding_ratio,
                asset_dir=asset_dir,
                focus_anchor_bboxes=crop_meta.get("focus_bboxes", crop_meta.get("item_anchor_bboxes", [])),
                target_barcodes=_evidence_barcode_values(evidence),
                page_item_anchors=item_anchors_by_page.get(page_key, []),
                page_ruling_rects=ruling_rects_by_page.get(page_key, []),
                page_raster_rects=raster_rects_by_page.get(page_key, []),
                page_raster_barcode_rects=raster_barcode_rects_by_page.get(page_key, {}),
            )
            crop_method = "evidence_union_bbox"
            if crop_quality.get("vector_rects_used"):
                crop_method = "vector_rect_anchor_bbox"
            elif crop_quality.get("raster_barcode_rects_used"):
                crop_method = "raster_barcode_rect_bbox"
            elif crop_quality.get("raster_rects_used"):
                crop_method = "raster_rect_anchor_bbox"
            elif any(x.startswith("item_anchors:") for x in crop_basis):
                crop_method = "item_anchor_evidence_bbox"
            items.append({
                "item_number": result.item_number,
                "cell_id": result.cell_id,
                "source_pdf": str(source_path),
                "page": evidence.cell.page + 1,
                "status": "OK",
                "image": str(crop_path),
                "bbox_px": box,
                "bbox_pdf": [round(float(v), 3) for v in crop_bbox],
                "crop_method": crop_method,
                "crop_basis": crop_basis,
                "crop_quality": crop_quality,
            })
        except Exception as exc:
            message = f"{result.item_number}:CROP_FAILED:{type(exc).__name__}:{exc}"
            events.append(message)
            items.append(_manifest_item(result, str(source_path), "CROP_FAILED", message))

    manifest = {
        "asset_dir": str(asset_dir),
        "items": _apply_residual_raster_crops(items, rendered_pages, raster_rects_by_page, asset_dir),
        "events": events,
    }
    (output_dir / "mark_images_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _apply_residual_raster_crops(
    items: list[dict[str, Any]],
    rendered_pages: dict[tuple[str, int], tuple[Path, float, float, int, int]],
    raster_rects_by_page: dict[tuple[str, int], list[tuple[int, int, int, int]]],
    asset_dir: Path,
) -> list[dict[str, Any]]:
    fallback_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    used_by_page: dict[tuple[str, int], list[tuple[int, int, int, int]]] = {}

    for item in items:
        page_key = _manifest_page_key(item)
        if page_key is None:
            continue
        status = item.get("status")
        if status == "PAGE_FALLBACK":
            fallback_by_page.setdefault(page_key, []).append(item)
            continue
        if status != "OK":
            continue
        box = _manifest_bbox(item)
        if box is not None:
            used_by_page.setdefault(page_key, []).append(box)

    for page_key, fallback_items in fallback_by_page.items():
        page_rects = raster_rects_by_page.get(page_key, [])
        if len(fallback_items) != 1 or not page_rects:
            continue
        used_rects = _dedupe_pixel_bboxes(used_by_page.get(page_key, []))
        residual_rects = [
            rect for rect in page_rects
            if not any(_pixel_bboxes_match(rect, used) for used in used_rects)
        ]
        if len(residual_rects) != 1:
            continue
        rendered = rendered_pages.get(page_key)
        if not rendered:
            continue
        page_image, _pdf_w, _pdf_h, image_w, image_h = rendered
        fallback_item = fallback_items[0]
        rect = residual_rects[0]
        try:
            image = Image.open(page_image)
            crop_path = asset_dir / f"mark_{_safe_stem(str(fallback_item.get('item_number', 'residual')))}.png"
            image.crop(rect).save(crop_path)
        except Exception:
            continue
        area_ratio = _pixel_bbox_area(rect) / max(image_w * image_h, 1)
        fallback_item.update({
            "status": "OK",
            "image": str(crop_path),
            "bbox_px": rect,
            "crop_method": "raster_residual_rect_bbox",
            "crop_basis": ["unique_unassigned_raster_rect"],
            "crop_quality": {
                "touches_page_edge": rect[0] <= 0 or rect[1] <= 0 or rect[2] >= image_w or rect[3] >= image_h,
                "area_ratio": round(area_ratio, 6),
                "raster_rects_available": len(page_rects),
                "raster_residual_rects_used": True,
                "raster_residual_rects_selected": 1,
                "association_unverified": True,
            },
            "message": "唯一剩余栅格外框裁剪候选；条码/OCR未确认，不代表核对通过。",
        })

    return items


def _manifest_page_key(item: dict[str, Any]) -> tuple[str, int] | None:
    source = item.get("source_pdf")
    page = item.get("page")
    if not source or page in {"", None}:
        return None
    try:
        return (str(Path(source)), int(page) - 1)
    except Exception:
        return None


def _manifest_bbox(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = item.get("bbox_px")
    if not bbox or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(value) for value in bbox]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _pixel_bboxes_match(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    tolerance: int = 3,
) -> bool:
    return all(abs(a[index] - b[index]) <= tolerance for index in range(4))


def _first_pdf_page_preview(
    evidences: list[CellEvidence],
    rendered_pages: dict[tuple[str, int], tuple[Path, float, float, int, int]],
    asset_dir: Path,
    zoom: float,
) -> dict[str, Any] | None:
    for evidence in evidences:
        source_path = Path(evidence.cell.file_path)
        if not _is_renderable_source(source_path):
            continue
        page_key = (str(source_path), evidence.cell.page)
        try:
            if page_key not in rendered_pages:
                rendered_pages[page_key] = _render_page(source_path, evidence.cell.page, asset_dir, zoom)
            page_image, _pdf_w, _pdf_h, image_w, image_h = rendered_pages[page_key]
            return {
                "source_pdf": str(source_path),
                "page": evidence.cell.page + 1,
                "image": str(page_image),
                "bbox_px": (0, 0, image_w, image_h),
            }
        except Exception:
            continue
    return None


def _fallback_manifest_item(result: VerificationResult, fallback_page: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_number": result.item_number,
        "cell_id": "",
        "source_pdf": fallback_page.get("source_pdf", ""),
        "page": fallback_page.get("page", ""),
        "status": "PAGE_FALLBACK",
        "image": fallback_page.get("image", ""),
        "bbox_px": fallback_page.get("bbox_px", ""),
        "message": "未定位到该货号的单个唛头，当前显示整页预览供人工查看。",
    }


def _render_page(
    source_path: Path,
    page_index: int,
    asset_dir: Path,
    zoom: float,
) -> tuple[Path, float, float, int, int]:
    if _is_image_source(source_path):
        image = Image.open(source_path).convert("RGB")
        stem = _safe_stem(source_path.stem)
        page_path = asset_dir / f"{stem}_page_{page_index + 1}.png"
        image.save(page_path)
        return page_path, float(image.width), float(image.height), int(image.width), int(image.height)

    doc = fitz.open(source_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        stem = _safe_stem(source_path.stem)
        page_path = asset_dir / f"{stem}_page_{page_index + 1}.png"
        pix.save(page_path)
        return page_path, float(page.rect.width), float(page.rect.height), int(pix.width), int(pix.height)
    finally:
        doc.close()


def _is_renderable_source(path: Path) -> bool:
    return path.suffix.lower() == ".pdf" or _is_image_source(path)


def _is_image_source(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _source_zoom(path: Path, zoom: float) -> float:
    return 1.0 if _is_image_source(path) else zoom


def _page_ruling_rects(pdf_path: Path, page_index: int) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    if pdf_path.suffix.lower() != ".pdf":
        return rects
    try:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_index]
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            for drawing in page.get_drawings():
                draw_type = str(drawing.get("type", ""))
                if "s" not in draw_type:
                    continue
                for item in drawing.get("items", []):
                    if not item or item[0] != "re":
                        continue
                    rect = item[1]
                    bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                    if _looks_like_label_rect(bbox, page_w, page_h):
                        rects.append(bbox)
        finally:
            doc.close()
    except Exception:
        return []
    return _dedupe_bboxes(rects)


def _page_raster_rects(page_image: Path) -> list[tuple[int, int, int, int]]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return []

    try:
        gray = np.array(Image.open(page_image).convert("L"))
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
    return _dedupe_pixel_bboxes(rects)


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


def _page_raster_barcode_rects(
    page_image: Path,
    page_raster_rects: list[tuple[int, int, int, int]],
) -> dict[str, list[tuple[int, int, int, int]]]:
    if not page_raster_rects:
        return {}

    try:
        image = Image.open(page_image).convert("RGB")
    except Exception:
        return {}

    breaker = CircuitBreaker(max_seconds=8.0, max_errors=4)
    by_value: dict[str, list[tuple[int, int, int, int]]] = {}
    for rect in page_raster_rects:
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        try:
            crop = image.crop(rect)
            result = decode_barcodes(crop, 0, float(crop.width), float(crop.height), breaker)
        except Exception:
            continue
        for evidence in result.evidences:
            value = normalize_barcode(evidence.value)
            if not value:
                continue
            values = by_value.setdefault(value, [])
            if rect not in values:
                values.append(rect)
    return by_value


def _looks_like_label_rect(
    bbox: tuple[float, float, float, float],
    page_w: float,
    page_h: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return False
    page_area = max(page_w * page_h, 1.0)
    area = width * height
    return (
        width >= max(70.0, page_w * 0.08)
        and height >= max(70.0, page_h * 0.05)
        and area >= page_area * 0.006
        and area <= page_area * 0.08
    )


def _dedupe_bboxes(bboxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    seen = set()
    deduped = []
    for bbox in sorted(bboxes, key=lambda item: (item[1], item[0], item[2], item[3])):
        key = tuple(round(value, 1) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bbox)
    return deduped


def _crop_cell(
    page_image: Path,
    item_number: str,
    bbox: tuple[float, float, float, float],
    pdf_w: float,
    pdf_h: float,
    image_w: int,
    image_h: int,
    zoom: float,
    padding_ratio: float,
    asset_dir: Path,
    focus_anchor_bboxes: list[tuple[float, float, float, float]] | None = None,
    target_barcodes: list[str] | None = None,
    page_item_anchors: list[dict[str, Any]] | None = None,
    page_ruling_rects: list[tuple[float, float, float, float]] | None = None,
    page_raster_rects: list[tuple[int, int, int, int]] | None = None,
    page_raster_barcode_rects: dict[str, list[tuple[int, int, int, int]]] | None = None,
) -> tuple[Path, tuple[int, int, int, int], dict[str, Any]]:
    x0, y0, x1, y1 = [float(x) for x in bbox]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid bbox: {bbox}")

    pad_x = (x1 - x0) * padding_ratio
    pad_y = (y1 - y0) * padding_ratio
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(pdf_w, x1 + pad_x)
    y1 = min(pdf_h, y1 + pad_y)
    evidence_box = (
        max(0, min(image_w, round(x0 * zoom))),
        max(0, min(image_h, round(y0 * zoom))),
        max(0, min(image_w, round(x1 * zoom))),
        max(0, min(image_h, round(y1 * zoom))),
    )
    if evidence_box[2] <= evidence_box[0] or evidence_box[3] <= evidence_box[1]:
        raise ValueError(f"invalid pixel box: {evidence_box}")

    image = Image.open(page_image)
    focus_anchors_px = [_pdf_bbox_to_px(bbox, zoom, image_w, image_h) for bbox in (focus_anchor_bboxes or [])]
    vector_box, vector_quality = _crop_box_from_vector_rects(
        focus_anchor_bboxes or [],
        page_ruling_rects or [],
        zoom,
        image_w,
        image_h,
    )
    raster_barcode_box, raster_barcode_quality = _crop_box_from_raster_barcode_values(
        target_barcodes or [],
        page_raster_barcode_rects or {},
    )
    raster_box, raster_quality = _crop_box_from_raster_rects(
        focus_anchors_px,
        page_raster_rects or [],
    )
    base_box = vector_box or raster_barcode_box or raster_box or evidence_box
    page_anchors_px = [
        {"item_number": item.get("item_number", ""), "bbox": _pdf_bbox_to_px(item["bbox"], zoom, image_w, image_h)}
        for item in (page_item_anchors or [])
        if item.get("bbox")
    ]
    box, optimization_quality = _optimize_crop_box(
        image=image,
        base_box=base_box,
        item_number=item_number,
        focus_anchors=focus_anchors_px,
        page_item_anchors=page_anchors_px,
        prefer_fixed_box=vector_box is not None or raster_barcode_box is not None or raster_box is not None,
    )
    crop = image.crop(box)
    crop_path = asset_dir / f"mark_{_safe_stem(item_number)}.png"
    crop.save(crop_path)
    final_area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(image_w * image_h, 1)
    quality = {
        "touches_page_edge": box[0] <= 0 or box[1] <= 0 or box[2] >= image_w or box[3] >= image_h,
        "area_ratio": round(final_area_ratio, 6),
        "possible_other_items": bool(optimization_quality.get("foreign_item_anchors")),
        **vector_quality,
        **raster_barcode_quality,
        **raster_quality,
        **optimization_quality,
    }
    return crop_path, box, quality


def _crop_box_from_vector_rects(
    focus_anchor_bboxes: list[tuple[float, float, float, float]],
    page_ruling_rects: list[tuple[float, float, float, float]],
    zoom: float,
    image_w: int,
    image_h: int,
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    selected: list[tuple[float, float, float, float]] = []
    for anchor in focus_anchor_bboxes:
        cx = (anchor[0] + anchor[2]) / 2
        cy = (anchor[1] + anchor[3]) / 2
        candidates = [
            rect for rect in page_ruling_rects
            if _rect_contains_point(rect, cx, cy, tolerance=6.0)
        ]
        if not candidates:
            continue
        selected.append(min(candidates, key=_bbox_area))

    selected = _dedupe_bboxes(selected)
    if not selected:
        return None, {
            "vector_rects_available": len(page_ruling_rects),
            "vector_rects_used": False,
            "vector_rects_selected": 0,
        }

    union = (
        min(rect[0] for rect in selected),
        min(rect[1] for rect in selected),
        max(rect[2] for rect in selected),
        max(rect[3] for rect in selected),
    )
    box = _pdf_bbox_to_px(union, zoom, image_w, image_h)
    return box, {
        "vector_rects_available": len(page_ruling_rects),
        "vector_rects_used": True,
        "vector_rects_selected": len(selected),
        "vector_rects_bbox_pdf": [round(value, 3) for value in union],
    }


def _crop_box_from_raster_barcode_values(
    target_barcodes: list[str],
    page_raster_barcode_rects: dict[str, list[tuple[int, int, int, int]]],
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    selected: list[tuple[int, int, int, int]] = []
    matched_values: list[str] = []
    for raw_value in target_barcodes:
        value = normalize_barcode(raw_value)
        if not value:
            continue
        rects = page_raster_barcode_rects.get(value, [])
        if not rects:
            continue
        matched_values.append(value)
        selected.extend(rects)

    selected = _dedupe_pixel_bboxes(selected)
    if not selected:
        return None, {
            "raster_barcode_rects_available": len(page_raster_barcode_rects),
            "raster_barcode_rects_used": False,
            "raster_barcode_rects_selected": 0,
        }

    union = (
        min(rect[0] for rect in selected),
        min(rect[1] for rect in selected),
        max(rect[2] for rect in selected),
        max(rect[3] for rect in selected),
    )
    return union, {
        "raster_barcode_rects_available": len(page_raster_barcode_rects),
        "raster_barcode_rects_used": True,
        "raster_barcode_rects_selected": len(selected),
        "raster_barcode_values_matched": sorted(dict.fromkeys(matched_values)),
        "raster_barcode_rects_bbox_px": list(union),
    }


def _crop_box_from_raster_rects(
    focus_bboxes_px: list[tuple[int, int, int, int]],
    page_raster_rects: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    selected: list[tuple[int, int, int, int]] = []
    for focus in focus_bboxes_px:
        cx = (focus[0] + focus[2]) / 2
        cy = (focus[1] + focus[3]) / 2
        candidates = [
            rect for rect in page_raster_rects
            if _pixel_rect_contains_point(rect, cx, cy, tolerance=8)
        ]
        if not candidates:
            continue
        selected.append(min(candidates, key=_pixel_bbox_area))

    selected = _dedupe_pixel_bboxes(selected)
    if not selected:
        return None, {
            "raster_rects_available": len(page_raster_rects),
            "raster_rects_used": False,
            "raster_rects_selected": 0,
        }

    union = (
        min(rect[0] for rect in selected),
        min(rect[1] for rect in selected),
        max(rect[2] for rect in selected),
        max(rect[3] for rect in selected),
    )
    return union, {
        "raster_rects_available": len(page_raster_rects),
        "raster_rects_used": True,
        "raster_rects_selected": len(selected),
        "raster_rects_bbox_px": list(union),
    }


def _pixel_rect_contains_point(
    rect: tuple[int, int, int, int],
    x: float,
    y: float,
    tolerance: int = 0,
) -> bool:
    return (
        rect[0] - tolerance <= x <= rect[2] + tolerance
        and rect[1] - tolerance <= y <= rect[3] + tolerance
    )


def _pixel_bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _rect_contains_point(
    rect: tuple[float, float, float, float],
    x: float,
    y: float,
    tolerance: float = 0.0,
) -> bool:
    return (
        rect[0] - tolerance <= x <= rect[2] + tolerance
        and rect[1] - tolerance <= y <= rect[3] + tolerance
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _optimize_crop_box(
    image: Image.Image,
    base_box: tuple[int, int, int, int],
    item_number: str,
    focus_anchors: list[tuple[int, int, int, int]],
    page_item_anchors: list[dict[str, Any]],
    max_iterations: int = 4,
    prefer_fixed_box: bool = False,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    box = base_box
    stop_reason = "max_iterations"

    for iteration in range(max_iterations):
        candidate = _score_crop_candidate(image, box, item_number, focus_anchors, page_item_anchors)
        candidate["iteration"] = iteration
        candidates.append(candidate)
        if prefer_fixed_box:
            stop_reason = "vector_rect_candidate"
            break
        if iteration > 0 and candidate["score"] >= 92 and not candidate["foreign_item_anchors"]:
            stop_reason = "quality_target_met"
            break

        next_box, step_quality = _next_crop_candidate(image, box, iteration)
        if next_box == box:
            stop_reason = step_quality.get("stop_reason", "no_change")
            break
        box = next_box

    selected = max(candidates, key=_crop_candidate_rank)
    selected_box = selected["box"]
    return selected_box, {
        "crop_loop": {
            "iterations": len(candidates),
            "stop_reason": stop_reason,
            "selected_iteration": selected["iteration"],
            "selected_score": round(selected["score"], 3),
            "candidates": [
                {
                    "iteration": item["iteration"],
                    "score": round(item["score"], 3),
                    "box": list(item["box"]),
                    "foreign_item_anchors": item["foreign_item_anchors"],
                    "area_ratio": round(item["area_ratio"], 6),
                    "focus_missing": item["focus_missing"],
                    "line_support": item["line_support"],
                    "edge_cut_risk": item["edge_cut_risk"],
                }
                for item in candidates
            ],
        },
        "foreign_item_anchors": selected["foreign_item_anchors"],
        "line_support": selected["line_support"],
        "edge_cut_risk": selected["edge_cut_risk"],
    }


def _crop_candidate_rank(item: dict[str, Any]) -> tuple[bool, float, int, float, float]:
    complete_without_neighbor = (
        item["focus_missing"] == 0
        and not item["foreign_item_anchors"]
        and item["area_ratio"] <= 0.08
    )
    line_support_count = sum(1 for value in item["line_support"].values() if value)
    oversize_penalty = max(0.0, float(item["area_ratio"]) - 0.08)
    return (
        complete_without_neighbor,
        float(item["score"]),
        line_support_count,
        -float(item["edge_cut_risk"]),
        -oversize_penalty,
    )


def _next_crop_candidate(
    image: Image.Image,
    box: tuple[int, int, int, int],
    iteration: int,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    if iteration == 0:
        return _expand_box_to_whitespace(image, box)
    if iteration == 1:
        return _snap_box_to_ruling_lines(image, box)
    if iteration == 2:
        snapped, _line_quality = _snap_box_to_ruling_lines(image, box)
        return _expand_box_to_whitespace(image, snapped)
    return box, {"stop_reason": "candidate_loop_exhausted"}


def _score_crop_candidate(
    image: Image.Image,
    box: tuple[int, int, int, int],
    item_number: str,
    focus_anchors: list[tuple[int, int, int, int]],
    page_item_anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    width, height = image.size
    area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(width * height, 1)
    focus_missing = [anchor for anchor in focus_anchors if not _box_contains_center(box, anchor)]
    foreign_items = sorted({
        item.get("item_number", "")
        for item in page_item_anchors
        if item.get("item_number") != item_number and _box_contains_center(box, item["bbox"])
    })
    line_support = _ruling_line_support(image, box)
    edge_cut_risk = _edge_cut_risk(image, box, line_support)

    score = 100.0
    score -= len(focus_missing) * 35.0
    score -= len(foreign_items) * 45.0
    score -= max(0.0, area_ratio - 0.08) * 220.0
    score -= edge_cut_risk * 18.0
    score += sum(1 for value in line_support.values() if value) * 1.5

    return {
        "box": box,
        "score": score,
        "area_ratio": area_ratio,
        "focus_missing": len(focus_missing),
        "foreign_item_anchors": foreign_items,
        "line_support": line_support,
        "edge_cut_risk": round(edge_cut_risk, 3),
    }


def _expand_box_to_whitespace(
    image: Image.Image,
    box: tuple[int, int, int, int],
    max_expand_ratio: float = 0.08,
    min_band_px: int = 18,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    x0, y0, x1, y1 = box
    max_expand_y = max(24, round(height * max_expand_ratio))

    if _edge_has_ink(pixels, width, height, x0, x1, y0, vertical=False):
        new_y0, top_found = _scan_whitespace_boundary(
            pixels, width, height, x0, x1, y0, -1, max_expand_y, min_band_px
        )
    else:
        new_y0, top_found = y0, False

    if _edge_has_ink(pixels, width, height, x0, x1, y1 - 1, vertical=False):
        new_y1, bottom_found = _scan_whitespace_boundary(
            pixels, width, height, x0, x1, y1, 1, max_expand_y, min_band_px
        )
    else:
        new_y1, bottom_found = y1, False

    new_x0, left_found = x0, False
    new_x1, right_found = x1, False

    expanded = (
        max(0, min(width, new_x0)),
        max(0, min(height, new_y0)),
        max(0, min(width, new_x1)),
        max(0, min(height, new_y1)),
    )
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        return box, {"whitespace_expand_applied": False}

    return expanded, {
        "whitespace_expand_applied": expanded != box,
        "whitespace_boundaries_found": {
            "top": top_found,
            "bottom": bottom_found,
            "left": left_found,
            "right": right_found,
        },
    }


def _scan_whitespace_boundary(
    pixels,
    width: int,
    height: int,
    span0: int,
    span1: int,
    start: int,
    direction: int,
    max_expand: int,
    min_band_px: int,
    vertical: bool = False,
) -> tuple[int, bool]:
    pos = max(0, min(width if vertical else height, start))
    limit = max(0, min(width if vertical else height, start + direction * max_expand))
    band = 0

    while 0 <= pos < (width if vertical else height):
        if direction > 0 and pos >= limit:
            break
        if direction < 0 and pos <= limit:
            break

        ink_ratio = _line_ink_ratio(pixels, width, height, span0, span1, pos, vertical)
        if ink_ratio < 0.006:
            band += 1
            if band >= min_band_px:
                boundary = pos - direction * (min_band_px - 1)
                return (boundary if direction < 0 else boundary + min_band_px), True
        else:
            band = 0
        pos += direction

    return max(0, min(width if vertical else height, start)), False


def _line_ink_ratio(
    pixels,
    width: int,
    height: int,
    span0: int,
    span1: int,
    pos: int,
    vertical: bool,
) -> float:
    low = max(0, min(height if vertical else width, span0))
    high = max(0, min(height if vertical else width, span1))
    if high <= low:
        return 0.0
    ink = 0
    total = high - low
    if vertical:
        x = max(0, min(width - 1, pos))
        for y in range(low, high):
            if pixels[x, y] < 235:
                ink += 1
    else:
        y = max(0, min(height - 1, pos))
        for x in range(low, high):
            if pixels[x, y] < 235:
                ink += 1
    return ink / max(total, 1)


def _edge_has_ink(
    pixels,
    width: int,
    height: int,
    span0: int,
    span1: int,
    pos: int,
    vertical: bool,
    window_px: int = 4,
) -> bool:
    axis_max = width if vertical else height
    start = max(0, min(axis_max - 1, pos))
    lo = max(0, start - window_px)
    hi = min(axis_max - 1, start + window_px)
    for edge_pos in range(lo, hi + 1):
        if _line_ink_ratio(pixels, width, height, span0, span1, edge_pos, vertical) >= 0.006:
            return True
    return False


def _snap_box_to_ruling_lines(
    image: Image.Image,
    box: tuple[int, int, int, int],
    search_ratio: float = 0.08,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    x0, y0, x1, y1 = box
    search_x = max(24, round(width * search_ratio))
    search_y = max(24, round(height * search_ratio))

    top, top_found = _nearest_ruling_line(pixels, width, height, x0, x1, y0, -1, search_y, vertical=False)
    bottom, bottom_found = _nearest_ruling_line(pixels, width, height, x0, x1, y1, 1, search_y, vertical=False)
    left, left_found = _nearest_ruling_line(pixels, width, height, y0, y1, x0, -1, search_x, vertical=True)
    right, right_found = _nearest_ruling_line(pixels, width, height, y0, y1, x1, 1, search_x, vertical=True)

    snapped = (
        max(0, min(width, left if left_found else x0)),
        max(0, min(height, top if top_found else y0)),
        max(0, min(width, right if right_found else x1)),
        max(0, min(height, bottom if bottom_found else y1)),
    )
    if snapped[2] <= snapped[0] or snapped[3] <= snapped[1]:
        return box, {"stop_reason": "invalid_ruling_snap"}
    return snapped, {
        "ruling_snap_applied": snapped != box,
        "ruling_lines_found": {
            "top": top_found,
            "bottom": bottom_found,
            "left": left_found,
            "right": right_found,
        },
    }


def _nearest_ruling_line(
    pixels,
    width: int,
    height: int,
    span0: int,
    span1: int,
    start: int,
    direction: int,
    max_distance: int,
    vertical: bool,
) -> tuple[int, bool]:
    axis_max = width if vertical else height
    pos = max(0, min(axis_max - 1, start))
    limit = max(0, min(axis_max - 1, start + direction * max_distance))
    best_pos = pos
    best_ratio = 0.0

    while 0 <= pos < axis_max:
        if direction > 0 and pos > limit:
            break
        if direction < 0 and pos < limit:
            break
        ratio = _line_ink_ratio(pixels, width, height, span0, span1, pos, vertical)
        if ratio > best_ratio:
            best_pos = pos
            best_ratio = ratio
        pos += direction

    return best_pos, best_ratio >= 0.18


def _ruling_line_support(image: Image.Image, box: tuple[int, int, int, int]) -> dict[str, bool]:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    x0, y0, x1, y1 = box
    return {
        "top": _nearby_max_ink_ratio(pixels, width, height, x0, x1, y0, vertical=False) >= 0.18,
        "bottom": _nearby_max_ink_ratio(pixels, width, height, x0, x1, y1 - 1, vertical=False) >= 0.18,
        "left": _nearby_max_ink_ratio(pixels, width, height, y0, y1, x0, vertical=True) >= 0.18,
        "right": _nearby_max_ink_ratio(pixels, width, height, y0, y1, x1 - 1, vertical=True) >= 0.18,
    }


def _nearby_max_ink_ratio(
    pixels,
    width: int,
    height: int,
    span0: int,
    span1: int,
    pos: int,
    vertical: bool,
    radius: int = 3,
) -> float:
    axis_max = width if vertical else height
    center = max(0, min(axis_max - 1, pos))
    return max(
        _line_ink_ratio(pixels, width, height, span0, span1, value, vertical)
        for value in range(max(0, center - radius), min(axis_max - 1, center + radius) + 1)
    )


def _edge_cut_risk(
    image: Image.Image,
    box: tuple[int, int, int, int],
    line_support: dict[str, bool],
) -> float:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    x0, y0, x1, y1 = box
    checks = [
        ("top", _nearby_max_ink_ratio(pixels, width, height, x0, x1, y0, vertical=False)),
        ("bottom", _nearby_max_ink_ratio(pixels, width, height, x0, x1, y1 - 1, vertical=False)),
        ("left", _nearby_max_ink_ratio(pixels, width, height, y0, y1, x0, vertical=True)),
        ("right", _nearby_max_ink_ratio(pixels, width, height, y0, y1, x1 - 1, vertical=True)),
    ]
    risk = 0.0
    for side, ratio in checks:
        if ratio >= 0.006 and not line_support.get(side, False):
            risk += min(1.0, ratio * 12.0)
    return min(risk, 4.0)


def _pdf_bbox_to_px(
    bbox: tuple[float, float, float, float],
    zoom: float,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (
        max(0, min(image_w, round(x0 * zoom))),
        max(0, min(image_h, round(y0 * zoom))),
        max(0, min(image_w, round(x1 * zoom))),
        max(0, min(image_h, round(y1 * zoom))),
    )


def _box_contains_center(
    box: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _page_item_anchors(
    evidence_by_cell: dict[str, CellEvidence],
    results: list[VerificationResult],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for result in results:
        evidence = evidence_by_cell.get(result.cell_id)
        if not evidence:
            continue
        page_key = (str(Path(evidence.cell.file_path)), evidence.cell.page)
        for bbox in _item_anchor_bboxes(evidence, result.item_number):
            by_page.setdefault(page_key, []).append({
                "item_number": result.item_number,
                "bbox": bbox,
            })
    return by_page


def _evidence_barcode_values(evidence: CellEvidence) -> list[str]:
    values: list[str] = []
    for raw_value in [*evidence.barcodes, *evidence.barcode_scans, *evidence.barcode_texts]:
        value = normalize_barcode(raw_value)
        if value and value not in values:
            values.append(value)
    for barcode in evidence.cell.barcodes:
        value = normalize_barcode(barcode.value)
        if value and value not in values:
            values.append(value)
    return values


def _evidence_crop_bbox(evidence: CellEvidence, item_number: str) -> tuple[tuple[float, float, float, float], list[str], dict[str, Any]]:
    bboxes: list[tuple[float, float, float, float]] = []
    basis: list[str] = []
    anchor_bboxes = _item_anchor_bboxes(evidence, item_number)

    if anchor_bboxes:
        bboxes.extend(anchor_bboxes)
        basis.append(f"item_anchors:{len(anchor_bboxes)}")
        anchor_top = min(b[1] for b in anchor_bboxes)
        min_content_y = anchor_top - 25.0
    else:
        min_content_y = None

    span_count = 0
    for span in evidence.cell.spans:
        if span.page != evidence.cell.page:
            continue
        if min_content_y is not None and span.y0 < min_content_y:
            continue
        if _append_valid_bbox(bboxes, span.bbox):
            span_count += 1
    if span_count:
        basis.append(f"spans:{span_count}")

    barcode_count = 0
    for barcode in evidence.cell.barcodes:
        if not barcode.bbox or barcode.page != evidence.cell.page:
            continue
        if min_content_y is not None and float(barcode.bbox[1]) < min_content_y:
            continue
        if _append_valid_bbox(bboxes, barcode.bbox):
            barcode_count += 1
    if barcode_count:
        basis.append(f"barcodes:{barcode_count}")

    if not bboxes and _append_valid_bbox(bboxes, evidence.cell.bbox):
        basis.append("cell_fallback")

    if not bboxes:
        raise ValueError("no valid crop bbox")
    focus_bboxes = anchor_bboxes or list(bboxes)

    return (
        (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ),
        basis,
        {
            "item_anchor_bboxes": anchor_bboxes,
            "focus_bboxes": focus_bboxes,
        },
    )


def _item_anchor_bboxes(evidence: CellEvidence, item_number: str) -> list[tuple[float, float, float, float]]:
    target = "".join(ch for ch in item_number if ch.isalnum()).upper()
    if not target:
        return []

    rows: list[list[Any]] = []
    candidates = [
        span for span in evidence.cell.spans
        if span.page == evidence.cell.page and _anchor_fragment(span.text)
    ]
    for span in sorted(candidates, key=lambda item: (item.y0, item.x0)):
        placed = False
        tolerance = max(12.0, float(span.size or 0) * 0.35)
        for row in rows:
            if abs(row[0].y0 - span.y0) <= tolerance:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    anchors: list[tuple[float, float, float, float]] = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        for start in range(len(row)):
            text = ""
            group = []
            last_x1 = None
            for span in row[start:]:
                if last_x1 is not None:
                    gap = span.x0 - last_x1
                    threshold = max(10.0, min(45.0, float(span.size or 0) * 0.35))
                    if gap > threshold:
                        break
                text += "".join(ch for ch in span.text if ch.isalnum()).upper()
                group.append(span)
                last_x1 = span.x1
                if text == target:
                    anchors.append((
                        min(item.x0 for item in group),
                        min(item.y0 for item in group),
                        max(item.x1 for item in group),
                        max(item.y1 for item in group),
                    ))
                    break
                if len(text) >= len(target):
                    break
    return anchors


def _anchor_fragment(text: str) -> bool:
    compact = "".join(ch for ch in text if ch.isalnum()).upper()
    if not compact:
        return False
    return compact.isdigit() or compact == text.strip().replace(" ", "").upper()


def _append_valid_bbox(
    bboxes: list[tuple[float, float, float, float]],
    bbox: tuple[float, float, float, float],
) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return False
    bboxes.append((x0, y0, x1, y1))
    return True


def _manifest_item(result: VerificationResult, source: str, status: str, message: str) -> dict[str, Any]:
    return {
        "item_number": result.item_number,
        "cell_id": result.cell_id,
        "source_pdf": source,
        "page": "",
        "status": status,
        "image": "",
        "message": message,
    }


def _safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "mark"
