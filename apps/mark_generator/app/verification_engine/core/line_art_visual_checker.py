from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from .models import ContractItem, FieldCheck

try:
    import cv2
except Exception:  # pragma: no cover - environment without OpenCV
    cv2 = None


# Production policy: no network vision model.  When both product evidence and
# lineart exist, run a cheap deterministic fuzzy shape check so the lineart is
# at least roughly the same product family/object silhouette as the photo.
PROMPT_VERSION = "line-art-fuzzy-shape-v1"
VISUAL_OC_ENABLED = False
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_INK_RATIO = 0.003
# Fuzzy thresholds tuned on real bottle/frame matches vs cross-product swaps.
PASS_SCORE = 0.28
REVIEW_SCORE = 0.18
COMPARE_SIZE = 192


def build_line_art_fields(
    items: list[ContractItem],
    evidence_dir: Path | None,
    output_dir: Path,
    *,
    ai_reviewer: Callable[[ContractItem, Path, Path], dict[str, Any]] | None = None,
) -> dict[str, FieldCheck]:
    """Build lineart field checks without calling a vision model.

    ``ai_reviewer`` remains accepted for source compatibility and is ignored.
    Blank/unreadable lineart still fails closed.  When a product photo/cutout is
    present, a fuzzy silhouette/edge comparison must show the lineart is roughly
    the same object; clear mismatches become MISMATCH, borderline scores become
    NEEDS_REVIEW.  Lineart without product evidence keeps the non-blank integrity
    check only (manual no-photo path).
    """

    evidence = discover_evidence(evidence_dir)
    fields: dict[str, FieldCheck] = {}
    audit_items: list[dict[str, Any]] = []
    for item in items:
        pair = evidence.get(item.item_number, {})
        source = pair.get("source")
        lineart = pair.get("lineart")
        audit: dict[str, Any] = {
            "item_number": item.item_number,
            "source": artifact(source),
            "lineart": artifact(lineart),
            "prompt_version": PROMPT_VERSION,
            "visual_oc_enabled": VISUAL_OC_ENABLED,
            "reviewer_callback_supplied": ai_reviewer is not None,
            "fuzzy_shape_enabled": True,
        }

        if not lineart:
            audit.update(
                {
                    "verdict": "SKIPPED_NO_LINEART",
                    "reason": "no lineart artifact supplied for fuzzy/integrity checks",
                }
            )
            audit_items.append(audit)
            continue

        try:
            metrics = deterministic_metrics(lineart)
        except Exception as exc:
            field = FieldCheck(
                "line_art",
                "线稿文件",
                "不可读",
                "NEEDS_REVIEW",
                0.0,
                f"线稿文件不可读，未执行模糊外形核对：{type(exc).__name__}",
                "line_art_deterministic_qc",
            )
            fields[item.item_number] = field
            audit.update({"verdict": "UNREADABLE_LINEART", "error": type(exc).__name__})
            audit_items.append(audit)
            continue

        audit["deterministic_metrics"] = metrics
        if metrics["ink_ratio"] < MIN_INK_RATIO:
            field = FieldCheck(
                "line_art",
                "线稿内容",
                "空白/近空白",
                "MISMATCH",
                0.0,
                "线稿为空或接近空白，无法让客户识别产品",
                "line_art_deterministic_qc",
            )
            fields[item.item_number] = field
            audit["verdict"] = "BLANK_LINEART"
            audit_items.append(audit)
            continue

        if source is None:
            field = FieldCheck(
                "line_art",
                "无产品图",
                f"{metrics['width']}x{metrics['height']} 非空线稿",
                "PASS",
                1.0,
                "无产品图可对照，仅完成线稿可读/非空检查（缺图手工路径）；未做外形模糊核对",
                "line_art_nonblank_no_source",
            )
            fields[item.item_number] = field
            audit.update(
                {
                    "verdict": "NONBLANK_NO_SOURCE",
                    "reason": "no product image/cutout evidence; fuzzy shape skipped",
                }
            )
            audit_items.append(audit)
            continue

        try:
            comparison = fuzzy_product_lineart_match(source, lineart)
        except Exception as exc:
            field = FieldCheck(
                "line_art",
                "产品外形",
                f"比对失败 {type(exc).__name__}",
                "NEEDS_REVIEW",
                0.0,
                f"线稿模糊外形核对失败，需人工确认：{type(exc).__name__}",
                "line_art_fuzzy_shape",
            )
            fields[item.item_number] = field
            audit.update({"verdict": "FUZZY_COMPARE_ERROR", "error": type(exc).__name__})
            audit_items.append(audit)
            continue

        audit["fuzzy_shape"] = comparison
        score = float(comparison.get("score") or 0.0)
        if score >= PASS_SCORE:
            field = FieldCheck(
                "line_art",
                "产品外形",
                f"模糊分 {score:.2f}",
                "PASS",
                score,
                f"线稿与产品外形大致一致（模糊分 {score:.2f}）",
                "line_art_fuzzy_shape",
            )
            verdict = "FUZZY_MATCH"
        elif score >= REVIEW_SCORE:
            field = FieldCheck(
                "line_art",
                "产品外形",
                f"模糊分 {score:.2f}",
                "NEEDS_REVIEW",
                score,
                f"线稿与产品外形仅部分相似（模糊分 {score:.2f}），需人工确认是否同一产品",
                "line_art_fuzzy_shape",
            )
            verdict = "FUZZY_BORDERLINE"
        else:
            field = FieldCheck(
                "line_art",
                "产品外形",
                f"模糊分 {score:.2f}",
                "MISMATCH",
                score,
                f"线稿与产品外形差异过大（模糊分 {score:.2f}），疑似不是同一产品",
                "line_art_fuzzy_shape",
            )
            verdict = "FUZZY_MISMATCH"

        fields[item.item_number] = field
        audit.update({"verdict": verdict, "reason": field.note})
        audit_items.append(audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "line_art_verification.json").write_text(
        json.dumps(
            {
                "schema_version": "1.2.0",
                "ai_enabled": False,
                "visual_oc_enabled": VISUAL_OC_ENABLED,
                "trust_policy": "fuzzy_shape_when_source_available",
                "prompt_version": PROMPT_VERSION,
                "thresholds": {"pass": PASS_SCORE, "review": REVIEW_SCORE},
                "items": audit_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return fields


def discover_evidence(evidence_dir: Path | None) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    if not evidence_dir or not evidence_dir.is_dir():
        return result
    for path in evidence_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS or "__" not in path.stem:
            continue
        item_no, role = path.stem.split("__", 1)
        if item_no and role in {"source", "lineart"}:
            result.setdefault(item_no, {})[role] = path
    return result


def deterministic_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
        histogram = gray.histogram()
        total = max(sum(histogram), 1)
        ink_ratio = sum(histogram[:220]) / total
        return {
            "width": gray.width,
            "height": gray.height,
            "aspect_ratio": round(gray.width / float(max(gray.height, 1)), 4),
            "ink_ratio": round(ink_ratio, 4),
        }


def fuzzy_product_lineart_match(source: Path, lineart: Path) -> dict[str, Any]:
    """Cheap silhouette/edge similarity: not pixel-perfect, catches wrong product."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required for fuzzy lineart/product comparison")

    source_gray = _load_square_gray(source)
    lineart_gray = _load_square_gray(lineart)
    source_edges = _edges(source_gray)
    lineart_edges = _edges(lineart_gray)

    corr = _normalized_cross_correlation(source_edges, lineart_edges)
    aspect = _aspect_similarity(source, lineart)
    shape = _contour_shape_similarity(source_edges, lineart_edges)
    score = max(0.0, min(1.0, 0.55 * max(corr, 0.0) + 0.25 * aspect + 0.20 * shape))

    return {
        "score": round(float(score), 4),
        "edge_correlation": round(float(corr), 4),
        "aspect_similarity": round(float(aspect), 4),
        "contour_shape_similarity": round(float(shape), 4),
        "compare_size": COMPARE_SIZE,
        "method": "edge_ncc_aspect_matchshapes_v1",
    }


def _load_square_gray(path: Path, size: int = COMPARE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.array(ImageOps.grayscale(ImageOps.exif_transpose(image)), dtype=np.uint8)
    height, width = gray.shape
    side = max(height, width, 1)
    fill = 255 if float(gray.mean()) > 127 else 0
    canvas = np.full((side, side), fill, dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = gray
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(edges, kernel)


def _normalized_cross_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float32)
    b = right.astype(np.float32)
    a = (a - float(a.mean())) / (float(a.std()) + 1e-6)
    b = (b - float(b.mean())) / (float(b.std()) + 1e-6)
    return float((a * b).mean())


def _ink_aspect_ratio(path: Path, *, lineart: bool) -> float:
    with Image.open(path) as image:
        gray = np.array(ImageOps.grayscale(ImageOps.exif_transpose(image)), dtype=np.uint8)
    if lineart:
        mask = gray < 220
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = np.logical_or(edges > 0, threshold > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return width / float(max(height, 1))


def _aspect_similarity(source: Path, lineart: Path) -> float:
    left = _ink_aspect_ratio(source, lineart=False)
    right = _ink_aspect_ratio(lineart, lineart=True)
    if left <= 0 or right <= 0:
        return 0.0
    return float(min(left, right) / max(left, right))


def _contour_shape_similarity(left_edges: np.ndarray, right_edges: np.ndarray) -> float:
    left_contours, _ = cv2.findContours(left_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours, _ = cv2.findContours(right_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not left_contours or not right_contours:
        return 0.0
    left = max(left_contours, key=cv2.contourArea)
    right = max(right_contours, key=cv2.contourArea)
    if cv2.contourArea(left) < 20 or cv2.contourArea(right) < 20:
        return 0.0
    distance = float(cv2.matchShapes(left, right, cv2.CONTOURS_MATCH_I1, 0.0))
    return float(1.0 / (1.0 + distance))


def ai_enabled() -> bool:
    """Legacy compatibility flag; network visual OC remains disabled."""

    return False


def review_with_optional_ai(item: ContractItem, source: Path, lineart: Path) -> dict[str, Any]:
    """Legacy entry point that must never perform network I/O."""

    del item
    try:
        comparison = fuzzy_product_lineart_match(source, lineart)
    except Exception as exc:
        return {
            "verdict": "DISABLED_NETWORK_OC",
            "reason": f"network visual OC disabled; fuzzy compare failed: {type(exc).__name__}",
            "visual_oc_enabled": False,
        }
    score = float(comparison.get("score") or 0.0)
    if score >= PASS_SCORE:
        verdict = "PASS"
    elif score >= REVIEW_SCORE:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "MISMATCH"
    return {
        "verdict": verdict,
        "reason": "network visual OC disabled; deterministic fuzzy shape used instead",
        "visual_oc_enabled": False,
        "fuzzy_shape": comparison,
    }


def data_url(path: Path) -> str:
    """Retained for compatibility; no production path calls this helper."""

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def artifact(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.is_file():
        return None
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
