from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .lineart_canvas import LineartCanvasError, load_lineart_grayscale, normalize_lineart_bytes


DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "templates"
    / "kmt_style_v1"
    / "template.json"
)
TARGET_DPIS = (96, 150, 300)
GOLDEN_STYLE_PROFILE = {
    "target_dpi": 96,
    "black_ratio_min": 0.04,
    "black_ratio_max": 0.14,
    "combined_survival_ratio_min": 0.65,
    "endpoint_density_max": 0.20,
}


class TargetSizeQcError(ValueError):
    pass


def evaluate_target_size(
    lineart_data: bytes,
    layer_data: dict[str, bytes] | None = None,
    *,
    template_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Evaluate print readability using the template's lineart rectangle.

    The function is deterministic and review-only.  It does not approve a
    candidate and never changes the supplied raster.
    """

    template_path = template_path or DEFAULT_TEMPLATE
    try:
        config = json.loads(template_path.read_text(encoding="utf-8"))
        rect = config["front"]["lineart"]["rect"]
        units = str(config.get("units") or "pt")
        template_id = str(config.get("template_id") or template_path.parent.name)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TargetSizeQcError(f"Could not read template lineart rectangle: {exc}") from exc
    if units != "pt" or not isinstance(rect, list) or len(rect) != 4:
        raise TargetSizeQcError("Template lineart rectangle must contain four point coordinates.")
    try:
        x0, y0, x1, y1 = (float(value) for value in rect)
    except (TypeError, ValueError) as exc:
        raise TargetSizeQcError("Template lineart rectangle is invalid.") from exc
    width_pt = x1 - x0
    height_pt = y1 - y0
    if width_pt <= 0 or height_pt <= 0:
        raise TargetSizeQcError("Template lineart rectangle has no positive area.")

    try:
        combined_image, normalization = normalize_lineart_bytes(lineart_data)
    except LineartCanvasError as exc:
        raise TargetSizeQcError(str(exc)) from exc
    combined = grayscale_to_ink(combined_image)
    source_size = tuple(normalization["source_size"])
    shared_crop_box = tuple(normalization["crop_box"])
    layers: dict[str, np.ndarray] = {}
    for name, data in (layer_data or {}).items():
        if name not in {"primary", "secondary", "texture"}:
            continue
        try:
            layer_image = load_lineart_grayscale(data)
            if layer_image.size == source_size:
                layer_image = layer_image.crop(shared_crop_box)
            else:
                layer_image, _ = normalize_lineart_bytes(data)
            layers[name] = grayscale_to_ink(layer_image)
        except LineartCanvasError as exc:
            raise TargetSizeQcError(str(exc)) from exc
    results: list[dict[str, Any]] = []
    previews: dict[str, bytes] = {}
    failures: list[str] = []
    if normalization["ink_bbox"] is None:
        failures.append("target_lineart_has_no_effective_ink")
    elif normalization["low_effective_resolution"]:
        failures.append("target_lineart_effective_pixels_too_low")
    for dpi in TARGET_DPIS:
        box_width = max(1, round(width_pt / 72.0 * dpi))
        box_height = max(1, round(height_pt / 72.0 * dpi))
        rendered_width, rendered_height = contain_size(
            combined.shape[1], combined.shape[0], box_width, box_height
        )
        target = resize_ink(combined, rendered_width, rendered_height)
        layer_metrics = {
            name: survival_metrics(layer, rendered_width, rendered_height)
            for name, layer in layers.items()
        }
        metrics = image_metrics(target)
        metrics.update(
            {
                "dpi": dpi,
                "box_pixels": [box_width, box_height],
                "rendered_pixels": [rendered_width, rendered_height],
                "combined_survival_ratio": survival_metrics(
                    combined, rendered_width, rendered_height
                )["survival_ratio"],
                "layers": layer_metrics,
            }
        )
        results.append(metrics)
        previews[f"target-{dpi}dpi"] = encode_preview(target, box_width, box_height)

    smallest = results[0]
    primary = smallest["layers"].get("primary")
    secondary = smallest["layers"].get("secondary")
    if smallest["combined_survival_ratio"] < 0.55:
        failures.append("target_combined_structure_does_not_survive")
    if primary and primary["source_ink_pixels"] and primary["survival_ratio"] < 0.65:
        failures.append("target_primary_structure_does_not_survive")
    if secondary and secondary["source_ink_pixels"] and secondary["survival_ratio"] < 0.35:
        failures.append("target_secondary_structure_does_not_survive")
    if smallest["black_ratio"] > 0.32:
        failures.append("target_lineart_merges_into_black_area")
    if smallest["black_ratio"] < 0.003:
        failures.append("target_lineart_is_too_sparse")
    if smallest["endpoint_density_per_skeleton_pixel"] > 0.30:
        failures.append("target_lineart_is_excessively_fragmented")

    return (
        {
            "template_id": template_id,
            "template_path": str(template_path),
            "lineart_rect_pt": [x0, y0, x1, y1],
            "lineart_size_pt": [width_pt, height_pt],
            "canvas_normalization": normalization,
            "checks": results,
            "technical_passed": not failures,
            "failure_reasons": failures,
        },
        previews,
    )


def evaluate_golden_style_alignment(target_size_qc: dict[str, Any]) -> dict[str, Any]:
    """Compare target-size behaviour with broad cross-category golden bands.

    The profile intentionally measures readability characteristics rather than
    subject geometry.  It must not force a bottle, multi-piece set, or future
    product category to resemble either reference's content.
    """

    checks = list(target_size_qc.get("checks") or [])
    target = next(
        (entry for entry in checks if entry.get("dpi") == GOLDEN_STYLE_PROFILE["target_dpi"]),
        None,
    )
    if not target:
        return {
            "passed": False,
            "failure_reasons": ["golden_style_target_metrics_missing"],
            "profile": dict(GOLDEN_STYLE_PROFILE),
        }
    failures: list[str] = []
    black_ratio = float(target.get("black_ratio") or 0.0)
    survival = float(target.get("combined_survival_ratio") or 0.0)
    endpoint_density = float(target.get("endpoint_density_per_skeleton_pixel") or 0.0)
    if black_ratio < GOLDEN_STYLE_PROFILE["black_ratio_min"]:
        failures.append("golden_style_target_too_light")
    if black_ratio > GOLDEN_STYLE_PROFILE["black_ratio_max"]:
        failures.append("golden_style_target_too_heavy")
    if survival < GOLDEN_STYLE_PROFILE["combined_survival_ratio_min"]:
        failures.append("golden_style_structure_does_not_survive")
    if endpoint_density > GOLDEN_STYLE_PROFILE["endpoint_density_max"]:
        failures.append("golden_style_lines_are_too_fragmented")
    return {
        "passed": not failures,
        "failure_reasons": failures,
        "profile": dict(GOLDEN_STYLE_PROFILE),
        "observed": {
            "black_ratio": black_ratio,
            "combined_survival_ratio": survival,
            "endpoint_density_per_skeleton_pixel": endpoint_density,
        },
    }


def decode_ink(data: bytes) -> np.ndarray:
    try:
        image, _ = normalize_lineart_bytes(data)
    except LineartCanvasError as exc:
        raise TargetSizeQcError(f"Could not decode lineart diagnostic: {exc}") from exc
    return grayscale_to_ink(image)


def grayscale_to_ink(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    return (gray < 192).astype(np.uint8) * 255


def contain_size(
    source_width: int, source_height: int, box_width: int, box_height: int
) -> tuple[int, int]:
    scale = min(
        box_width / float(max(source_width, 1)),
        box_height / float(max(source_height, 1)),
    )
    return (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )


def resize_ink(ink: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA)
    return (resized >= 96).astype(np.uint8) * 255


def survival_metrics(ink: np.ndarray, width: int, height: int) -> dict[str, Any]:
    target = resize_ink(ink, width, height)
    restored = cv2.resize(target, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)
    covered = np.count_nonzero((ink > 0) & (cv2.dilate(restored, np.ones((3, 3), np.uint8)) > 0))
    source_pixels = int(np.count_nonzero(ink))
    return {
        "source_ink_pixels": source_pixels,
        "target_ink_pixels": int(np.count_nonzero(target)),
        "survival_ratio": round(covered / float(max(source_pixels, 1)), 4),
    }


def image_metrics(ink: np.ndarray) -> dict[str, Any]:
    line_pixels = int(np.count_nonzero(ink))
    total = int(ink.size)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8), connectivity=8
    )
    areas = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)),
        reverse=True,
    )
    cumulative = 0
    components_85 = 0
    for area in areas:
        cumulative += area
        components_85 += 1
        if cumulative >= line_pixels * 0.85:
            break
    skeleton = thin(ink)
    skeleton_pixels = int(np.count_nonzero(skeleton))
    neighbours = cv2.filter2D(
        (skeleton > 0).astype(np.uint8),
        cv2.CV_16U,
        np.ones((3, 3), dtype=np.uint8),
    ) - (skeleton > 0).astype(np.uint8)
    endpoints = int(np.count_nonzero((skeleton > 0) & (neighbours == 1)))
    distance = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    widths = distance[distance > 0] * 2.0
    return {
        "black_ratio": round(line_pixels / float(max(total, 1)), 4),
        "connected_components": len(areas),
        "components_to_85_percent_ink": components_85,
        "largest_component_ink_share": round(
            (areas[0] if areas else 0) / float(max(line_pixels, 1)), 4
        ),
        "top5_component_ink_share": round(
            sum(areas[:5]) / float(max(line_pixels, 1)), 4
        ),
        "skeleton_pixels": skeleton_pixels,
        "skeleton_endpoints": endpoints,
        "endpoint_density_per_skeleton_pixel": round(
            endpoints / float(max(skeleton_pixels, 1)), 4
        ),
        "median_stroke_width_pixels": round(float(np.median(widths)) if widths.size else 0.0, 3),
    }


def thin(ink: np.ndarray) -> np.ndarray:
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning((ink > 0).astype(np.uint8) * 255)
    work = (ink > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton


def encode_preview(ink: np.ndarray, box_width: int, box_height: int) -> bytes:
    canvas = np.full((box_height, box_width), 255, dtype=np.uint8)
    y = (box_height - ink.shape[0]) // 2
    x = (box_width - ink.shape[1]) // 2
    canvas[y : y + ink.shape[0], x : x + ink.shape[1]][ink > 0] = 0
    payload = BytesIO()
    Image.fromarray(canvas).convert("RGB").save(payload, format="PNG", optimize=True)
    return payload.getvalue()
