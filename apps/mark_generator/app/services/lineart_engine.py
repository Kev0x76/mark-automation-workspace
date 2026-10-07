"""Deterministic, non-generative line-art MVP.

The engine has two routes:

* existing line-art / near-monochrome input: normalize the existing strokes;
* continuous-tone product input: combine a mask-derived silhouette with edges
  that are directly supported by source-image gradients.

It never synthesizes product parts.  Quality checks are evidence for review
routing only; ``auto_qc_passed`` is not approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


ALGORITHM_VERSION = "opencv-lineart-v2-mvp"
V3_ALGORITHM_VERSION = "opencv-lineart-v3.2-mvp"
MAX_LINEART_SIZE = 1600


@dataclass(frozen=True)
class LineartEngineResult:
    data: bytes
    mode: str
    algorithm_version: str
    quality: dict[str, Any]
    warnings: list[dict[str, Any]]
    diagnostics: dict[str, bytes] = field(default_factory=dict)


def generate_lineart_file(region_path: Path, mask_path: Path | None = None) -> LineartEngineResult:
    mask_data = mask_path.read_bytes() if mask_path and mask_path.exists() else None
    return generate_lineart_bytes(region_path.read_bytes(), mask_data=mask_data)


def generate_lineart_bytes(data: bytes, *, mask_data: bytes | None = None) -> LineartEngineResult:
    image = load_rgb(data)
    image = resize_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mode, input_metrics = classify_input(gray, hsv)
    supplied_mask = load_mask(mask_data, gray.shape) if mask_data else None

    if mode == "existing_lineart":
        lines, source_support, object_mask, route_metrics = existing_lineart_route(gray, supplied_mask)
    else:
        lines, source_support, object_mask, route_metrics = photo_edge_route(image, gray, supplied_mask)

    quality = quality_metrics(lines, source_support, object_mask, mode)
    quality["input"] = input_metrics
    quality["route"] = route_metrics
    quality["algorithm_version"] = ALGORITHM_VERSION
    warnings: list[dict[str, Any]] = []
    if quality["auto_qc_passed"]:
        warnings.append(
            {
                "severity": "info",
                "code": "lineart_auto_qc_passed",
                "message": "Deterministic lineart passed MVP quality checks; human approval is still required.",
            }
        )
    else:
        warnings.append(
            {
                "severity": "needs_review",
                "code": "lineart_auto_qc_failed",
                "message": "Lineart failed one or more deterministic quality checks.",
                "reasons": quality["failure_reasons"],
            }
        )
    if quality.get("low_resolution"):
        warnings.append(
            {
                "severity": "warning",
                "code": "lineart_low_resolution",
                "message": "Selected region has limited source pixels; no upsampling can recover missing detail.",
            }
        )

    output = np.full(gray.shape, 255, dtype=np.uint8)
    output[lines > 0] = 0
    payload = BytesIO()
    Image.fromarray(output).convert("RGB").save(payload, format="PNG", optimize=True)
    return LineartEngineResult(
        data=payload.getvalue(),
        mode=mode,
        algorithm_version=ALGORITHM_VERSION,
        quality=quality,
        warnings=warnings,
    )


def generate_lineart_file_v3(
    region_path: Path,
    mask_path: Path | None = None,
    *,
    text_mask_path: Path | None = None,
    text_regions: list[dict[str, Any]] | None = None,
    text_mask_status: str | None = None,
    mask_semantics: str = "foreground_evidence",
) -> LineartEngineResult:
    """Generate a source-supported v3 candidate from one selected region.

    ``text_mask_path`` and ``text_regions`` use crop coordinates, not source
    sales-image coordinates.  A missing OCR mask is not treated as proof that
    no text exists; it is recorded as a QC failure instead.
    """

    mask_data = mask_path.read_bytes() if mask_path and mask_path.exists() else None
    text_mask_data = (
        text_mask_path.read_bytes() if text_mask_path and text_mask_path.exists() else None
    )
    return generate_lineart_bytes_v3(
        region_path.read_bytes(),
        mask_data=mask_data,
        text_mask_data=text_mask_data,
        text_regions=text_regions,
        text_mask_status=text_mask_status,
        mask_semantics=mask_semantics,
    )


def generate_lineart_bytes_v3(
    data: bytes,
    *,
    mask_data: bytes | None = None,
    text_mask_data: bytes | None = None,
    text_regions: list[dict[str, Any]] | None = None,
    text_mask_status: str | None = None,
    mask_semantics: str = "foreground_evidence",
) -> LineartEngineResult:
    """Deterministic mask-outline plus multi-scale-Canny line-art route.

    The route only selects pixels supported by the supplied product mask or by
    source-image gradients.  OCR evidence can remove fine edges but is never
    allowed to erase the mask-derived silhouette, so it cannot change product
    structure silently.
    """

    original = load_rgb(data)
    original_shape = original.shape[:2]
    image = resize_rgb(original)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    classified_mode, input_metrics = classify_input(gray, hsv)
    supplied_mask = load_mask(mask_data, gray.shape) if mask_data else None

    parsed_text_regions = 0
    if text_mask_data is not None:
        text_mask = load_mask(text_mask_data, gray.shape)
        effective_text_status = text_mask_status or "provided"
    elif text_regions:
        text_mask, parsed_text_regions = text_mask_from_regions(
            text_regions,
            target_shape=gray.shape,
            source_shape=original_shape,
        )
        effective_text_status = (
            text_mask_status
            or ("text_regions" if parsed_text_regions else "invalid_text_regions")
        )
    else:
        text_mask = np.zeros_like(gray)
        effective_text_status = text_mask_status or "missing"

    if classified_mode == "existing_lineart":
        lines, source_support, object_mask, route_metrics = existing_lineart_route(
            gray, supplied_mask
        )
        layer_maps = {
            "primary": lines.copy(),
            "secondary": np.zeros_like(lines),
            "texture": np.zeros_like(lines),
        }
        mode = "existing_lineart"
    else:
        lines, source_support, object_mask, route_metrics, layer_maps = photo_edge_route_v3(
            image,
            gray,
            supplied_mask,
            text_mask,
            mask_semantics=mask_semantics,
        )
        mode = "photo_edges"

    route_metrics.update(
        {
            "text_mask_status": effective_text_status,
            "text_mask_pixels": int(np.count_nonzero(text_mask)),
            "parsed_text_regions": parsed_text_regions,
            "mask_semantics": mask_semantics,
        }
    )
    quality = quality_metrics_v3(
        lines,
        source_support,
        object_mask,
        mode,
        text_mask=text_mask,
        text_mask_status=effective_text_status,
        mask_semantics=mask_semantics,
        supplied_mask_present=supplied_mask is not None,
    )
    quality["input"] = input_metrics
    quality["route"] = route_metrics
    quality["algorithm_version"] = V3_ALGORITHM_VERSION
    if float(route_metrics.get("inserted_pixel_ratio") or 0.0) > 0.35:
        quality["failure_reasons"] = list(
            dict.fromkeys(
                list(quality.get("failure_reasons") or [])
                + ["stroke_styling_exceeds_source_evidence_budget"]
            )
        )
        quality["auto_qc_passed"] = False
        quality["technical_qc_passed"] = False

    warnings: list[dict[str, Any]] = []
    if quality["auto_qc_passed"]:
        warnings.append(
            {
                "severity": "info",
                "code": "lineart_v3_technical_qc_passed",
                "message": (
                    "Deterministic v3 lineart passed technical QC; explicit human approval "
                    "is still required."
                ),
            }
        )
    else:
        warnings.append(
            {
                "severity": "needs_review",
                "code": "lineart_v3_technical_qc_failed",
                "message": "V3 lineart failed one or more deterministic quality checks.",
                "reasons": quality["failure_reasons"],
            }
        )
    if effective_text_status not in {"provided", "text_regions", "ocr_success_no_text"}:
        warnings.append(
            {
                "severity": "needs_review",
                "code": "lineart_ocr_evidence_unavailable",
                "message": (
                    "OCR text evidence is missing or invalid; absence of sales text cannot be assumed."
                ),
            }
        )
    if quality.get("low_resolution"):
        warnings.append(
            {
                "severity": "warning",
                "code": "lineart_low_resolution",
                "message": "Selected region has limited source pixels; no upsampling can recover missing detail.",
            }
        )

    output = np.full(gray.shape, 255, dtype=np.uint8)
    output[lines > 0] = 0
    payload = BytesIO()
    Image.fromarray(output).convert("RGB").save(payload, format="PNG", optimize=True)
    return LineartEngineResult(
        data=payload.getvalue(),
        mode=mode,
        algorithm_version=V3_ALGORITHM_VERSION,
        quality=quality,
        warnings=warnings,
        diagnostics=encode_layer_diagnostics(layer_maps),
    )


def load_rgb(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            canvas = Image.new("RGBA", image.size, "WHITE")
            canvas.alpha_composite(image.convert("RGBA"))
            image = canvas.convert("RGB")
        else:
            image = image.convert("RGB")
        return np.asarray(image).copy()


def encode_layer_diagnostics(layer_maps: dict[str, np.ndarray]) -> dict[str, bytes]:
    """Encode review-only layer images without altering the formal output."""

    diagnostics: dict[str, bytes] = {}
    if not layer_maps:
        return diagnostics
    shape = next(iter(layer_maps.values())).shape
    for name in ("primary", "secondary", "texture"):
        layer = layer_maps.get(name, np.zeros(shape, dtype=np.uint8))
        canvas = np.full(shape, 255, dtype=np.uint8)
        canvas[layer > 0] = 0
        payload = BytesIO()
        Image.fromarray(canvas).convert("RGB").save(payload, format="PNG", optimize=True)
        diagnostics[name] = payload.getvalue()

    composite = np.full((*shape, 3), 255, dtype=np.uint8)
    colors = {
        "primary": (20, 20, 20),
        "secondary": (40, 90, 210),
        "texture": (220, 90, 40),
    }
    for name in ("texture", "secondary", "primary"):
        layer = layer_maps.get(name)
        if layer is not None:
            composite[layer > 0] = colors[name]
    payload = BytesIO()
    Image.fromarray(composite).save(payload, format="PNG", optimize=True)
    diagnostics["layers-color"] = payload.getvalue()
    return diagnostics


def resize_rgb(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_LINEART_SIZE / float(max(width, height)))
    if scale >= 1.0:
        return image
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def load_mask(data: bytes, target_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(BytesIO(data)) as source:
        mask = np.asarray(ImageOps.exif_transpose(source).convert("L"))
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask >= 128).astype(np.uint8) * 255


def classify_input(gray: np.ndarray, hsv: np.ndarray) -> tuple[str, dict[str, float]]:
    near_white_ratio = float(np.mean(gray >= 238))
    nonwhite_ratio = 1.0 - near_white_ratio
    neutral_ratio = float(np.mean(hsv[:, :, 1] <= 12))
    extreme_ratio = float(np.mean((gray <= 64) | (gray >= 238)))
    saturated_ratio = float(np.mean(hsv[:, :, 1] >= 32))
    lineart_like = (
        near_white_ratio >= 0.65
        and neutral_ratio >= 0.95
        and extreme_ratio >= 0.70
        and 0.01 <= nonwhite_ratio <= 0.40
    )
    return (
        "existing_lineart" if lineart_like else "photo_edges",
        {
            "near_white_ratio": round(near_white_ratio, 4),
            "nonwhite_ratio": round(nonwhite_ratio, 4),
            "neutral_ratio": round(neutral_ratio, 4),
            "extreme_ratio": round(extreme_ratio, 4),
            "saturated_ratio": round(saturated_ratio, 4),
        },
    )


def existing_lineart_route(
    gray: np.ndarray, supplied_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preservation_threshold = int(np.clip(otsu_threshold + 35, 190, 215))
    # Do not morphologically simplify an existing drawing: even a one-pixel
    # mark may be a real measurement tick or product detail.
    lines = (gray < preservation_threshold).astype(np.uint8) * 255
    source_support = (gray <= min(248, preservation_threshold + 32)).astype(np.uint8) * 255
    evidence = cv2.bitwise_or(lines, supplied_mask) if supplied_mask is not None else lines
    object_mask = envelope_from_evidence(evidence)
    return lines, source_support, object_mask, {
        "otsu_threshold": round(float(otsu_threshold), 2),
        "preservation_threshold": preservation_threshold,
    }


def photo_edge_route(
    image: np.ndarray, gray: np.ndarray, supplied_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    object_mask = normalize_object_mask(supplied_mask, image, gray)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(object_mask, cv2.MORPH_GRADIENT, kernel3)
    interior = cv2.erode(object_mask, kernel3)

    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
    canny = cv2.Canny(smooth, 45, 135, L2gradient=True)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    magnitudes = []
    for channel in cv2.split(lab):
        gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        magnitudes.append(cv2.magnitude(gx, gy))
    magnitude = np.maximum.reduce(magnitudes)
    values = magnitude[object_mask > 0]
    strong_threshold = float(np.percentile(values, 82)) if values.size else 0.0
    weak_threshold = float(np.percentile(values, 65)) if values.size else 0.0
    strong = (magnitude >= strong_threshold).astype(np.uint8) * 255 if strong_threshold > 0 else np.zeros_like(gray)
    weak = (magnitude >= weak_threshold).astype(np.uint8) * 255 if weak_threshold > 0 else np.zeros_like(gray)

    internal = cv2.bitwise_or(canny, strong)
    internal = cv2.bitwise_and(internal, interior)
    lines = cv2.bitwise_or(boundary, internal)
    lines = remove_tiny_components(lines, minimum_area=3)
    source_support = cv2.bitwise_or(boundary, cv2.dilate(cv2.bitwise_or(canny, weak), kernel3))
    return lines, source_support, object_mask, {
        "strong_gradient_threshold": round(strong_threshold, 2),
        "weak_gradient_threshold": round(weak_threshold, 2),
        "mask_source": "selected_region_mask" if supplied_mask is not None else "derived_foreground",
        "mask_semantics": "foreground_evidence",
    }


def photo_edge_route_v3(
    image: np.ndarray,
    gray: np.ndarray,
    supplied_mask: np.ndarray | None,
    text_mask: np.ndarray,
    *,
    mask_semantics: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Extract only mask- or source-gradient-supported structure.

    Fine edges inside OCR regions are retained only when a separately smoothed
    coarse Canny pass supports them.  The silhouette is composed afterwards,
    which means OCR can flag a product/text collision but cannot erase the
    product outline.
    """

    object_mask = normalize_object_mask_v3(
        supplied_mask,
        image,
        gray,
        mask_semantics=mask_semantics,
    )
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary_mask = (
        (supplied_mask > 0).astype(np.uint8) * 255
        if supplied_mask is not None and mask_semantics == "product_foreground"
        else object_mask
    )
    boundary = cv2.morphologyEx(boundary_mask, cv2.MORPH_GRADIENT, kernel3)
    interior = cv2.erode(object_mask, kernel3)

    fine_smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
    intermediate_sigma = 1.5
    intermediate_smooth = cv2.GaussianBlur(
        fine_smooth,
        (0, 0),
        sigmaX=intermediate_sigma,
        sigmaY=intermediate_sigma,
    )
    coarse_smooth = cv2.GaussianBlur(fine_smooth, (0, 0), sigmaX=2.0, sigmaY=2.0)
    fine_edges = cv2.Canny(fine_smooth, 35, 105, L2gradient=True)
    intermediate_edges = cv2.Canny(intermediate_smooth, 45, 135, L2gradient=True)
    coarse_edges = cv2.Canny(coarse_smooth, 45, 135, L2gradient=True)

    # A global gradient-percentile gate lets a dense, high-contrast texture
    # dominate sparse low-contrast product structure.  Use local fine-edge
    # density instead: sparse Canny evidence is retained regardless of global
    # contrast, while dense texture falls back to the independently smoothed
    # intermediate/coarse passes.  Every selected pixel is still a source
    # Canny edge; no line is joined, extrapolated, or synthesized.
    density_window = max(9, round(min(gray.shape) * 0.06))
    if density_window % 2 == 0:
        density_window += 1
    fine_density = cv2.boxFilter(
        (fine_edges > 0).astype(np.float32),
        ddepth=-1,
        ksize=(density_window, density_window),
        normalize=True,
    )
    density_values = np.clip(
        np.rint(fine_density[fine_edges > 0] * 255.0), 0, 255
    ).astype(np.uint8)
    if density_values.size:
        otsu_density, _ = cv2.threshold(
            density_values.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        fine_density_cutoff = max(0.18, float(otsu_density) / 255.0)
    else:
        fine_density_cutoff = 0.18
    sparse_fine = cv2.bitwise_and(
        fine_edges,
        (fine_density < fine_density_cutoff).astype(np.uint8) * 255,
    )
    # EdgeDrawing links locally coherent pixels into longer chains.  It is an
    # optional OpenCV-contrib primitive: when unavailable, the deterministic
    # Canny route remains the fail-closed fallback.  The chains are source
    # observations only; no gap closing or geometry synthesis is performed.
    edge_drawing, edge_drawing_metrics = edge_drawing_edges(gray)
    # EdgeDrawing can still return every visible weave strand.  Apply the
    # same local-density principle as the fine Canny pass so long coherent
    # rims survive while dense photographic texture is sampled sparsely.
    if np.any(edge_drawing):
        ed_density = cv2.boxFilter(
            (edge_drawing > 0).astype(np.float32),
            ddepth=-1,
            ksize=(density_window := max(9, round(min(gray.shape) * 0.06)), density_window),
            normalize=True,
        )
        edge_drawing = cv2.bitwise_and(
            edge_drawing,
            (ed_density < 0.24).astype(np.uint8) * 255,
        )
        edge_drawing_metrics["edge_drawing_pixels_after_density"] = int(
            np.count_nonzero(edge_drawing)
        )
    gradient_support = cv2.bitwise_or(fine_edges, intermediate_edges)
    gradient_support = cv2.bitwise_or(gradient_support, coarse_edges)
    edge_drawing = cv2.bitwise_and(edge_drawing, cv2.dilate(gradient_support, kernel3))
    coarse_primary = cv2.bitwise_and(coarse_edges, interior)
    primary = cv2.bitwise_or(boundary, coarse_primary)
    secondary = cv2.bitwise_and(intermediate_edges, interior)
    texture = cv2.bitwise_or(edge_drawing, sparse_fine)
    texture = cv2.bitwise_and(texture, interior)

    dense_fine = cv2.bitwise_and(fine_edges, cv2.bitwise_not(sparse_fine))
    represented_dense = cv2.dilate(
        cv2.bitwise_or(intermediate_edges, coarse_edges), kernel3
    )
    dense_fine_suppressed = int(
        np.count_nonzero((dense_fine > 0) & (represented_dense == 0))
    )

    # OCR is layout-text negative evidence only.  It can suppress secondary
    # and texture strokes, but never the mask boundary or a coarse primary
    # source contour.  A primary/text collision remains a QC blocker below.
    primary_text_support = cv2.dilate(primary, kernel3)
    removable_text = cv2.bitwise_and(text_mask, cv2.bitwise_not(primary_text_support))
    before_suppression = int(np.count_nonzero(secondary)) + int(np.count_nonzero(texture))
    secondary[removable_text > 0] = 0
    texture[removable_text > 0] = 0
    suppressed_pixels = before_suppression - (
        int(np.count_nonzero(secondary)) + int(np.count_nonzero(texture))
    )

    # Render hierarchy after structure selection.  A one-pixel widening is a
    # stroke-style operation, not geometry completion.  It is clipped to a
    # two-pixel source-evidence corridor and measured as inserted ink below.
    primary_rendered = cv2.bitwise_or(
        boundary,
        cv2.bitwise_and(cv2.dilate(coarse_primary, kernel3), object_mask),
    )
    secondary = cv2.bitwise_and(secondary, cv2.bitwise_not(primary_rendered))
    texture = cv2.bitwise_and(
        texture,
        cv2.bitwise_not(cv2.bitwise_or(primary_rendered, secondary)),
    )
    lines = cv2.bitwise_or(primary_rendered, secondary)
    lines = cv2.bitwise_or(lines, texture)
    observed_structure = cv2.bitwise_or(primary, secondary)
    observed_structure = cv2.bitwise_or(observed_structure, texture)
    evidence_corridor = cv2.bitwise_or(
        cv2.dilate(boundary, np.ones((5, 5), dtype=np.uint8)),
        cv2.dilate(gradient_support, np.ones((5, 5), dtype=np.uint8)),
    )
    lines = cv2.bitwise_and(lines, evidence_corridor)
    minimum_area = max(3, round(lines.size * 0.000008))
    lines = remove_tiny_components(lines, minimum_area=minimum_area)
    source_support = evidence_corridor
    inserted_pixels = int(
        np.count_nonzero((lines > 0) & (observed_structure == 0))
    )
    line_pixels = int(np.count_nonzero(lines))
    layer_maps = {
        "primary": cv2.bitwise_and(primary_rendered, lines),
        "secondary": cv2.bitwise_and(secondary, lines),
        "texture": cv2.bitwise_and(texture, lines),
    }
    total_layer_pixels = max(sum(int(np.count_nonzero(value)) for value in layer_maps.values()), 1)
    return lines, source_support, object_mask, {
        "fine_canny_thresholds": [35, 105],
        "intermediate_canny_thresholds": [45, 135],
        "intermediate_canny_sigma": intermediate_sigma,
        "coarse_canny_thresholds": [45, 135],
        "fine_density_window": density_window,
        "fine_density_cutoff": round(fine_density_cutoff, 4),
        "sparse_fine_pixels_retained": int(np.count_nonzero(sparse_fine)),
        "dense_fine_pixels_suppressed": dense_fine_suppressed,
        "suppressed_fine_text_edge_pixels": suppressed_pixels,
        "layers": {
            name: {
                "pixels": int(np.count_nonzero(layer)),
                "ink_share": round(int(np.count_nonzero(layer)) / float(total_layer_pixels), 4),
                "nominal_stroke_width": 2 if name == "primary" else 1,
            }
            for name, layer in layer_maps.items()
        },
        "primary_text_overlap_pixels": int(np.count_nonzero((primary_rendered > 0) & (text_mask > 0))),
        "inserted_pixels": inserted_pixels,
        "inserted_pixel_ratio": round(inserted_pixels / float(max(line_pixels, 1)), 4),
        **edge_drawing_metrics,
        "mask_source": "selected_region_mask" if supplied_mask is not None else "derived_foreground",
        "mask_semantics": mask_semantics,
        "non_generative": True,
    }, layer_maps


def edge_drawing_edges(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return coherent source-supported chains when ximgproc is available.

    EdgeDrawing is deliberately additive to the existing multi-scale route:
    it improves continuity of rims and woven strips, while Canny remains the
    compatibility baseline.  A missing contrib build never blocks generation.
    """

    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "createEdgeDrawing"):
        return np.zeros_like(gray), {
            "edge_drawing_available": False,
            "edge_drawing_segments": 0,
            "edge_drawing_pixels": 0,
        }
    # OpenCV wheels expose the parameter class with either a module-level or
    # ximgproc-qualified name across releases.
    params_factory = getattr(cv2, "ximgproc_EdgeDrawing_Params", None)
    if params_factory is None:
        params_factory = getattr(ximgproc, "EdgeDrawing_Params", None)
    if params_factory is None:
        return np.zeros_like(gray), {
            "edge_drawing_available": False,
            "edge_drawing_segments": 0,
            "edge_drawing_pixels": 0,
        }
    params = params_factory()
    # Keep only chains long enough to represent a rim, panel boundary, or
    # coherent strip.  Short chains are predominantly photographic texture
    # at this crop size and make the output unreadable when printed small.
    params.MinPathLength = max(24, round(min(gray.shape) * 0.09))
    params.MinLineLength = max(12, round(min(gray.shape) * 0.04))
    params.GradientThresholdValue = 18
    params.AnchorThresholdValue = 8
    detector = ximgproc.createEdgeDrawing()
    detector.setParams(params)
    detector.detectEdges(gray)
    segments = detector.getSegments() or []
    output = np.zeros_like(gray)
    kept = 0
    for segment in segments:
        points = np.asarray(segment).reshape(-1, 2)
        if len(points) < 2:
            continue
        length = float(np.linalg.norm(np.diff(points.astype(np.float32), axis=0), axis=1).sum())
        if length < params.MinPathLength:
            continue
        cv2.polylines(output, [np.rint(points).astype(np.int32)], False, 255, 1, cv2.LINE_AA)
        kept += 1
    return output, {
        "edge_drawing_available": True,
        "edge_drawing_segments": kept,
        "edge_drawing_pixels": int(np.count_nonzero(output)),
        "edge_drawing_min_path_length": int(params.MinPathLength),
    }


def normalize_object_mask(supplied_mask: np.ndarray | None, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
    if supplied_mask is not None and np.any(supplied_mask):
        return fill_external_components(supplied_mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    evidence = ((gray < 245) | (hsv[:, :, 1] > 28)).astype(np.uint8) * 255
    return envelope_from_evidence(evidence)


def normalize_object_mask_v3(
    supplied_mask: np.ndarray | None,
    image: np.ndarray,
    gray: np.ndarray,
    *,
    mask_semantics: str,
) -> np.ndarray:
    if supplied_mask is None or not np.any(supplied_mask):
        return normalize_object_mask(None, image, gray)
    if mask_semantics != "product_foreground":
        # Compatibility route for v1 evidence masks.  It remains review-only in
        # QC because filling an evidence envelope is not precise segmentation.
        return fill_external_components(supplied_mask)

    binary = (supplied_mask > 0).astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3)
    return retain_mask_components(binary, minimum_area=max(4, round(binary.size * 0.0002)))


def retain_mask_components(mask: np.ndarray, *, minimum_area: int) -> np.ndarray:
    """Remove isolated mask noise without filling real holes or handles."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    output = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            output[labels == label] = 255
    return output


def text_mask_from_regions(
    regions: list[dict[str, Any]],
    *,
    target_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Rasterize common OCR bbox/polygon shapes into crop coordinates."""

    output = np.zeros(target_shape, dtype=np.uint8)
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    scale_x = target_width / float(max(source_width, 1))
    scale_y = target_height / float(max(source_height, 1))
    parsed = 0
    for region in regions:
        box = text_region_bbox(region)
        if box is None:
            continue
        x, y, width, height = box
        if max(abs(x), abs(y), abs(width), abs(height)) <= 1.0:
            x *= source_width
            width *= source_width
            y *= source_height
            height *= source_height
        x1 = int(np.floor(x * scale_x))
        y1 = int(np.floor(y * scale_y))
        x2 = int(np.ceil((x + width) * scale_x))
        y2 = int(np.ceil((y + height) * scale_y))
        padding = max(1, round(min(target_shape) * 0.002))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(target_width, x2 + padding)
        y2 = min(target_height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        output[y1:y2, x1:x2] = 255
        parsed += 1
    return output, parsed


def text_region_bbox(region: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value: Any = region.get("bbox", region)
    if isinstance(value, dict):
        try:
            x = float(value.get("x", value.get("left")))
            y = float(value.get("y", value.get("top")))
            if value.get("width") is not None and value.get("height") is not None:
                return x, y, float(value["width"]), float(value["height"])
            x2 = float(value.get("x2", value.get("right")))
            y2 = float(value.get("y2", value.get("bottom")))
            return x, y, x2 - x, y2 - y
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value):
            try:
                xs = [float(point[0]) for point in value]
                ys = [float(point[1]) for point in value]
            except (TypeError, ValueError):
                return None
            return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        try:
            x, y, width, height = (float(part) for part in value)
            return x, y, width, height
        except (TypeError, ValueError):
            return None
    polygon = region.get("polygon") or region.get("points")
    if isinstance(polygon, (list, tuple)) and polygon:
        try:
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
        except (IndexError, TypeError, ValueError):
            return None
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return None


def envelope_from_evidence(evidence: np.ndarray) -> np.ndarray:
    height, width = evidence.shape
    kernel_size = max(3, round(min(width, height) * 0.018))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    merged = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, kernel)
    merged = cv2.dilate(merged, kernel)
    return fill_external_components(merged)


def fill_external_components(mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = mask.size * 0.0005
    kept = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if kept:
        cv2.drawContours(output, kept, -1, 255, thickness=cv2.FILLED)
    return output


def remove_tiny_components(lines: np.ndarray, *, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((lines > 0).astype(np.uint8), connectivity=8)
    output = np.zeros_like(lines)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            output[labels == label] = 255
    return output


def quality_metrics(
    lines: np.ndarray, source_support: np.ndarray, object_mask: np.ndarray, mode: str
) -> dict[str, Any]:
    total_pixels = int(lines.size)
    line_pixels = int(np.count_nonzero(lines))
    mask_pixels = int(np.count_nonzero(object_mask))
    black_ratio = line_pixels / float(total_pixels) if total_pixels else 0.0
    mask_ratio = mask_pixels / float(total_pixels) if total_pixels else 0.0

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_tolerance = cv2.dilate(object_mask, kernel3)
    outside_pixels = int(np.count_nonzero((lines > 0) & (mask_tolerance == 0)))
    outside_ratio = outside_pixels / float(max(line_pixels, 1))
    boundary = cv2.morphologyEx(object_mask, cv2.MORPH_GRADIENT, kernel3)
    boundary_pixels = int(np.count_nonzero(boundary))
    covered_boundary = cv2.bitwise_and(boundary, cv2.dilate(lines, kernel3))
    boundary_coverage = int(np.count_nonzero(covered_boundary)) / float(max(boundary_pixels, 1))
    supported = cv2.bitwise_and(lines, cv2.dilate(source_support, kernel3))
    source_support_ratio = int(np.count_nonzero(supported)) / float(max(line_pixels, 1))

    component_count, _, stats, _ = cv2.connectedComponentsWithStats((lines > 0).astype(np.uint8), connectivity=8)
    component_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count)]
    short_pixels = sum(area for area in component_areas if area < 8)
    short_component_ratio = short_pixels / float(max(line_pixels, 1))
    border = np.zeros_like(lines)
    border_width = max(1, round(min(lines.shape) * 0.005))
    border[:border_width, :] = 255
    border[-border_width:, :] = 255
    border[:, :border_width] = 255
    border[:, -border_width:] = 255
    border_ink_ratio = int(np.count_nonzero((lines > 0) & (border > 0))) / float(max(line_pixels, 1))
    line_points = cv2.findNonZero((lines > 0).astype(np.uint8))
    if line_points is None:
        line_bbox_width_ratio = 0.0
        line_bbox_height_ratio = 0.0
    else:
        _, _, line_width, line_height = cv2.boundingRect(line_points)
        line_bbox_width_ratio = line_width / float(max(lines.shape[1], 1))
        line_bbox_height_ratio = line_height / float(max(lines.shape[0], 1))
    touches_edge = bool(
        np.any(object_mask[0, :])
        or np.any(object_mask[-1, :])
        or np.any(object_mask[:, 0])
        or np.any(object_mask[:, -1])
    )

    failures = []
    if line_pixels < max(20, round(total_pixels * (0.01 if mode == "existing_lineart" else 0.0015))):
        failures.append("lineart_is_blank_or_too_sparse")
    if black_ratio > 0.32:
        failures.append("lineart_has_excessive_black_area")
    if mask_ratio < 0.01:
        failures.append("object_mask_is_empty_or_too_small")
    if outside_ratio > 0.02:
        failures.append("lines_extend_outside_selected_mask")
    # A near-monochrome input already contains the source-supported strokes we
    # want to preserve.  Its outline need not be a single closed contour, so a
    # mask-boundary coverage test would reject valid sparse technical drawings.
    # For photo-derived output the current mask is only foreground evidence,
    # not an object silhouette.  Coverage is therefore a technical consistency
    # check, while the route remains review-only regardless of the score.
    if mode == "photo_edges" and boundary_coverage < 0.55:
        failures.append("foreground_evidence_envelope_is_incomplete")
    if mode == "existing_lineart" and (line_bbox_width_ratio < 0.15 or line_bbox_height_ratio < 0.15):
        failures.append("existing_lineart_has_insufficient_spatial_extent")
    if source_support_ratio < 0.75:
        failures.append("lines_lack_source_image_evidence")
    if short_component_ratio > 0.40:
        failures.append("lineart_contains_too_many_short_fragments")
    if mode == "existing_lineart" and short_component_ratio > 0.03:
        failures.append("existing_lineart_contains_excessive_speckle")
    if mode == "existing_lineart" and border_ink_ratio > 0.02:
        failures.append("existing_lineart_touches_canvas_border")
    if touches_edge and mode == "photo_edges":
        failures.append("product_mask_touches_crop_edge")
    # There is no validated photo-to-lineart ground truth in the current A2/A5
    # corpus.  Keep this useful candidate route review-only until such evidence
    # exists instead of presenting an unverified heuristic as automatic truth.
    if mode == "photo_edges":
        failures.append("photo_edge_route_requires_review")

    return {
        "auto_qc_passed": not failures,
        "failure_reasons": failures,
        "black_ratio": round(black_ratio, 4),
        "mask_ratio": round(mask_ratio, 4),
        "outside_mask_ratio": round(outside_ratio, 4),
        "foreground_envelope_coverage": round(boundary_coverage, 4),
        "source_support_ratio": round(source_support_ratio, 4),
        "connected_components": len(component_areas),
        "short_component_ratio": round(short_component_ratio, 4),
        "border_ink_ratio": round(border_ink_ratio, 4),
        "line_bbox_width_ratio": round(line_bbox_width_ratio, 4),
        "line_bbox_height_ratio": round(line_bbox_height_ratio, 4),
        "mask_touches_edge": touches_edge,
        "width_pixels": int(lines.shape[1]),
        "height_pixels": int(lines.shape[0]),
        "low_resolution": min(lines.shape) < 150 or max(lines.shape) < 300,
    }


def quality_metrics_v3(
    lines: np.ndarray,
    source_support: np.ndarray,
    object_mask: np.ndarray,
    mode: str,
    *,
    text_mask: np.ndarray,
    text_mask_status: str,
    mask_semantics: str,
    supplied_mask_present: bool,
) -> dict[str, Any]:
    """Add silhouette and OCR-residual checks without changing v2 QC."""

    result = quality_metrics(lines, source_support, object_mask, mode)
    # Keep the photo route's review gate in the public result.  Technical
    # metrics can be clean while a photograph-to-lineart candidate still has
    # no validated ground truth; filtering this reason here used to turn that
    # candidate into a misleading ``passed`` status downstream.
    failures = list(result["failure_reasons"])

    silhouette_coverage = float(result["foreground_envelope_coverage"])
    if mode == "photo_edges" and silhouette_coverage < 0.70:
        failures = [
            reason
            for reason in failures
            if reason != "foreground_evidence_envelope_is_incomplete"
        ]
        failures.append("product_silhouette_is_incomplete")

    if not supplied_mask_present:
        failures.append("product_mask_missing")
    elif mask_semantics != "product_foreground":
        failures.append("legacy_foreground_evidence_mask_requires_review")

    valid_text_statuses = {"provided", "text_regions", "ocr_success_no_text"}
    if text_mask_status not in valid_text_statuses:
        failures.append("ocr_text_evidence_missing_or_invalid")

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(object_mask, cv2.MORPH_GRADIENT, kernel3)
    text_pixels = int(np.count_nonzero(text_mask))
    residual_text_pixels = int(np.count_nonzero((lines > 0) & (text_mask > 0)))
    text_residual_density = residual_text_pixels / float(max(text_pixels, 1))
    text_line_share = residual_text_pixels / float(max(np.count_nonzero(lines), 1))
    text_boundary_pixels = int(np.count_nonzero((boundary > 0) & (text_mask > 0)))
    text_boundary_overlap = text_boundary_pixels / float(max(np.count_nonzero(boundary), 1))

    if text_pixels and text_boundary_overlap > 0.02:
        failures.append("ocr_text_overlaps_product_silhouette")
    if text_pixels and text_residual_density > 0.08:
        failures.append("lineart_contains_text_residual")

    result.update(
        {
            "auto_qc_passed": not failures,
            "technical_qc_passed": not any(
                reason != "photo_edge_route_requires_review" for reason in failures
            ),
            "visual_qc_required": mode == "photo_edges",
            "failure_reasons": list(dict.fromkeys(failures)),
            "silhouette_coverage": round(silhouette_coverage, 4),
            "text_mask_status": text_mask_status,
            "text_mask_pixels": text_pixels,
            "text_residual_pixels": residual_text_pixels,
            "text_residual_density": round(text_residual_density, 4),
            "text_line_share": round(text_line_share, 4),
            "text_boundary_overlap_ratio": round(text_boundary_overlap, 4),
            "mask_semantics": mask_semantics,
        }
    )
    return result
