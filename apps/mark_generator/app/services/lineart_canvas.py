from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


INK_THRESHOLD = 245
CANVAS_MARGIN_RATIO = 0.04
MIN_EFFECTIVE_SIDE_PIXELS = 24
MIN_EFFECTIVE_INK_PIXELS = 64
MAX_EMBED_SIDE_PIXELS = 700


class LineartCanvasError(ValueError):
    pass


def load_lineart_grayscale(data: bytes) -> Image.Image:
    """Decode a lineart raster onto an opaque white grayscale canvas."""

    try:
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                background = Image.new("RGBA", image.size, "white")
                background.alpha_composite(image.convert("RGBA"))
                image = background.convert("RGB")
            return image.convert("L").copy()
    except Exception as exc:
        raise LineartCanvasError(f"Could not decode lineart canvas: {exc}") from exc


def normalize_lineart_canvas(
    image: Image.Image,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    ink_threshold: int = INK_THRESHOLD,
    margin_ratio: float = CANVAS_MARGIN_RATIO,
) -> tuple[Image.Image, dict[str, Any]]:
    """Crop unused white canvas while preserving every detected ink pixel.

    The operation is strictly geometric: it never stretches, rotates, fills,
    traces, or invents product detail.  A small white margin is retained around
    the complete ink envelope so antialiased shoulders and edge strokes are not
    clipped by the PDF image rectangle.
    """

    gray = image.convert("L")
    source_width, source_height = gray.size
    ink_mask = gray.point(lambda value: 255 if value < ink_threshold else 0)
    ink_bbox = ink_mask.getbbox()
    histogram = gray.histogram()
    effective_ink_pixels = int(sum(histogram[:ink_threshold]))

    if ink_bbox:
        ink_width = ink_bbox[2] - ink_bbox[0]
        ink_height = ink_bbox[3] - ink_bbox[1]
    else:
        ink_width = 0
        ink_height = 0

    if crop_box is None and ink_bbox:
        margin = max(3, math.ceil(max(ink_width, ink_height) * margin_ratio))
        crop_box = (
            max(0, ink_bbox[0] - margin),
            max(0, ink_bbox[1] - margin),
            min(source_width, ink_bbox[2] + margin),
            min(source_height, ink_bbox[3] + margin),
        )
    elif crop_box is not None:
        left, top, right, bottom = crop_box
        crop_box = (
            max(0, min(source_width, int(left))),
            max(0, min(source_height, int(top))),
            max(0, min(source_width, int(right))),
            max(0, min(source_height, int(bottom))),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise LineartCanvasError("Lineart crop box has no positive area.")

    if crop_box is None:
        crop_box = (0, 0, source_width, source_height)
    normalized = gray.crop(crop_box)
    low_effective_resolution = (
        ink_bbox is None
        or min(ink_width, ink_height) < MIN_EFFECTIVE_SIDE_PIXELS
        or effective_ink_pixels < MIN_EFFECTIVE_INK_PIXELS
    )
    metrics = {
        "source_size": [source_width, source_height],
        "ink_bbox": list(ink_bbox) if ink_bbox else None,
        "ink_bbox_size": [ink_width, ink_height],
        "crop_box": list(crop_box),
        "normalized_size": list(normalized.size),
        "effective_ink_pixels": effective_ink_pixels,
        "ink_threshold": ink_threshold,
        "margin_ratio": margin_ratio,
        "crop_applied": crop_box != (0, 0, source_width, source_height),
        "low_effective_resolution": low_effective_resolution,
    }
    return normalized, metrics


def normalize_lineart_bytes(
    data: bytes,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    return normalize_lineart_canvas(load_lineart_grayscale(data), crop_box=crop_box)


def normalized_lineart_png_bytes(
    data: bytes,
    *,
    max_side: int | None = MAX_EMBED_SIDE_PIXELS,
    reject_low_effective_resolution: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Return a lossless, proportion-preserving normalized lineart PNG."""

    image, metrics = normalize_lineart_bytes(data)
    if reject_low_effective_resolution and metrics["low_effective_resolution"]:
        raise LineartCanvasError(
            "Lineart effective ink envelope is too small for reliable rendering."
        )
    if max_side is not None:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    metrics = dict(metrics)
    metrics["output_size"] = list(image.size)
    return output.getvalue(), metrics


def inspect_lineart_file(path: Path) -> dict[str, Any]:
    _, metrics = normalize_lineart_bytes(path.read_bytes())
    return metrics
