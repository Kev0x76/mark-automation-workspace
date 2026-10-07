from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .lineart_canvas import normalize_lineart_canvas


MAX_LINEART_SIZE = 1600
GENERATIVE_CANVAS_MARGIN_RATIO = 0.04


def postprocess_lineart_bytes(data: bytes, *, threshold: int = 210) -> tuple[bytes, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGBA", image.size, "WHITE")
            background.alpha_composite(image.convert("RGBA"))
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")

        image.thumbnail((MAX_LINEART_SIZE, MAX_LINEART_SIZE), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(image)
        binary = gray.point(lambda value: 0 if value < threshold else 255, mode="1")
        output_image, canvas_metrics = normalize_lineart_canvas(binary.convert("L"))

        output = BytesIO()
        output_image.convert("RGB").save(output, format="PNG", optimize=True)
        if image.width == MAX_LINEART_SIZE or image.height == MAX_LINEART_SIZE:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "lineart_resized",
                    "message": "Lineart image was resized to the maximum candidate size.",
                }
            )
        if canvas_metrics["low_effective_resolution"]:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "lineart_low_effective_pixels",
                    "message": "Lineart ink envelope has too few effective pixels for reliable output.",
                }
            )
        return output.getvalue(), warnings


def postprocess_lineart_file(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    return postprocess_lineart_bytes(path.read_bytes())


def pad_generative_canvas(data: bytes) -> bytes:
    """Add a small opaque white safety margin around model output.

    Image models occasionally draw a valid product stroke on the outermost
    pixel even when the product itself is complete.  That is a canvas-format
    defect, not new product evidence, and can be corrected deterministically
    before lineart QC.  The source-faithful pixels are only scaled by the
    later renderer; no product content is synthesized or cropped here.
    """

    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGBA", image.size, "WHITE")
            background.alpha_composite(image.convert("RGBA"))
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        margin = max(8, round(min(image.size) * GENERATIVE_CANVAS_MARGIN_RATIO))
        canvas = Image.new("RGB", (image.width + margin * 2, image.height + margin * 2), "white")
        canvas.paste(image, (margin, margin))
        output = BytesIO()
        canvas.save(output, format="PNG", optimize=True)
        return output.getvalue()


def postprocess_generative_lineart_bytes(data: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    """Normalize model output without destroying clean antialiased edges.

    Generated drawings are already illustrations, not photographs.  Applying
    the legacy high-threshold binary route expands every pale antialias pixel
    into solid black, which makes small shipping-mark art look heavy and
    jagged.  This route removes color and background haze while retaining the
    grayscale edge ramp seen in the temporary golden references.
    """

    warnings: list[dict[str, Any]] = []
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGBA", image.size, "WHITE")
            background.alpha_composite(image.convert("RGBA"))
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        image.thumbnail((MAX_LINEART_SIZE, MAX_LINEART_SIZE), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(image)
        # Preserve dark strokes and their antialiased shoulders, but force the
        # near-white model canvas to an actually white printable background.
        gray = gray.point(lambda value: 255 if value >= 248 else (0 if value <= 24 else value))
        gray, canvas_metrics = normalize_lineart_canvas(gray)
        output = BytesIO()
        gray.convert("RGB").save(output, format="PNG", optimize=True)
        if image.width == MAX_LINEART_SIZE or image.height == MAX_LINEART_SIZE:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "lineart_resized",
                    "message": "Lineart image was resized to the maximum candidate size.",
                }
            )
        if canvas_metrics["low_effective_resolution"]:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "lineart_low_effective_pixels",
                    "message": "Lineart ink envelope has too few effective pixels for reliable output.",
                }
            )
        return output.getvalue(), warnings
