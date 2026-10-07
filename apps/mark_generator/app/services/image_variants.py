"""Bandwidth-friendly display variants of project images and mark previews.

The workbench is used over slow, high-latency links (public server ~400 ms RTT,
~5 Mbit/s). Contract product photos are ~850 KB and full mark previews are
~180 KB PNGs, while most of the UI shows them as 52–360 px thumbnails. This
module derives downscaled WebP variants, caches them on disk keyed by the source
file's identity, and exposes an ETag so browsers can revalidate cheaply.

Variants are display-only. Originals stay the authoritative files: production
PDFs, verification and manual cropping never read from this cache.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from .mark_preview import render_standard_mark_pair_pixmap, standard_mark_preview_sources


# Widths are snapped to a small ladder so the cache stays bounded and shared
# between views that ask for similar sizes.
WIDTH_LADDER = (160, 320, 480, 720, 960, 1280, 1600)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
LOSSY_QUALITY = 82
CACHE_VERSION = "1"
MAX_SOURCE_PIXELS = 60_000_000


class ImageVariantError(ValueError):
    """The requested variant cannot be produced from this source."""


@dataclass(frozen=True)
class ImageVariant:
    body: bytes
    etag: str
    media_type: str = "image/webp"


def snap_width(width: int | None) -> int | None:
    if width is None:
        return None
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ImageVariantError("Width must be a positive integer.")
    for step in WIDTH_LADDER:
        if width <= step:
            return step
    return WIDTH_LADDER[-1]


def _file_identity(path: Path) -> str:
    stat = path.stat()
    return f"{path.as_posix()}|{stat.st_mtime_ns}|{stat.st_size}"


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join((CACHE_VERSION, *parts)).encode("utf-8"))
    return digest.hexdigest()[:32]


def _cache_dir(workspace_root: Path | str) -> Path:
    return Path(workspace_root) / ".runtime" / "image-variants"


def _cached(workspace_root: Path | str, key: str, build) -> bytes:
    cache_dir = _cache_dir(workspace_root)
    target = cache_dir / key[:2] / f"{key}.webp"
    try:
        return target.read_bytes()
    except FileNotFoundError:
        pass
    body = build()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publish: concurrent requests may build the same variant.
        handle, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        with os.fdopen(handle, "wb") as stream:
            stream.write(body)
        os.replace(temp_name, target)
    except OSError:
        # A read-only or full disk must not break image display.
        pass
    return body


def _encode_webp(image: Image.Image, *, lossless: bool) -> bytes:
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    buffer = BytesIO()
    if lossless:
        image.save(buffer, format="WEBP", lossless=True, quality=80, method=4)
    else:
        image.save(buffer, format="WEBP", quality=LOSSY_QUALITY, method=4)
    return buffer.getvalue()


def project_image_variant(workspace_root: Path | str, source: Path, width: int) -> ImageVariant:
    """Return a WebP no wider than ``width`` (snapped) for a project image file."""

    snapped = snap_width(width)
    if source.suffix.lower() not in IMAGE_SUFFIXES:
        raise ImageVariantError("Only PNG, JPEG and WebP project images have display variants.")
    if not source.is_file():
        raise FileNotFoundError("Project image not found.")
    key = _cache_key("file", _file_identity(source), str(snapped))

    def build() -> bytes:
        try:
            with Image.open(source) as opened:
                if opened.width * opened.height > MAX_SOURCE_PIXELS:
                    raise ImageVariantError("Source image is too large for a display variant.")
                opened.load()
                image = opened.copy()
        except ImageVariantError:
            raise
        except Exception as exc:
            raise ImageVariantError("Project image could not be decoded.") from exc
        if image.width > snapped:
            height = max(1, round(image.height * snapped / image.width))
            image = image.resize((snapped, height), Image.Resampling.LANCZOS)
        # Line art and cutouts keep transparency; photos become lossy WebP.
        return _encode_webp(image, lossless=False)

    return ImageVariant(body=_cached(workspace_root, key, build), etag=f'"v{key}"')


def standard_mark_preview_variant(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    width: int | None,
) -> ImageVariant:
    """Return one front/side pair as WebP.

    ``width`` given: a lossy thumbnail for galleries. ``width`` omitted: a
    lossless 144 dpi render, pixel-identical in content to ``preview.png``.
    """

    snapped = snap_width(width)
    sources = standard_mark_preview_sources(workspace_root, project_id, item_no)
    key = _cache_key(
        "mark",
        project_id,
        item_no,
        *(_file_identity(path) for path in sources),
        str(snapped or "full"),
    )

    def build() -> bytes:
        pixmap = render_standard_mark_pair_pixmap(
            workspace_root, project_id, item_no, target_width=snapped
        )
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        if snapped and image.width != snapped:
            # MuPDF rounds the clip outward; pin the ladder width exactly.
            height = max(1, round(image.height * snapped / image.width))
            image = image.resize((snapped, height), Image.Resampling.LANCZOS)
        return _encode_webp(image, lossless=snapped is None)

    return ImageVariant(body=_cached(workspace_root, key, build), etag=f'"v{key}"')


def standard_mark_preview_etag(workspace_root: Path | str, project_id: str, item_no: str) -> str:
    sources = standard_mark_preview_sources(workspace_root, project_id, item_no)
    key = _cache_key("mark-png", project_id, item_no, *(_file_identity(path) for path in sources))
    return f'"v{key}"'


__all__ = [
    "ImageVariant",
    "ImageVariantError",
    "WIDTH_LADDER",
    "project_image_variant",
    "snap_width",
    "standard_mark_preview_etag",
    "standard_mark_preview_variant",
]
