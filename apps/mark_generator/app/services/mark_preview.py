"""Render one Kmt Style v1 front/side mark pair from a generated grid PDF."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .lineart import resolve_project_dir, resolve_project_file
from .template_renderer import ASSETS_DIR


PREVIEW_DPI = 144
PDF_CANDIDATES = ("generated/standard_mark.pdf",)
PNG_SIGNATURE = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))


class MarkPreviewError(ValueError):
    """The requested preview cannot be derived safely from project data."""


class MarkPreviewNotFound(FileNotFoundError):
    """The requested item or generated PDF does not exist."""


def render_standard_mark_pair_png(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    *,
    dpi: int = PREVIEW_DPI,
) -> bytes:
    """Return a deterministic PNG containing one complete front/side pair.

    The item position is derived only from the order in
    ``data/items.normalized.json`` and the installed Kmt Style v1 grid
    configuration. Only the generated standard PDF is authoritative. A
    diagnostic preview PDF must never be presented through this endpoint as a
    standard-mark artifact.
    """

    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 72 <= dpi <= 300:
        raise MarkPreviewError("Preview DPI must be an integer between 72 and 300.")
    pixmap = render_standard_mark_pair_pixmap(workspace_root, project_id, item_no, dpi=dpi)
    try:
        png = pixmap.tobytes("png")
    except Exception as exc:
        raise MarkPreviewError("Generated mark pair could not be rendered.") from exc
    if not png.startswith(PNG_SIGNATURE):
        raise MarkPreviewError("Rendered mark preview is not a valid PNG.")
    return png


def standard_mark_preview_sources(
    workspace_root: Path | str, project_id: str, item_no: str
) -> tuple[Path, ...]:
    """Files whose content fully determines one pair preview (cache identity)."""

    _validate_item_no(item_no)
    project_dir = resolve_project_dir(workspace_root, project_id)
    normalized_path = resolve_project_file(project_dir, "data/items.normalized.json")
    if not normalized_path.is_file():
        raise MarkPreviewNotFound("data/items.normalized.json not found.")
    return (
        _select_generated_pdf(project_dir),
        normalized_path,
        ASSETS_DIR / "template.json",
    )


def render_standard_mark_pair_pixmap(
    workspace_root: Path | str,
    project_id: str,
    item_no: str,
    *,
    dpi: int = PREVIEW_DPI,
    target_width: int | None = None,
) -> fitz.Pixmap:
    """Render one pair crop as an RGB pixmap at ``dpi`` or at ``target_width`` px."""

    _validate_item_no(item_no)
    project_dir = resolve_project_dir(workspace_root, project_id)
    items = _load_items(project_dir)
    item_index = _find_unique_item_index(items, item_no)
    config = _load_kmt_config()
    page_index, pair_rect = _pair_location(config, item_index)
    pdf_path = _select_generated_pdf(project_dir)

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise MarkPreviewError("Generated mark PDF could not be opened.") from exc

    try:
        if page_index >= document.page_count:
            raise MarkPreviewError(
                "Generated mark PDF does not contain the page required for this item."
            )
        page = document[page_index]
        _ensure_rect_within_page(pair_rect, page.rect)
        scale = target_width / pair_rect.width if target_width else dpi / 72.0
        return page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=pair_rect,
            colorspace=fitz.csRGB,
            alpha=False,
            annots=False,
        )
    except MarkPreviewError:
        raise
    except Exception as exc:
        raise MarkPreviewError("Generated mark pair could not be rendered.") from exc
    finally:
        document.close()


def _validate_item_no(item_no: str) -> None:
    if (
        not isinstance(item_no, str)
        or not item_no
        or len(item_no) > 200
        or item_no in {".", ".."}
        or "/" in item_no
        or "\\" in item_no
        or "\x00" in item_no
    ):
        raise MarkPreviewError("Invalid item_no.")


def _load_items(project_dir: Path) -> list[dict[str, Any]]:
    normalized_path = resolve_project_file(project_dir, "data/items.normalized.json")
    if not normalized_path.is_file():
        raise MarkPreviewNotFound("data/items.normalized.json not found.")
    try:
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MarkPreviewError("data/items.normalized.json is not readable JSON.") from exc
    items = normalized.get("items")
    if not isinstance(items, list):
        raise MarkPreviewError("items.normalized.json must contain an items list.")
    if not all(isinstance(item, dict) for item in items):
        raise MarkPreviewError("items.normalized.json contains an invalid item entry.")
    return items


def _find_unique_item_index(items: list[dict[str, Any]], item_no: str) -> int:
    matches = [
        index
        for index, item in enumerate(items)
        if item.get("item_no") == item_no
    ]
    if not matches:
        raise MarkPreviewNotFound(f"Item not found: {item_no}")
    if len(matches) > 1:
        raise MarkPreviewError(f"Duplicate item_no in normalized data: {item_no}")
    return matches[0]


def _load_kmt_config() -> dict[str, Any]:
    config_path = ASSETS_DIR / "template.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MarkPreviewError("Kmt Style v1 template configuration is unavailable.") from exc
    if config.get("template_id") != "kmt_style_v1":
        raise MarkPreviewError("Installed template is not Kmt Style v1.")
    return config


def _pair_location(config: dict[str, Any], item_index: int) -> tuple[int, fitz.Rect]:
    try:
        grid = config["grid"]
        panel = config["panel"]
        per_page = _positive_int(grid["pairs_per_page"], "pairs_per_page")
        columns = _positive_int(grid["columns"], "columns")
        rows = _positive_int(grid["rows"], "rows")
        panel_width = _positive_number(panel["width"], "panel.width")
        panel_height = _positive_number(panel["height"], "panel.height")
        margin_left = _number(grid["margin_left"], "margin_left")
        margin_top = _number(grid["margin_top"], "margin_top")
        pair_pitch_x = _positive_number(grid["pair_pitch_x"], "pair_pitch_x")
        row_pitch_y = _positive_number(grid["row_pitch_y"], "row_pitch_y")
        side_offset_x = _positive_number(grid["side_offset_x"], "side_offset_x")
    except (KeyError, TypeError) as exc:
        raise MarkPreviewError("Kmt Style v1 grid configuration is incomplete.") from exc

    if per_page > columns * rows:
        raise MarkPreviewError("Kmt Style v1 grid capacity is inconsistent.")
    page_index, slot = divmod(item_index, per_page)
    if grid.get("fill_order", "column_major") == "column_major":
        col, row = divmod(slot, rows)
    else:
        row, col = divmod(slot, columns)
    if not 0 <= col < columns or not 0 <= row < rows:
        raise MarkPreviewError("Item position is outside the configured Kmt grid.")

    front_x = margin_left + col * pair_pitch_x
    top = margin_top + row * row_pitch_y
    front = fitz.Rect(front_x, top, front_x + panel_width, top + panel_height)
    side_x = front_x + side_offset_x
    side = fitz.Rect(side_x, top, side_x + panel_width, top + panel_height)
    pair_rect = front | side
    if pair_rect.is_empty or pair_rect.is_infinite:
        raise MarkPreviewError("Kmt Style v1 pair crop is invalid.")
    return page_index, pair_rect


def _select_generated_pdf(project_dir: Path) -> Path:
    for relative_path in PDF_CANDIDATES:
        path = resolve_project_file(project_dir, relative_path)
        if path.exists():
            if not path.is_file():
                raise MarkPreviewError(f"Generated mark path is not a file: {relative_path}")
            return path
    raise MarkPreviewNotFound(
        "No generated standard mark PDF found (expected standard_mark.pdf)."
    )


def _ensure_rect_within_page(rect: fitz.Rect, page_rect: fitz.Rect) -> None:
    tolerance = 0.5
    if (
        rect.x0 < page_rect.x0 - tolerance
        or rect.y0 < page_rect.y0 - tolerance
        or rect.x1 > page_rect.x1 + tolerance
        or rect.y1 > page_rect.y1 + tolerance
    ):
        raise MarkPreviewError("Configured item crop lies outside the generated PDF page.")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarkPreviewError(f"Kmt grid value {label} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise MarkPreviewError(f"Kmt grid value {label} must be finite.")
    return number


def _positive_number(value: object, label: str) -> float:
    number = _number(value, label)
    if number <= 0:
        raise MarkPreviewError(f"Kmt grid value {label} must be positive.")
    return number


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MarkPreviewError(f"Kmt grid value {label} must be a positive integer.")
    return value
