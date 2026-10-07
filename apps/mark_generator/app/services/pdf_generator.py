from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.graphics.barcode import createBarcodeDrawing
from reportlab.graphics import renderPDF
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .ean13 import is_valid_ean13
from .generation_report import write_generation_report
from .lineart import (
    resolve_project_dir,
    resolve_review_lineart_candidate,
    validate_lineart_provenance,
)
from .lineart_canvas import LineartCanvasError, inspect_lineart_file, normalized_lineart_png_bytes
from .manifest import write_manifest
from .template_renderer import TemplateAssetError, render_grid_pdf


PAIRS_PER_PAGE = 24

ADAPTIVE_TEMPLATE_MODE = "kmt_style_v1_adaptive"


def get_template_mode() -> str:
    """Which renderer to use.

    ``kmt_style_v1``          golden-base grid sheet (default, unchanged)
    ``kmt_style_v1_minimal``  legacy per-SKU pages
    ``kmt_style_v1_adaptive`` carton-size adaptive, one page per SKU

    Adaptive is opt-in. It needs a carton dimension table, and it changes the
    output page geometry, so it must never become the default by accident.
    """
    return os.environ.get("MARK_TEMPLATE_MODE", "kmt_style_v1")


class GenerationError(ValueError):
    pass


def generate_project_pdf(workspace_root: Path | str, project_id: str, *, mode: str) -> dict[str, Any]:
    if mode not in {"preview", "standard"}:
        raise GenerationError("mode must be preview or standard.")
    project_dir = resolve_project_dir(workspace_root, project_id)
    generated_dir = project_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    normalized = load_normalized(project_dir)
    items = normalized.get("items", [])
    warnings: list[dict[str, Any]] = []
    errors = validate_items(project_dir, items, mode)
    item_results = build_item_results(project_dir, items, mode, errors)

    output_name = "preview.pdf" if mode == "preview" else "standard_mark.pdf"
    output_path = generated_dir / output_name

    template_mode = get_template_mode()
    if errors and mode == "standard":
        if output_path.exists():
            output_path.unlink()
        report = write_generation_report(
            project_dir,
            project_id=project_id,
            mode=mode,
            status="needs_review",
            items=item_results,
            warnings=warnings,
            errors=errors,
            template_mode=template_mode,
        )
        manifest = write_manifest(project_dir, project_id, "needs_review", warnings, errors, template_mode=template_mode)
        return {"project_id": project_id, "status": "needs_review", "report": report, "manifest": manifest}

    if mode == "preview":
        warnings.extend(preview_warnings(project_dir, items))

    if template_mode == ADAPTIVE_TEMPLATE_MODE:
        adaptive_errors = render_carton_adaptive(
            project_dir, items, output_path, mode=mode, warnings=warnings
        )
        if adaptive_errors:
            # Fail closed: a carton-adaptive request with no usable carton data
            # must not quietly fall back to the fixed panel size, because the
            # resulting mark would not match the carton it is printed on.
            if output_path.exists():
                output_path.unlink()
            report = write_generation_report(
                project_dir,
                project_id=project_id,
                mode=mode,
                status="needs_review",
                items=item_results,
                warnings=warnings,
                errors=adaptive_errors,
                template_mode=template_mode,
                report_relative_path=("reports/preview_generation_report.json" if mode == "preview"
                                      else "reports/generation_report.json"),
            )
            manifest = _read_existing_manifest(project_dir) if mode == "preview" else write_manifest(
                project_dir, project_id, "needs_review", warnings, adaptive_errors,
                template_mode=template_mode,
            )
            return {
                "project_id": project_id,
                "status": "needs_review",
                "report": report,
                "manifest": manifest,
            }
    elif template_mode == "kmt_style_v1_minimal":
        fonts = register_fonts(warnings)
        draw_pdf(project_dir, items, output_path, mode=mode, fonts=fonts, warnings=warnings)
    else:
        try:
            render_grid_pdf(
                project_dir,
                items,
                output_path,
                mode=mode,
                resolve_image=lambda item: select_lineart_or_preview_image(project_dir, item, mode),
                warnings=warnings,
            )
        except TemplateAssetError as exc:
            raise GenerationError(str(exc)) from exc
    status = "success_with_warnings" if warnings else "success"
    report = write_generation_report(
        project_dir,
        project_id=project_id,
        mode=mode,
        status=status,
        items=build_item_results(project_dir, items, mode, []),
        warnings=warnings,
        errors=[],
        template_mode=template_mode,
        report_relative_path=(
            "reports/preview_generation_report.json"
            if mode == "preview"
            else "reports/generation_report.json"
        ),
    )
    if mode == "preview":
        manifest = _read_existing_manifest(project_dir)
    else:
        manifest = write_manifest(
            project_dir,
            project_id,
            "generated",
            warnings,
            [],
            template_mode=template_mode,
        )
    result = {
        "project_id": project_id,
        "status": status,
        "pdf": str(output_path),
        "pdf_path": f"generated/{output_name}",
        "report": report,
        "manifest": manifest,
    }
    if mode == "preview":
        result["diagnostic_only"] = True
        result["authoritative_state_unchanged"] = True
    return result


def render_carton_adaptive(
    project_dir: Path,
    items: list[dict[str, Any]],
    output_path: Path,
    *,
    mode: str,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render one page per SKU, sized to that SKU's real carton faces.

    Returns blocking errors; an empty list means the PDF was written. The layout
    decisions are archived to ``reports/layout_revision.json`` so every placed
    component's millimetre box, the family chosen and why, and the barcode's
    achieved GS1 magnification stay auditable after the fact.
    """
    from .adaptive_layout import render_adaptive_marks
    from .carton_specs import carton_facts_for_items, resolve_project_carton_specs

    spec = resolve_project_carton_specs(project_dir)
    errors = [issue for issue in spec.issues if issue.get("severity") == "error"]
    warnings.extend(issue for issue in spec.issues if issue.get("severity") != "error")
    if not spec.facts:
        return errors or [
            issue(
                "carton_spec_source_not_found",
                "Carton-adaptive layout needs carton dimensions and none were found.",
                severity="error",
            )
        ]

    cartons, missing = carton_facts_for_items(spec, items)
    if missing:
        # Every SKU in the run must have a carton, otherwise the PDF would mix
        # adapted and un-adapted marks with no way to tell them apart.
        return [{**entry, "severity": "error"} for entry in missing]

    result = render_adaptive_marks(
        project_dir,
        items,
        cartons,
        output_path,
        mode=mode,
        resolve_image=lambda item: select_lineart_or_preview_image(project_dir, item, mode),
        warnings=warnings,
    )

    revision = result.get("layout_revision")
    if revision is not None:
        reports_dir = project_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "layout_revision.json").write_text(
            json.dumps(revision, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    if result.get("status") != "rendered":
        return result.get("errors") or [
            issue(
                "manual_layout_required",
                "No SKU produced a valid carton-adaptive layout; see reports/layout_revision.json.",
                severity="error",
            )
        ]
    return []


def _read_existing_manifest(project_dir: Path) -> dict[str, Any] | None:
    """Read, but never rewrite, production state during diagnostic preview."""

    path = project_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_normalized(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "data" / "items.normalized.json"
    if not path.exists():
        raise FileNotFoundError("data/items.normalized.json not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_items(project_dir: Path, items: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for item in items:
        item_no = item.get("item_no")
        if not item_no:
            errors.append(issue("missing_item_no", "Item number is required.", item_no=item_no, severity="error"))
        qty = item.get("qty")
        if not isinstance(qty, (int, float)) or isinstance(qty, bool):
            errors.append(issue("invalid_qty", "Quantity must be numeric.", item_no=item_no, severity="error"))
        barcode = item.get("barcode")
        if not isinstance(barcode, str) or not barcode.isdigit() or len(barcode) != 13:
            errors.append(issue("invalid_barcode_format", "Barcode must be 13 digits.", item_no=item_no, severity="error"))
        elif not is_valid_ean13(barcode):
            errors.append(issue("invalid_ean13_checksum", "Barcode failed EAN-13 checksum validation.", item_no=item_no, severity="error"))
        if mode == "standard":
            binding_scope = (item.get("lineart_provenance") or {}).get("binding_scope")
            manual_without_region = binding_scope == "manual_item_without_product_region"
            if (
                not manual_without_region
                and (item.get("product_region_status") != "selected" or not item.get("selected_product_region"))
            ):
                errors.append(
                    issue(
                        "product_region_not_selected",
                        "Standard PDF requires a selected product region before lineart can be accepted.",
                        item_no=item_no,
                        severity="needs_review",
                    )
                )
            asset = item.get("lineart_asset")
            if item.get("lineart_status") == "stale":
                errors.append(
                    issue(
                        "stale_lineart",
                        "Lineart became stale after the selected product region changed.",
                        item_no=item_no,
                        severity="needs_review",
                    )
                )
            elif item.get("lineart_status") != "approved":
                errors.append(issue("missing_approved_lineart", "Standard PDF requires approved lineart.", item_no=item_no, severity="needs_review"))
            elif not asset or not (project_dir / asset).exists():
                errors.append(issue("missing_lineart_asset", "Approved lineart file does not exist.", item_no=item_no, severity="error", path=asset))
            else:
                provenance_issue = validate_lineart_provenance(project_dir, item)
                if provenance_issue:
                    errors.append(
                        issue(
                            provenance_issue["code"],
                            provenance_issue["message"],
                            item_no=item_no,
                            severity="needs_review",
                        )
                    )
                try:
                    canvas_metrics = inspect_lineart_file(project_dir / asset)
                    if canvas_metrics["low_effective_resolution"]:
                        errors.append(
                            issue(
                                "lineart_effective_pixels_too_low",
                                "Approved lineart has too few effective ink pixels for reliable rendering.",
                                item_no=item_no,
                                severity="needs_review",
                                path=asset,
                            )
                        )
                except (OSError, LineartCanvasError) as exc:
                    errors.append(
                        issue(
                            "lineart_canvas_invalid",
                            f"Approved lineart canvas could not be normalized: {exc}",
                            item_no=item_no,
                            severity="needs_review",
                            path=asset,
                        )
                    )
    return errors


def preview_warnings(project_dir: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings = []
    for item in items:
        if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
            continue
        candidate = resolve_review_lineart_candidate(project_dir, item)
        if candidate:
            warnings.append(
                issue(
                    "preview_uses_unapproved_lineart",
                    "Preview PDF used a region-bound lineart candidate that still requires approval.",
                    item_no=item.get("item_no"),
                    path=str(candidate.relative_to(project_dir).as_posix()),
                )
            )
        else:
            warnings.append(
                issue(
                    "preview_missing_lineart",
                    "Preview PDF left a NEEDS LINEART placeholder; source sales images are never used in mark layouts.",
                    item_no=item.get("item_no"),
                )
            )
    return warnings


def register_fonts(warnings: list[dict[str, Any]]) -> dict[str, str]:
    fonts = {"cn": "Helvetica", "latin": "Helvetica"}
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = {
        "cn": [
            windir / "Fonts" / "msyh.ttc",
            Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"),
        ],
        "latin": [
            windir / "Fonts" / "arial.ttf",
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ],
    }
    for key, paths in candidates.items():
        for path in paths:
            if path.exists():
                name = "MarkCnFont" if key == "cn" else "MarkLatinFont"
                try:
                    pdfmetrics.registerFont(TTFont(name, str(path)))
                    fonts[key] = name
                    break
                except Exception as exc:
                    warnings.append(issue("font_register_failed", f"Font registration failed: {exc}", path=str(path)))
        else:
            warnings.append(issue("font_fallback", f"{key} font was not found; using ReportLab default font."))
    return fonts


def draw_pdf(project_dir: Path, items: list[dict[str, Any]], output_path: Path, *, mode: str, fonts: dict[str, str], warnings: list[dict[str, Any]]) -> None:
    page_size = landscape(A4)
    pdf = canvas.Canvas(str(output_path), pagesize=page_size)
    width, height = page_size
    for item in items:
        draw_combined_mark(pdf, project_dir, item, width, height, mode, fonts, warnings)
        pdf.showPage()
    pdf.save()


def draw_combined_mark(
    pdf: canvas.Canvas,
    project_dir: Path,
    item: dict[str, Any],
    page_width: float,
    page_height: float,
    mode: str,
    fonts: dict[str, str],
    warnings: list[dict[str, Any]],
) -> None:
    margin = 8 * mm
    gap = 3 * mm
    panel_width = (page_width - margin * 2 - gap) / 2
    panel_height = page_height - margin * 2
    front = (margin, margin, panel_width, panel_height)
    side = (margin + panel_width + gap, margin, panel_width, panel_height)
    draw_front_panel(pdf, project_dir, item, front, mode, fonts, warnings)
    draw_side_panel(pdf, item, side, fonts)
    pdf.setFont(fonts["latin"], 5.5)
    pdf.drawRightString(page_width - margin, 3 * mm, "kmt_style_v1_minimal")


def draw_front_panel(
    pdf: canvas.Canvas,
    project_dir: Path,
    item: dict[str, Any],
    panel: tuple[float, float, float, float],
    mode: str,
    fonts: dict[str, str],
    warnings: list[dict[str, Any]],
) -> None:
    x, y, width, height = panel
    pad = 4 * mm
    pdf.setLineWidth(1.2)
    pdf.rect(x, y, width, height)
    header_h = 31 * mm
    draw_sku_header(pdf, item.get("item_no", ""), x + pad, y + height - pad - header_h, width - pad * 2, header_h)

    content_top = y + height - pad - header_h - 3 * mm
    pdf.setFont(fonts["latin"], 21)
    pdf.drawString(x + pad + 1 * mm, content_top - 7 * mm, "Kmt")
    pdf.setFont(fonts["latin"], 17)
    pdf.drawString(x + pad + 22 * mm, content_top - 7 * mm, "Style")

    text_x = x + pad + 1 * mm
    text_y = content_top - 17 * mm
    draw_fit_line(pdf, item.get("cn_name", ""), text_x, text_y, 80 * mm, fonts["cn"], 8.5, bold=True)
    draw_fit_line(pdf, str(item.get("it_name", "")).upper(), text_x, text_y - 6 * mm, 80 * mm, fonts["latin"], 7.8, bold=True)
    draw_fit_line(pdf, str(item.get("es_name", "")).upper(), text_x, text_y - 12 * mm, 80 * mm, fonts["latin"], 7.8, bold=True)
    draw_fit_line(pdf, str(item.get("en_name", "")).upper(), text_x, text_y - 18 * mm, 80 * mm, fonts["latin"], 7.8, bold=True)

    image_path = select_lineart_or_preview_image(project_dir, item, mode)
    image_box = (x + width - pad - 75 * mm, y + 45 * mm, 34 * mm, 54 * mm)
    draw_image_or_placeholder(pdf, image_path, image_box, item.get("item_no", ""), warnings)

    barcode_box = (x + width - pad - 18 * mm, y + 45 * mm, 14 * mm, 52 * mm)
    draw_barcode_vertical(pdf, item["barcode"], *barcode_box)

    pdf.setFont(fonts["latin"], 6.5)
    pdf.drawString(x + pad + 1 * mm, y + 24 * mm, "MADE IN CHINA")
    pdf.drawString(x + pad + 32 * mm, y + 24 * mm, "HECHO EN CHINA")
    draw_icon_placeholders(pdf, x + pad + 1 * mm, y + 12 * mm, 8)


def draw_side_panel(pdf: canvas.Canvas, item: dict[str, Any], panel: tuple[float, float, float, float], fonts: dict[str, str]) -> None:
    x, y, width, height = panel
    pad = 4 * mm
    pdf.setLineWidth(1.2)
    pdf.rect(x, y, width, height)
    header_h = 31 * mm
    draw_sku_header(pdf, item.get("item_no", ""), x + pad, y + height - pad - header_h, width - pad * 2, header_h)

    table_top = y + height - pad - header_h
    left_x = x + pad
    left_w = 68 * mm
    right_x = x + pad + left_w + 4 * mm
    barcode_box = (x + width - pad - 20 * mm, y + 24 * mm, 16 * mm, 58 * mm)

    draw_side_quantity_box(pdf, left_x, table_top - 38 * mm, left_w, 29 * mm, item, fonts)
    draw_side_meas_box(pdf, left_x, table_top - 68 * mm, left_w, 24 * mm, fonts)
    draw_side_info_table(pdf, right_x, table_top - 68 * mm, width - (right_x - x) - 24 * mm, 59 * mm, item, fonts)
    draw_barcode_vertical(pdf, item["barcode"], *barcode_box)


def draw_sku_header(pdf: canvas.Canvas, item_no: str, x: float, y: float, width: float, height: float) -> None:
    pdf.setFillColorRGB(0.08, 0.06, 0.05)
    pdf.rect(x, y, width, height, fill=1, stroke=0)
    pdf.setFillColorRGB(1, 1, 1)
    draw_fit_line(pdf, item_no, x + 2 * mm, y + 5 * mm, width - 4 * mm, "Helvetica-Bold", height * 0.9, bold=True)
    pdf.setFillColorRGB(0, 0, 0)


def draw_fit_line(
    pdf: canvas.Canvas,
    text: Any,
    x: float,
    y: float,
    max_width: float,
    font_name: str,
    font_size: float,
    *,
    bold: bool = False,
) -> None:
    value = "" if text is None else str(text)
    if not value:
        return
    selected_font = font_name
    if bold and font_name == "Helvetica":
        selected_font = "Helvetica-Bold"
    size = font_size
    while size > 4 and pdfmetrics.stringWidth(value, selected_font, size) > max_width:
        size -= 0.5
    pdf.setFont(selected_font, size)
    pdf.drawString(x, y, value)


def draw_multiline_fit(
    pdf: canvas.Canvas,
    text: Any,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
    font_name: str,
    font_size: float,
) -> None:
    value = "" if text is None else str(text)
    if not value:
        return
    words = value.split()
    if len(words) <= 1:
        draw_fit_line(pdf, value, x, y + max_height / 2 - 3, max_width, font_name, font_size, bold=True)
        return
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    lines = lines[:2]
    size = font_size
    while size > 4:
        line_height = size + 2
        if line_height * len(lines) <= max_height and all(pdfmetrics.stringWidth(line, font_name, size) <= max_width for line in lines):
            break
        size -= 0.5
    pdf.setFont(font_name, size)
    start_y = y + max_height - size
    for index, line in enumerate(lines):
        pdf.drawString(x, start_y - index * (size + 2), line)


def draw_icon_placeholders(pdf: canvas.Canvas, x: float, y: float, count: int) -> None:
    size = 7 * mm
    for index in range(count):
        ix = x + index * (size + 1.5 * mm)
        pdf.rect(ix, y, size, size)
        pdf.setFont("Helvetica-Bold", 4.8)
        pdf.drawCentredString(ix + size / 2, y + size / 2 - 1.5, str(index + 1))


def draw_side_quantity_box(pdf: canvas.Canvas, x: float, y: float, width: float, height: float, item: dict[str, Any], fonts: dict[str, str]) -> None:
    label_w = 22 * mm
    pdf.rect(x, y, width, height)
    pdf.setFillColorRGB(0.08, 0.06, 0.05)
    pdf.rect(x, y, label_w, height, fill=1, stroke=0)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(x + label_w / 2, y + height / 2 - 4, "Q.TY")
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 19)
    qty_text = str(item.get("qty", ""))
    pdf.drawCentredString(x + label_w + (width - label_w) * 0.43, y + height / 2 - 6, qty_text)
    pdf.setFont(fonts["latin"], 10)
    pdf.drawString(x + width - 17 * mm, y + height / 2 - 4, "PCS")


def draw_side_meas_box(pdf: canvas.Canvas, x: float, y: float, width: float, height: float, fonts: dict[str, str]) -> None:
    label_w = 22 * mm
    pdf.rect(x, y, width, height)
    pdf.setFillColorRGB(0.08, 0.06, 0.05)
    pdf.rect(x, y, label_w, height, fill=1, stroke=0)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawCentredString(x + label_w / 2, y + height / 2 - 4, "MEAS")
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont(fonts["latin"], 10)
    pdf.drawRightString(x + width - 4 * mm, y + height / 2 - 4, "cm")


def draw_side_info_table(pdf: canvas.Canvas, x: float, y: float, width: float, height: float, item: dict[str, Any], fonts: dict[str, str]) -> None:
    rows = [
        ("NAME", f"ES.{str(item.get('es_name', '')).upper()}"),
        ("N.W.", "kg"),
        ("G.W.", "kg"),
    ]
    row_h = height / len(rows)
    label_w = 21 * mm
    for index, (label, value) in enumerate(rows):
        ry = y + height - (index + 1) * row_h
        pdf.rect(x, ry, width, row_h)
        pdf.setFillColorRGB(0.08, 0.06, 0.05)
        pdf.rect(x, ry, label_w, row_h, fill=1, stroke=0)
        pdf.setFillColorRGB(1, 1, 1)
        pdf.setFont("Helvetica-Bold", 7.5)
        pdf.drawCentredString(x + label_w / 2, ry + row_h / 2 - 3, label)
        pdf.setFillColorRGB(0, 0, 0)
        if label == "NAME":
            draw_multiline_fit(pdf, value, x + label_w + 2 * mm, ry + 4 * mm, width - label_w - 4 * mm, row_h - 7 * mm, fonts["latin"], 8.5)
        else:
            pdf.setFont(fonts["latin"], 10)
            pdf.drawRightString(x + width - 4 * mm, ry + row_h / 2 - 4, value)


def select_lineart_or_preview_image(project_dir: Path, item: dict[str, Any], mode: str) -> Path | None:
    if item.get("lineart_status") == "approved" and item.get("lineart_asset"):
        path = project_dir / item["lineart_asset"]
        if path.exists():
            return path
    if mode == "preview":
        return resolve_review_lineart_candidate(project_dir, item)
    return None


def draw_image_or_placeholder(pdf: canvas.Canvas, image_path: Path | None, box: tuple[float, float, float, float], item_no: str, warnings: list[dict[str, Any]]) -> None:
    x, y, width, height = box
    pdf.rect(x, y, width, height)
    if not image_path:
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(x + width / 2, y + height / 2, "NEEDS LINEART")
        return
    try:
        pdf.drawImage(ImageReader(optimized_image_reader(image_path)), x + 2 * mm, y + 2 * mm, width - 4 * mm, height - 4 * mm, preserveAspectRatio=True, mask="auto")
    except Exception as exc:
        warnings.append(issue("lineart_draw_failed", f"Could not draw lineart image: {exc}", item_no=item_no, path=str(image_path)))
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(x + width / 2, y + height / 2, "IMAGE ERROR")


def optimized_image_reader(image_path: Path) -> BytesIO:
    normalized, _ = normalized_lineart_png_bytes(image_path.read_bytes())
    output = BytesIO(normalized)
    output.seek(0)
    return output


def draw_barcode(pdf: canvas.Canvas, barcode: str, x: float, y: float, width: float, height: float) -> None:
    drawing = createBarcodeDrawing("EAN13", value=barcode, barHeight=height * 0.72, humanReadable=True)
    scale = min(width / drawing.width, height / drawing.height)
    drawing.scale(scale, scale)
    renderPDF.draw(drawing, pdf, x, y)


def draw_barcode_vertical(pdf: canvas.Canvas, barcode: str, x: float, y: float, width: float, height: float) -> None:
    pdf.saveState()
    pdf.translate(x + width, y)
    pdf.rotate(90)
    draw_barcode(pdf, barcode, 0, 0, height, width)
    pdf.restoreState()


def build_item_results(project_dir: Path, items: list[dict[str, Any]], mode: str, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    error_items = {error.get("item_no") for error in errors}
    grid_layout = get_template_mode() != "kmt_style_v1_minimal"
    results = []
    for index, item in enumerate(items):
        item_no = item.get("item_no")
        lineart_status = item.get("lineart_status", "missing")
        status = "generated" if mode == "standard" and item_no not in error_items else "preview_only"
        if item_no in error_items:
            status = "needs_review"
        page = index // PAIRS_PER_PAGE + 1 if grid_layout else index + 1
        results.append(
            {
                "item_no": item_no or "",
                "status": status,
                "generated_marks": ["front", "side"] if item_no not in error_items or mode == "preview" else [],
                "lineart_status": lineart_status,
                "lineart_asset": item.get("lineart_asset"),
                "barcode_status": "valid" if is_valid_ean13(item.get("barcode", "")) else "invalid",
                "pages": {"front": page, "side": page},
                "warnings": [],
                "errors": [error for error in errors if error.get("item_no") == item_no],
            }
        )
    return results


def issue(code: str, message: str, *, severity: str = "warning", item_no: str | None = None, path: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if item_no:
        payload["item_no"] = item_no
    if path:
        payload["path"] = path
    return payload
