"""Kmt Style v1 grid renderer.

Replicates the golden sheet layout: front/side mark pairs on one large page,
3 pairs per row x 8 rows (24 SKUs per page, more SKUs paginate). Static
artwork comes from the extracted base template
(assets/templates/kmt_style_v1/base.pdf); variable fields are drawn on top
using the coordinates in template.json (panel-local pt, produced by
tools/build_template.py from the golden PDF).

Route B hook: every coordinate lives in template.json, so a later
carton-size adaptation only needs to scale the panel target rects and field
coordinates (see the route_b note in template.json).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF

from .barcode_renderer import BarcodeRenderError, draw_ean13_bars, format_ean13_hri
from .ean13 import is_valid_ean13
from .font_subset import collection_face_buffer, preferred_families, subset_large_embedded_fonts
from .lineart_canvas import normalized_lineart_png_bytes


ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "templates" / "kmt_style_v1"


class TemplateAssetError(RuntimeError):
    pass


def issue(code: str, message: str, *, severity: str = "warning", item_no: str | None = None, path: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if item_no:
        payload["item_no"] = item_no
    if path:
        payload["path"] = path
    return payload


class KmtTemplateRenderer:
    """Renders the grid sheet for one project. Create per render call."""

    def __init__(self, assets_dir: Path | None = None) -> None:
        assets_dir = assets_dir or ASSETS_DIR
        template_json = assets_dir / "template.json"
        base_pdf = assets_dir / "base.pdf"
        if not template_json.exists() or not base_pdf.exists():
            raise TemplateAssetError(
                f"Template assets missing under {assets_dir}. Run tools/build_template.py first."
            )
        self.config = json.loads(template_json.read_text(encoding="utf-8"))
        self.base = fitz.open(base_pdf)
        self.font_warnings: list[dict[str, Any]] = []
        self.synthetic_bold_fonts: set[str] = set()
        self.fonts = self._load_fonts()

    def _load_fonts(self) -> dict[str, fitz.Font]:
        windir = os.environ.get("WINDIR", "C:/Windows")
        fonts: dict[str, fitz.Font] = {}
        self.font_files: dict[str, str | None] = {}
        self.font_fallbacks: dict[str, str] = {}
        for key, spec in self.config["fonts"].items():
            font = None
            self.font_fallbacks[key] = spec["fallback"]
            self.font_files[key] = None
            configured_paths = [
                Path(candidate.replace("{WINDIR}", windir))
                for candidate in spec["candidates"]
            ]
            candidate_paths = [
                *configured_paths,
                *_fontconfig_font_paths(spec.get("fontconfig_families", [])),
            ]
            families = preferred_families(spec.get("fontconfig_families", []))
            for path in dict.fromkeys(candidate_paths):
                if path.exists():
                    try:
                        # A collection opens at face 0 (Noto CJK: JP); pick the preferred face.
                        face = collection_face_buffer(path, families)
                        candidate_font = fitz.Font(fontbuffer=face) if face else fitz.Font(fontfile=str(path))
                        requirement_issue = _font_requirement_issue(candidate_font, spec)
                        if requirement_issue:
                            self.font_warnings.append(
                                issue(
                                    "font_candidate_rejected",
                                    f"Could not use font {path} for {key}: {requirement_issue}",
                                    path=str(path),
                                )
                            )
                            continue
                        font = candidate_font
                        self.font_files[key] = str(path)
                        break
                    except Exception as exc:
                        self.font_warnings.append(issue("font_load_failed", f"Could not load font {path}: {exc}", path=str(path)))
            if font is None:
                self.font_warnings.append(issue("font_fallback", f"{key} font not found; using built-in {spec['fallback']}."))
                font = fitz.Font(spec["fallback"])
                requirement_issue = _font_requirement_issue(font, spec)
                if requirement_issue and spec.get("required_flags", {}).get("bold"):
                    self.synthetic_bold_fonts.add(key)
                    self.font_warnings.append(
                        issue(
                            "font_synthetic_bold",
                            f"{key} has no real bold font; applying a stroked fallback: {requirement_issue}",
                        )
                    )
                elif requirement_issue:
                    raise TemplateAssetError(f"Fallback font for {key} is unusable: {requirement_issue}")
            fonts[key] = font
        return fonts

    # --- public API ----------------------------------------------------------

    def render(
        self,
        project_dir: Path,
        items: list[dict[str, Any]],
        output_path: Path,
        *,
        mode: str,
        resolve_image: Callable[[dict[str, Any]], Path | None],
        warnings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        warnings.extend(self.font_warnings)
        grid = self.config["grid"]
        per_page = grid["pairs_per_page"]
        columns = grid["columns"]
        page_count = max(1, math.ceil(len(items) / per_page))

        grid_rows = grid["rows"]
        column_major = grid.get("fill_order", "column_major") == "column_major"
        doc = fitz.open()
        for page_index in range(page_count):
            chunk = items[page_index * per_page : (page_index + 1) * per_page]
            if column_major:
                # golden sheet order: SKUs run down each column (8 per column)
                rows = max(1, min(len(chunk), grid_rows))
            else:
                rows = max(1, math.ceil(len(chunk) / columns))
            height = grid["margin_top"] + rows * grid["row_pitch_y"] - (grid["row_pitch_y"] - self.config["panel"]["height"]) + grid["margin_bottom"]
            page = doc.new_page(width=grid["page_width"], height=height)
            for slot, item in enumerate(chunk):
                if column_major:
                    col, row = divmod(slot, grid_rows)
                else:
                    row, col = divmod(slot, columns)
                fx = grid["margin_left"] + col * grid["pair_pitch_x"]
                fy = grid["margin_top"] + row * grid["row_pitch_y"]
                front_origin = fitz.Point(fx, fy)
                side_origin = fitz.Point(fx + grid["side_offset_x"], fy)
                self._render_pair(page, front_origin, side_origin, item, project_dir, mode, resolve_image, warnings)
            if mode == "preview":
                page.insert_text(
                    fitz.Point(grid["margin_left"], height - grid["margin_bottom"] / 2),
                    f"PREVIEW - {self.config['template_id']} - NOT FOR PRODUCTION",
                    fontsize=28,
                    fontname="helv",
                    color=(0.6, 0.6, 0.6),
                )
        doc.subset_fonts()
        warnings.extend(subset_large_embedded_fonts(doc))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path, garbage=3, deflate=True)
        doc.close()
        return {"pages": page_count, "pairs_per_page": per_page}

    # --- pair rendering -------------------------------------------------------

    def _render_pair(
        self,
        page: fitz.Page,
        front_origin: fitz.Point,
        side_origin: fitz.Point,
        item: dict[str, Any],
        project_dir: Path,
        mode: str,
        resolve_image: Callable[[dict[str, Any]], Path | None],
        warnings: list[dict[str, Any]],
    ) -> None:
        panel_w = self.config["panel"]["width"]
        panel_h = self.config["panel"]["height"]
        for origin, base_page in ((front_origin, 0), (side_origin, 1)):
            rect = fitz.Rect(origin.x, origin.y, origin.x + panel_w, origin.y + panel_h)
            page.show_pdf_page(rect, self.base, base_page)

        item_no = str(item.get("item_no") or "")
        front = self.config["front"]
        side = self.config["side"]
        self._draw_item_no(page, front_origin, front["item_no"], item_no)
        self._draw_item_no(page, side_origin, side["item_no"], item_no)
        self._draw_names(page, front_origin, front["names"], item)
        self._draw_lineart(page, front_origin, front["lineart"], item, project_dir, mode, resolve_image, warnings)
        barcode = str(item.get("barcode") or "")
        if is_valid_ean13(barcode):
            self._draw_barcode(page, front_origin, front["barcode"], barcode)
            self._draw_barcode(page, side_origin, side["barcode"], barcode)
        else:
            warnings.append(issue("barcode_skipped", "Barcode missing or invalid; barcode area left empty.", item_no=item_no or None))
        self._draw_qty(page, side_origin, side["qty"], item)
        self._draw_es_name(page, side_origin, side["es_name"], item)

    def _write_text(
        self,
        page: fitz.Page,
        pos: fitz.Point,
        text: str,
        font: fitz.Font,
        fontsize: float,
        *,
        color: tuple[float, float, float] = (0, 0, 0),
        hscale: float = 1.0,
        synthetic_bold: bool = False,
    ) -> None:
        writer = fitz.TextWriter(page.rect)
        writer.append(pos, text, font=font, fontsize=fontsize)
        morph = None
        if hscale < 0.999:
            morph = (pos, fitz.Matrix(hscale, 0, 0, 1, 0, 0))
        writer.write_text(page, color=color, morph=morph, render_mode=2 if synthetic_bold else 0)

    def _draw_item_no(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], item_no: str) -> None:
        """Draw SKU white text centered inside the black bar (bar_rect).

        Golden template measured baseline sat near the bottom of the bar; digits
        looked bottom-heavy. Prefer true center in bar_rect for both axes, and
        keep horizontal centering even when h-scale shrinks long item numbers.
        """
        if not item_no:
            return
        font = self.fonts[cfg["font"]]
        fontsize = float(cfg["fontsize"])
        width = float(font.text_length(item_no, fontsize))
        color = tuple(cfg["color"])

        if cfg.get("bar_rect"):
            bar = fitz.Rect(cfg["bar_rect"])
            # Keep a small inset so glyphs do not kiss the printed black edge.
            pad_x = max(2.0, min(bar.width * 0.02, 8.0))
            pad_y = max(2.0, min(bar.height * 0.04, 10.0))
            content = fitz.Rect(bar.x0 + pad_x, bar.y0 + pad_y, bar.x1 - pad_x, bar.y1 - pad_y)
            max_width = max(1.0, content.width)
            hscale = (max_width / width) if width > max_width > 0 else 1.0
            drawn_width = width * hscale
            x0 = content.x0 + (content.width - drawn_width) / 2.0
            # Vertical center by lining-figure ink, not full em-box.
            # Digit ink mid ≈ baseline - 0.36*fs (measured on Arial Bold 0-9 in this bar).
            # Em-box mid leaves digits high; golden baseline_y left them low.
            mid_y = (content.y0 + content.y1) / 2.0
            baseline_y = mid_y + 0.36 * fontsize
        else:
            # Legacy fields without bar_rect (should not happen for kmt_style_v1).
            max_width = float(cfg["max_width"])
            x0 = float(cfg["x0"])
            if width > max_width > 0:
                hscale = max_width / width
                drawn_width = width * hscale
                x0 = x0 + (max_width - drawn_width) / 2.0
            else:
                hscale = 1.0
                x0 += (max_width - width) / 2.0
            baseline_y = float(cfg["baseline_y"])

        pos = fitz.Point(origin.x + x0, origin.y + baseline_y)
        self._write_text(page, pos, item_no, font, fontsize, color=color, hscale=hscale)

    def _draw_names(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], item: dict[str, Any]) -> None:
        max_width = cfg["max_width"]
        min_fontsize = cfg["min_fontsize"]
        entries: list[dict[str, Any]] = []
        for line in cfg["lines"]:
            text = str(item.get(line["field"]) or "").strip()
            if line["field"] == "cn_name":
                text = abbreviate_cn_display_name(text)
            if line["field"] in {"it_name", "es_name", "en_name"}:
                text = text.upper()
            if not text:
                continue
            font = self.fonts[line["font"]]
            entries.append({"cfg": line, "text": text, "font": font})

        if not entries:
            return

        top_y = float(cfg.get("top_y", min(entry["cfg"]["baseline"][1] for entry in entries) - 18.0))
        bottom_y = float(cfg.get("bottom_y", max(entry["cfg"]["baseline"][1] for entry in entries) + 5.0))
        leading_ratio = float(cfg.get("leading_ratio", 1.15))
        block_gap = float(cfg.get("block_gap", 3.0))
        chosen: list[tuple[dict[str, Any], float, list[str]]] | None = None

        # Allocate the four languages as one typographic block. This avoids
        # shrinking one long name to a tiny single line while the other rows
        # remain oversized and visually unrelated.
        scale = 1.0
        while scale >= 0.55:
            layout: list[tuple[dict[str, Any], float, list[str]]] = []
            total_height = block_gap * max(0, len(entries) - 1)
            valid = True
            for entry in entries:
                fontsize = max(min_fontsize, float(entry["cfg"]["fontsize"]) * scale)
                max_lines = int(entry["cfg"].get("max_lines", cfg.get("max_lines_per_name", 2)))
                lines = wrap_display_text(entry["font"], entry["text"], fontsize, max_width)
                if len(lines) > max_lines:
                    valid = False
                    break
                layout.append((entry, fontsize, lines))
                total_height += len(lines) * fontsize * leading_ratio
            if valid and total_height <= bottom_y - top_y:
                chosen = layout
                break
            scale -= 0.05

        if chosen is None:
            # Extremely long data remains complete and legible at the declared
            # minimum. Horizontal squeezing is preferable to silent truncation.
            chosen = []
            for entry in entries:
                fontsize = min_fontsize
                lines = wrap_display_text(entry["font"], entry["text"], fontsize, max_width)
                max_lines = int(entry["cfg"].get("max_lines", 2))
                if len(lines) > max_lines:
                    lines = force_balanced_lines(entry["text"], max_lines)
                chosen.append((entry, fontsize, lines))

        cursor_y = top_y
        for entry, fontsize, lines in chosen:
            x0 = float(entry["cfg"]["baseline"][0])
            for text_line in lines:
                width = entry["font"].text_length(text_line, fontsize)
                hscale = min(1.0, max_width / width) if width else 1.0
                baseline_y = cursor_y + fontsize
                self._write_text(
                    page,
                    fitz.Point(origin.x + x0, origin.y + baseline_y),
                    text_line,
                    entry["font"],
                    fontsize,
                    hscale=hscale,
                    synthetic_bold=entry["cfg"]["font"] in self.synthetic_bold_fonts,
                )
                cursor_y += fontsize * leading_ratio
            cursor_y += block_gap

    def _draw_lineart(
        self,
        page: fitz.Page,
        origin: fitz.Point,
        cfg: dict[str, Any],
        item: dict[str, Any],
        project_dir: Path,
        mode: str,
        resolve_image: Callable[[dict[str, Any]], Path | None],
        warnings: list[dict[str, Any]],
    ) -> None:
        rect = fitz.Rect(cfg["rect"]) + (origin.x, origin.y, origin.x, origin.y)
        image_path = resolve_image(item)
        if not image_path:
            page.draw_rect(rect, color=(0.65, 0.65, 0.65), width=0.8)
            label = "NEEDS LINEART"
            fontsize = 11
            label_width = fitz.get_text_length(label, fontname="helv", fontsize=fontsize)
            page.insert_text(
                fitz.Point(rect.x0 + max(2, (rect.width - label_width) / 2), rect.y0 + rect.height / 2),
                label,
                fontsize=fontsize,
                fontname="helv",
                color=(0.55, 0.55, 0.55),
            )
            return
        try:
            # The line art is stretched freely when the layout says so. This used
            # to be hardcoded to keep_proportion=True, which meant no plan could
            # ever ask for a non-uniform line art and the owner's 2026-09-15
            # "线稿只能等比例放大，改成自由放大" could not be honoured by any rule.
            # "stretch" is now a real option the plan can choose; everything else
            # keeps the original aspect.
            keep_proportion = str(cfg.get("fit") or "contain_center") != "stretch"
            page.insert_image(
                rect, stream=optimized_image_bytes(image_path), keep_proportion=keep_proportion
            )
            # Approval state is recorded by the preview footer, manifest, and
            # reports.  A red slot outline is misleading production artwork
            # and falsely implies that every product must fill that boundary.
        except Exception as exc:
            warnings.append(
                issue("lineart_draw_failed", f"Could not draw lineart image: {exc}", item_no=item.get("item_no"), path=str(image_path))
            )

    def _draw_barcode(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], value: str) -> None:
        bars_rect = fitz.Rect(cfg["bars_rect"]) + (origin.x, origin.y, origin.x, origin.y)
        draw_ean13_bars(page, bars_rect, value, cfg.get("regular_bar_inset", 9.8))
        self._draw_barcode_readout(page, origin, cfg, value)

    def _draw_barcode_readout(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], value: str) -> None:
        if cfg.get("hri_layout", "module_groups") == "module_groups":
            self._draw_barcode_readout_groups(page, origin, cfg, value)
            return
        self._draw_barcode_readout_single_span(page, origin, cfg, value)

    def _draw_barcode_readout_groups(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], value: str) -> None:
        bars_rect = fitz.Rect(cfg["bars_rect"]) + (origin.x, origin.y, origin.x, origin.y)
        font_key = cfg.get("font", "barcode_hri")
        font = self.fonts.get(font_key)
        xscale = cfg.get("readout_xscale", 1.0)
        fontsize = cfg["text_fontsize"]
        character_spacing = cfg.get("character_spacing", 0.0)
        visual_gap = cfg.get("readout_gap", 0.8)
        regular_x1 = bars_rect.x1 - cfg.get("regular_bar_inset", 9.8)
        ascender = font.ascender if font else 0.8
        pivot_x = regular_x1 + visual_gap + fontsize * ascender * xscale
        first_digit_height = cfg.get("first_digit_height", 19.0)
        first_digit_offset = cfg.get("first_digit_offset", 16.0)
        font_args = self._barcode_font_args(font_key)
        self._insert_rotated_barcode_text(
            page,
            value[0],
            pivot_x,
            bars_rect.y1 + first_digit_offset - first_digit_height,
            bars_rect.y1 + first_digit_offset,
            fontsize,
            xscale,
            font,
            font_args,
        )

        main_texts = [value[1:7], value[7:]]
        main_gap = cfg.get("main_group_gap", 4.0)
        main_bottom = bars_rect.y1 - cfg.get("main_bottom_inset", 10.7)
        main_top = bars_rect.y0 + cfg.get("main_top_inset", 6.0)
        main_fontsize = fontsize
        if font:
            total_advance = sum(
                self._barcode_text_advance(text, main_fontsize, font, character_spacing) for text in main_texts
            ) + main_gap
            available = main_bottom - main_top
            if total_advance > available and total_advance > 0:
                main_fontsize *= available / total_advance

        lower_advance = self._barcode_text_advance(main_texts[0], main_fontsize, font, character_spacing)
        upper_advance = self._barcode_text_advance(main_texts[1], main_fontsize, font, character_spacing)
        lower_y1 = main_bottom
        lower_y0 = lower_y1 - lower_advance
        upper_y1 = lower_y0 - main_gap
        upper_y0 = upper_y1 - upper_advance
        self._insert_rotated_barcode_text(
            page,
            main_texts[0],
            pivot_x,
            lower_y0,
            lower_y1,
            main_fontsize,
            xscale,
            font,
            font_args,
            character_spacing,
        )
        self._insert_rotated_barcode_text(
            page,
            main_texts[1],
            pivot_x,
            upper_y0,
            upper_y1,
            main_fontsize,
            xscale,
            font,
            font_args,
            character_spacing,
        )

    def _barcode_text_advance(
        self, text: str, fontsize: float, font: fitz.Font | None, character_spacing: float = 0.0
    ) -> float:
        if font:
            text_advance = font.text_length(text, fontsize)
        else:
            text_advance = len(text) * fontsize * 0.55
        return text_advance + max(0, len(text) - 1) * character_spacing

    def _insert_rotated_barcode_text(
        self,
        page: fitz.Page,
        text: str,
        pivot_x: float,
        y0: float,
        y1: float,
        fontsize: float,
        xscale: float,
        font: fitz.Font | None,
        font_args: dict[str, str],
        character_spacing: float = 0.0,
    ) -> None:
        text_advance = self._barcode_text_advance(text, fontsize, font, character_spacing)
        pivot_y = (y0 + y1) / 2 + text_advance / 2
        if not character_spacing:
            pivot = fitz.Point(pivot_x, pivot_y)
            morph = (pivot, fitz.Matrix(xscale, 1.0)) if abs(xscale - 1.0) > 0.001 else None
            page.insert_text(pivot, text, fontsize=fontsize, rotate=90, morph=morph, **font_args)
            return

        cursor_y = pivot_y
        for index, character in enumerate(text):
            pivot = fitz.Point(pivot_x, cursor_y)
            morph = (pivot, fitz.Matrix(xscale, 1.0)) if abs(xscale - 1.0) > 0.001 else None
            page.insert_text(pivot, character, fontsize=fontsize, rotate=90, morph=morph, **font_args)
            if index < len(text) - 1:
                cursor_y -= self._barcode_text_advance(character, fontsize, font) + character_spacing

    def _draw_barcode_readout_single_span(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], value: str) -> None:
        try:
            text = format_ean13_hri(value, cfg.get("hri_grouping", "golden_1_6_6"))
        except BarcodeRenderError:
            text = value
        pivot = fitz.Point(origin.x + cfg["readout_origin"][0], origin.y + cfg["readout_origin"][1])
        font_key = cfg.get("font", "barcode_hri")
        xscale = cfg.get("readout_xscale", 1.0)
        morph = (pivot, fitz.Matrix(xscale, 1.0)) if abs(xscale - 1.0) > 0.001 else None
        font_args = self._barcode_font_args(font_key)
        page.insert_text(pivot, text, fontsize=cfg["text_fontsize"], rotate=90, morph=morph, **font_args)

    def _barcode_font_args(self, font_key: str) -> dict[str, str]:
        if self.font_files.get(font_key):
            return {"fontname": "KmtBarcodeHRI", "fontfile": self.font_files[font_key]}
        return {"fontname": self.font_fallbacks.get(font_key, "helv")}

    def _draw_qty(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], item: dict[str, Any]) -> None:
        qty = item.get("qty")
        if qty is None or isinstance(qty, bool):
            return
        if isinstance(qty, float) and qty.is_integer():
            qty = int(qty)
        text = str(qty)
        font = self.fonts[cfg["font"]]
        fontsize = cfg["fontsize"]
        rect = fitz.Rect(cfg["rect"])
        width = font.text_length(text, fontsize)
        if width > rect.width:
            fontsize = fontsize * rect.width / width
            width = font.text_length(text, fontsize)
        x = rect.x0 + (rect.width - width) / 2
        pos = fitz.Point(origin.x + x, origin.y + cfg["baseline_y"])
        self._write_text(page, pos, text, font, fontsize)

    def _draw_es_name(self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], item: dict[str, Any]) -> None:
        name = str(item.get("es_name") or "").strip().upper()
        if not name:
            return
        text = f"{cfg['prefix']}{name}"
        font = self.fonts[cfg["font"]]
        rect = fitz.Rect(cfg["rect"])
        layout = fit_side_name_text(
            font,
            text,
            max_width=rect.width,
            max_height=rect.height,
            fontsize=float(cfg["fontsize"]),
            min_fontsize=float(cfg["min_fontsize"]),
            preferred_max_lines=int(cfg.get("preferred_max_lines", 2)),
            max_lines=int(cfg["max_lines"]),
            leading_ratio=float(cfg.get("leading_ratio", 1.0)),
        )
        if layout is None:
            raise TemplateAssetError(
                "Side-mark Spanish name cannot fit the NAME value cell without clipping or horizontal distortion: "
                + text
            )
        fontsize, lines, baselines = layout
        for line, baseline in zip(lines, baselines):
            pos = fitz.Point(origin.x + rect.x0, origin.y + rect.y0 + baseline)
            self._write_text(page, pos, line, font, fontsize)


def wrap_words(font: fitz.Font, text: str, fontsize: float, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.text_length(candidate, fontsize) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fontconfig_font_paths(families: list[str]) -> list[Path]:
    executable = shutil.which("fc-match")
    if not executable:
        return []
    paths: list[Path] = []
    for family in families:
        try:
            result = subprocess.run(
                [executable, "-f", "%{file}\n", family],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for raw_path in result.stdout.splitlines():
            path = Path(raw_path.strip())
            if raw_path.strip() and path not in paths:
                paths.append(path)
    return paths


def _font_requirement_issue(font: fitz.Font, spec: dict[str, Any]) -> str:
    for flag, expected in spec.get("required_flags", {}).items():
        if bool(font.flags.get(flag)) != bool(expected):
            return f"required font flag {flag}={bool(expected)} is missing"
    missing = [character for character in spec.get("required_glyphs", "") if not font.has_glyph(ord(character))]
    if missing:
        return "required CJK glyph coverage is missing"
    return ""


def wrap_words_balanced(
    font: fitz.Font,
    text: str,
    fontsize: float,
    max_width: float,
    max_lines: int = 2,
) -> list[str]:
    """Prefer visually balanced two-line Latin names over greedy wrapping."""

    balanced = balanced_word_lines(font, text, fontsize, max_width, max_lines)
    if balanced:
        return balanced
    return wrap_words(font, text, fontsize, max_width)


def balanced_word_lines(
    font: fitz.Font,
    text: str,
    fontsize: float,
    max_width: float,
    max_lines: int,
) -> list[str] | None:
    """Return the best complete 1..N-line word partition that fits exactly."""

    words = text.split()
    if not words or max_lines < 1:
        return []
    best: tuple[tuple[int, float, float], list[str]] | None = None
    maximum = min(max_lines, len(words))
    # A word cannot be split; omit impossible partitions before enumerating
    # them (four-line fallback otherwise makes long rejected names costly).
    word_widths = [font.text_length(word, fontsize) for word in words]
    if any(width > max_width + 0.01 for width in word_widths):
        return None
    space_width = font.text_length(" ", fontsize)
    for line_count in range(1, maximum + 1):
        if sum(word_widths) + (len(words) - line_count) * space_width > line_count * (max_width + 0.01):
            continue
        for breaks in combinations(range(1, len(words)), line_count - 1):
            starts = (0, *breaks)
            ends = (*breaks, len(words))
            lines = [" ".join(words[start:end]) for start, end in zip(starts, ends)]
            widths = [font.text_length(line, fontsize) for line in lines]
            if any(width > max_width + 0.01 for width in widths):
                continue
            width_spread = max(widths) - min(widths)
            score = (line_count, max(widths), width_spread)
            if best is None or score < best[0]:
                best = (score, lines)
        if best is not None:
            break
    return best[1] if best else None


def fit_side_name_text(
    font: fitz.Font,
    text: str,
    *,
    max_width: float,
    max_height: float,
    fontsize: float,
    min_fontsize: float,
    preferred_max_lines: int,
    max_lines: int,
    leading_ratio: float,
) -> tuple[float, list[str], list[float]] | None:
    """Fit a complete side name without truncation or horizontal scaling."""

    line_limits = list(dict.fromkeys([min(preferred_max_lines, max_lines), max_lines]))
    for line_limit in line_limits:
        candidate_size = fontsize
        while candidate_size >= min_fontsize - 0.001:
            lines = balanced_word_lines(font, text, candidate_size, max_width, line_limit)
            if lines:
                line_step = candidate_size * leading_ratio
                ink_height = (font.ascender - font.descender) * candidate_size
                block_height = ink_height + max(0, len(lines) - 1) * line_step
                if block_height <= max_height + 0.01:
                    top = (max_height - block_height) / 2.0
                    first_baseline = top + font.ascender * candidate_size
                    baselines = [first_baseline + index * line_step for index in range(len(lines))]
                    return candidate_size, lines, baselines
            candidate_size = round(candidate_size - 0.5, 3)
    return None


_OPENING_PUNCTUATION = frozenset("([{（【「『《")
_CLOSING_PUNCTUATION = frozenset(")]}）】」』》，。、；：！？,;:!?")


def _attach_break_punctuation(tokens: list[str]) -> list[str]:
    """Line-break rule: an opening bracket never ends a line, closing punctuation never starts one.

    Without it "透明喷雾瓶50ML (带刻度)" wrapped with a lone ")" on its own line.
    """
    out: list[str] = []
    opening = ""
    for token in tokens:
        if token in _OPENING_PUNCTUATION:
            opening += token
            continue
        token, opening = opening + token, ""
        if out and token in _CLOSING_PUNCTUATION:
            out[-1] += token
        else:
            out.append(token)
    if opening:
        out.append(opening)
    return out


def wrap_display_text(font: fitz.Font, text: str, fontsize: float, max_width: float) -> list[str]:
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return wrap_words(font, text, fontsize, max_width)
    # Keep Latin specifications such as 50*70CM*5MM together while allowing
    # Chinese product wording to wrap at natural character boundaries.
    import re

    tokens = _attach_break_punctuation(re.findall(r"[A-Za-z0-9*×./%+-]+|[\u4e00-\u9fff]|\S", text))
    lines: list[str] = []
    current = ""
    for token in tokens:
        separator = " " if current and token[0].isascii() and current[-1].isascii() else ""
        candidate = f"{current}{separator}{token}" if current else token
        if current and font.text_length(candidate, fontsize) > max_width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def abbreviate_cn_display_name(text: str) -> str:
    """Create a display-only Chinese abbreviation without changing contract truth."""

    compact = " ".join(text.split())
    if len(compact) <= 18:
        return compact
    replacements = (
        ("生日快乐拍照道具框", "生日拍照框"),
        ("生日快乐拍照框", "生日拍照框"),
        ("拍照道具框", "拍照框"),
    )
    for source, target in replacements:
        compact = compact.replace(source, target)
    return compact


def force_balanced_lines(text: str, max_lines: int) -> list[str]:
    """Keep every character when an extreme value needs horizontal fitting."""

    import re

    if max_lines <= 1:
        return [text]
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        tokens = re.findall(r"[A-Za-z0-9*×./%+-]+|[\u4e00-\u9fff]|\S", text)
        separator = ""
    else:
        tokens = text.split()
        separator = " "
    if not tokens:
        return []
    lines: list[str] = []
    remaining = list(tokens)
    for index in range(max_lines):
        slots = max_lines - index
        take = max(1, (len(remaining) + slots - 1) // slots)
        lines.append(separator.join(remaining[:take]))
        remaining = remaining[take:]
        if not remaining:
            break
    if remaining:
        lines[-1] += separator + separator.join(remaining)
    return lines


def optimized_image_bytes(image_path: Path) -> bytes:
    normalized, _ = normalized_lineart_png_bytes(image_path.read_bytes())
    return normalized


def render_grid_pdf(
    project_dir: Path,
    items: list[dict[str, Any]],
    output_path: Path,
    *,
    mode: str,
    resolve_image: Callable[[dict[str, Any]], Path | None],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    renderer = KmtTemplateRenderer()
    return renderer.render(project_dir, items, output_path, mode=mode, resolve_image=resolve_image, warnings=warnings)
