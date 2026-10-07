"""Render a chosen LayoutPlan to a deterministic vector PDF.

Two reuse decisions keep the adaptive output visually identical to the current
standard mark rather than a look-alike:

* Static artwork is stamped from ``base.pdf`` by clip rectangle, so the logo,
  icon row, table cells and barcode brackets are the same vector objects the
  approved mark uses.
* Variable text and the barcode are drawn by the *existing*
  ``KmtTemplateRenderer`` methods, fed transformed coordinates. Wrapping,
  balancing, shrink-to-fit, item-number centring and the barcode's
  human-readable grouping therefore behave exactly as they do today, with no
  second typography implementation to drift.

Page topology: one page per SKU carrying the front and side faces side by side.
That is what lets the built-in V5 verifier aggregate both faces as a single SKU
cell - its text-anchored segmenter keeps one row as one cell when the same item
number appears twice - so no verifier change is needed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import fitz

from ..barcode_renderer import draw_ean13_bars_horizontal, format_ean13_hri
from ..layout_audit.layers import LayerTagger
from ..ean13 import is_valid_ean13
from ..font_subset import subset_large_embedded_fonts
from ..template_renderer import KmtTemplateRenderer, abbreviate_cn_display_name, issue, wrap_display_text
from .atlas import Atlas, load_atlas
from .model import Box, FacePlan, LayoutPlan, mm, pt


DEFAULT_GUTTER_MM = 4.0
DEFAULT_PAGE_MARGIN_MM = 0.0

#: ``show_pdf_page`` angle for a ``rotate90_uniform`` tile. PyMuPDF turns the
#: shown page counter-clockwise for a positive angle. Every barcode tile in
#: base.pdf is the vertical symbol, physical 270 (digits on the right, reading
#: upward); a rotated tile has to print flat with digits below, physical 0 -
#: owner rule 2026-09-15 section 4.5, the only other permitted direction. That
#: is a quarter turn clockwise. ``rotate=90`` turned it the other way: measured
#: on the 305 x 95 mm wide-band front, the bars and brackets printed at 180 deg,
#: digits above and upside down, while the plan said 0.
ROTATE90_UNIFORM_ANGLE = -90


class AdaptiveRenderer:
    """Draws LayoutPlans. One instance per render call."""

    def __init__(self, atlas: Atlas | None = None, *, audit_layers: bool = False) -> None:
        """``audit_layers`` wraps every element in its own optional content group.

        Off by default and byte-for-byte inert when off: the audit needs to
        isolate each element's ink, and production must not pay for that.
        """
        self.atlas = atlas or load_atlas()
        self.audit_layers = audit_layers
        # KmtTemplateRenderer owns font loading, text fitting and the barcode
        # drawing that the adaptive layer reuses verbatim.
        self.kmt = KmtTemplateRenderer()
        self.base = self.kmt.base
        self.font_warnings = list(self.kmt.font_warnings)
        self._barcode_masters: dict[tuple[str, str], fitz.Document] = {}

    # -- static artwork ----------------------------------------------------- #

    def _stamp(self, page: fitz.Page, origin: fitz.Point, placement) -> None:
        if placement.target.width <= 0 or placement.target.height <= 0:
            return
        rotate = ROTATE90_UNIFORM_ANGLE if placement.fit == "rotate90_uniform" else 0
        # show_pdf_page keeps proportion by default, which letterboxes the tile
        # inside the target and silently shrinks it. That is right for artwork
        # whose shape must not change (logo, icons, brackets) - and those targets
        # are already aspect-matched, so it is a no-op there. It is wrong for the
        # solid black bar and the table cell frames, which are meant to take the
        # width they are given.
        #
        # A slice_x cell is three draws of the same tile: label and unit at the
        # uniform scale, the empty interior stretched between them. Each draw
        # appends its own content stream and the caller holds one
        # tagger.element context around this whole call, so all three land in
        # the cell's single audit layer.
        for clip, box, keep_proportion in placement.slice_segments():
            target = fitz.Rect(origin.x + box.x0, origin.y + box.y0, origin.x + box.x1, origin.y + box.y1)
            if target.width <= 0 or target.height <= 0:
                continue
            page.show_pdf_page(
                target,
                self.base,
                placement.source_page,
                clip=fitz.Rect(*clip),
                rotate=rotate,
                keep_proportion=keep_proportion,
            )

    # -- variable fields ---------------------------------------------------- #

    def _draw_fields(
        self,
        page: fitz.Page,
        origin: fitz.Point,
        plan: FacePlan,
        item: dict[str, Any],
        project_dir: Path,
        mode: str,
        resolve_image: Callable[[dict[str, Any]], Path | None],
        warnings: list[dict[str, Any]],
        tagger: LayerTagger,
    ) -> None:
        fields = plan.fields
        item_no = str(item.get("item_no") or "")
        face = plan.kind

        if "item_no" in fields:
            with tagger.element(page, face, "item_no"):
                self.kmt._draw_item_no(page, origin, fields["item_no"], item_no)

        for block in ("names", "cn_name"):
            if block in fields:
                with tagger.element(page, face, block):
                    self.kmt._draw_names(page, origin, self._names_start_size(fields[block], item), item)

        if "lineart" in fields:
            with tagger.element(page, face, "lineart"):
                self.kmt._draw_lineart(
                    page, origin, fields["lineart"], item, project_dir, mode, resolve_image, warnings
                )

        if "qty" in fields:
            with tagger.element(page, face, "qty"):
                self.kmt._draw_qty(page, origin, fields["qty"], item)

        if "es_name" in fields:
            with tagger.element(page, face, "es_name"):
                self.kmt._draw_es_name(page, origin, fields["es_name"], item)

        barcode_cfg = fields.get("barcode")
        barcode_value = str(item.get("barcode") or "")
        if barcode_cfg and is_valid_ean13(barcode_value):
            with tagger.element(page, face, "barcode"):
                self._draw_barcode_group(page, origin, plan, barcode_value)
        elif barcode_cfg:
            warnings.append(
                issue(
                    "barcode_skipped",
                    "Barcode missing or invalid; barcode area left empty.",
                    item_no=item_no or None,
                )
            )

    def _names_start_size(self, cfg: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        """Lower a names block's start size until the template's own search can succeed.

        Opt-in (``fit == "shrink_fontsize_continuous"``), so every other family
        draws exactly as before. ``fit`` carries the flag because it is what
        the field already uses to name its sizing behaviour, and
        ``_draw_names`` itself never reads it. ``KmtTemplateRenderer._draw_names`` tries the start
        size x 1.00, 0.95 ... 0.55 and, if none fits, drops straight to
        ``min_fontsize``. A band family's start size comes from its region
        height, which on a narrow column can be more than 1/0.55 of what the
        width carries. Measured 2026-09-15 on the column front of a
        780 x 200 mm face: the Latin names printed at 14.9 mm and the Chinese
        name, one step past the search, at 4.3 mm; a long-name SKU printed all
        four blocks at the 4.3 mm floor on 780-1014 mm fronts.

        The fit test below is the template renderer's own, line for line
        (same wrap helper, same max_lines, leading, block gap and height rule).
        It first walks the template's own ladder with the template's own float
        arithmetic: if that finds a size, the block is returned untouched and
        prints exactly as it did. Only past the template's last step does it
        continue, in 5% geometric steps, and the first size that passes becomes
        the new start, so the template's first attempt accepts and draws it;
        typography is still only drawn by ``_draw_names``. When nothing passes
        above the floor the block is returned unchanged and the template's own
        fallback applies.
        """
        if cfg.get("fit") != "shrink_fontsize_continuous":
            return cfg
        entries = []
        for line in cfg["lines"]:
            text = str(item.get(line["field"]) or "").strip()
            if line["field"] == "cn_name":
                text = abbreviate_cn_display_name(text)
            if line["field"] in {"it_name", "es_name", "en_name"}:
                text = text.upper()
            if text:
                entries.append((line, text, self.kmt.fonts[line["font"]]))
        if not entries:
            return cfg
        floor = float(cfg["min_fontsize"])
        width = float(cfg["max_width"])
        slot = float(cfg.get("bottom_y", 0.0)) - float(cfg.get("top_y", 0.0))
        leading = float(cfg.get("leading_ratio", 1.15))
        gap = float(cfg.get("block_gap", 3.0))

        def fits(factor: float) -> bool:
            total = gap * max(0, len(entries) - 1)
            for line, text, font in entries:
                size = max(floor, float(line["fontsize"]) * factor)
                count = len(wrap_display_text(font, text, size, width))
                if count > int(line.get("max_lines", cfg.get("max_lines_per_name", 2))):
                    return False
                total += count * size * leading
            return total <= slot

        factor = 1.0
        while factor >= 0.55:  # KmtTemplateRenderer._draw_names' ladder, verbatim
            if fits(factor):
                return cfg
            factor -= 0.05
        while any(float(line["fontsize"]) * factor > floor for line, _, _ in entries):
            if fits(factor):
                break
            factor *= 0.95
        else:
            return cfg
        resized = dict(cfg)
        resized["lines"] = [
            {**line, "fontsize": float(line["fontsize"]) * factor} for line in cfg["lines"]
        ]
        return resized

    def _draw_barcode_group(
        self, page: fitz.Page, origin: fitz.Point, plan: FacePlan, value: str
    ) -> None:
        """Stamp native bars and HRI with exactly the brackets' transform.

        The two audit layers remain independently measurable, but share one
        source clip, one destination and one rotation. Slot or face dimensions
        cannot resize the digits independently of the bars. The payload is
        drawn afresh for each distinct barcode, never copied from another SKU.
        """
        bracket = next(s for s in plan.statics if s.name == "barcode_brackets")
        key = (plan.kind, value)
        if key not in self._barcode_masters:
            master = fitz.open()
            source_page = self.base[bracket.source_page]
            native = master.new_page(width=source_page.rect.width, height=source_page.rect.height)
            cfg = getattr(self.atlas.frame, plan.kind)["barcode"]
            self.kmt._draw_barcode(native, fitz.Point(0, 0), cfg, value)
            self._barcode_masters[key] = master
        target = fitz.Rect(*bracket.target.as_tuple()) + (origin.x, origin.y, origin.x, origin.y)
        page.show_pdf_page(
            target, self._barcode_masters[key], 0,
            clip=fitz.Rect(*bracket.clip),
            # the brackets' own angle, so bars, digits and corners turn as one group
            rotate=ROTATE90_UNIFORM_ANGLE if bracket.fit == "rotate90_uniform" else 0,
            keep_proportion=True,
        )

    def _draw_barcode_upright(
        self, page: fitz.Page, origin: fitz.Point, cfg: dict[str, Any], value: str
    ) -> None:
        """Upright EAN-13 for the wide-band family, with HRI under the bars."""
        bars = fitz.Rect(*cfg["bars_rect"]) + (origin.x, origin.y, origin.x, origin.y)
        inset = float(cfg.get("regular_bar_inset", 9.8))
        draw_ean13_bars_horizontal(page, bars, value, inset)

        font_key = cfg.get("font", "barcode_hri")
        font = self.kmt.fonts.get(font_key)
        # The digits live in the notch left by the shortened data bars, so the
        # notch depth is the ceiling: sizing them from the module alone made
        # them ride up over the barcode.
        fontsize = min(float(cfg.get("text_fontsize", 10.0)), inset * 0.92)
        args = self.kmt._barcode_font_args(font_key)
        baseline = bars.y1 - inset * 0.06

        # EAN-13 keeps its guard bars full height, so they cut straight through the
        # notch. Drawing the HRI as one centred span therefore puts the centre guard
        # on top of a digit. Split it the way the symbology intends: first digit in
        # the left quiet zone, then 6 digits per half, each centred between guards.
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        module = bars.width / 95.0

        def _measure(s: str, size: float) -> float:
            return font.text_length(s, size) if font else len(s) * size * 0.55

        if len(digits) == 13:
            # module spans: left guard 0-2, left data 3-44, centre guard 45-49,
            # right data 50-91, right guard 92-94
            pieces = [
                (digits[0], bars.x0 - module * 1.5, "right"),
                (digits[1:7], bars.x0 + module * 24.0, "center"),
                (digits[7:], bars.x0 + module * 71.0, "center"),
            ]
            widest = max(_measure(p[0], fontsize) for p in pieces)
            room = module * 40.0
            if widest > room > 0:
                fontsize *= room / widest
            for text, anchor, align in pieces:
                w = _measure(text, fontsize)
                x = anchor - w if align == "right" else anchor - w / 2.0
                page.insert_text(fitz.Point(x, baseline), text, fontsize=fontsize, **args)
            return

        # Not a 13-digit payload: fall back to one centred span.
        text = format_ean13_hri(value, cfg.get("hri_grouping", "golden_1_6_6"))
        width = _measure(text, fontsize)
        available = bars.width - 2.0
        if width > available > 0:
            fontsize *= available / width
            width = _measure(text, fontsize)
        page.insert_text(
            fitz.Point(bars.x0 + (bars.width - width) / 2.0, baseline),
            text, fontsize=fontsize, **args,
        )

    # -- pages -------------------------------------------------------------- #

    def render(
        self,
        project_dir: Path,
        plans: list[LayoutPlan],
        items_by_no: dict[str, dict[str, Any]],
        output_path: Path,
        *,
        mode: str,
        resolve_image: Callable[[dict[str, Any]], Path | None],
        warnings: list[dict[str, Any]],
        draw_cut_marks: bool = True,
    ) -> dict[str, Any]:
        warnings.extend(self.font_warnings)
        doc = fitz.open()
        tagger = LayerTagger(doc, enabled=self.audit_layers)
        for plan in plans:
            item = items_by_no.get(plan.item_no, {"item_no": plan.item_no})
            page = doc.new_page(width=plan.page_size_pt[0], height=plan.page_size_pt[1])
            for face in (plan.front, plan.side):
                origin = fitz.Point(face.panel.x0, face.panel.y0)
                local = FacePlan(
                    kind=face.kind,
                    family=face.family,
                    panel=Box(0, 0, face.panel.width, face.panel.height),
                    content=_shift_box(face.content, -face.panel.x0, -face.panel.y0),
                    scale=face.scale,
                    inset_mm=face.inset_mm,
                    statics=[_shift_static(s, -face.panel.x0, -face.panel.y0) for s in face.statics],
                    fields=_shift_fields(face.fields, -face.panel.x0, -face.panel.y0),
                    barcode=face.barcode,
                    notes=face.notes,
                    safe_margin_mm=face.safe_margin_mm,
                )
                for placement in local.statics:
                    with tagger.element(page, face.kind, placement.name):
                        self._stamp(page, origin, placement)
                self._draw_fields(
                    page, origin, local, item, project_dir, mode, resolve_image, warnings, tagger
                )
                if draw_cut_marks:
                    page.draw_rect(
                        fitz.Rect(
                            face.panel.x0, face.panel.y0, face.panel.x1, face.panel.y1
                        ),
                        color=(0.75, 0.75, 0.75),
                        width=0.3,
                        dashes="[2 2] 0",
                    )
            if mode == "preview":
                page.insert_text(
                    fitz.Point(6, page.rect.height - 6),
                    f"PREVIEW - {self.atlas.template_id} adaptive - NOT FOR PRODUCTION",
                    fontsize=8,
                    fontname="helv",
                    color=(0.6, 0.6, 0.6),
                )
        doc.subset_fonts()
        warnings.extend(subset_large_embedded_fonts(doc))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path, garbage=3, deflate=True)
        page_count = doc.page_count
        doc.close()
        for master in self._barcode_masters.values():
            master.close()
        self._barcode_masters.clear()
        return {"pages": page_count, "layout": "carton_adaptive", "pairs_per_page": 1}


def _shift_box(box: Box, dx: float, dy: float) -> Box:
    return Box(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy)


def _shift_static(placement, dx: float, dy: float):
    # replace() rather than re-listing fields: a translated placement must keep
    # every piece of its geometry, including a slice_x cell's measured interior.
    return replace(placement, target=_shift_box(placement.target, dx, dy))


def _shift_fields(fields: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Translate every coordinate in a field config into panel-local space."""
    out: dict[str, Any] = {}
    for key, cfg in fields.items():
        if not isinstance(cfg, dict):
            out[key] = cfg
            continue
        moved = dict(cfg)
        for rect_key in ("rect", "bar_rect", "bars_rect"):
            if rect_key in moved:
                x0, y0, x1, y1 = (float(v) for v in moved[rect_key])
                moved[rect_key] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
        if "x0" in moved:
            moved["x0"] = float(moved["x0"]) + dx
        for y_key in ("baseline_y", "top_y", "bottom_y"):
            if y_key in moved:
                moved[y_key] = float(moved[y_key]) + dy
        if "readout_origin" in moved:
            ox, oy = moved["readout_origin"]
            moved["readout_origin"] = [float(ox) + dx, float(oy) + dy]
        if "lines" in moved:
            lines = []
            for line in moved["lines"]:
                entry = dict(line)
                bx, by = entry["baseline"]
                entry["baseline"] = [float(bx) + dx, float(by) + dy]
                lines.append(entry)
            moved["lines"] = lines
        out[key] = moved
    return out


def page_geometry(
    front_face_mm: tuple[float, float],
    side_face_mm: tuple[float, float],
    *,
    gutter_mm: float = DEFAULT_GUTTER_MM,
    margin_mm: float = DEFAULT_PAGE_MARGIN_MM,
) -> tuple[tuple[float, float], Box, Box, float]:
    """Front and side laid out side by side on one page, in points."""
    fw, fh = pt(front_face_mm[0]), pt(front_face_mm[1])
    sw, sh = pt(side_face_mm[0]), pt(side_face_mm[1])
    gutter = pt(gutter_mm)
    margin = pt(margin_mm)
    height = max(fh, sh)
    page = (margin * 2 + fw + gutter + sw, margin * 2 + height)
    front = Box(margin, margin, margin + fw, margin + fh)
    side_x0 = margin + fw + gutter
    side = Box(side_x0, margin, side_x0 + sw, margin + sh)
    return page, front, side, gutter


__all__ = ["AdaptiveRenderer", "DEFAULT_GUTTER_MM", "page_geometry"]
