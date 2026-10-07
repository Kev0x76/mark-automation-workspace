"""Box-model layout: fill the face by distributing space, not by scaling a template.

Why this exists
---------------
The first implementation projected template.json's coordinates onto the target
face with one uniform scale. That is an affine transform, not a layout: when a
face is much taller than the template, nothing claims the extra height, so the
mark sits in a band with large blanks above and below. Measured against real
factory panels, our side marks had a 34.5% largest-blank ratio where the factory
worst case is 15.2% - same ink coverage, worse distribution.

The model here treats the mark as nested boxes with a preferred size and a share
of the surplus, the way flexbox does:

    final = basis + surplus * (own_grow / total_grow)

Two element classes behave differently, and conflating them was the old bug:

* **Aspect-locked art** (logo, icon row, line art slot, barcode, table cells).
  Its size follows whichever axis binds; it may never be stretched on one axis.
  A table cell carries its label as baked vector text, so stretching it
  vertically would distort "Q.TY".
* **Elastic content** (the item-number bar, text blocks, gaps). It absorbs the
  surplus.

Filling therefore comes from allocation, not from enlarging artwork past what
its proportions allow.
"""

from __future__ import annotations

from typing import Any

from .atlas import Atlas, ComponentSpec
from .model import BarcodePlacement, Box, FaceKind, FacePlan, StaticPlacement, mm, pt


# Five cells stacked at full width occupy roughly 2x their width in height.
# Below this height-to-width ratio the stack cannot fit without narrowing the
# cells badly, so the two-column arrangement is used instead.
SIDE_STACK_MIN_HEIGHT_RATIO = 1.25

# How far the stack may narrow before it stops being worth it.
SIDE_STACK_MIN_SCALE = 0.55

SIDE_CELLS_STACKED = ("cell_qty", "cell_name", "cell_meas", "cell_nw", "cell_gw")


def _inset_pt(atlas: Atlas, panel: Box) -> float:
    rule = atlas.policy["inset"]
    raw = mm(panel.height) * float(rule["ratio"])
    clamped = max(float(rule["min_mm"]), min(float(rule["max_mm"]), raw))
    return min(pt(clamped), min(panel.width, panel.height) * 0.15)


def _gap_pt(content: Box) -> float:
    """Breathing room between blocks: proportional, with a physical floor."""
    return max(pt(2.0), content.height * 0.02)


def _static(spec: ComponentSpec, target: Box, fit: str | None = None) -> StaticPlacement:
    return StaticPlacement(
        name=spec.name, clip=spec.clip, target=target,
        fit=fit or spec.fit, source_page=spec.source_page,
    )


def _fit_aspect(region: Box, aspect: float, *, ax: str = "center", ay: str = "middle") -> Box:
    return region.fit_uniform(aspect, align_x=ax, align_y=ay)


def _item_no_bar(atlas: Atlas, width: float, item_no: str, max_height: float) -> tuple[float, float]:
    """Bar height and font size for a bar of the given width.

    The template's own horizontal condensation for this digit count is the floor,
    so digits are never squeezed harder than today's approved mark squeezes them.
    """
    frame = atlas.frame
    digits = max(1, len(item_no))
    floor = frame.item_no_hscale_by_digits.get(
        digits, min(frame.item_no_hscale_by_digits.values(), default=1.0)
    )
    pad = max(2.0, min(width * 0.02, 8.0))
    fontsize = max(1.0, width - 2 * pad) / max(1e-6, digits * frame.digit_advance_em * floor)
    height = fontsize / max(1e-6, frame.item_no_fontsize_ratio)
    if height > max_height:
        height = max_height
        fontsize = height * frame.item_no_fontsize_ratio
    return height, fontsize


def _barcode(bars: Box, rotation: int, atlas: Atlas) -> BarcodePlacement:
    limits = atlas.constraints["barcode"]
    nominal = float(limits["nominal_x_dimension_mm"])
    # max_x_mm may be null (uncapped) - see families._barcode_placement
    _raw_max_x = limits.get("max_x_mm")
    max_x = float(_raw_max_x) if _raw_max_x else 0.0

    def metrics(box: Box) -> tuple[float, float]:
        if rotation == 90:
            return mm(box.height) / 95.0, mm(box.width)
        return mm(box.width) / 95.0, mm(box.height)

    x_dim, bar_len = metrics(bars)
    if max_x and x_dim > max_x:
        k = max_x / x_dim
        w, h = bars.width * k, bars.height * k
        cx = (bars.x0 + bars.x1) / 2.0
        cy = (bars.y0 + bars.y1) / 2.0
        bars = Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        x_dim, bar_len = metrics(bars)
    return BarcodePlacement(
        bars=bars, rotation_deg=rotation, x_dimension_mm=x_dim,
        bar_length_mm=bar_len, magnification=x_dim / nominal if nominal else 0.0,
    )


def _bars_inside_brackets(frame, brackets: ComponentSpec, placed: Box, *, side: bool,
                          rotated: bool = False) -> Box:
    bars = (frame.side if side else frame.front)["barcode"]["bars_rect"]
    bx0, by0, bx1, by1 = (float(v) for v in bars)
    cx0, cy0, cx1, cy1 = brackets.clip
    fw, fh = max(1e-6, cx1 - cx0), max(1e-6, cy1 - cy0)
    f = ((bx0 - cx0) / fw, (by0 - cy0) / fh, (bx1 - cx0) / fw, (by1 - cy0) / fh)
    if rotated:
        f = (f[1], f[0], f[3], f[2])
    return placed.fraction(*f)


def _names_cfg(atlas: Atlas, template_names: dict[str, Any], region: Box,
               fields: list[str], *, widen_max: float = 1.25) -> dict[str, Any]:
    """Size the name block to fill its region instead of inheriting a scale.

    Line height comes from the region, not from a global factor, which is what
    stops a tall face from leaving the text stranded at the top.
    """
    lines = [dict(l) for l in template_names["lines"] if l["field"] in fields]
    if not lines:
        return {}
    leading = float(template_names.get("leading_ratio", 1.15))
    gap = region.height * 0.04
    per = max(1.0, (region.height - gap * (len(lines) - 1)) / len(lines))
    fontsize = per / leading

    floors = atlas.constraints["min_font_mm"]
    floor_pt = pt(max(float(floors["latin_name"]), float(floors["cn_name"])))

    y = region.y0
    for line in lines:
        line["fontsize"] = fontsize
        line["baseline"] = [region.x0, y + per * 0.82]
        line["max_lines"] = 1
        y += per + gap
    return {
        "top_y": region.y0, "bottom_y": region.y1,
        "leading_ratio": leading, "block_gap": gap,
        "max_lines_per_name": 1, "lines": lines,
        "max_width": region.width, "fit": "shrink_fontsize",
        "min_fontsize": max(floor_pt, fontsize * 0.55),
        "widen_max": widen_max,
        "rect": list(region.as_tuple()),
    }


def _barcode_cfg(template_cfg: dict[str, Any], bars: Box, *, upright: bool) -> dict[str, Any]:
    """Human-readable digits are sized from the symbol, never from the face.

    Binding the HRI to the panel scale is what made the digits look unbalanced
    against their barcode; tying them to the module width keeps the pair in
    proportion at every size.
    """
    out = dict(template_cfg)
    out["bars_rect"] = list(bars.as_tuple())
    module = (bars.height if not upright else bars.width) / 95.0
    out["text_fontsize"] = module * 9.0
    out["regular_bar_inset"] = (bars.width if not upright else bars.height) * 0.16
    out["character_spacing"] = module * 0.35
    out["readout_gap"] = module * 0.6
    out["first_digit_height"] = module * 6.0
    out["first_digit_offset"] = module * 9.0
    out["main_group_gap"] = module * 4.5
    out["main_bottom_inset"] = module * 3.0
    out["main_top_inset"] = module * 3.0
    out["readout_xscale"] = 1.0
    if upright:
        out["orientation"] = "horizontal"
        out["hri_layout"] = "upright_single_span"
    return out


# --------------------------------------------------------------------------- #
# Front
# --------------------------------------------------------------------------- #


def build_front(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    gap = _gap_pt(content)

    logo = atlas.component("front", "kmt_logo")
    icons = atlas.component("front", "icon_row")
    origin = atlas.component("front", "origin_text")
    brackets = atlas.component("front", "barcode_brackets")

    # --- row 1: item-number bar. Elastic, but never taller than its share. ---
    bar_h, fontsize = _item_no_bar(atlas, content.width, str(item.get("item_no") or ""),
                                   content.height * 0.42)
    bar = Box(content.x0, content.y0, content.x1, content.y0 + bar_h)

    # --- row 3: compliance footer, sized by the icon row's locked aspect ---
    footer_h = min(content.height * 0.22, content.width / icons.aspect + origin.size[1] * 1.6)
    footer = Box(content.x0, content.y1 - footer_h, content.x1, content.y1)
    icon_box = _fit_aspect(Box(footer.x0, footer.y0 + footer.height * 0.34, footer.x1, footer.y1),
                           icons.aspect, ax="left", ay="bottom")
    origin_box = _fit_aspect(Box(footer.x0, footer.y0, footer.x1, footer.y0 + footer.height * 0.30),
                             origin.aspect, ax="left", ay="middle")

    # --- row 2: everything between them gets the remaining height ---
    body = Box(content.x0, bar.y1 + gap, content.x1, footer.y0 - gap)
    body_h = max(1.0, body.height)

    # Barcode column. Vertical is the Kmt default, but on a short face the band
    # height cannot carry 95 modules, so the symbol turns upright - which is
    # exactly what the factory does on wide, short panels.
    vertical_box = _fit_aspect(Box(body.x1 - body_h * brackets.aspect, body.y0, body.x1, body.y1),
                               brackets.aspect, ax="right", ay="middle")
    vertical = _barcode(_bars_inside_brackets(frame, brackets, vertical_box, side=False), 90, atlas)

    warn = float(atlas.constraints["barcode"]["warn_below_x_mm"])
    if vertical.x_dimension_mm >= warn:
        barcode, rotated = vertical, False
        bracket_box = _rebuild_brackets(frame, brackets, barcode.bars, side=False)
    else:
        slot_w = min(body.width * 0.42, body_h * 2.4)
        upright_box = _fit_aspect(Box(body.x1 - slot_w, body.y0, body.x1, body.y1),
                                  1.0 / brackets.aspect, ax="right", ay="middle")
        upright = _barcode(
            _bars_inside_brackets(frame, brackets, upright_box, side=False, rotated=True), 0, atlas
        )
        if upright.x_dimension_mm > vertical.x_dimension_mm:
            barcode, rotated = upright, True
            bracket_box = _rebuild_brackets(frame, brackets, barcode.bars, side=False, rotated=True)
        else:
            barcode, rotated = vertical, False
            bracket_box = _rebuild_brackets(frame, brackets, barcode.bars, side=False)

    # left column: logo over the four names
    left_w = max(content.width * 0.28, min(content.width * 0.40, body_h * 1.15))
    left = Box(body.x0, body.y0, body.x0 + left_w, body.y1)
    logo_box = _fit_aspect(Box(left.x0, left.y0, left.x1, left.y0 + left.height * 0.22),
                           logo.aspect, ax="left", ay="top")
    names_region = Box(left.x0, logo_box.y1 + gap * 0.6, left.x1, left.y1)

    # line art takes the whole middle, so the band has no structural hole
    art = Box(left.x1 + gap, body.y0, bracket_box.x0 - gap, body.y1)

    statics = [
        _static(atlas.component("front", "item_no_bar"), bar),
        _static(logo, logo_box),
        _static(origin, origin_box),
        _static(icons, icon_box),
        StaticPlacement(name="barcode_brackets", clip=brackets.clip, target=bracket_box,
                        fit="rotate90_uniform" if rotated else brackets.fit,
                        source_page=brackets.source_page),
    ]
    fields = {
        "item_no": _item_no_field(frame, bar, fontsize, front=True),
        "names": _names_cfg(atlas, frame.front["names"], names_region,
                            ["cn_name", "it_name", "es_name", "en_name"]),
        "lineart": {"rect": list(art.as_tuple()), "fit": "stretch"},
        "barcode": _barcode_cfg(frame.front["barcode"], barcode.bars, upright=rotated),
    }
    return FacePlan(
        kind="front", family="front_boxfill", panel=panel, content=content,
        scale=body_h / frame.content_height, inset_mm=mm(inset),
        statics=statics, fields=fields, barcode=barcode,
        safe_margin_mm=float(atlas.policy.get("safe_area", {}).get("margin_mm", 2.0)),
    )


def _rebuild_brackets(frame, brackets: ComponentSpec, bars: Box, *, side: bool,
                      rotated: bool = False) -> Box:
    """Corner brackets follow the (possibly GS1-capped) symbol."""
    ref = (frame.side if side else frame.front)["barcode"]["bars_rect"]
    bx0, by0, bx1, by1 = (float(v) for v in ref)
    cx0, cy0, cx1, cy1 = brackets.clip
    fw, fh = max(1e-6, cx1 - cx0), max(1e-6, cy1 - cy0)
    f = ((bx0 - cx0) / fw, (by0 - cy0) / fh, (bx1 - cx0) / fw, (by1 - cy0) / fh)
    if rotated:
        f = (f[1], f[0], f[3], f[2])
    w = bars.width / max(1e-6, f[2] - f[0])
    h = bars.height / max(1e-6, f[3] - f[1])
    x0 = bars.x0 - w * f[0]
    y0 = bars.y0 - h * f[1]
    return Box(x0, y0, x0 + w, y0 + h)


def _item_no_field(frame, bar: Box, fontsize: float, *, front: bool) -> dict[str, Any]:
    cfg = frame.front["item_no"] if front else frame.side["item_no"]
    return {
        "bar_rect": list(bar.as_tuple()),
        "x0": bar.x0,
        "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
        "max_width": bar.width,
        "fontsize": fontsize,
        "font": cfg["font"],
        "color": list(cfg["color"]),
        "fit": "center_in_bar",
    }


# --------------------------------------------------------------------------- #
# Side
# --------------------------------------------------------------------------- #


def build_side(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    gap = _gap_pt(content)
    brackets = atlas.component("side", "barcode_brackets")

    bar_h, fontsize = _item_no_bar(atlas, content.width, str(item.get("item_no") or ""),
                                   content.height * 0.42)
    bar = Box(content.x0, content.y0, content.x1, content.y0 + bar_h)
    below = Box(content.x0, bar.y1 + gap, content.x1, content.y1)

    # Choose by whether the arrangement actually fits, not by a width threshold.
    # Five cells stacked at full width need about 2x their width in height; on a
    # short face that forced the cells down to a fifth of the width, which left
    # worse blanks than the layout it replaced.
    stacked = below.height >= below.width * SIDE_STACK_MIN_HEIGHT_RATIO
    plan = _side_stacked(atlas, frame, brackets, below, gap) if stacked else None
    if plan is None:
        stacked = False
        plan = _side_two_column(atlas, frame, brackets, below, gap)
    statics, fields, barcode = plan

    statics.insert(0, _static(atlas.component("side", "item_no_bar"), bar))
    fields["item_no"] = _item_no_field(frame, bar, fontsize, front=False)
    return FacePlan(
        kind="side", family="side_boxfill_stacked" if stacked else "side_boxfill",
        panel=panel, content=content, scale=below.height / frame.content_height,
        inset_mm=mm(inset), statics=statics, fields=fields, barcode=barcode,
        safe_margin_mm=float(atlas.policy.get("safe_area", {}).get("margin_mm", 2.0)),
        notes=["side table stacked into one column: face too narrow for two"] if stacked else [],
    )


def _cell_value_box(frame, atlas: Atlas, cell_name: str, field_name: str, placed: Box) -> Box:
    spec = atlas.component("side", cell_name)
    cx0, cy0, cx1, cy1 = spec.clip
    fw, fh = max(1e-6, cx1 - cx0), max(1e-6, cy1 - cy0)
    vx0, vy0, vx1, vy1 = (float(v) for v in frame.side[field_name]["rect"])
    return placed.fraction(
        max(0.0, (vx0 - cx0) / fw), max(0.0, (vy0 - cy0) / fh),
        min(1.0, (vx1 - cx0) / fw), min(1.0, (vy1 - cy0) / fh),
    )


def _side_stacked(atlas: Atlas, frame, brackets: ComponentSpec, region: Box, gap: float):
    """Narrow face: five cells in one column plus an upright barcode below.

    Cell heights follow their locked aspect at full width, so the column's
    natural height is what fills the face - which is exactly how the factory
    fills a tall side panel.
    """
    specs = [atlas.component("side", n) for n in SIDE_CELLS_STACKED]
    # Reserve exactly what an upright barcode needs at the GS1 ceiling, not a
    # fixed share of the height: a fixed share starved the table on tall faces
    # and forced a fallback that scattered the cells.
    max_x = float(atlas.constraints["barcode"]["max_x_mm"])
    bar_w = min(region.width, pt(95.0 * max_x) / 0.845)
    bar_slot = min(region.height * 0.30, bar_w * brackets.aspect * 1.15)
    table_h_avail = region.height - bar_slot - gap * len(specs)

    natural = sum(region.width / s.aspect for s in specs)
    k = min(1.0, table_h_avail / max(1e-6, natural))
    if k < SIDE_STACK_MIN_SCALE:
        return None  # caller falls back to the two-column arrangement
    cell_w = region.width * k
    x0 = region.x0 + (region.width - cell_w) / 2.0

    # Surplus height goes into the gaps between cells, never into stretching a
    # cell: the label text inside is baked vector art and would distort.
    spare = max(0.0, table_h_avail - natural * k)
    row_gap = gap + min(spare / max(1, len(specs)), gap * 1.5)

    statics, boxes = [], {}
    y = region.y0
    for spec in specs:
        h = cell_w / spec.aspect
        box = Box(x0, y, x0 + cell_w, y + h)
        statics.append(_static(spec, box))
        boxes[spec.name] = box
        y += h + row_gap

    # upright barcode fills the remaining strip
    strip = Box(region.x0, y, region.x1, region.y1)
    if strip.height > gap:
        bracket_box = _fit_aspect(strip, 1.0 / brackets.aspect, ax="center", ay="middle")
        bars = _bars_inside_brackets(frame, brackets, bracket_box, side=True, rotated=True)
        barcode = _barcode(bars, 0, atlas)
        bracket_box = _rebuild_brackets(frame, brackets, barcode.bars, side=True, rotated=True)
        statics.append(StaticPlacement(name="barcode_brackets", clip=brackets.clip,
                                       target=bracket_box, fit="rotate90_uniform",
                                       source_page=brackets.source_page))
        cfg = _barcode_cfg(frame.side["barcode"], barcode.bars, upright=True)
    else:
        barcode, cfg = None, {}

    fields: dict[str, Any] = {}
    if cfg:
        fields["barcode"] = cfg
    fields.update(_table_fields(atlas, frame, boxes))
    return statics, fields, barcode


def _side_two_column(atlas: Atlas, frame, brackets: ComponentSpec, region: Box, gap: float):
    """Wider face: Kmt's own two-column table beside a vertical barcode."""
    bar_w = min(region.width * 0.22, region.height * brackets.aspect)
    bracket_box = _fit_aspect(Box(region.x1 - bar_w, region.y0, region.x1, region.y1),
                              brackets.aspect, ax="right", ay="middle")
    bars = _bars_inside_brackets(frame, brackets, bracket_box, side=True)
    barcode = _barcode(bars, 90, atlas)
    bracket_box = _rebuild_brackets(frame, brackets, barcode.bars, side=True)

    table = Box(region.x0, region.y0, bracket_box.x0 - gap, region.y1)
    left_w = table.width * 0.46
    left = Box(table.x0, table.y0, table.x0 + left_w, table.y1)
    right = Box(left.x1 + gap, table.y0, table.x1, table.y1)

    statics = [_static(brackets, bracket_box)]
    boxes = {}
    for column, names in ((left, ("cell_qty", "cell_meas")), (right, ("cell_name", "cell_nw", "cell_gw"))):
        specs = [atlas.component("side", n) for n in names]
        natural = sum(column.width / s.aspect for s in specs)
        k = min(1.0, (column.height - gap * (len(specs) - 1)) / max(1e-6, natural))
        cell_w = column.width * k
        # Surplus is capped so the cells stay a group. Spreading a short table
        # over a tall column makes the elements read as scattered, which is
        # worse than a little blank at top and bottom.
        spare = max(0.0, column.height - natural * k)
        row_gap = min(gap + spare / max(1, len(specs) + 1), gap * 2.5)
        used = natural * k + row_gap * (len(specs) - 1)
        y = column.y0 + max(0.0, (column.height - used) / 2.0)
        for spec in specs:
            h = cell_w / spec.aspect
            box = Box(column.x0, y, column.x0 + cell_w, y + h)
            statics.append(_static(spec, box))
            boxes[spec.name] = box
            y += h + row_gap

    fields: dict[str, Any] = {"barcode": _barcode_cfg(frame.side["barcode"], barcode.bars, upright=False)}
    fields.update(_table_fields(atlas, frame, boxes))
    return statics, fields, barcode


def _table_fields(atlas: Atlas, frame, boxes: dict[str, Box]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "cell_qty" in boxes:
        v = _cell_value_box(frame, atlas, "cell_qty", "qty", boxes["cell_qty"])
        size = v.height * 0.62
        out["qty"] = {
            "rect": list(v.as_tuple()),
            "baseline_y": v.y0 + v.height * 0.5 + 0.36 * size,
            "fontsize": size, "font": frame.side["qty"]["font"], "fit": "shrink_center",
        }
    if "cell_name" in boxes:
        v = _cell_value_box(frame, atlas, "cell_name", "es_name", boxes["cell_name"])
        floor = pt(float(atlas.constraints["min_font_mm"]["table_value"]))
        # Long Spanish names ("ES.SET 4 BOTES DE VIAJE 50ML") need room to wrap.
        # The fitter refuses to clip or horizontally distort, so it must be given
        # both enough lines and a low enough floor, or it fails the whole render.
        out["es_name"] = {
            **frame.side["es_name"], "rect": list(v.as_tuple()),
            "fontsize": v.height * 0.5,
            "min_fontsize": floor,
            "preferred_max_lines": 2, "max_lines": 3,
        }
    return out


BUILDERS = {"front": build_front, "side": build_side}


__all__ = ["BUILDERS", "build_front", "build_side"]
