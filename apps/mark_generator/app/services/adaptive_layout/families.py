"""Layout families: how the mark rearranges itself on a given printed face.

The families are not invented. They were measured from real factory mark files
(20 SKUs, each page stating its carton dimensions) plus dieline references
supplied by the project owner:

  face aspect 0.93 - 2.36  the standard arrangement, only re-proportioned
  face aspect 2.62 - 3.68  a genuinely different arrangement (a wide band)

The standard family is *derived from template.json*, so at the template's own
face proportions the adaptive engine reproduces today's mark exactly. That is
the regression guarantee: adaptivity is a superset of current behaviour, not a
replacement for it.

Two sizing laws come out of the measurements:

  * Everything scales with face **height**, not width or area. Factory product
    names sit at a near-constant 4.9-5.6% of panel height across panel heights
    from 95 mm to 305 mm.
  * The item-number bar's height is capped by what its width can carry, so a
    tall narrow face gets a short bar instead of grotesquely condensed digits.
"""

from __future__ import annotations

from typing import Any

from .atlas import AdaptiveAssetError, Atlas, ComponentSpec
from .owner_rules import FRONT_COLUMN_REGIONS, OWNER_RULE_REVISION, OWNER_SIDE_COLUMNS, wants_owner_column
from .model import (
    BarcodePlacement,
    Box,
    FaceKind,
    FacePlan,
    StaticPlacement,
    mm,
    pt,
)


# Minimum visual breathing room between neighbouring blocks in a reflow band.
_MIN_GAP_MM = 2.0

# Widest slot the line art may claim, as a multiple of its band height. Keeps a
# very wide face from centring the artwork in a huge empty box.
_LINEART_MAX_ASPECT = 1.8


def _inset_pt(atlas: Atlas, panel: Box) -> float:
    rule = atlas.policy["inset"]
    raw_mm = mm(panel.height) * float(rule["ratio"])
    clamped = max(float(rule["min_mm"]), min(float(rule["max_mm"]), raw_mm))
    # Never let the inset eat a face so small that nothing fits.
    return min(pt(clamped), min(panel.width, panel.height) * 0.15)


def _safe_margin_mm(atlas: Atlas) -> float:
    return float(atlas.policy.get("safe_area", {}).get("margin_mm", 2.0))


def _project_point(
    frame_content: Box, target_content: Box, x: float, y: float,
    scale: float, anchor_x: str, scale_x: float | None = None,
) -> tuple[float, float]:
    """Map a template panel-local point onto the target content box.

    Vertical offsets always scale from the top. Horizontal offsets scale from
    whichever edge the component is anchored to, which is what makes a wider
    face spread out rather than stretch: the factory keeps names hard left,
    line art centred and the barcode hard right.

    ``scale_x`` defaults to ``scale``, i.e. uniform. On a face much wider than
    the template the uniform scale is governed by height alone (2.08: the face
    is 2.05x the template wide but only 1.217x tall, so 40% of the width is
    left unused), and the factory does not leave that space empty - it enlarges
    the marks. Callers pass an explicitly derived ``scale_x`` when the family
    is allowed to spread horizontally; it is never guessed here.
    """
    sx = scale if scale_x is None else scale_x
    dy = (y - frame_content.y0) * scale
    ty = target_content.y0 + dy
    if anchor_x == "right":
        tx = target_content.x1 - (frame_content.x1 - x) * sx
    elif anchor_x == "center":
        frame_center = (frame_content.x0 + frame_content.x1) / 2.0
        target_center = (target_content.x0 + target_content.x1) / 2.0
        tx = target_center + (x - frame_center) * sx
    else:
        tx = target_content.x0 + (x - frame_content.x0) * sx
    return tx, ty


def _project_box(
    frame_content: Box, target_content: Box, rect: Any, scale: float, anchor_x: str,
    scale_x: float | None = None,
) -> Box:
    sx = scale if scale_x is None else scale_x
    x0, y0, x1, y1 = (float(v) for v in rect)
    tx0, ty0 = _project_point(frame_content, target_content, x0, y0, scale, anchor_x, sx)
    width = (x1 - x0) * sx
    height = (y1 - y0) * scale
    return Box(tx0, ty0, tx0 + width, ty0 + height)


# A face wider than this multiple of the template cannot be served by the
# uniform scale alone: the scale is set by height, so the extra width is spent
# on nothing. Measured on 2.08 (face 2.05x the template wide, 1.217x tall) the
# engine's own style check reports largest_empty_rect 0.132 against the
# 0.040-0.095 band measured on 19 factory panels, and balance 0.301 against
# 0.102-0.196 - i.e. the layout is visibly emptier and more lopsided than any
# real carton. 1.45 is where the spare width starts to matter: at 1.45 the
# uniform design already fits width-wise and the slack is under 10%.
_WIDE_FACE_SPREAD_AT = 1.45
# Ceiling on the horizontal stretch. Names and titles are allowed to grow and
# spread, but a logo stretched 2x would no longer read as the logo, so the
# horizontal scale is capped at twice the uniform one.
_MAX_HORIZONTAL_STRETCH = 2.0


def _spread_scale_x(frame_content: Box, target_content: Box, scale: float) -> float:
    """Horizontal scale for a wide face: how much room there genuinely is.

    Only the *width* budget is spent here. The vertical rhythm is untouched, so
    this cannot make a face crowded - it fills width the uniform scale left
    idle. Bounded by ``_MAX_HORIZONTAL_STRETCH`` so no single element is
    stretched into a different aspect than the artwork it came from.
    """
    if target_content.aspect <= frame_content.aspect * _WIDE_FACE_SPREAD_AT:
        return scale
    available = target_content.width / frame_content.width
    return min(available, scale * _MAX_HORIZONTAL_STRETCH)


def _item_no_geometry(
    atlas: Atlas, content: Box, item_no: str, scale: float, bar_width: float
) -> tuple[float, float]:
    """Return (bar_height, fontsize) for an item-number bar of ``bar_width``.

    Height-driven size keeps the template's proportion on a normal face.
    Width-driven size caps it so digits are never condensed harder than the
    template already condenses them. The smaller of the two wins.
    """
    frame = atlas.frame
    digits = max(1, len(item_no))
    height_driven = frame.item_no_bar_height_ratio * content.height

    # The template's own condensation for this digit count is the floor: the
    # adaptive bar may never squeeze digits harder than today's mark does.
    hscale_floor = frame.item_no_hscale_by_digits.get(
        digits, min(frame.item_no_hscale_by_digits.values(), default=1.0)
    )

    # Padding scales with the bar, so it has to be solved for rather than
    # subtracted first. Writing the width as
    #
    #     bar_width = 2*padding + text_width
    #               = bar_height * (2*padding_ratio
    #                               + digits*advance*fontsize_ratio*hscale)
    #
    # inverts to the line below. The previous version subtracted the template
    # renderer's absolute max(2, min(w*0.02, 8)) instead, which is right on the
    # template's own panel - where the 8 pt ceiling binds - and wrong everywhere
    # else: a face twice the size kept the same 8 pt margin, so the bar and
    # everything positioned against it stopped scaling with the mark. At the
    # template's own proportions this expression returns its bar height to
    # within 3e-14 pt, so today's mark is reproduced exactly.
    per_height = (
        2.0 * frame.item_no_bar_padding_ratio
        + digits * frame.digit_advance_em * frame.item_no_fontsize_ratio * hscale_floor
    )
    width_driven = bar_width / max(1e-6, per_height)

    bar_height = min(height_driven, width_driven)
    fontsize = bar_height * frame.item_no_fontsize_ratio
    return bar_height, fontsize


def _band_bar(
    atlas: Atlas, content: Box, region: Box, item_no: str, *, clear_right: bool = True
) -> tuple[Box, float]:
    """An item-number bar filling ``region``, with the digits centred inside it.

    This used to hug the digits: the bar ended where the number ended, on the
    reasoning that "a bar pinned to a fixed fraction leaves the number floating
    in black space". Measurement says the factory's bar does exactly that, and
    that hugging contradicts the source this very region came from.

    ``front_wide_band``'s item_no_bar region, x 0.0-0.74, was measured off
    factory pages at aspect 2.62 / 3.21 / 3.68, and ``side_compressed``'s
    0.0-0.88 off the 305x95 mm side panel. Filling those regions reproduces the
    factory's own bar-to-face-width curve to within 9 points and straddles it
    (-6.3, +4.1, +9.0 at aspect 2.60 / 3.70 / 4.50 against a measured
    78.2 / 67.9 / 63.8%). Hugging came out at 25-26% on the same faces: 38 to 52
    points short, and short on every single one. An engine that shrinks the bar
    inside a box it measured off the factory is disagreeing with its own
    evidence, which is a self-consistency defect and not a question of taste.

    The digits keep their template condensation and are centred; they are only
    shrunk when the region genuinely cannot hold them.
    """
    frame = atlas.frame
    digits = max(1, len(item_no))
    hscale_floor = frame.item_no_hscale_by_digits.get(
        digits, min(frame.item_no_hscale_by_digits.values(), default=1.0)
    )

    # The band the family config allocated *is* the intended bar height here.
    # Real factory marks give the item number a larger share of a short face,
    # and the reflow band already encodes that; the template's own
    # bar-to-content ratio would under-size it.
    bar_height = region.height
    fontsize = bar_height * frame.item_no_fontsize_ratio
    text_width = digits * frame.digit_advance_em * fontsize * hscale_floor

    # Scaled from the bar height, not the absolute 2 pt / 8 pt clamp the
    # template renderer uses on its own fixed panel. Copying that clamp here
    # meant a face twice as large kept the same 8 pt margin, so the mark stopped
    # being a scaled copy of itself - the frozen-constant defect the invariance
    # check exists to catch.
    padding = bar_height * frame.item_no_bar_padding_ratio

    # The bar stops one reflow gap short of its region's right edge, because
    # what sits immediately beyond that edge differs by family and one of them
    # leaves no room. front_wide_band declares the bar at 0.0-0.74 and the line
    # art at 0.77, so there is a gap in the policy; side_compressed declares the
    # bar at 0.0-0.88 and the barcode column at 0.88-1.0, so there is none. A
    # bar that filled its region exactly therefore ended on the same coordinate
    # as the barcode brackets and 51 side faces between aspect 2.88 and 3.66
    # were rejected for components_overlap - correctly, since a black field
    # butted against a symbol's quiet zone is not something to print.
    # A caller whose bar has nothing beyond its right edge passes
    # clear_right=False and keeps the full region.
    gap = pt(_MIN_GAP_MM) if clear_right else 0.0
    right = max(region.x0 + region.width * 0.35, region.x1 - gap)

    needed = text_width + 2 * padding
    available = right - region.x0
    if needed > available:
        # Width is the binding constraint: shrink the number so the template's
        # condensation is still not exceeded. The bar keeps its region - it is
        # the number that has to give, not the field it sits in.
        fontsize *= available / needed

    return Box(region.x0, region.y0, right, region.y0 + bar_height), fontsize


def _static(spec: ComponentSpec, target: Box) -> StaticPlacement:
    return StaticPlacement(
        name=spec.name, clip=spec.clip, target=target, fit=spec.fit, source_page=spec.source_page
    )


def _place_uniform(
    spec: ComponentSpec, region: Box, *, align_x: str = "left", align_y: str = "top"
) -> StaticPlacement:
    return _static(spec, region.fit_uniform(spec.aspect, align_x=align_x, align_y=align_y))


def _barcode_placement(bars: Box, rotation_deg: int, atlas: Atlas) -> BarcodePlacement:
    """Derive the GS1 metrics a placed barcode actually achieves.

    The 95 modules run along the box's height when rotated 90 degrees and along
    its width when upright; bar length is the other axis.

    The symbol stops growing at the GS1 200% ceiling. Scaling it with the panel
    would take a 1300 mm face to 1187% magnification, which is both wasteful and
    outside spec - and real factory artwork for a 900 mm carton uses a small
    barcode, not a panel-sized one. The capped symbol keeps its anchor edge so it
    stays where the layout put it.
    """
    limits = atlas.constraints["barcode"]
    nominal = float(limits["nominal_x_dimension_mm"])
    # max_x_mm may be null: the owner uncapped magnification on 2026-08-26, so the
    # symbol scales with its slot and the ceiling now lives in the layout, not here.
    _raw_max_x = limits.get("max_x_mm")
    max_x = float(_raw_max_x) if _raw_max_x else 0.0

    def metrics(box: Box) -> tuple[float, float]:
        if rotation_deg == 90:
            return mm(box.height) / 95.0, mm(box.width)
        return mm(box.width) / 95.0, mm(box.height)

    x_dimension, bar_length = metrics(bars)
    if max_x and x_dimension > max_x:
        shrink = max_x / x_dimension
        width, height = bars.width * shrink, bars.height * shrink
        if rotation_deg == 90:
            # vertical symbol sits at the right edge, vertically centred
            y0 = bars.y0 + (bars.height - height) / 2.0
            bars = Box(bars.x1 - width, y0, bars.x1, y0 + height)
        else:
            # upright symbol sits centred in its slot
            x0 = bars.x0 + (bars.width - width) / 2.0
            y0 = bars.y0 + (bars.height - height) / 2.0
            bars = Box(x0, y0, x0 + width, y0 + height)
        x_dimension, bar_length = metrics(bars)

    return BarcodePlacement(
        bars=bars,
        rotation_deg=rotation_deg,
        x_dimension_mm=x_dimension,
        bar_length_mm=bar_length,
        magnification=x_dimension / nominal if nominal else 0.0,
    )


# --------------------------------------------------------------------------- #
# Front face
# --------------------------------------------------------------------------- #


def build_front_standard(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """The current mark, re-proportioned onto a real face.

    At the template's own face size and proportions this reproduces
    template.json coordinates exactly.
    """
    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)
    fc = frame.content
    # On a face much wider than the template the uniform scale is set by height
    # and leaves width unused; spread the marks horizontally to use it. 1.0 at
    # the template's own proportions, so normal faces are bit-identical.
    scale_x = _spread_scale_x(fc, content, scale)

    # When width governs the scale there is spare height. The owner's 1.00 anchor
    # (2026-09-15, "正唛上下调整了一下") puts the bar at the top edge and the origin
    # and icon rows at the bottom, with the middle group halfway between, so
    # the spare height is split into two equal gaps instead of centring the
    # whole block. With no slack (the template's own proportions) nothing moves.
    slack = max(0.0, content.height - frame.content_height * scale)
    design_top = content.y0
    middle_drop, bottom_drop = slack / 2.0, slack

    item_no = str(item.get("item_no") or "")
    bar_width = content.width
    bar_height, fontsize = _item_no_geometry(atlas, content, item_no, scale, bar_width)
    bar_box = Box(content.x0, design_top, content.x0 + bar_width, design_top + bar_height)

    statics = [_static(atlas.component("front", "item_no_bar"), bar_box)]

    # Everything below the bar keeps the template's vertical rhythm, measured
    # from the bar's real bottom edge so a shortened bar does not leave a hole.
    tpl_bar_bottom = float(frame.front["item_no"]["bar_rect"][3])
    y_shift = bar_box.y1 - (content.y0 + (tpl_bar_bottom - fc.y0) * scale)

    def project(rect: Any, anchor_x: str, *, spread: bool = True, drop: float = middle_drop) -> Box:
        box = _project_box(fc, content, rect, scale, anchor_x,
                           scale_x if spread else None)
        return Box(box.x0, box.y0 + y_shift + drop, box.x1, box.y1 + y_shift + drop)

    # project() already applies the uniform scale, and each clip has exactly the
    # size of its template artwork, so the projected box is the placement.
    for name in ("kmt_logo", "origin_text", "icon_row"):
        spec = atlas.component("front", name)
        drop = middle_drop if name == "kmt_logo" else bottom_drop
        statics.append(_static(spec, project(spec.clip, "left", drop=drop)))

    brackets = atlas.component("front", "barcode_brackets")
    bars_box = project(frame.front["barcode"]["bars_rect"], "right")
    barcode = _barcode_placement(bars_box, 90, atlas)
    statics.append(_static(brackets, _bracket_for_bars(frame, brackets, barcode.bars, side=False)))

    names_cfg = _scale_names_cfg(frame.front["names"], fc, content, scale, y_shift + middle_drop, atlas,
                                 scale_x=scale_x) if scale_x != scale else _scale_names_cfg(
        frame.front["names"], fc, content, scale, y_shift + middle_drop, atlas)
    lineart_box = project(frame.front["lineart"]["rect"], "center")

    fields: dict[str, Any] = {
        "item_no": {
            "bar_rect": list(bar_box.as_tuple()),
            "x0": bar_box.x0,
            "baseline_y": bar_box.y0 + bar_height * 0.5 + 0.36 * fontsize,
            "max_width": bar_width,
            "fontsize": fontsize,
            "font": frame.front["item_no"]["font"],
            "color": list(frame.front["item_no"]["color"]),
            "fit": "center_in_bar",
        },
        "names": names_cfg,
        "lineart": {"rect": list(lineart_box.as_tuple()), "fit": "stretch"},
        "barcode": _scale_barcode_cfg(frame.front["barcode"], barcode.bars, scale),
    }

    return FacePlan(
        kind="front",
        family="front_standard",
        panel=panel,
        content=content,
        scale=scale,
        inset_mm=mm(inset),
        safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics,
        fields=fields,
        barcode=barcode,
    )


def _scale_names_cfg(
    cfg: dict[str, Any], fc: Box, content: Box, scale: float, y_shift: float, atlas: Atlas,
    scale_x: float | None = None,
) -> dict[str, Any]:
    sx = scale if scale_x is None else scale_x
    out = dict(cfg)
    out["top_y"] = content.y0 + (float(cfg["top_y"]) - fc.y0) * scale + y_shift
    out["bottom_y"] = content.y0 + (float(cfg["bottom_y"]) - fc.y0) * scale + y_shift
    # Horizontal quantities spread with scale_x (a name block may use the idle
    # width of a wide face); the glyph size itself stays on the uniform scale,
    # because a font may not be stretched sideways.
    out["max_width"] = float(cfg["max_width"]) * sx
    # The gap between name blocks is a length like every other, and its sibling
    # _band_names_cfg already scales it. Left unscaled it stayed at the
    # template's 3.0 pt on every face, so on a 1100 mm panel the names sat a
    # millimetre apart while everything around them grew.
    out["block_gap"] = float(cfg.get("block_gap", 3.0)) * scale
    # Clamp the shrink-to-fit floor up to the physical legibility minimum. The
    # renderer is then structurally unable to print below it, instead of relying
    # on a validator to notice after the fact.
    floors = atlas.constraints["min_font_mm"]
    floor_pt = pt(max(float(floors["latin_name"]), float(floors["cn_name"])))
    out["min_fontsize"] = max(floor_pt, float(cfg["min_fontsize"]) * scale)
    lines = []
    for line in cfg["lines"]:
        entry = dict(line)
        bx, by = line["baseline"]
        entry["baseline"] = [
            content.x0 + (float(bx) - fc.x0) * sx,
            content.y0 + (float(by) - fc.y0) * scale + y_shift,
        ]
        entry["fontsize"] = float(line["fontsize"]) * scale
        lines.append(entry)
    out["lines"] = lines
    return out


def _scale_barcode_cfg(cfg: dict[str, Any], bars: Box, scale: float) -> dict[str, Any]:
    out = dict(cfg)
    out["bars_rect"] = list(bars.as_tuple())
    for key in (
        "text_fontsize",
        "readout_gap",
        "character_spacing",
        "first_digit_height",
        "first_digit_offset",
        "main_group_gap",
        "main_bottom_inset",
        "main_top_inset",
        "regular_bar_inset",
    ):
        if key in out:
            out[key] = float(out[key]) * scale
    if "readout_origin" in out:
        out["readout_origin"] = [float(v) * scale for v in out["readout_origin"]]
    return out


def build_front_column(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """Owner-confirmed order: bar | brand/names | art/origin/icons | barcode.

    Each column is independent of the others. Regions are an initial shared
    allocation, while artwork keeps its original proportions and names fit the
    available width. No interpolation through hand-dragged anchor coordinates
    is used to predict a new carton.
    """
    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)

    def region(name: str) -> Box:
        spec = FRONT_COLUMN_REGIONS[name]
        return content.region(spec["x"], spec["y"])

    bar, fontsize = _band_bar(atlas, content, region("item_no_bar"), str(item.get("item_no") or ""))
    statics = [_static(atlas.component("front", "item_no_bar"), bar)]
    for name in ("kmt_logo", "origin_text", "icon_row"):
        statics.append(_place_uniform(atlas.component("front", name), region(name), align_y="middle"))

    brackets = atlas.component("front", "barcode_brackets")
    tile = region("barcode").fit_uniform(brackets.aspect, align_x="right", align_y="middle")
    barcode = _barcode_placement(_bracket_inner_box(frame, brackets, tile, rotated=False), 90, atlas)
    statics.append(_static(brackets, _bracket_for_bars(frame, brackets, barcode.bars, side=False)))
    barcode_scale = barcode.bars.height / (frame.front["barcode"]["bars_rect"][3] - frame.front["barcode"]["bars_rect"][1])
    barcode_cfg = _scale_barcode_cfg(frame.front["barcode"], barcode.bars, barcode_scale)
    barcode_cfg["physical_rotation_deg"] = 270

    # One block: the Chinese name directly above the IT/ES/EN names.
    names = _band_names_cfg(
        frame.front["names"], region("names"), scale, ["cn_name", "it_name", "es_name", "en_name"], atlas
    )
    # A column may wrap a longer name onto two lines. The renderer's common
    # shrink-to-fit path still enforces the physical legibility floor.
    #
    # The start size comes from the region height, which a narrow column's
    # width often cannot carry. The template search only looks down to 0.55 of
    # the start before dropping to the floor, so the size is searched
    # continuously instead (AdaptiveRenderer._names_start_size): on the
    # 780 x 200 mm front the Chinese name went from the 4.3 mm floor to the
    # largest size the column actually holds.
    names["max_lines_per_name"] = 2
    names["fit"] = "shrink_fontsize_continuous"
    for line in names["lines"]:
        line["max_lines"] = 2

    return FacePlan(
        kind="front", family="column", panel=panel, content=content,
        scale=scale, inset_mm=mm(inset), safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics, barcode=barcode,
        fields={
            "item_no": {
                "bar_rect": list(bar.as_tuple()), "x0": bar.x0,
                "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
                "max_width": bar.width, "fontsize": fontsize,
                "font": frame.front["item_no"]["font"],
                "color": list(frame.front["item_no"]["color"]), "fit": "center_in_bar",
            },
            "names": names,
            "lineart": {"rect": list(region("lineart").as_tuple()), "fit": "stretch"},
            "barcode": barcode_cfg,
        },
        notes=[f"rule={OWNER_RULE_REVISION}", "four columns; 55% bar allocation is a prototype, not an aspect curve"],
    )


def build_front_wide_band(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """The arrangement the factory switches to on a very wide, short face.

    The bar retreats to the left so line art can own the right edge at nearly
    full height, and the remainder becomes one bottom row of
    [logo + Latin names] [upright barcode] [Chinese name + origin + icons].
    """
    frame = atlas.frame
    spec = atlas.families["front_wide_band"]
    regions = spec["regions"]
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)

    item_no = str(item.get("item_no") or "")
    bar_region = content.region(regions["item_no_bar"]["x"], regions["item_no_bar"]["y"])
    bar_box, fontsize = _band_bar(atlas, content, bar_region, item_no)
    bar_height = bar_box.height

    statics = [_static(atlas.component("front", "item_no_bar"), bar_box)]

    logo = atlas.component("front", "kmt_logo")
    statics.append(_place_uniform(logo, content.region(regions["kmt_logo"]["x"], regions["kmt_logo"]["y"]),
                                  align_x="left", align_y="middle"))

    origin = atlas.component("front", "origin_text")
    statics.append(_place_uniform(origin, content.region(regions["origin_text"]["x"], regions["origin_text"]["y"]),
                                  align_x="left", align_y="middle"))

    icons = atlas.component("front", "icon_row")
    statics.append(_place_uniform(icons, content.region(regions["icon_row"]["x"], regions["icon_row"]["y"]),
                                  align_x="left", align_y="bottom"))

    # Upright barcode: 95 modules along the width. The bracket tile is drawn
    # from the same clip but rotated by placing it in a transposed box, so the
    # bracket proportions stay locked to the symbol.
    barcode_region = content.region(regions["barcode"]["x"], regions["barcode"]["y"])
    brackets = atlas.component("front", "barcode_brackets")
    bracket_box = barcode_region.fit_uniform(1.0 / brackets.aspect, align_x="center", align_y="middle")
    statics.append(
        StaticPlacement(
            name="barcode_brackets",
            clip=brackets.clip,
            target=bracket_box,
            fit="rotate90_uniform",
            source_page=brackets.source_page,
        )
    )
    inner = _bracket_inner_box(frame, brackets, bracket_box, rotated=True)
    barcode = _barcode_placement(inner, 0, atlas)
    bracket_box = _bracket_for_bars(frame, brackets, barcode.bars, side=False, rotated=True)
    statics[-1] = StaticPlacement(
        name="barcode_brackets", clip=brackets.clip, target=bracket_box,
        fit="rotate90_uniform", source_page=brackets.source_page,
    )

    # Line art owns the right edge of the top band, as the factory does. Its
    # width is capped relative to the band height so a wide face does not leave
    # the artwork floating in the middle of an over-wide slot, and it may expand
    # leftward into whatever the hugging bar did not use.
    lineart_region = content.region(regions["lineart"]["x"], regions["lineart"]["y"])
    gap = pt(_MIN_GAP_MM)
    left_limit = max(bar_box.x1 + gap, content.x0)
    available = max(1.0, content.x1 - left_limit)
    lineart_width = min(available, lineart_region.height * _LINEART_MAX_ASPECT)
    lineart_box = Box(
        content.x1 - lineart_width, lineart_region.y0, content.x1, lineart_region.y1
    )
    names_region = content.region(regions["names"]["x"], regions["names"]["y"])
    cn_region = content.region(regions["cn_name"]["x"], regions["cn_name"]["y"])

    fields = {
        "item_no": {
            "bar_rect": list(bar_box.as_tuple()),
            "x0": bar_box.x0,
            "baseline_y": bar_box.y0 + bar_height * 0.5 + 0.36 * fontsize,
            "max_width": bar_box.width,
            "fontsize": fontsize,
            "font": frame.front["item_no"]["font"],
            "color": list(frame.front["item_no"]["color"]),
            "fit": "center_in_bar",
        },
        "names": _band_names_cfg(frame.front["names"], names_region, scale, regions["names"]["fields"], atlas),
        "cn_name": _band_names_cfg(frame.front["names"], cn_region, scale, ["cn_name"], atlas),
        "lineart": {"rect": list(lineart_box.as_tuple()), "fit": "stretch"},
        "barcode": _upright_barcode_cfg(frame.front["barcode"], barcode.bars, scale),
    }

    return FacePlan(
        kind="front",
        family="front_wide_band",
        panel=panel,
        content=content,
        scale=scale,
        inset_mm=mm(inset),
        safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics,
        fields=fields,
        barcode=barcode,
        notes=["barcode rotated to 0 degrees: 95 modules do not fit along a short face height"],
    )


def build_front_three_column(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """Owner anchors 2.08 and 2.62 (2026-09-15): bar on top, then three columns.

    Full-width item-number bar; below it logo over the names block | line art |
    horizontal barcode over origin and icons. Each element fills its own region
    (owner: 满版), which is what the uniform projection of front_standard could
    not do on a face twice as wide as the template.
    """
    frame = atlas.frame
    regions = atlas.families["front_three_column"]["regions"]
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)

    def region(name: str) -> Box:
        return content.region(regions[name]["x"], regions[name]["y"])

    # Full-width bar: nothing sits beyond its right edge.
    bar, fontsize = _band_bar(atlas, content, region("item_no_bar"), str(item.get("item_no") or ""), clear_right=False)
    statics = [_static(atlas.component("front", "item_no_bar"), bar)]
    for name, align_y in (("kmt_logo", "middle"), ("origin_text", "middle"), ("icon_row", "bottom")):
        statics.append(_place_uniform(atlas.component("front", name), region(name), align_x="left", align_y=align_y))

    # Horizontal symbol, brackets rotated with it (as front_wide_band). The HRI
    # scales with the symbol itself so the whole group stays one uniform scale.
    brackets = atlas.component("front", "barcode_brackets")
    tile = region("barcode").fit_uniform(1.0 / brackets.aspect, align_x="left", align_y="middle")
    barcode = _barcode_placement(_bracket_inner_box(frame, brackets, tile, rotated=True), 0, atlas)
    statics.append(StaticPlacement(
        name="barcode_brackets", clip=brackets.clip, fit="rotate90_uniform", source_page=brackets.source_page,
        target=_bracket_for_bars(frame, brackets, barcode.bars, side=False, rotated=True),
    ))
    template_bars = frame.front["barcode"]["bars_rect"]
    barcode_scale = barcode.bars.width / (float(template_bars[3]) - float(template_bars[1]))

    names = _band_names_cfg(frame.front["names"], region("names"), scale, regions["names"]["fields"], atlas)
    names["max_lines_per_name"] = 2
    names["fit"] = "shrink_fontsize_continuous"
    for line in names["lines"]:
        line["max_lines"] = 2

    return FacePlan(
        kind="front", family="front_three_column", panel=panel, content=content,
        scale=scale, inset_mm=mm(inset), safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics, barcode=barcode,
        fields={
            "item_no": {
                "bar_rect": list(bar.as_tuple()), "x0": bar.x0,
                "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
                "max_width": bar.width, "fontsize": fontsize,
                "font": frame.front["item_no"]["font"],
                "color": list(frame.front["item_no"]["color"]), "fit": "center_in_bar",
            },
            "names": names,
            "lineart": {"rect": list(region("lineart").as_tuple()), "fit": "stretch"},
            "barcode": _upright_barcode_cfg(frame.front["barcode"], barcode.bars, barcode_scale),
        },
        notes=["R1 2026-09-16: owner anchors 2.08 / 2.62 structure"],
    )


def _turn_fractions_clockwise(
    fx0: float, fy0: float, fx1: float, fy1: float
) -> tuple[float, float, float, float]:
    """Where a box given as fractions of a tile lands after the tile turns a quarter clockwise.

    The renderer stamps every ``rotate90_uniform`` tile a quarter turn
    clockwise (``renderer.ROTATE90_UNIFORM_ANGLE``), which takes native
    ``(x, y)`` to ``(1 - y, x)``. This used to be a transpose ``(y, x)``, which
    is neither quarter turn: base.pdf's bars sit asymmetrically in their
    brackets (front x 0.134-0.845, y 0.078-0.894 of the tile), so the planned
    bars were 0.02-0.03 of the tile away from the printed ones, and under the
    clockwise turn that offset would lie along the module axis, where the quiet
    zones are measured.
    """
    return 1.0 - fy1, fx0, 1.0 - fy0, fx1


def _bars_fractions(frame, brackets: ComponentSpec, *, side: bool) -> tuple[float, float, float, float]:
    """Where the bars sit inside the bracket tile, as fractions of that tile."""
    bars = (frame.side if side else frame.front)["barcode"]["bars_rect"]
    bx0, by0, bx1, by1 = (float(v) for v in bars)
    cx0, cy0, cx1, cy1 = brackets.clip
    fw = max(1e-6, cx1 - cx0)
    fh = max(1e-6, cy1 - cy0)
    return ((bx0 - cx0) / fw, (by0 - cy0) / fh, (bx1 - cx0) / fw, (by1 - cy0) / fh)


def _bracket_for_bars(
    frame, brackets: ComponentSpec, bars: Box, *, side: bool, rotated: bool = False
) -> Box:
    """Inverse of ``_bracket_inner_box``: the bracket tile enclosing ``bars``.

    Needed because the barcode may be capped at the GS1 200% ceiling after its
    slot was chosen. The printed corner brackets have to shrink with it, or a
    small symbol would sit inside oversized brackets.
    """
    fx0, fy0, fx1, fy1 = _bars_fractions(frame, brackets, side=side)
    if rotated:
        fx0, fy0, fx1, fy1 = _turn_fractions_clockwise(fx0, fy0, fx1, fy1)
    width = bars.width / max(1e-6, fx1 - fx0)
    height = bars.height / max(1e-6, fy1 - fy0)
    x0 = bars.x0 - width * fx0
    y0 = bars.y0 - height * fy0
    return Box(x0, y0, x0 + width, y0 + height)


def _bracket_inner_box(frame, brackets: ComponentSpec, placed: Box, *, rotated: bool) -> Box:
    """Where the bars sit inside the corner-bracket tile.

    Derived from the template's own bracket-to-bars relationship so the bars
    never drift out of the printed brackets at any scale.
    """
    bars = frame.front["barcode"]["bars_rect"]
    bx0, by0, bx1, by1 = (float(v) for v in bars)
    cx0, cy0, cx1, cy1 = brackets.clip
    fw = max(1e-6, cx1 - cx0)
    fh = max(1e-6, cy1 - cy0)
    fx0, fy0 = (bx0 - cx0) / fw, (by0 - cy0) / fh
    fx1, fy1 = (bx1 - cx0) / fw, (by1 - cy0) / fh
    if rotated:
        return placed.fraction(*_turn_fractions_clockwise(fx0, fy0, fx1, fy1))
    return placed.fraction(fx0, fy0, fx1, fy1)


def _band_names_cfg(
    cfg: dict[str, Any], region: Box, scale: float, fields: list[str], atlas: Atlas
) -> dict[str, Any]:
    """A names block re-flowed into an explicit region of a band layout.

    ``min_fontsize`` is clamped up to the policy's physical legibility floor.
    Without that clamp the shrink-to-fit path could quietly reach an unreadable
    size on a short face and still look like it "fitted".
    """
    lines = [dict(line) for line in cfg["lines"] if line["field"] in fields]
    count = max(1, len(lines))
    leading = float(cfg.get("leading_ratio", 1.15))
    block_gap = float(cfg.get("block_gap", 3.0)) * scale
    usable = region.height - block_gap * (count - 1)
    per_line = max(1.0, usable / count)
    fontsize = per_line / leading
    for index, line in enumerate(lines):
        line["fontsize"] = fontsize
        line["baseline"] = [region.x0, region.y0 + (index + 1) * per_line]
        line["max_lines"] = 1

    floors = atlas.constraints["min_font_mm"]
    floor_mm = max(
        float(floors["cn_name" if field == "cn_name" else "latin_name"])
        for field in (fields or ["latin_name"])
    )
    return {
        "top_y": region.y0,
        "bottom_y": region.y1,
        "leading_ratio": leading,
        "block_gap": block_gap,
        "max_lines_per_name": 1,
        "lines": lines,
        "max_width": region.width,
        "fit": "shrink_fontsize",
        "min_fontsize": max(pt(floor_mm), min(fontsize, float(cfg["min_fontsize"]) * scale)),
        "rect": list(region.as_tuple()),
    }


def _upright_barcode_cfg(cfg: dict[str, Any], bars: Box, scale: float) -> dict[str, Any]:
    out = _scale_barcode_cfg(cfg, bars, scale)
    out["orientation"] = "horizontal"
    out["hri_layout"] = "upright_single_span"
    return out


# --------------------------------------------------------------------------- #
# Side face
# --------------------------------------------------------------------------- #


SIDE_CELLS = ("cell_qty", "cell_meas", "cell_name", "cell_nw", "cell_gw")


def _uniform_cell_group(atlas: Atlas, slots: dict[str, Box]) -> tuple[list[StaticPlacement], dict[str, Box], float]:
    """All five original label/frame tiles use one uniform source scale.

    The old independent region rectangles stretched each label differently.
    Shared scaling preserves the original label-to-value relationship; moving
    a cell only changes translation. Value fields are subsequently projected
    from the same placed tile, so their registration stays intact.
    """
    scale = min(
        min(slot.width / atlas.component("side", name).size[0], slot.height / atlas.component("side", name).size[1])
        for name, slot in slots.items()
    )
    placements = []
    boxes = {}
    for name, slot in slots.items():
        spec = atlas.component("side", name)
        width, height = (value * scale for value in spec.size)
        box = Box(slot.x0, slot.y0, slot.x0 + width, slot.y0 + height)
        boxes[name] = box
        placements.append(StaticPlacement(name=name, clip=spec.clip, target=box, fit="uniform", source_page=spec.source_page))
    return placements, boxes, scale


class OwnerGridUnavailable(ValueError):
    """The owner side grid cannot be laid out on this face at all.

    Raised only for geometry that has no answer (no room below the bar, no
    width left beside the barcode). ``build_candidates`` turns it into a failed
    candidate, and because the owner rule makes the grid the only side family
    for a triggering carton, the SKU then fails closed as
    ``manual_layout_required`` instead of silently changing arrangement.
    """


def template_side_bar_share(frame) -> float:
    """The side item-number bar's share of the design content height.

    Read from template.json (side bar_rect 18.5-199.2 pt against the 431 pt
    content height = 0.4193), not typed in. It lies inside the owner's observed
    0.383-0.419 band and is the owner's 4.50-side value to three decimals, so
    the grid's bar is the approved mark's own proportion rather than a new one.
    """
    bar = frame.side["item_no"]["bar_rect"]
    return (float(bar[3]) - float(bar[1])) / frame.content_height


def build_side_owner_grid(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """Owner side structure: bar on top, QTY/MEAS | NAME/N.W./G.W. | barcode.

    This is template.json's own side table, widened. base.pdf already draws
    the side as a full-width bar, a two-column table and a vertical barcode on
    the right; what the owner changed on 2026-09-15 is that the table fills the
    width up to the barcode instead of floating at its source width. The
    construction therefore keeps every source relationship and only chooses
    two numbers, the table scale and how much the empty cell interiors widen:

    * One scale ``s`` for the whole table - labels, units, row heights, row
      gaps, column gap and the bar-to-table gap all come from base.pdf times
      ``s``. ``s`` is the largest scale at which the table reaches the content
      bottom (height-bound) while the table at source proportions plus a
      barcode as tall as the table still fits the width (width-bound).
    * The table's right edge stops ``min_component_gap_mm`` short of the
      barcode brackets. A vertical symbol's quiet zones run along y, above and
      below the bars, so ``_validate_quiet_zones`` does not bind sideways; the
      binding checks are components_overlap (hard) and the clearance warning.
    * The width that remains is shared between the two columns in their source
      width ratio (222.9 : 199.2 pt), and every cell in a column takes the
      column's width. The owner's own columns disagree with each other on this
      split (QTY:NAME 1.31 on the 4.50 side, 0.95 on 7.35), so neither is
      copied. Each cell is drawn ``slice_x``: label and unit at ``s``, only the
      empty interior stretched, never compressed below its source width.

    When the width binds - a square or taller-than-wide side - the interiors
    stay at source width, the table is the template table at scale ``s``
    hanging below the bar, and the space under it stays empty. The owner's
    own 170 x 170 mm side leaves 10-13% of its height free there as well.
    """
    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)
    item_no = str(item.get("item_no") or "")

    share = template_side_bar_share(frame)
    bar_region = Box(content.x0, content.y0, content.x1, content.y0 + content.height * share)
    bar, fontsize = _band_bar(atlas, content, bar_region, item_no)
    bar_spec = atlas.component("side", "item_no_bar")
    statics = [_static(bar_spec, bar)]

    columns = OWNER_SIDE_COLUMNS
    specs = {name: atlas.component("side", name) for column in columns for name in column}
    table_y0 = min(spec.clip[1] for spec in specs.values())
    table_height = max(spec.clip[3] for spec in specs.values()) - table_y0
    bar_gap = table_y0 - bar_spec.clip[3]
    column_gap = min(specs[n].clip[0] for n in columns[1]) - max(specs[n].clip[2] for n in columns[0])
    column_widths = [max(specs[n].size[0] for n in column) for column in columns]
    brackets = atlas.component("side", "barcode_brackets")
    barcode_gap = pt(float(atlas.constraints.get("min_component_gap_mm", 1.5)))

    height_scale = (content.y1 - bar.y1) / (bar_gap + table_height)
    width_scale = (content.width - barcode_gap) / (
        sum(column_widths) + column_gap + brackets.aspect * table_height
    )
    cell_scale = min(height_scale, width_scale)
    if not cell_scale > 0:
        raise OwnerGridUnavailable(
            f"no room for the side table: height scale {height_scale:.4f}, width scale {width_scale:.4f}"
        )

    table_top = bar.y1 + bar_gap * cell_scale
    table_bottom = table_top + table_height * cell_scale
    # Width-bound (square or tall side): the table at scale s leaves height
    # free under it. Owner 2026-09-23 - no blank under the table; the factory
    # photo 557099 grows the rows to the bottom and keeps the words as they are.
    # So the rows take that height in their padding bands (atlas.slice_y) and
    # the barcode group, kept uniform, is centred on the taller table band.
    slack = content.y1 - table_bottom if height_scale > width_scale else 0.0
    grow_rows = slack > 1e-3 * content.height
    shift = slack / 2.0 if grow_rows else 0.0
    tile_width = brackets.aspect * table_height * cell_scale
    tile = Box(content.x1 - tile_width, table_top + shift, content.x1, table_bottom + shift)
    barcode = _barcode_placement(_bracket_inner_box_side(frame, brackets, tile), 90, atlas)
    bracket_box = _bracket_for_bars(frame, brackets, barcode.bars, side=True)

    # The clearance check tests ``x1 + gap <= x0``; a table edge computed as
    # exactly ``x0 - gap`` can round one ulp past that and report a clearance
    # warning at the very gap it was built to keep (seen on the 273 x 193 mm
    # side). A micro-point of float headroom is not a design margin.
    available = bracket_box.x0 - barcode_gap - 1e-6 - content.x0 - column_gap * cell_scale
    widths = [available * width / sum(column_widths) for width in column_widths]
    starts = [content.x0, content.x0 + widths[0] + column_gap * cell_scale]
    cells: dict[str, StaticPlacement] = {}
    for column, x0, width in zip(columns, starts, widths):
        ordered = sorted(column, key=lambda n: specs[n].clip[1])
        heights = [specs[n].size[1] * cell_scale for n in ordered]
        column_bottom = table_top + (specs[ordered[-1]].clip[3] - table_y0) * cell_scale
        extra = (content.y1 - column_bottom) if grow_rows else 0.0
        grown_above = 0.0
        for name, height in zip(ordered, heights):
            spec = specs[name]
            add = extra * height / sum(heights)
            y0 = table_top + (spec.clip[1] - table_y0) * cell_scale + grown_above
            grown_above += add
            cells[name] = StaticPlacement(
                name=name, clip=spec.clip, source_page=spec.source_page, fit="slice_x",
                target=Box(x0, y0, x0 + width, y0 + height + add),
                slice_x=atlas.slice_x("side", name),
                **({"slice_y": atlas.slice_y("side", name), "glyph_scale": cell_scale} if grow_rows else {}),
                **_label_kwargs(atlas, name),
            )
    statics.extend(cells[name] for name in SIDE_CELLS)
    statics.append(_static(brackets, bracket_box))

    barcode_scale = barcode.bars.height / (frame.side["barcode"]["bars_rect"][3] - frame.side["barcode"]["bars_rect"][1])
    barcode_cfg = _scale_barcode_cfg(frame.side["barcode"], barcode.bars, barcode_scale)
    barcode_cfg["physical_rotation_deg"] = 270

    # Values keep their registration by going through the cell's own
    # piecewise mapping: the template's qty rect sits in the QTY interior
    # between label and PCS, so it stays centred there however wide the
    # interior becomes; the NAME value likewise stays inside NAME's interior.
    qty_cell, name_cell = cells["cell_qty"], cells["cell_name"]
    qty_box = qty_cell.map_source_rect(frame.side["qty"]["rect"])
    qty_size = float(frame.side["qty"]["fontsize"]) * cell_scale
    name_box = name_cell.map_source_rect(frame.side["es_name"]["rect"])
    binding = "height" if height_scale <= width_scale else "width"
    qty_cfg = {"rect": list(qty_box.as_tuple()),
               "baseline_y": qty_cell.map_source_y(float(frame.side["qty"]["baseline_y"])),
               "fontsize": qty_size, "font": frame.side["qty"]["font"], "fit": "shrink_center"}
    name_cfg = {**frame.side["es_name"], "rect": list(name_box.as_tuple()),
                "fontsize": float(frame.side["es_name"]["fontsize"]) * cell_scale,
                "min_fontsize": max(pt(float(atlas.constraints["min_font_mm"]["table_value"])), float(frame.side["es_name"]["min_fontsize"]) * cell_scale)}
    _grow_table_text(atlas, cells, qty_cfg, name_cfg)
    return FacePlan(
        kind="side", family="side_owner_grid", panel=panel, content=content,
        scale=scale, inset_mm=mm(inset), safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics, barcode=barcode,
        fields={
            "item_no": {
                "bar_rect": list(bar.as_tuple()), "x0": bar.x0,
                "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
                "max_width": bar.width, "fontsize": fontsize,
                "font": frame.side["item_no"]["font"],
                "color": list(frame.side["item_no"]["color"]), "fit": "center_in_bar",
            },
            "qty": qty_cfg,
            "es_name": name_cfg,
            "barcode": barcode_cfg,
        },
        notes=[
            f"rule={OWNER_RULE_REVISION}",
            f"owner side grid: bar share {share:.4f} (template.json side bar); table scale {cell_scale:.4f}, "
            f"{binding}-bound; cells slice_x, interiors widened, labels at the table scale",
        ],
    )


def build_side_standard(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """The current side mark, re-proportioned. Two-column table, vertical barcode."""
    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)
    fc = frame.content

    slack = max(0.0, content.height - frame.content_height * scale)
    design_top = content.y0 + slack / 2.0

    item_no = str(item.get("item_no") or "")
    bar_height, fontsize = _item_no_geometry(atlas, content, item_no, scale, content.width)
    bar_box = Box(content.x0, design_top, content.x1, design_top + bar_height)
    statics = [_static(atlas.component("side", "item_no_bar"), bar_box)]

    tpl_bar_bottom = float(frame.side["item_no"]["bar_rect"][3])
    y_shift = bar_box.y1 - (content.y0 + (tpl_bar_bottom - fc.y0) * scale)

    def project(rect: Any, anchor_x: str) -> Box:
        box = _project_box(fc, content, rect, scale, anchor_x)
        return Box(box.x0, box.y0 + y_shift, box.x1, box.y1 + y_shift)

    for name in SIDE_CELLS:
        spec = atlas.component("side", name)
        statics.append(_static(spec, project(spec.clip, "left")))

    brackets = atlas.component("side", "barcode_brackets")
    bars_box = project(frame.side["barcode"]["bars_rect"], "right")
    barcode = _barcode_placement(bars_box, 90, atlas)
    statics.append(_static(brackets, _bracket_for_bars(frame, brackets, barcode.bars, side=True)))

    qty_box = project(frame.side["qty"]["rect"], "left")
    es_box = project(frame.side["es_name"]["rect"], "left")

    fields = {
        "item_no": {
            "bar_rect": list(bar_box.as_tuple()),
            "x0": bar_box.x0,
            "baseline_y": bar_box.y0 + bar_height * 0.5 + 0.36 * fontsize,
            "max_width": bar_box.width,
            "fontsize": fontsize,
            "font": frame.side["item_no"]["font"],
            "color": list(frame.side["item_no"]["color"]),
            "fit": "center_in_bar",
        },
        "qty": {
            "rect": list(qty_box.as_tuple()),
            "baseline_y": qty_box.y0 + qty_box.height * 0.5 + 0.36 * float(frame.side["qty"]["fontsize"]) * scale,
            "fontsize": float(frame.side["qty"]["fontsize"]) * scale,
            "font": frame.side["qty"]["font"],
            "fit": "shrink_center",
        },
        "es_name": {
            **frame.side["es_name"],
            "rect": list(es_box.as_tuple()),
            "fontsize": float(frame.side["es_name"]["fontsize"]) * scale,
            "min_fontsize": float(frame.side["es_name"]["min_fontsize"]) * scale,
        },
        "barcode": _scale_barcode_cfg(frame.side["barcode"], barcode.bars, scale),
    }

    return FacePlan(
        kind="side",
        family="side_standard",
        panel=panel,
        content=content,
        scale=scale,
        inset_mm=mm(inset),
        safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics,
        fields=fields,
        barcode=barcode,
    )


def build_side_compressed(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """Wide short side face: bar and a 3-column table share the left, the
    vertical barcode holds the right edge at full height."""
    frame = atlas.frame
    spec = atlas.families["side_compressed"]
    regions = spec["regions"]
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)

    item_no = str(item.get("item_no") or "")
    bar_region = content.region(regions["item_no_bar"]["x"], regions["item_no_bar"]["y"])
    bar_box, fontsize = _band_bar(atlas, content, bar_region, item_no)
    bar_height = bar_box.height
    statics = [_static(atlas.component("side", "item_no_bar"), bar_box)]

    barcode_region = content.region(regions["barcode"]["x"], regions["barcode"]["y"])

    # Nothing in the left area may run all the way to the barcode column.
    # side_compressed declares the table at 0.68-0.88 and the barcode column at
    # 0.88-1.0, so the two share a coordinate exactly - unlike front_wide_band,
    # which declares 0.0-0.74 against 0.77 and leaves a gap in the policy. The
    # shared edge made cell_nw and cell_gw report components_overlap against
    # barcode_brackets, and because side_standard is not a candidate above
    # aspect 2.5 there was no fallback: 51 side faces between aspect 2.88 and
    # 3.66 were rejected outright. Measured 2026-09-10 by re-running the
    # whole-domain scan; the same 51 fail identically under the pre-2026-09-10
    # bar rule, so this is older than that change and was simply never scanned
    # for.
    left_limit = barcode_region.x0 - pt(_MIN_GAP_MM)

    cell_slots: dict[str, Box] = {}
    for name in SIDE_CELLS:
        if name not in regions:
            continue
        region = content.region(regions[name]["x"], regions[name]["y"])
        if region.x1 > left_limit:
            region = Box(region.x0, region.y0, max(region.x0 + 1.0, left_limit), region.y1)
        cell_slots[name] = region
    cells, cell_boxes, cell_scale = _uniform_cell_group(atlas, cell_slots)
    statics.extend(cells)
    brackets = atlas.component("side", "barcode_brackets")
    bracket_box = barcode_region.fit_uniform(brackets.aspect, align_x="right", align_y="middle")
    inner = _bracket_inner_box_side(frame, brackets, bracket_box)
    barcode = _barcode_placement(inner, 90, atlas)
    statics.append(_static(brackets, _bracket_for_bars(frame, brackets, barcode.bars, side=True)))

    qty_cell = cell_boxes.get("cell_qty", content)
    name_cell = cell_boxes.get("cell_name", content)
    qty_value = _cell_value_box(frame, "cell_qty", "qty", qty_cell, atlas)
    es_value = _cell_value_box(frame, "cell_name", "es_name", name_cell, atlas)

    fields = {
        "item_no": {
            "bar_rect": list(bar_box.as_tuple()),
            "x0": bar_box.x0,
            "baseline_y": bar_box.y0 + bar_height * 0.5 + 0.36 * fontsize,
            "max_width": bar_box.width,
            "fontsize": fontsize,
            "font": frame.side["item_no"]["font"],
            "color": list(frame.side["item_no"]["color"]),
            "fit": "center_in_bar",
        },
        "qty": {
            "rect": list(qty_value.as_tuple()),
            "baseline_y": qty_value.y0 + qty_value.height * 0.5 + 0.36 * (qty_value.height * 0.62),
            "fontsize": qty_value.height * 0.62,
            "font": frame.side["qty"]["font"],
            "fit": "shrink_center",
        },
        "es_name": {
            **frame.side["es_name"],
            "rect": list(es_value.as_tuple()),
            "fontsize": es_value.height * 0.55,
            "min_fontsize": max(pt(float(atlas.constraints["min_font_mm"]["table_value"])), es_value.height * 0.28),
            "max_lines": 2,
            "preferred_max_lines": 1,
        },
        "barcode": _scale_barcode_cfg(frame.side["barcode"], barcode.bars, scale),
    }

    return FacePlan(
        kind="side",
        family="side_compressed",
        panel=panel,
        content=content,
        scale=scale,
        inset_mm=mm(inset),
        safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics,
        fields=fields,
        barcode=barcode,
        notes=["side table regrouped into 3 columns x 2 rows to fit a wide short face"],
    )


def _slice_cell_group(
    atlas: Atlas, slots: dict[str, Box], *, fill_slots: bool
) -> tuple[dict[str, StaticPlacement], float]:
    """The five table cells, each in its own slot.

    Two readings of the same table are kept, chosen by ``fill_slots`` and
    therefore by ``side_table.row_heights`` in the policy:

    ``fill_slots=True`` (``row_heights="reference"``) - the row a cell sits in
    is the row it is *drawn* at, so the drawn heights are exactly the slots',
    which :func:`_cell_stack_slots` apportioned from the factory reference's
    1.489. This is what the reference actually draws: on p17 its ``Q.TY`` plate
    is 76.39 pt against 51.26 pt for the other four, and ``Q.TY``'s own label
    glyphs are 24.05 pt against 17.72 pt - a bigger box, not a shared scale
    that drifted.

    ``fill_slots=False`` (``row_heights="artwork"``) - the slots decide only how
    much room each cell has; the cell is drawn at its own tile's source height
    at the frame scale, centred in its slot. That reproduces base.pdf's own
    proportions (128.5 / 66.1 / 83.8 / 70.7 / 70.6 pt, i.e. 1.765 with a 1.268
    spread across the other four) and is what the template happened to produce.

    Both paths leave the stack inside its band by construction: whichever way a
    cell is drawn, its height never exceeds its slot, and the slots are cut from
    the band. Both also keep one scale per cell rather than one scale for the
    table, so the units (PCS / cm / KG) beside the labels are never stretched -
    the deformation a per-cell ``slice_x`` takes out of the empty interior.
    """
    cells: dict[str, StaticPlacement] = {}
    for name, slot in slots.items():
        spec = atlas.component("side", name)
        if not slot.height > 0:
            raise OwnerGridUnavailable(f"no height left for side table cell {name}")
        if fill_slots:
            target = slot
        else:
            height = spec.size[1] * _table_scale(atlas, slots)
            y0 = slot.y0 + (slot.height - height) / 2.0
            target = Box(slot.x0, y0, slot.x1, y0 + height)
        cells[name] = StaticPlacement(
            name=name, clip=spec.clip, source_page=spec.source_page, fit="slice_x",
            target=target, slice_x=atlas.slice_x("side", name), **_label_kwargs(atlas, name),
        )
    return cells, min(cell.uniform_scale for cell in cells.values())


def _label_kwargs(atlas: Atlas, name: str) -> dict[str, Any]:
    """Keep a widened cell's black label block at its template share (``side_table.label_block``).

    ``template_share`` (default): the block stays a third of the cell as drawn
    in base.pdf and on the factory's boxes; only black padding beside the word
    widens (owner 2026-09-29, "黑框框就一点点"). ``fixed``: the block stays at
    the tile's uniform scale and only the white interior widens - the drawing
    before 2026-09-29.

    ``label_word`` ``block_share`` (default) grows the word with the block - its
    template share of the block's width, no taller than
    ``label_word_max_row_share`` of the row (owner 2026-09-29, "QTY 等标志也可以
    跟着变大"); ``fixed`` keeps the word at the tile's own scale.
    """
    cfg = atlas.policy.get("side_table") or {}
    if str(cfg.get("label_block", "template_share")) == "fixed":
        return {}
    out: dict[str, Any] = {"label_pad": atlas.label_pad("side", name)}
    if str(cfg.get("label_word", "block_share")) != "fixed":
        out["label_word"] = atlas.label_word("side", name)
        out["label_word_row_share"] = float(cfg.get("label_word_max_row_share", 0.5))
        out["unit_word"] = atlas.unit_word("side", name)
    return out


#: Cap height of the value fonts over their font size, for the row-share check
#: on the QTY digits (Arial Bold / Helvetica digits are 0.716 em).
_DIGIT_CAP = 0.716


def _grow_table_text(
    atlas: Atlas, cells: dict[str, StaticPlacement], qty_cfg: dict[str, Any], name_cfg: dict[str, Any]
) -> float:
    """Grow every word in each side-table cell by that cell's one factor (owner 2026-09-29).

    "单位和品名装箱数这些一起放大，这些字符一起定个规则". The rule, per cell:

    * the factor ``g`` is set by the label word - it keeps its template share
      of the black block's width (0.55-0.78, as on the factory's boxes) and is
      no taller than ``label_word_max_row_share`` of the row; never below 1;
    * the unit (PCS / cm / kg) and the value (Q.TY digits, NAME text) in that
      cell grow by the same ``g``, each also no taller than the row share;
    * a long NAME then wraps or shrinks inside its cell by its own rules.

    Per cell rather than one factor for the table: kg is already 0.42 of its
    row in base.pdf, so one table factor would hold every label to 1.2x - the
    owner had just approved the per-label sizes ("差不多"). Mutates ``cells``
    and the two field configs; returns the Q.TY cell's factor.
    """
    if not all(cell.label_word is not None for cell in cells.values()):
        return 1.0
    for cell in cells.values():
        cell.text_growth = cell.text_growth_limit()
    qty_cell = cells["cell_qty"]
    cap = qty_cell.label_word_row_share * qty_cell.target.height / (_DIGIT_CAP * float(qty_cfg["fontsize"]))
    growth = max(1.0, min(qty_cell.text_growth, cap))
    centre = float(qty_cfg["baseline_y"]) - 0.5 * _DIGIT_CAP * float(qty_cfg["fontsize"])
    qty_cfg["fontsize"] = float(qty_cfg["fontsize"]) * growth
    qty_cfg["baseline_y"] = centre + 0.5 * _DIGIT_CAP * qty_cfg["fontsize"]
    name_cell = cells["cell_name"]
    name_growth = name_cell.text_growth
    inner = name_cell.interior_box()
    x0, y0, x1, y1 = (float(v) for v in name_cfg["rect"])
    height = min(inner.height * 0.8, (y1 - y0) * name_growth)
    middle = (inner.y0 + inner.y1) / 2.0
    name_cfg["rect"] = [x0, middle - height / 2.0, x1, middle + height / 2.0]
    name_cfg["fontsize"] = float(name_cfg["fontsize"]) * name_growth
    return growth


def _cell_grid_slots(atlas: Atlas, content: Box, grid: dict[str, Any]) -> dict[str, Box]:
    """Slots for the five cells in columns side by side, each column filling the band.

    The factory's two-column table (owner photos 2026-09-29: 501174, 507021,
    557962, 556102, 556093): Q.TY over MEAS on the left, NAME / N.W. / G.W. on
    the right, both columns the full height of the table. Column widths follow
    the template's own columns (widest tile of each). Vertically the table is
    base.pdf's own table - every cell at its source position and height, row
    gaps included - scaled by one factor to the band, so all five cells draw at
    one scale (layout_audit info_cells); both source columns are 225 pt tall.
    """
    area = content.region(grid["x"], grid["y"])
    columns = [list(column) for column in grid["columns"]]
    column_gap = content.width * float(grid.get("column_gap", 0.0))
    specs = {name: atlas.component("side", name) for column in columns for name in column}
    top = min(spec.clip[1] for spec in specs.values())
    k = area.height / (max(spec.clip[3] for spec in specs.values()) - top)
    widths = [max(specs[name].size[0] for name in column) for column in columns]
    free_width = area.width - column_gap * (len(columns) - 1)
    if not (k > 0 and free_width > 0):
        raise OwnerGridUnavailable(f"no room for the side table columns in {area}")
    slots: dict[str, Box] = {}
    x = area.x0
    for column, width in zip(columns, widths):
        column_width = free_width * width / sum(widths)
        for name in column:
            y0 = area.y0 + (specs[name].clip[1] - top) * k
            slots[name] = Box(x, y0, x + column_width, y0 + specs[name].size[1] * k)
        x += column_width + column_gap
    return slots


def _table_scale(atlas: Atlas, slots: dict[str, Box]) -> float:
    """The one scale at which every tile fits its slot, for the artwork path.

    A single scale - not one per cell - because that is the arrangement
    ``row_heights="artwork"`` means: the table is one block drawn at one scale,
    with the tiles' own heights carrying the row proportions.
    """
    scale = min(
        min(
            slot.width / atlas.component("side", name).size[0],
            slot.height / atlas.component("side", name).size[1],
        )
        for name, slot in slots.items()
    )
    if not scale > 0:
        raise OwnerGridUnavailable(f"no room for the side table cells (scale {scale:.4f})")
    return scale


def _cell_stack_slots(atlas: Atlas, content: Box, stack: dict[str, Any]) -> dict[str, Box]:
    """Slots for the five cells stacked top to bottom in ``stack["order"]``.

    The band is cut by weight rather than into equal rows: the factory reference
    draws ``Q.TY`` taller than the other four and the other four equal, so
    ``Q.TY`` takes ``qty_row_ratio`` units of the band and each of the others
    one. That apportionment is the geometry both readings are measured against,
    so it is produced whatever ``row_heights`` says.

    What ``row_heights`` decides is whether a cell is drawn to fill its slot
    (``reference``) or at its tile's own height inside it (``artwork``); see
    :func:`_slice_cell_group`. The slots are cut the same way either way, which
    is what keeps the two paths comparable and the band respected.

    ``order`` is the source of both the stacking order and the weights; a cell
    without a ratio falls back to one unit, so a newer policy that names a cell
    this code has never seen degrades to an equal row instead of failing.
    """
    area = content.region(stack["x"], stack["y"])
    order = list(stack["order"])
    if not order:
        raise OwnerGridUnavailable("side table has no cells to place")
    gap = content.height * float(stack["gap"])
    ratios = _side_row_ratios(atlas, order)
    total = sum(ratios)
    free = area.height - gap * (len(order) - 1)
    if not free > 0:
        raise OwnerGridUnavailable(
            f"no room for the side table rows (band {area.height:.1f} pt with {len(order) - 1} gaps of {gap:.4f} pt)"
        )
    unit = free / total
    slots: dict[str, Box] = {}
    y = area.y0
    for name, weight in zip(order, ratios):
        height = unit * weight
        slots[name] = Box(area.x0, y, area.x1, y + height)
        y += height + gap
    return slots


def _side_row_ratios(atlas: Atlas, order: list[str]) -> list[float]:
    """Relative row heights for the side table, from the factory reference.

    ``Q.TY`` is taller than the rest and the rest are equal - see the
    ``side_table`` block in adaptive_layout.json for the measurement and its
    per-face spread. Read from the policy rather than typed in so the ratio has
    one home and the evidence travels with it.

    Under ``row_heights="artwork"`` the weights are deliberately all equal: that
    path does not apportion rows at all, it lets each tile's own height state
    the proportion, so the band is cut evenly only to give each cell room.
    """
    cfg = atlas.policy.get("side_table") or {}
    if _row_heights_mode(atlas) != "reference":
        return [1.0 for _ in order]
    ratio = float(cfg.get("qty_row_ratio", 1.0))
    return [ratio if name == "cell_qty" else 1.0 for name in order]


def _row_heights_mode(atlas: Atlas) -> str:
    """Which reading of the side table the policy asks for.

    One accessor, so the code, the tests and the audit all name the same switch.
    An unrecognised value is treated as the delivered default (``reference``)
    rather than crashing a production run over a policy typo.
    """
    cfg = atlas.policy.get("side_table") or {}
    mode = str(cfg.get("row_heights", "reference"))
    return "artwork" if mode == "artwork" else "reference"


def _build_side_from_regions(atlas: Atlas, panel: Box, item: dict[str, Any], family: str) -> FacePlan:
    """A side face laid out from the region map of an owner-confirmed structure."""
    frame = atlas.frame
    regions = atlas.families[family]["regions"]
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)

    def region(name: str) -> Box:
        return content.region(regions[name]["x"], regions[name]["y"])

    bar_region = region("item_no_bar")
    # A bar spanning the full content width has nothing beyond its right edge,
    # so it keeps the whole width instead of giving up the reflow gap.
    bar, fontsize = _band_bar(atlas, content, bar_region, str(item.get("item_no") or ""),
                              clear_right=bar_region.x1 < content.x1 - 1e-6)
    statics: list[StaticPlacement] = [_static(atlas.component("side", "item_no_bar"), bar)]
    if "cell_grid" in regions:
        order = [name for column in regions["cell_grid"]["columns"] for name in column]
        slots, fill = _cell_grid_slots(atlas, content, regions["cell_grid"]), True
    else:
        stack = regions["cell_stack"]
        order = list(stack["order"])
        slots, fill = _cell_stack_slots(atlas, content, stack), _row_heights_mode(atlas) == "reference"
    cells, cell_scale = _slice_cell_group(atlas, slots, fill_slots=fill)
    statics.extend(cells[name] for name in order)

    brackets = atlas.component("side", "barcode_brackets")
    horizontal = int(regions["barcode"].get("rotation_deg", 90)) == 0
    if horizontal:
        # The side tile is clipped upright; lying it down means rotating the tile
        # too, or a wide short slot binds it by height and the X dimension drops
        # under the GS1 floor.
        tile = region("barcode").fit_uniform(1.0 / brackets.aspect, align_x="center", align_y="middle")
        fit = "rotate90_uniform"
    else:
        tile = region("barcode").fit_uniform(brackets.aspect, align_x="right", align_y="middle")
        fit = brackets.fit
    barcode = _barcode_placement(
        _bracket_inner_box_side(frame, brackets, tile, rotated=horizontal), 0 if horizontal else 90, atlas
    )
    statics.append(StaticPlacement(
        name="barcode_brackets", clip=brackets.clip, fit=fit, source_page=brackets.source_page,
        target=_bracket_for_bars(frame, brackets, barcode.bars, side=True, rotated=horizontal),
    ))
    template_bars = frame.side["barcode"]["bars_rect"]
    module_length = barcode.bars.width if horizontal else barcode.bars.height
    barcode_cfg = _scale_barcode_cfg(
        frame.side["barcode"], barcode.bars, module_length / (float(template_bars[3]) - float(template_bars[1]))
    )
    if horizontal:
        barcode_cfg["orientation"] = "horizontal"
        barcode_cfg["hri_layout"] = "upright_single_span"
    else:
        barcode_cfg["physical_rotation_deg"] = 270

    # Values keep their registration through each cell's own piecewise mapping.
    qty_cell, name_cell = cells["cell_qty"], cells["cell_name"]
    qty_box = qty_cell.map_source_rect(frame.side["qty"]["rect"])
    name_box = name_cell.map_source_rect(frame.side["es_name"]["rect"])
    qty_cfg = {"rect": list(qty_box.as_tuple()),
               "baseline_y": qty_cell.map_source_y(float(frame.side["qty"]["baseline_y"])),
               "fontsize": float(frame.side["qty"]["fontsize"]) * cell_scale,
               "font": frame.side["qty"]["font"], "fit": "shrink_center"}
    name_cfg = {**frame.side["es_name"], "rect": list(name_box.as_tuple()),
                "fontsize": float(frame.side["es_name"]["fontsize"]) * cell_scale,
                "min_fontsize": max(pt(float(atlas.constraints["min_font_mm"]["table_value"])),
                                    float(frame.side["es_name"]["min_fontsize"]) * cell_scale)}
    _grow_table_text(atlas, cells, qty_cfg, name_cfg)
    return FacePlan(
        kind="side", family=family, panel=panel, content=content,
        scale=scale, inset_mm=mm(inset), safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics, barcode=barcode,
        fields={
            "item_no": {
                "bar_rect": list(bar.as_tuple()), "x0": bar.x0,
                "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
                "max_width": bar.width, "fontsize": fontsize,
                "font": frame.side["item_no"]["font"],
                "color": list(frame.side["item_no"]["color"]), "fit": "center_in_bar",
            },
            "qty": qty_cfg,
            "es_name": name_cfg,
            "barcode": barcode_cfg,
        },
        notes=[f"R1 2026-09-16: owner structure {family}; table scale {cell_scale:.4f}, cells slice_x"],
    )


def build_side_rows(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """Tall side: bar on top, five rows as wide as the bar, barcode lying at the foot.

    Owner anchor 1.00 side (120 x 250 mm, 2026-09-15), re-proportioned by the owner on
    2026-09-29 after the factory photo 517393 (bar 0.18, table to 0.76, barcode band 0.20).
    """
    return _build_side_from_regions(atlas, panel, item, "side_rows")


def build_side_bar_left(atlas: Atlas, panel: Box, item: dict[str, Any]) -> FacePlan:
    """Owner 2026-09-29 on the 3.21 side (305 x 95 mm): bar on the left half; on the right the two-column
    table over a lying barcode (replaces the 2026-09-15 anchor's one column + standing barcode)."""
    return _build_side_from_regions(atlas, panel, item, "side_bar_left")


def _bracket_inner_box_side(
    frame, brackets: ComponentSpec, placed: Box, *, rotated: bool = False
) -> Box:
    """Where the bars sit inside the side face's corner-bracket tile.

    ``rotated`` mirrors the front helper: the side symbol stands upright in
    production, so its tile is clipped from base.pdf in that orientation, and an
    arrangement that lays the symbol across the face has to swap the fractions
    or the bars land in a box of the wrong proportion. Default False leaves
    every production call byte-for-byte unchanged.
    """
    bars = frame.side["barcode"]["bars_rect"]
    bx0, by0, bx1, by1 = (float(v) for v in bars)
    cx0, cy0, cx1, cy1 = brackets.clip
    fw = max(1e-6, cx1 - cx0)
    fh = max(1e-6, cy1 - cy0)
    fx0, fy0 = (bx0 - cx0) / fw, (by0 - cy0) / fh
    fx1, fy1 = (bx1 - cx0) / fw, (by1 - cy0) / fh
    if rotated:
        fx0, fy0, fx1, fy1 = _turn_fractions_clockwise(fx0, fy0, fx1, fy1)
    return placed.fraction(fx0, fy0, fx1, fy1)


def _cell_value_box(frame, cell_name: str, field_name: str, placed: Box, atlas: Atlas) -> Box:
    """Project a value's position inside a re-placed table cell tile."""
    spec = atlas.component("side", cell_name)
    cx0, cy0, cx1, cy1 = spec.clip
    fw = max(1e-6, cx1 - cx0)
    fh = max(1e-6, cy1 - cy0)
    rect = frame.side[field_name]["rect"]
    vx0, vy0, vx1, vy1 = (float(v) for v in rect)
    return placed.fraction(
        max(0.0, (vx0 - cx0) / fw), max(0.0, (vy0 - cy0) / fh),
        min(1.0, (vx1 - cx0) / fw), min(1.0, (vy1 - cy0) / fh),
    )


# --------------------------------------------------------------------------- #
# Candidate enumeration
# --------------------------------------------------------------------------- #

FRONT_BUILDERS = {
    "front_standard": build_front_standard,
    "front_three_column": build_front_three_column,
    "front_wide_band": build_front_wide_band,
    "column": build_front_column,
}
SIDE_BUILDERS = {
    "side_standard": build_side_standard,
    "side_rows": build_side_rows,
    "side_owner_grid": build_side_owner_grid,
    "side_compressed": build_side_compressed,
    "side_bar_left": build_side_bar_left,
}

#: Families chosen by aspect band, in band order. Every owner-confirmed anchor
#: (2026-09-15) sits inside the band of the structure the owner confirmed for it,
#: and the bands in adaptive_layout.json tile the whole aspect range, so exactly
#: one family applies to any face. side_standard is not listed: side_owner_grid
#: is the owner's widened version of the same table and covers its band.
FRONT_BAND_FAMILIES = ("front_standard", "front_three_column", "front_wide_band")
SIDE_BAND_FAMILIES = ("side_rows", "side_owner_grid", "side_compressed", "side_bar_left")

def candidate_families(
    atlas: Atlas, kind: FaceKind, aspect: float, *,
    front_aspect: float | None = None, side_aspect: float | None = None,
) -> list[str]:
    """The family the owner's confirmed structures assign to this face.

    Until R1 (2026-09-16) neighbouring families competed on a score near a 2.5
    threshold. The owner has since confirmed a structure at each anchor, and the
    score was never calibrated (its area term peaks at 45% coverage while the
    owner asks for 满版), so the band alone decides.

    A face the assigned family cannot carry fails closed as
    ``manual_layout_required`` rather than silently swapping in a neighbour.
    """
    if kind == "front" and wants_owner_column(front_aspect, side_aspect):
        return ["column"]
    for name in FRONT_BAND_FAMILIES if kind == "front" else SIDE_BAND_FAMILIES:
        rule = atlas.families.get(name, {}).get("applies_when", {})
        if float(rule.get("aspect_min", 0.0)) <= aspect < float(rule.get("aspect_max", 1e9)):
            return [name]
    raise AdaptiveAssetError(f"No {kind} family band in adaptive_layout.json covers aspect {aspect:.4f}.")


def _grow_lineart(atlas: Atlas, plan: FacePlan) -> FacePlan:
    """Keep the line art in proportion and give it the free space around its slot.

    The owner's R3 (2026-09-15, "线稿只能等比例放大，改成自由放大") was built as a
    non-uniform stretch, which drew the 3210808 bottle 2.3-5.3x too wide; on
    2026-09-23 the owner confirmed R3 meant *keep the proportion*. A drawing of a
    real product must never be distorted, so the slot is grown instead: up and
    down first, then left and right, until it meets another component (keeping
    ``min_component_gap_mm``) or the content edge, and the drawing is fitted into
    it at its own aspect and centred. The family's own slot is kept as
    ``slot_rect`` so the growth stays visible in the plan.

    The gap scales with the plan (``min_component_gap_mm`` at the template's own
    size) so the mark stays scale-invariant, and in the owner's column structure
    the slot grows only inside the owner's art column (``FRONT_COLUMN_REGIONS``).
    """
    cfg = plan.fields.get("lineart")
    if not isinstance(cfg, dict) or "rect" not in cfg:
        return plan
    gap = pt(float(atlas.constraints.get("min_component_gap_mm", 1.5))) * plan.scale
    slot = Box(*cfg["rect"])
    others = [box for name, box in plan.occupied_boxes() if name != "lineart"]
    bound = plan.content
    if plan.family == "column":
        spec = FRONT_COLUMN_REGIONS["lineart"]
        bound = plan.content.region(spec["x"], spec["y"])
    eps = 1e-6
    x0, y0, x1, y1 = slot.x0, slot.y0, slot.x1, slot.y1

    in_column = [o for o in others if o.x0 < x1 - eps and o.x1 > x0 + eps]
    y0 = min(y0, max([bound.y0] + [o.y1 + gap for o in in_column if o.y1 <= slot.y0 + eps]))
    y1 = max(y1, min([bound.y1] + [o.y0 - gap for o in in_column if o.y0 >= slot.y1 - eps]))
    in_row = [o for o in others if o.y0 < y1 - eps and o.y1 > y0 + eps]
    x0 = min(x0, max([bound.x0] + [o.x1 + gap for o in in_row if o.x1 <= slot.x0 + eps]))
    x1 = max(x1, min([bound.x1] + [o.x0 - gap for o in in_row if o.x0 >= slot.x1 - eps]))

    cfg["slot_rect"] = list(slot.as_tuple())
    cfg["rect"] = [x0, y0, x1, y1]
    cfg["fit"] = "contain_center"
    return plan


def build_candidates(
    atlas: Atlas, kind: FaceKind, panel: Box, item: dict[str, Any], *,
    front_aspect: float | None = None, side_aspect: float | None = None,
) -> list[FacePlan]:
    builders = FRONT_BUILDERS if kind == "front" else SIDE_BUILDERS
    plans: list[FacePlan] = []
    for name in candidate_families(atlas, kind, panel.aspect, front_aspect=front_aspect, side_aspect=side_aspect):
        try:
            plans.append(_grow_lineart(atlas, builders[name](atlas, panel, item)))
        except Exception as exc:  # a family that cannot express this face is not fatal
            plans.append(
                FacePlan(
                    kind=kind,
                    family=name,
                    panel=panel,
                    content=panel,
                    scale=0.0,
                    inset_mm=0.0,
                    notes=[f"family failed to build: {exc}"],
                )
            )
    return plans


__all__ = [
    "FRONT_BUILDERS",
    "SIDE_BUILDERS",
    "build_candidates",
    "build_front_standard",
    "build_front_column",
    "build_front_three_column",
    "build_front_wide_band",
    "build_side_bar_left",
    "build_side_rows",
    "build_side_compressed",
    "build_side_standard",
    "build_side_owner_grid",
    "candidate_families",
]
