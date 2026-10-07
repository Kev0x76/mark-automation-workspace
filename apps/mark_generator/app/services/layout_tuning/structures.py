"""Whole-face arrangements to put in front of the owner, as region maps.

Round 1 asked the owner to compare marks that differed by one parameter, and on
the line-art ladder the two images differed by 0.34% of their pixels. When the
answer to "which is better" is "I cannot see a difference", the honest reading
is that the question was not worth asking. The fix is not more questions - it is
questions about things that actually differ.

So the candidates here differ by **arrangement**, not by a multiplier. Each
structure says where every element sits as a fraction of the content box, which
is how ``adaptive_layout.json`` already describes ``front_wide_band``, and is
therefore directly transcribable into policy if the owner confirms one.

Provenance of each structure - none of them is invented for variety's sake:

``standard``
    The production ``front_standard`` builder: the template projected onto the
    face. Factory evidence at aspect 0.93-2.36.
``wide_band``
    The production ``front_wide_band`` builder. Factory evidence at 2.62-3.68.
``wide_band_full_bar``
    ``wide_band`` with the item bar filling its region instead of hugging the
    digits. Exists because the bar measures 26-34 points of face width short of
    the factory curve on wide faces, and ``_hugging_bar``'s rule is the cause
    (docs/owner-preference-study-20260907.md section 5).
``column``
    One row across the whole face, every element sized off face height, the bar
    taking the width slack. Measured from the 竹海 1250x170 mm factory sample,
    where the factory did *not* use the two-band arrangement
    (docs/wide-face-column-layout-evidence-20260824.md).
``bar_left``
    Item bar as a tall left block. From the 78x28x20 experiment's B family,
    which its scorer ranked first (docs/adaptive-layout-78x28x20-experiment-20260909.md).
``bar_top_full``
    Full-width bar across the top, everything else in one reflowed row beneath.
    The hybrid between the two families that no factory sample covers, included
    because the 2.36-2.62 evidence hole is exactly where such a face would land.

Nothing here changes production behaviour: the module builds ``FacePlan``
objects with the production helpers and the production atlas, and is only
imported by the study tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..adaptive_layout.atlas import Atlas
from ..adaptive_layout.families import (
    _MIN_GAP_MM,
    SIDE_CELLS,
    _band_names_cfg,
    _barcode_placement,
    _bracket_for_bars,
    _bracket_inner_box,
    _bracket_inner_box_side,
    _cell_value_box,
    _inset_pt,
    _place_uniform,
    _safe_margin_mm,
    _scale_barcode_cfg,
    _static,
    build_front_standard,
    build_front_wide_band,
    build_side_compressed,
    build_side_standard,
)
from ..adaptive_layout.model import Box, FacePlan, StaticPlacement, mm, pt


@dataclass(frozen=True)
class Structure:
    """One arrangement, and the aspect range it is offered on.

    ``builder`` is set for the two production families, which are used as they
    ship rather than re-implemented. Everything else is data: ``bar`` plus
    ``regions`` fully determine the face.
    """

    key: str
    label: str
    provenance: str
    aspect_min: float = 0.0
    aspect_max: float = 1e9
    builder: Callable[..., FacePlan] | None = None
    bar: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, Any] = field(default_factory=dict)


#: The three Latin product names always travel together; Chinese is separate
#: because every measured factory arrangement gives it its own slot.
_LATIN = ["it_name", "es_name", "en_name"]


STRUCTURES: list[Structure] = [
    Structure(
        key="standard",
        label="标准式：货号满宽横条在上，下方左品名、中线稿、右竖条码",
        provenance="production front_standard; factory evidence aspect 0.93-2.36",
        aspect_max=3.2,
        builder=build_front_standard,
    ),
    Structure(
        key="wide_band",
        label="上栏式：货号条占左上，线稿占右上，下方一整行",
        provenance="production front_wide_band; factory evidence aspect 2.62-3.68",
        aspect_min=1.6,
        builder=build_front_wide_band,
    ),
    Structure(
        key="wide_band_full_bar",
        label="上栏式·长条：同上，但货号黑条更长，占到面宽的八成半",
        # Until 2026-09-10 this candidate existed to ask whether the bar should
        # fill its region at all, because production hugged the digits and came
        # out 26-34 points under the factory. That is now fixed in the engine on
        # measured evidence, so `wide_band` already fills its region and this
        # would otherwise be a duplicate.
        #
        # What is genuinely still open is the residual: the region is one fixed
        # fraction (0.74) while the factory curve descends 78.2 -> 67.9 -> 63.8
        # across the band, so the engine lands -6.3 / +4.1 / +9.0 against it.
        # Three factory samples cannot settle whether that descent is real, and
        # fitting a curve through three points would be inventing a mechanism
        # nobody measured. So this candidate asks the owner's eye instead: a
        # markedly longer bar, held at one fraction, against the measured one.
        provenance=(
            "bar region widened to 0.86 against production's measured 0.74; asks whether a "
            "wide face wants the bar as dominant as a normal face gives it (~93%) or pulled "
            "back to the fraction measured off the factory wide-band pages"
        ),
        aspect_min=1.6,
        bar={"x": [0.0, 0.86], "y": [0.0, 0.5], "mode": "fill"},
        # max_lines is pinned to 1 on both name blocks because this structure is
        # a MATCHED PAIR with wide_band: it exists to ask one question, whether
        # the bar should be long. The automatic line budget below fired on this
        # slot but not on the production family that renders the same slot, so
        # the "long bar" option also arrived with half-size Latin names (40.4 pt
        # against 20.2 pt, measured on every anchor). The owner would have been
        # choosing between two things and the answer would have been filed under
        # one - which is exactly the defect that made three of round 1's five
        # dials unusable.
        regions={
            "lineart": {"x": [0.87, 1.0], "y": [0.0, 0.55], "anchor": "right"},
            "kmt_logo": {"x": [0.0, 0.24], "y": [0.56, 0.72]},
            "names": {"x": [0.0, 0.26], "y": [0.72, 1.0], "fields": _LATIN, "max_lines": 1},
            "barcode": {"x": [0.30, 0.55], "y": [0.56, 1.0], "rotation": 0},
            "cn_name": {"x": [0.58, 1.0], "y": [0.56, 0.70], "fields": ["cn_name"], "max_lines": 1},
            "origin_text": {"x": [0.58, 1.0], "y": [0.70, 0.82]},
            "icon_row": {"x": [0.58, 1.0], "y": [0.83, 1.0]},
        },
    ),
    Structure(
        key="column",
        label="纯分栏：全部元素排成横向一行，货号条吃掉多余宽度",
        provenance="measured from the 竹海 1250x170 mm factory sample (aspect 7.35)",
        aspect_min=2.2,
        bar={"x": [0.0, 0.46], "y": [0.06, 0.94], "mode": "fill"},
        regions={
            "kmt_logo": {"x": [0.48, 0.62], "y": [0.06, 0.28]},
            "cn_name": {"x": [0.48, 0.80], "y": [0.30, 0.50], "fields": ["cn_name"]},
            "names": {"x": [0.48, 0.80], "y": [0.52, 0.84], "fields": _LATIN},
            "origin_text": {"x": [0.48, 0.62], "y": [0.86, 0.98]},
            "lineart": {"x": [0.81, 0.89], "y": [0.06, 0.86], "anchor": "center"},
            "icon_row": {"x": [0.81, 0.89], "y": [0.87, 0.98]},
            "barcode": {"x": [0.90, 1.0], "y": [0.06, 0.94], "rotation": 90},
        },
    ),
    Structure(
        key="bar_left",
        label="左栏式：货号条竖占左半，右侧依次品牌品名、线稿、条码",
        provenance="B family of the 78x28x20 experiment, ranked first by its scorer",
        aspect_min=1.9,
        bar={"x": [0.0, 0.38], "y": [0.02, 0.98], "mode": "fill"},
        regions={
            "kmt_logo": {"x": [0.40, 0.54], "y": [0.02, 0.22]},
            "cn_name": {"x": [0.40, 0.72], "y": [0.24, 0.46], "fields": ["cn_name"]},
            "names": {"x": [0.40, 0.72], "y": [0.48, 0.80], "fields": _LATIN},
            "origin_text": {"x": [0.40, 0.54], "y": [0.82, 0.98]},
            "lineart": {"x": [0.73, 0.85], "y": [0.02, 0.78], "anchor": "center"},
            "icon_row": {"x": [0.73, 0.85], "y": [0.80, 0.98]},
            "barcode": {"x": [0.86, 1.0], "y": [0.02, 0.98], "rotation": 90},
        },
    ),
    Structure(
        key="stack",
        label="竖排堆叠式：货号条、线稿、品名、条码自上而下一层层排",
        provenance="the arrangement a tall face allows and a wide one does not; no factory sample exists below aspect 0.93",
        aspect_max=1.7,
        bar={"x": [0.0, 1.0], "y": [0.0, 0.15], "mode": "fill"},
        regions={
            "lineart": {"x": [0.18, 0.82], "y": [0.18, 0.55], "anchor": "center"},
            "kmt_logo": {"x": [0.0, 0.30], "y": [0.57, 0.66]},
            "cn_name": {"x": [0.0, 0.72], "y": [0.67, 0.75], "fields": ["cn_name"]},
            "names": {"x": [0.0, 0.72], "y": [0.75, 0.90], "fields": _LATIN},
            "origin_text": {"x": [0.0, 0.40], "y": [0.90, 0.97]},
            "icon_row": {"x": [0.42, 0.74], "y": [0.90, 1.0]},
            "barcode": {"x": [0.76, 1.0], "y": [0.57, 1.0], "rotation": 90},
        },
    ),
    Structure(
        key="lineart_hero",
        label="线稿为主式：线稿占据上半整幅，文字与条码压在下方",
        provenance="the opposite trade to stack - artwork dominant, text minimal; no factory sample takes this position",
        aspect_max=2.4,
        bar={"x": [0.0, 1.0], "y": [0.0, 0.13], "mode": "fill"},
        regions={
            "lineart": {"x": [0.0, 1.0], "y": [0.16, 0.62], "anchor": "center"},
            "kmt_logo": {"x": [0.0, 0.26], "y": [0.64, 0.74]},
            "cn_name": {"x": [0.0, 0.62], "y": [0.75, 0.84], "fields": ["cn_name"]},
            "names": {"x": [0.0, 0.62], "y": [0.84, 0.98], "fields": _LATIN},
            "origin_text": {"x": [0.28, 0.62], "y": [0.64, 0.74]},
            "icon_row": {"x": [0.64, 0.86], "y": [0.86, 0.99]},
            "barcode": {"x": [0.66, 1.0], "y": [0.64, 0.85], "rotation": 0},
        },
    ),
    Structure(
        key="two_column",
        label="双栏式：货号条在上，下方左栏文字、右栏线稿与条码",
        provenance="splits the space below the bar into two columns instead of three slots; the arrangement the template projection never produces",
        aspect_max=2.8,
        bar={"x": [0.0, 1.0], "y": [0.0, 0.16], "mode": "fill"},
        regions={
            "kmt_logo": {"x": [0.0, 0.24], "y": [0.20, 0.32]},
            "cn_name": {"x": [0.0, 0.52], "y": [0.34, 0.46], "fields": ["cn_name"]},
            "names": {"x": [0.0, 0.52], "y": [0.46, 0.74], "fields": _LATIN},
            "origin_text": {"x": [0.0, 0.34], "y": [0.76, 0.86]},
            "icon_row": {"x": [0.0, 0.36], "y": [0.87, 1.0]},
            "lineart": {"x": [0.55, 1.0], "y": [0.20, 0.66], "anchor": "center"},
            "barcode": {"x": [0.62, 1.0], "y": [0.70, 1.0], "rotation": 0},
        },
    ),
    Structure(
        key="stack_centered",
        label="居中堆叠式：同样自上而下，但主体居中而不是靠左（产地与图标行左右对称分列）",
        provenance="left alignment is a choice every measured factory face makes; centring is the alternative it rules out, and nothing tests whether it was a rule or a habit",
        aspect_max=1.7,
        # Symmetric all the way down, which means the barcode lies across the
        # foot rather than standing at the right: an upright symbol pinned to
        # one edge is the single thing that would break the symmetry.
        bar={"x": [0.0, 1.0], "y": [0.0, 0.14], "mode": "fill"},
        regions={
            "lineart": {"x": [0.24, 0.76], "y": [0.16, 0.50], "anchor": "center"},
            "kmt_logo": {"x": [0.38, 0.62], "y": [0.52, 0.60], "align": "center"},
            "cn_name": {"x": [0.18, 0.82], "y": [0.61, 0.69], "fields": ["cn_name"]},
            "names": {"x": [0.18, 0.82], "y": [0.69, 0.84], "fields": _LATIN},
            "origin_text": {"x": [0.02, 0.30], "y": [0.86, 0.96], "align": "center"},
            "icon_row": {"x": [0.70, 0.98], "y": [0.86, 1.0], "align": "center"},
            "barcode": {"x": [0.32, 0.68], "y": [0.85, 1.0], "rotation": 0},
        },
    ),
    Structure(
        key="barcode_bottom",
        label="横码在底式：条码横躺在底部整条，上方左文字右线稿",
        provenance="every measured factory front stands the barcode upright at the right edge; laying it across the foot is the arrangement that choice excludes",
        aspect_max=2.6,
        bar={"x": [0.0, 1.0], "y": [0.0, 0.16], "mode": "fill"},
        regions={
            "kmt_logo": {"x": [0.0, 0.24], "y": [0.20, 0.32]},
            "cn_name": {"x": [0.0, 0.55], "y": [0.34, 0.46], "fields": ["cn_name"]},
            "names": {"x": [0.0, 0.55], "y": [0.46, 0.72], "fields": _LATIN},
            "origin_text": {"x": [0.0, 0.34], "y": [0.73, 0.82]},
            "lineart": {"x": [0.58, 1.0], "y": [0.20, 0.72], "anchor": "center"},
            "icon_row": {"x": [0.62, 1.0], "y": [0.84, 1.0]},
            "barcode": {"x": [0.0, 0.58], "y": [0.84, 1.0], "rotation": 0},
        },
    ),
    Structure(
        key="column_wide_art",
        label="纯分栏·短货号条：同为横向一行，但黑条只占三分之一，其余元素分到更多空间",
        provenance="the 竹海 sample gives the width slack to the bar (49.4% of face width); this answers that trade the other way. Widening the art SLOT alone changes nothing - artwork is fitted uniformly and is height-bound on a short face - so the art also gets the full content height, and the text column gets the rest",
        aspect_min=2.2,
        bar={"x": [0.0, 0.30], "y": [0.06, 0.94], "mode": "fill"},
        regions={
            "kmt_logo": {"x": [0.32, 0.46], "y": [0.02, 0.24]},
            "cn_name": {"x": [0.32, 0.62], "y": [0.26, 0.50], "fields": ["cn_name"]},
            "names": {"x": [0.32, 0.62], "y": [0.52, 0.86], "fields": _LATIN},
            "origin_text": {"x": [0.32, 0.46], "y": [0.88, 1.0]},
            "lineart": {"x": [0.63, 0.87], "y": [0.0, 0.84], "anchor": "center"},
            "icon_row": {"x": [0.63, 0.87], "y": [0.86, 1.0]},
            "barcode": {"x": [0.88, 1.0], "y": [0.0, 1.0], "rotation": 90},
        },
    ),
    Structure(
        key="bar_top_full",
        label="满宽上条式：货号条横跨整个面宽，其余全部收在下方一行",
        provenance="hybrid of the two families; no factory sample covers the 2.36-2.62 hole where it would apply",
        aspect_min=1.4,
        bar={"x": [0.0, 1.0], "y": [0.0, 0.40], "mode": "fill"},
        regions={
            "kmt_logo": {"x": [0.0, 0.14], "y": [0.44, 0.56]},
            "cn_name": {"x": [0.0, 0.46], "y": [0.57, 0.70], "fields": ["cn_name"]},
            "names": {"x": [0.0, 0.46], "y": [0.70, 0.89], "fields": _LATIN},
            "origin_text": {"x": [0.0, 0.20], "y": [0.90, 1.0]},
            "icon_row": {"x": [0.22, 0.48], "y": [0.90, 1.0]},
            "lineart": {"x": [0.50, 0.80], "y": [0.44, 1.0], "anchor": "center"},
            "barcode": {"x": [0.82, 1.0], "y": [0.44, 1.0], "rotation": 90},
        },
    ),
]

STRUCTURES_BY_KEY = {s.key: s for s in STRUCTURES}


#: Side arrangements. Fewer than the front, and deliberately so: the side face
#: carries a fixed table plus a bar plus a barcode, with no artwork and no
#: product names, so the only real structural choices are where the barcode
#: goes, where the bar goes, and how the five cells group.
SIDE_STRUCTURES: list[Structure] = [
    Structure(
        key="side_standard",
        label="标准式：货号满宽横条在上，下方两栏五格，右侧竖排条码",
        provenance="production side_standard; factory evidence up to side aspect 2.5",
        aspect_max=2.5,
        builder=build_side_standard,
    ),
    Structure(
        key="side_compressed",
        label="压缩式：货号条与五格共享左侧，条码占右边一竖条",
        provenance=(
            "production side_compressed; measured on the factory 305x95 mm side panel, "
            "where the five stacked rows regroup into three columns"
        ),
        aspect_min=2.5,
        builder=build_side_compressed,
    ),
    Structure(
        key="side_bar_left",
        label="左栏式：货号条竖占左侧，五格在中间，条码在最右",
        provenance="the side counterpart of the front bar_left arrangement",
        aspect_min=1.9,
        bar={"x": [0.0, 0.30], "y": [0.02, 0.98], "mode": "fill"},
        regions={
            "cell_qty": {"x": [0.33, 0.52], "y": [0.04, 0.47]},
            "cell_meas": {"x": [0.33, 0.52], "y": [0.53, 0.96]},
            "cell_name": {"x": [0.55, 0.74], "y": [0.04, 0.47]},
            "cell_nw": {"x": [0.55, 0.74], "y": [0.53, 0.96]},
            "cell_gw": {"x": [0.77, 0.88], "y": [0.04, 0.47]},
            "barcode": {"x": [0.90, 1.0], "y": [0.02, 0.98], "rotation": 90},
        },
    ),
    Structure(
        key="side_barcode_bottom",
        label="横码在底式：条码横躺在底部，货号条在上，五格在中间",
        provenance="the side counterpart of the front barcode_bottom arrangement",
        aspect_max=2.6,
        bar={"x": [0.0, 1.0], "y": [0.0, 0.22], "mode": "fill"},
        regions={
            "cell_qty": {"x": [0.0, 0.31], "y": [0.28, 0.60]},
            "cell_meas": {"x": [0.0, 0.31], "y": [0.63, 0.95]},
            "cell_name": {"x": [0.34, 0.65], "y": [0.28, 0.60]},
            "cell_nw": {"x": [0.34, 0.65], "y": [0.63, 0.95]},
            "cell_gw": {"x": [0.68, 0.99], "y": [0.28, 0.60]},
            "barcode": {"x": [0.68, 1.0], "y": [0.64, 0.98], "rotation": 0},
        },
    ),
    Structure(
        key="side_column",
        label="纯分栏：货号条、五格、条码横向排成一行",
        provenance=(
            "the side counterpart of the front column arrangement, which was measured "
            "off the 竹海 1250x170 mm factory sample"
        ),
        aspect_min=2.2,
        bar={"x": [0.0, 0.40], "y": [0.06, 0.94], "mode": "fill"},
        regions={
            "cell_qty": {"x": [0.42, 0.57], "y": [0.06, 0.49]},
            "cell_meas": {"x": [0.42, 0.57], "y": [0.51, 0.94]},
            "cell_name": {"x": [0.59, 0.74], "y": [0.06, 0.49]},
            "cell_nw": {"x": [0.59, 0.74], "y": [0.51, 0.94]},
            "cell_gw": {"x": [0.76, 0.87], "y": [0.06, 0.49]},
            "barcode": {"x": [0.89, 1.0], "y": [0.06, 0.94], "rotation": 90},
        },
    ),
]

SIDE_STRUCTURES_BY_KEY = {s.key: s for s in SIDE_STRUCTURES}


#: Which side arrangement belongs with which front arrangement.
#:
#: Pairing by shared structural idea rather than by cross-product. The owner is
#: being asked to confirm one MARK, so the two faces have to read as one design
#: - a front that lays the barcode across its foot next to a side that stands it
#: upright is not a design anybody chose. A cross-product would also put two
#: independent changes on one screen and make the answer unattributable, which
#: is precisely what made round 1 unusable.
#:
#: The preference is a preference, not a guarantee: the side face has its own
#: shape and its own aspect range, and anchor 20 is front 7.35 with side 1.00.
#: ``side_for`` falls back to the production family the side's own shape calls
#: for whenever the preferred arrangement does not apply to it.
SIDE_FOR_FRONT: dict[str, str] = {
    "standard": "side_standard",
    "two_column": "side_standard",
    "stack": "side_standard",
    "stack_centered": "side_standard",
    "lineart_hero": "side_standard",
    "bar_top_full": "side_standard",
    "wide_band": "side_compressed",
    "wide_band_full_bar": "side_compressed",
    "column": "side_column",
    "column_wide_art": "side_column",
    "bar_left": "side_bar_left",
    "barcode_bottom": "side_barcode_bottom",
}


def side_for(front_key: str, side_aspect: float) -> Structure:
    """The side arrangement to pair with ``front_key`` on a face of this shape.

    Preferred counterpart if the side's own aspect allows it; otherwise the
    production family for that shape, which is always defined because the two
    production families partition the range at 2.5.
    """
    preferred = SIDE_FOR_FRONT.get(front_key)
    if preferred is not None:
        candidate = SIDE_STRUCTURES_BY_KEY[preferred]
        if candidate.aspect_min <= side_aspect <= candidate.aspect_max:
            return candidate
    return SIDE_STRUCTURES_BY_KEY[
        "side_compressed" if side_aspect >= 2.5 else "side_standard"
    ]


def build_side_structure(
    structure: Structure, atlas: Atlas, panel: Box, item: dict[str, Any]
) -> FacePlan:
    """One side face laid out by ``structure``.

    Production families are delegated to their own builders, so for those two
    the owner sees byte-for-byte what the engine ships today. The rest are laid
    out from the region map, using the same table tiles lifted from base.pdf.
    """
    if structure.builder is not None:
        plan = structure.builder(atlas, panel, item)
        plan.notes = list(plan.notes) + [f"structure={structure.key}"]
        return plan

    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)
    regions = structure.regions

    item_no = str(item.get("item_no") or "")
    bar_region = content.region(structure.bar["x"], structure.bar["y"])
    barcode_region = content.region(regions["barcode"]["x"], regions["barcode"]["y"])

    # Nothing may run into the barcode column - the same rule the production
    # side builder needs, and for the same reason: a black field butted against
    # a symbol's quiet zone is not something to print.
    gap = pt(_MIN_GAP_MM)
    if bar_region.x1 > barcode_region.x0 - gap and bar_region.y0 < barcode_region.y1:
        bar_region = Box(
            bar_region.x0, bar_region.y0,
            max(bar_region.x0 + 1.0, barcode_region.x0 - gap), bar_region.y1,
        )
    bar, fontsize = _bar_box(atlas, bar_region, item_no, structure.bar.get("mode", "fill"))
    statics = [_static(atlas.component("side", "item_no_bar"), bar)]

    cell_boxes: dict[str, Box] = {}
    for name in SIDE_CELLS:
        if name not in regions:
            continue
        spec = atlas.component("side", name)
        placed = content.region(regions[name]["x"], regions[name]["y"]).fit_uniform(
            spec.aspect, align_x="left", align_y="middle"
        )
        cell_boxes[name] = placed
        statics.append(_static(spec, placed))

    # The side's bracket tile is clipped from base.pdf standing upright, because
    # that is what production prints. An arrangement that lays the symbol across
    # the face has to rotate the tile as well as the bars - fitting the upright
    # tile into a wide short slot makes it height-bound, and the symbol came out
    # at X = 0.218 mm against a GS1 floor of 0.264, failing on 17 of 20 anchors.
    brackets = atlas.component("side", "barcode_brackets")
    rotation = int(regions["barcode"].get("rotation", 90))
    rotated = rotation == 0
    if rotated:
        bracket_box = barcode_region.fit_uniform(
            1.0 / brackets.aspect, align_x="right", align_y="middle"
        )
        bracket_fit = "rotate90_uniform"
    else:
        bracket_box = barcode_region.fit_uniform(
            brackets.aspect, align_x="right", align_y="middle"
        )
        bracket_fit = brackets.fit
    inner = _bracket_inner_box_side(frame, brackets, bracket_box, rotated=rotated)
    barcode = _barcode_placement(inner, rotation, atlas)
    statics.append(
        StaticPlacement(
            name="barcode_brackets",
            clip=brackets.clip,
            target=_bracket_for_bars(frame, brackets, barcode.bars, side=True, rotated=rotated),
            fit=bracket_fit,
            source_page=brackets.source_page,
        )
    )

    qty_cell = cell_boxes.get("cell_qty", content)
    name_cell = cell_boxes.get("cell_name", content)
    qty_value = _cell_value_box(frame, "cell_qty", "qty", qty_cell, atlas)
    es_value = _cell_value_box(frame, "cell_name", "es_name", name_cell, atlas)

    barcode_cfg = _scale_barcode_cfg(frame.side["barcode"], barcode.bars, scale)
    if int(regions["barcode"].get("rotation", 90)) == 0:
        barcode_cfg["orientation"] = "horizontal"
        barcode_cfg["hri_layout"] = "upright_single_span"

    fields = {
        "item_no": {
            "bar_rect": list(bar.as_tuple()),
            "x0": bar.x0,
            "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
            "max_width": bar.width,
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
            "min_fontsize": max(4.0, es_value.height * 0.28),
            "max_lines": 2,
        },
        "barcode": barcode_cfg,
    }

    return FacePlan(
        kind="side",
        family=structure.key,
        panel=panel,
        content=content,
        scale=scale,
        inset_mm=mm(inset),
        safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics,
        fields=fields,
        barcode=barcode,
        notes=[f"structure={structure.key}"],
    )


def _bar_box(
    atlas: Atlas, region: Box, item_no: str, mode: str
) -> tuple[Box, float]:
    """The item-number bar inside ``region``.

    ``hug`` reproduces the production rule, where the bar ends where the digits
    end. ``fill`` keeps the whole region and lets the digits centre inside it,
    which is what the factory does on every measured face: the bar is a field of
    black whose length comes from the layout, not from the number.
    """
    frame = atlas.frame
    digits = max(1, len(item_no))
    hscale_floor = frame.item_no_hscale_by_digits.get(
        digits, min(frame.item_no_hscale_by_digits.values(), default=1.0)
    )
    bar_height = region.height
    fontsize = bar_height * frame.item_no_fontsize_ratio
    text_width = digits * frame.digit_advance_em * fontsize * hscale_floor
    padding = max(2.0, min(region.width * 0.02, 8.0))
    needed = text_width + 2 * padding

    if mode == "fill":
        # The digits may be condensed to fit, exactly as the renderer already
        # condenses them on the standard family; the bar itself does not move.
        if needed > region.width:
            fontsize *= region.width / needed
        return Box(region.x0, region.y0, region.x1, region.y1), fontsize

    if needed > region.width:
        shrink = region.width / needed
        fontsize *= shrink
        bar_height *= shrink
        needed = region.width
    width = min(region.width, max(needed, region.width * 0.35))
    return Box(region.x0, region.y0, region.x0 + width, region.y0 + bar_height), fontsize


def _names_cfg(
    atlas: Atlas, region: Box, scale: float, fields: list[str], spec: dict[str, Any]
) -> dict[str, Any]:
    """A names block sized for the shape of the slot it was given.

    ``_band_names_cfg`` pins every name to a single line, which is the right
    rule for the production wide-band family where the slot is a long ribbon.
    Drop that same rule into a narrow column and the renderer shrinks the text
    until one line fits the width - the Chinese name fell to roughly a third of
    its proper size in the column and two-column arrangements, which would have
    handed the owner a comparison decided by a slot shape rather than by the
    arrangement being asked about.

    So the line budget follows the slot: a ribbon keeps one line, a narrow box
    gets two and each line takes half the height.
    """
    cfg = _band_names_cfg(atlas.frame.front["names"], region, scale, fields, atlas)
    allowed = int(spec.get("max_lines", 2 if region.width < region.height * 8.0 else 1))
    if allowed > 1:
        cfg["max_lines_per_name"] = allowed
        for line in cfg["lines"]:
            line["max_lines"] = allowed
            line["fontsize"] = float(line["fontsize"]) / allowed
    return cfg


def _place_barcode(
    atlas: Atlas, region: Box, rotation: int, scale: float
) -> tuple[list[StaticPlacement], Any, dict[str, Any]]:
    """Brackets plus bars in ``region``.

    ``rotation`` is the printed orientation of the symbol: 90 means the bars run
    down the face (95 modules along the region height), 0 means they run across
    it. The bracket tile is always re-derived from the placed bars, so a symbol
    capped at the GS1 ceiling keeps brackets that fit it.
    """
    frame = atlas.frame
    brackets = atlas.component("front", "barcode_brackets")
    rotated = rotation == 0
    if rotated:
        tile = region.fit_uniform(1.0 / brackets.aspect, align_x="center", align_y="middle")
        fit = "rotate90_uniform"
    else:
        tile = region.fit_uniform(brackets.aspect, align_x="center", align_y="middle")
        fit = brackets.fit
    inner = _bracket_inner_box(frame, brackets, tile, rotated=rotated)
    placement = _barcode_placement(inner, 0 if rotated else 90, atlas)
    tile = _bracket_for_bars(frame, brackets, placement.bars, side=False, rotated=rotated)
    static = StaticPlacement(
        name="barcode_brackets",
        clip=brackets.clip,
        target=tile,
        fit=fit,
        source_page=brackets.source_page,
    )
    cfg = _scale_barcode_cfg(frame.front["barcode"], placement.bars, scale)
    if rotated:
        cfg["orientation"] = "horizontal"
        cfg["hri_layout"] = "upright_single_span"
    return [static], placement, cfg


def build_structure(
    structure: Structure, atlas: Atlas, panel: Box, item: dict[str, Any]
) -> FacePlan:
    """One face laid out by ``structure``.

    Production families are delegated to their own builders so what the owner
    sees for those two is byte-for-byte what the engine ships today.
    """
    if structure.builder is not None:
        plan = structure.builder(atlas, panel, item)
        plan.notes = list(plan.notes) + [f"structure={structure.key}"]
        return plan

    frame = atlas.frame
    inset = _inset_pt(atlas, panel)
    content = panel.inset_by(inset)
    scale = frame.scale_for(content)
    regions = structure.regions

    def region_of(name: str) -> Box:
        spec = regions[name]
        return content.region(spec["x"], spec["y"])

    item_no = str(item.get("item_no") or "")
    bar_region = content.region(structure.bar["x"], structure.bar["y"])
    bar, fontsize = _bar_box(atlas, bar_region, item_no, structure.bar.get("mode", "hug"))
    statics = [_static(atlas.component("front", "item_no_bar"), bar)]

    for name, align_y in (("kmt_logo", "middle"), ("origin_text", "middle"), ("icon_row", "bottom")):
        if name in regions:
            statics.append(
                _place_uniform(
                    atlas.component("front", name),
                    region_of(name),
                    align_x=regions[name].get("align", "left"),
                    align_y=align_y,
                )
            )

    barcode_spec = regions["barcode"]
    bar_statics, barcode, barcode_cfg = _place_barcode(
        atlas, region_of("barcode"), int(barcode_spec.get("rotation", 90)), scale
    )
    statics.extend(bar_statics)

    # Line art keeps its own proportions inside its slot; the slot is not
    # allowed to run into the bar, which is what would happen on a face where
    # the bar needed more room than its region nominally allows.
    art_region = region_of("lineart")
    gap = pt(_MIN_GAP_MM)
    left_limit = max(bar.x1 + gap, content.x0) if bar.y1 > art_region.y0 else content.x0
    art_x0 = max(art_region.x0, left_limit)
    if art_x0 >= art_region.x1 - 1.0:
        raise ValueError("line art slot collapsed against the item-number bar")
    art_region = Box(art_x0, art_region.y0, art_region.x1, art_region.y1)

    fields: dict[str, Any] = {
        "item_no": {
            "bar_rect": list(bar.as_tuple()),
            "x0": bar.x0,
            "baseline_y": bar.y0 + bar.height * 0.5 + 0.36 * fontsize,
            "max_width": bar.width,
            "fontsize": fontsize,
            "font": frame.front["item_no"]["font"],
            "color": list(frame.front["item_no"]["color"]),
            "fit": "center_in_bar",
        },
        "names": _names_cfg(
            atlas, region_of("names"), scale, regions["names"]["fields"], regions["names"]
        ),
        "cn_name": _names_cfg(
            atlas, region_of("cn_name"), scale, regions["cn_name"]["fields"], regions["cn_name"]
        ),
        "lineart": {"rect": list(art_region.as_tuple()), "fit": "contain_center"},
        "barcode": barcode_cfg,
    }

    return FacePlan(
        kind="front",
        family=structure.key,
        panel=panel,
        content=content,
        scale=scale,
        inset_mm=mm(inset),
        safe_margin_mm=_safe_margin_mm(atlas),
        statics=statics,
        fields=fields,
        barcode=barcode,
        notes=[f"structure={structure.key}", structure.provenance],
    )


def applicable(aspect: float) -> list[Structure]:
    """Structures worth trying at ``aspect``.

    The range on each structure is a hint, not a guarantee - a structure inside
    its range can still fail to build or fail validation on a particular face,
    and that is the honest answer for that face rather than an error.
    """
    return [s for s in STRUCTURES if s.aspect_min <= aspect <= s.aspect_max]


__all__ = [
    "SIDE_FOR_FRONT",
    "SIDE_STRUCTURES",
    "SIDE_STRUCTURES_BY_KEY",
    "STRUCTURES",
    "STRUCTURES_BY_KEY",
    "Structure",
    "applicable",
    "build_side_structure",
    "build_structure",
    "side_for",
]
