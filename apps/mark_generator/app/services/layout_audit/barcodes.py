"""Barcodes, read back from the final PDF the way a scanner meets them.

Four questions, each answered from the artwork and never from the plan:

``scannable``
    zxing-cpp must decode the EAN-13 from an 8-bit raster of the finished face -
    everything that prints around the symbol included - at 300 DPI and at 600
    DPI (``docs/layout-acceptance-criteria-20260902.md``: "300 DPI and 600 DPI
    each once"), and the value must be the item's.

``direction``
    Owner rule 2026-09-15 (``docs/calibration-rules-from-owner-20260915.md``
    §4.5): 0 deg is flat with the digits BELOW the bars; 270 deg clockwise is
    vertical with the digits on the RIGHT. 90 and 180 are not used.

    Two independent signals must agree. zxing-cpp's ``orientation`` is the
    reading direction; :func:`locate_digits` finds the human-readable digits in
    the barcode layer's own ink, with no decoder involved. A mirrored symbol, or
    a decoder convention that changed under an upgrade, makes them disagree and
    is reported as indeterminate instead of being resolved by assumption.

    zxing-cpp's convention was measured, not looked up (2026-09-15, renders of
    SKU 3210808): a front_standard face (38.5 x 26.5 x 18.5 cm carton, digits on
    the right by eye) reports -90; ``np.rot90`` of that crop by 90/180/270 deg
    counter-clockwise reports 180/90/0. So ``orientation mod 360`` is the
    clockwise angle in image coordinates, i.e. the owner's physical angle.

    The same measurement found a live defect on its first run: the
    front_wide_band face of the 27.5 x 27.5 x 10.5 cm carton, planned as
    physical 0, decoded at 180 with its digits ABOVE the bars, upside down by
    eye as well. Fixed in the same round (the bracket tile and its barcode were
    turned the wrong way against each other); this module is what found it and
    ``test_the_wide_band_front_prints_its_barcode_in_the_planned_direction``
    is what keeps it fixed.

``one group``
    Owner rule 2026-09-07 (``docs/barcode-uniform-scaling-20260907.md``): bars,
    digits, spacing and corner brackets scale as one vector group. The barcode
    layer's ink box, expressed in the bracket layer's ink box, must sit where a
    native render of the same value on the template puts it, rotated by the
    measured physical orientation. Any separate stretch of bars or digits moves
    an edge of that relation.

Image-level functions take plain arrays so they can be exercised on synthetic
input; :func:`audit_barcode` wires them to one face through two callables the
audit supplies (layer isolation and gray rendering), which keeps PDF plumbing
in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import fitz
import numpy as np

from ..ean13 import is_valid_ean13
from .raster import ink_mask, mask_bbox, pixmap_array

try:  # pragma: no cover - exercised by the environment, not by a branch test
    import zxingcpp
except ImportError:  # pragma: no cover
    zxingcpp = None  # type: ignore[assignment]


#: Owner rule 2026-09-15 §4.5.
ALLOWED_PHYSICAL_ORIENTATIONS: tuple[int, ...] = (0, 270)
#: Where the digits sit for each clockwise physical angle of an unmirrored EAN-13.
#: Turning a flat symbol clockwise carries its bottom edge to the left (90),
#: the top (180) and the right (270).
DIGIT_SIDE_ORIENTATION = {"below": 0, "left": 90, "above": 180, "right": 270}
#: Acceptance criteria 2026-09-02: decode once at each.
DEFAULT_DECODE_DPI: tuple[float, ...] = (300.0, 600.0)
#: The native reference is rendered once per face, at ~8x: the bracket tile is
#: 86 x 189 pt, i.e. 688 x 1512 px, so its own +-1 px edge error is <= 0.3%.
NATIVE_ZOOM = 8.0
EDGE_QUANTISATION_PX = 1.0


def _issue(code: str, message: str, *, severity: str = "error", **extra: Any) -> dict[str, Any]:
    payload = {"severity": severity, "code": code, "message": message}
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


class BarcodeDecoderUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedSymbol:
    text: str
    #: Clockwise physical angle, snapped to a quarter turn.
    orientation_deg: int
    #: zxing-cpp's raw ``orientation``.
    raw_orientation: int
    quad: tuple[tuple[float, float], ...]
    dpi: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "orientation_deg": self.orientation_deg,
            "raw_orientation": self.raw_orientation,
            "dpi": round(self.dpi, 1),
        }


def physical_from_zxing(raw: int) -> int:
    """zxing-cpp ``orientation`` -> clockwise physical angle in ``{0, 90, 180, 270}``.

    See the module docstring for the measurement. Snapped to the nearest quarter
    turn because the vector artwork is axis-aligned; the raw value is kept on the
    symbol for the report.
    """
    return int(round((int(raw) % 360) / 90.0)) * 90 % 360


def decode_ean13(gray: np.ndarray, *, dpi: float = 0.0) -> list[DecodedSymbol]:
    """Every EAN-13 zxing-cpp finds in an 8-bit grayscale image, in any direction."""
    if zxingcpp is None:
        raise BarcodeDecoderUnavailable("zxingcpp is not installed")
    image = np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))
    results = zxingcpp.read_barcodes(
        image, formats=zxingcpp.BarcodeFormat.EAN13, try_rotate=True
    )
    out: list[DecodedSymbol] = []
    for result in results:
        raw = int(round(float(result.orientation)))
        position = result.position
        quad = tuple(
            (float(point.x), float(point.y))
            for point in (
                position.top_left,
                position.top_right,
                position.bottom_right,
                position.bottom_left,
            )
        )
        out.append(DecodedSymbol(str(result.text), physical_from_zxing(raw), raw, quad, float(dpi)))
    return out


# --------------------------------------------------------------------------- #
# Human-readable digits, located in the artwork
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DigitSide:
    """Which side of the bars the barcode layer's non-bar ink sits on."""

    side: str | None
    bar_axis: str | None
    position: float
    residual_share: float
    reason: str = ""

    @property
    def orientation_deg(self) -> int | None:
        return DIGIT_SIDE_ORIENTATION.get(self.side) if self.side else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "orientation_deg": self.orientation_deg,
            "bar_axis": self.bar_axis,
            "position": round(self.position, 4),
            "residual_share": round(self.residual_share, 4),
            "reason": self.reason or None,
        }


def _vertical_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(column, start_row, end_row)`` of every vertical ink run; ends exclusive.

    Transposed so that ``np.nonzero``'s row-major order walks one column at a
    time: starts and ends then pair up index for index.
    """
    lines = np.pad(np.ascontiguousarray(mask.T).astype(np.int8), ((0, 0), (1, 1)))
    diff = np.diff(lines, axis=1)
    columns, starts = np.nonzero(diff == 1)
    _, ends = np.nonzero(diff == -1)
    return columns, starts, ends


def locate_digits(
    mask: np.ndarray,
    *,
    long_run_share: float = 0.5,
    min_axis_ratio: float = 2.0,
    end_share: float = 0.25,
    min_residual_share: float = 0.01,
) -> DigitSide:
    """Find the human-readable digits beside the bars, from ink alone.

    1. The bars fix the axis: along the bar length ink runs are as long as the
       guard bars, across it no run is longer than four modules.
    2. Every run at least ``long_run_share`` of the longest is bar. EAN-13 data
       bars are the guard length minus the digit notch (9.8 of 154.6 pt on the
       template, 94%); a digit glyph is under a tenth of it.
    3. What is left is the digits. Their centroid along the bar axis sits at the
       notch end of the bars (the leading digit even beyond it), so it lies in
       the outer ``end_share`` at one end - that end is the digit side.
    """
    box = mask_bbox(mask)
    if box is None:
        return DigitSide(None, None, 0.0, 0.0, "no barcode ink")
    crop = mask[box[1] : box[3], box[0] : box[2]]

    _, v_start, v_end = _vertical_runs(crop)
    _, h_start, h_end = _vertical_runs(crop.T)
    longest_v = int((v_end - v_start).max()) if v_start.size else 0
    longest_h = int((h_end - h_start).max()) if h_start.size else 0
    if max(longest_v, longest_h) < min_axis_ratio * max(1, min(longest_v, longest_h)):
        return DigitSide(None, None, 0.0, 0.0, "no dominant bar direction")
    bar_axis = "y" if longest_v >= longest_h else "x"

    # Canonical frame: bars vertical. For horizontal bars the transpose maps
    # original columns to rows, so "far end" means right instead of below.
    canonical = crop if bar_axis == "y" else crop.T
    columns, starts, ends = _vertical_runs(canonical)
    lengths = ends - starts
    keep = lengths >= long_run_share * int(lengths.max())
    height, width = canonical.shape
    marks = np.zeros((width, height + 1), dtype=np.int32)
    np.add.at(marks, (columns[keep], starts[keep]), 1)
    np.add.at(marks, (columns[keep], ends[keep]), -1)
    bars = (np.cumsum(marks, axis=1)[:, :-1] > 0).T

    residual = canonical & ~bars
    ink = int(canonical.sum())
    share = float(residual.sum()) / ink if ink else 0.0
    if share < min_residual_share:
        return DigitSide(None, bar_axis, 0.0, share, "no human-readable digits beside the bars")

    first, last = int(starts[keep].min()), int(ends[keep].max())
    rows = np.nonzero(residual)[0]
    position = (float(rows.mean()) + 0.5 - first) / max(1, last - first)
    if position >= 1.0 - end_share:
        far = True
    elif position <= end_share:
        far = False
    else:
        return DigitSide(None, bar_axis, position, share, "digits are not at either end of the bars")
    if bar_axis == "y":
        side = "below" if far else "above"
    else:
        side = "right" if far else "left"
    return DigitSide(side, bar_axis, position, share)


def orientation_findings(
    symbols: Sequence[DecodedSymbol],
    digits: DigitSide,
    *,
    plan_physical: int | None,
    allowed: Iterable[int] = ALLOWED_PHYSICAL_ORIENTATIONS,
    face: str = "",
) -> tuple[list[dict[str, Any]], int | None]:
    """Physical orientation where decoder and digits agree, and what is wrong with it."""
    read = sorted({symbol.orientation_deg for symbol in symbols})
    from_digits = digits.orientation_deg
    evidence = {
        "decoder_orientations": read,
        "digit_side": digits.side,
        "digit_orientation": from_digits,
        "digit_reason": digits.reason or None,
    }
    if len(read) != 1 or from_digits is None or read[0] != from_digits:
        return [
            _issue(
                "barcode_orientation_indeterminate",
                f"The {face} face barcode's direction could not be established: the decoder "
                f"reads it at {read or 'no angle'} and its digits sit "
                f"{digits.side or 'nowhere identifiable'} ({from_digits}). Both must agree.",
                face=face,
                component="barcode",
                **evidence,
            )
        ], None

    physical = read[0]
    findings: list[dict[str, Any]] = []
    permitted = sorted({int(value) for value in allowed})
    if physical not in permitted:
        findings.append(
            _issue(
                "barcode_orientation_not_allowed",
                f"The {face} face barcode prints at {physical} deg clockwise (digits "
                f"{digits.side}); the owner allows only {permitted} - 0 flat with digits below, "
                "270 vertical with digits on the right.",
                face=face,
                component="barcode",
                physical_rotation_deg=physical,
                allowed=permitted,
                **evidence,
            )
        )
    if plan_physical is not None and physical != int(plan_physical):
        findings.append(
            _issue(
                "barcode_orientation_differs_from_plan",
                f"The {face} face barcode prints at {physical} deg but its plan says "
                f"{plan_physical} deg.",
                face=face,
                component="barcode",
                physical_rotation_deg=physical,
                plan_physical_rotation_deg=int(plan_physical),
                **evidence,
            )
        )
    return findings, physical


# --------------------------------------------------------------------------- #
# One vector group
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GroupRelation:
    """The barcode layer's ink box in the unit frame of the bracket ink box."""

    edges: tuple[float, float, float, float]
    bracket_px: tuple[float, float]

    def rotated(self, clockwise_deg: int) -> "GroupRelation":
        """The same relation after turning the whole group clockwise.

        In image coordinates a quarter turn clockwise takes (x, y) to (1 - y, x).
        """
        edges, extent = self.edges, self.bracket_px
        for _ in range((int(clockwise_deg) % 360) // 90):
            x0, y0, x1, y1 = edges
            edges = (1.0 - y1, x0, 1.0 - y0, x1)
            extent = (extent[1], extent[0])
        return GroupRelation(edges, extent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": [round(v, 4) for v in self.edges],
            "bracket_px": [round(v, 1) for v in self.bracket_px],
        }


def group_relation(barcode_mask: np.ndarray, bracket_mask: np.ndarray) -> GroupRelation | None:
    """Both masks must share one raster frame."""
    bars, brackets = mask_bbox(barcode_mask), mask_bbox(bracket_mask)
    if bars is None or brackets is None:
        return None
    width = float(brackets[2] - brackets[0])
    height = float(brackets[3] - brackets[1])
    if width <= 0 or height <= 0:
        return None
    return GroupRelation(
        (
            (bars[0] - brackets[0]) / width,
            (bars[1] - brackets[1]) / height,
            (bars[2] - brackets[0]) / width,
            (bars[3] - brackets[1]) / height,
        ),
        (width, height),
    )


def native_group_masks(
    renderer: Any,
    cfg: dict[str, Any],
    value: str,
    clip: Sequence[float],
    source_page: int,
    *,
    zoom: float = NATIVE_ZOOM,
) -> tuple[np.ndarray, np.ndarray]:
    """``(barcode, brackets)`` ink of the approved template, inside the bracket tile.

    The brackets are ``base.pdf`` within the clip the plan stamps; the barcode is
    ``KmtTemplateRenderer._draw_barcode`` - the native bars and digits the
    standard mark prints - on a blank page of the same size, cut by the same
    clip. That is the group before any adaptive transform touched it.
    """
    rect = fitz.Rect(*clip)
    matrix = fitz.Matrix(zoom, zoom)
    base_page = renderer.base[int(source_page)]
    brackets = ink_mask(pixmap_array(base_page.get_pixmap(clip=rect, matrix=matrix, alpha=False)))
    doc = fitz.open()
    try:
        page = doc.new_page(width=base_page.rect.width, height=base_page.rect.height)
        renderer._draw_barcode(page, fitz.Point(0, 0), cfg, value)
        barcode = ink_mask(pixmap_array(page.get_pixmap(clip=rect, matrix=matrix, alpha=False)))
    finally:
        doc.close()
    return barcode, brackets


def uniform_group_findings(
    final: GroupRelation,
    native: GroupRelation,
    rotation_deg: int,
    *,
    tolerance: float,
    edge_px: float = EDGE_QUANTISATION_PX,
    face: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Does the printed relation match the native one turned by ``rotation_deg``?

    Per edge ``r = (b - B0) / Bw``: ``b`` and ``B0`` are each +-1 px and ``Bw``
    +-2 px, so ``r`` is uncertain by ``(2 + 2|r|) px / Bw``; the same holds on the
    native side at its own resolution. A deviation counts only beyond that plus
    the policy tolerance.
    """
    expected = native.rotated(rotation_deg)
    rows = []
    worst = -1.0
    for index, (got, want) in enumerate(zip(final.edges, expected.edges)):
        axis = index % 2
        uncertainty = (2.0 + 2.0 * abs(got)) * edge_px / final.bracket_px[axis]
        uncertainty += (2.0 + 2.0 * abs(want)) * edge_px / expected.bracket_px[axis]
        deviation = abs(got - want)
        worst = max(worst, deviation - uncertainty)
        rows.append(
            {
                "edge": ("x0", "y0", "x1", "y1")[index],
                "final": round(got, 4),
                "native": round(want, 4),
                "deviation": round(deviation, 4),
                "pixel_uncertainty": round(uncertainty, 4),
            }
        )
    evidence = {
        "rotation_from_native_deg": int(rotation_deg) % 360,
        "tolerance": tolerance,
        "edges": rows,
        "final": final.to_dict(),
        "native": expected.to_dict(),
    }
    if worst <= tolerance:
        return [], evidence
    return [
        _issue(
            "barcode_group_not_uniformly_scaled",
            f"On the {face} face the bars and digits do not sit inside their brackets the "
            f"way one uniformly scaled group would: an edge is off by {worst * 100:.1f}% of "
            "the bracket extent beyond pixel uncertainty. Bars, digits and brackets must "
            "share one transform.",
            face=face,
            component="barcode",
            excess=round(worst, 4),
            **evidence,
        )
    ], evidence


# --------------------------------------------------------------------------- #
# One face
# --------------------------------------------------------------------------- #

#: ``isolate(element, box_px, zoom) -> (mask, zoom_used)``; box in face pixels.
Isolate = Callable[[str, tuple[int, int, int, int], float], "tuple[np.ndarray, float] | None"]
#: ``render_gray(box_px, margin_pt, dpi) -> (gray, dpi_used)``; box in face pixels.
RenderGray = Callable[[tuple[int, int, int, int], float, float], "tuple[np.ndarray, float]"]


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def audit_barcode(
    face: Any,
    plan_face: Any,
    item: dict[str, Any],
    *,
    isolate: Isolate,
    render_gray: RenderGray,
    renderer: Any,
    atlas: Any,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every barcode check for one face; ``(findings, measurement)``."""
    if "barcode" not in plan_face.fields or plan_face.barcode is None:
        return [], {"status": "not_in_plan"}
    value = str(item.get("barcode") or "")
    if not is_valid_ean13(value):
        # The renderer leaves the area empty and warns barcode_skipped.
        return [], {"status": "skipped_invalid_or_missing_value"}

    rules = policy.get("barcode_final_pdf", {})
    edge_px = float(rules.get("edge_quantisation_px", EDGE_QUANTISATION_PX))
    name = face.face
    report: dict[str, Any] = {"status": "checked", "value": value}

    barcode_px = mask_bbox(face.shape_of("barcode"))
    if barcode_px is None:
        return [
            _issue(
                "barcode_unreadable_in_final_pdf",
                f"The {name} face was planned with barcode {value} but carries no barcode ink.",
                face=name,
                component="barcode",
                expected=value,
            )
        ], {**report, "status": "no_barcode_ink"}
    if zxingcpp is None:
        return [
            _issue(
                "barcode_decoder_unavailable",
                "zxingcpp is not installed, so the barcode cannot be read back. An unavailable "
                "check is not a passing one.",
                face=name,
                component="barcode",
            )
        ], {**report, "status": "decoder_unavailable"}

    findings: list[dict[str, Any]] = []

    # -- scannable, at each acceptance resolution --------------------------- #
    extent_pt = max(barcode_px[2] - barcode_px[0], barcode_px[3] - barcode_px[1]) / face.zoom
    margin_pt = max(
        float(rules.get("decode_margin_mm", 4.0)) * 72.0 / 25.4,
        float(rules.get("decode_margin_share", 0.2)) * extent_pt,
    )
    decoded: list[tuple[float, list[DecodedSymbol]]] = []
    for dpi in rules.get("decode_dpi", DEFAULT_DECODE_DPI):
        gray, used = render_gray(barcode_px, margin_pt, float(dpi))
        decoded.append((used, decode_ean13(gray, dpi=used)))
    report["decoded"] = [
        {"dpi": round(used, 1), "symbols": [symbol.to_dict() for symbol in symbols]}
        for used, symbols in decoded
    ]
    symbols = [symbol for _, found in decoded for symbol in found]
    failed = [round(used, 1) for used, found in decoded if not found]
    if failed:
        findings.append(
            _issue(
                "barcode_unreadable_in_final_pdf",
                f"The {name} face barcode does not decode from the final PDF at {failed} DPI.",
                face=name,
                component="barcode",
                expected=value,
                failed_dpi=failed,
            )
        )
    wrong = sorted({symbol.text for symbol in symbols if symbol.text != value})
    if wrong:
        findings.append(
            _issue(
                "barcode_value_mismatch",
                f"The {name} face barcode decodes as {wrong}, not the item's {value}.",
                face=name,
                component="barcode",
                expected=value,
                decoded=wrong,
            )
        )

    # -- the group, measured sharply enough to judge it --------------------- #
    bracket_px = mask_bbox(face.shape_of("barcode_brackets"))
    reference_box = bracket_px or barcode_px
    short_pt = min(reference_box[2] - reference_box[0], reference_box[3] - reference_box[1]) / face.zoom
    wanted = max(face.zoom, float(rules.get("group_min_bracket_px", 600)) / max(short_pt, 1e-6))
    barcode_mask = face.shape_of("barcode")
    bracket_mask = face.shape_of("barcode_brackets") if bracket_px else None
    group_zoom = face.zoom
    if wanted > face.zoom * 1.05:
        union = _union(barcode_px, bracket_px)
        sharp_bars = isolate("barcode", union, wanted)
        sharp_brackets = isolate("barcode_brackets", union, wanted) if bracket_px else None
        if sharp_bars is not None and (bracket_px is None or sharp_brackets is not None):
            barcode_mask, group_zoom = sharp_bars
            bracket_mask = sharp_brackets[0] if sharp_brackets is not None else None
    report["group_zoom"] = round(group_zoom, 3)

    # -- direction ---------------------------------------------------------- #
    digits = locate_digits(barcode_mask)
    report["digits"] = digits.to_dict()
    physical: int | None = None
    if symbols:
        plan_physical = plan_face.barcode.to_dict().get("physical_rotation_deg")
        found, physical = orientation_findings(
            symbols,
            digits,
            plan_physical=plan_physical,
            allowed=rules.get("allowed_physical_orientations", ALLOWED_PHYSICAL_ORIENTATIONS),
            face=name,
        )
        findings.extend(found)
        report["plan_physical_rotation_deg"] = plan_physical
    report["physical_rotation_deg"] = physical

    # -- one vector group --------------------------------------------------- #
    placement = next((s for s in plan_face.statics if s.name == "barcode_brackets"), None)
    if physical is None or bracket_mask is None or placement is None:
        report["group"] = {"status": "not_measured", "reason": "needs a determined orientation and both layers"}
        return findings, report
    final = group_relation(barcode_mask, bracket_mask)
    cfg = getattr(atlas.frame, plan_face.kind)["barcode"]
    native_bars, native_brackets = native_group_masks(
        renderer, cfg, value, tuple(placement.clip), placement.source_page
    )
    native = group_relation(native_bars, native_brackets)
    native_digits = locate_digits(native_bars)
    if final is None or native is None or native_digits.orientation_deg is None:
        report["group"] = {"status": "not_measured", "reason": "native reference not measurable"}
        return findings, report
    found, evidence = uniform_group_findings(
        final,
        native,
        (physical - native_digits.orientation_deg) % 360,
        tolerance=float(rules.get("group_uniform_tolerance", 0.01)),
        edge_px=edge_px,
        face=name,
    )
    findings.extend(found)
    report["group"] = {"status": "measured", "native_orientation_deg": native_digits.orientation_deg, **evidence}
    return findings, report


__all__ = [
    "ALLOWED_PHYSICAL_ORIENTATIONS",
    "BarcodeDecoderUnavailable",
    "DEFAULT_DECODE_DPI",
    "DIGIT_SIDE_ORIENTATION",
    "DecodedSymbol",
    "DigitSide",
    "GroupRelation",
    "audit_barcode",
    "decode_ean13",
    "group_relation",
    "locate_digits",
    "native_group_masks",
    "orientation_findings",
    "physical_from_zxing",
    "uniform_group_findings",
]
