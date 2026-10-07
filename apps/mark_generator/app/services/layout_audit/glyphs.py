"""Glyph fidelity: is the type still the shape the font drew?

The 2026-09-02 review found labels squeezed to 0.46 of their natural width, and
- more telling - squeezed *unevenly*: Q, T, Y and S at 0.40 while W sat at 0.64.
A font never does that. Uneven per-glyph distortion is proof that something
scaled a rendered run on one axis, and it is invisible to any check that looks
at the box the run was supposed to occupy.

The measurement avoids needing to know font size, side bearings or baselines:
render the same string with the same font and no transform, then compare shapes.

    distortion = (w_actual / h_actual) / (w_reference / h_reference)

1.00 means undistorted. The template's own item number is condensed to 0.796 on
purpose, so the rule is "matches the ratio this element declares", not "is 1.00";
whether that 0.796 counts as an approved design parameter is the owner's call and
is recorded, not assumed, here.

Measurement resolution
----------------------
Every extent here is a bounding box of thresholded, anti-aliased ink, so each
edge lands somewhere inside its boundary pixel and each extent is uncertain by
up to one pixel. On small type that is not noise, it is the whole finding: the
anchor-4 side face (30.5 x 30.5 x 9.5 cm carton) measured its "36" quantity at
74 x 51 px at the audit's zoom 3 and reported 1.030 - a 3% "stretch" - while the
very same run measured 1.009 at zoom 6 and 1.009 at zoom 12, with the renderer
drawing it through ``_write_text`` at hscale 1.0 in the same arialbd.ttf the
reference uses. One pixel on 51 is already 2%, the entire policy tolerance.

So a result carries the worst-case interval its pixels allow
(:func:`aspect_ratio_bounds`), and callers judge the interval, not the point.
A verdict the interval cannot settle is re-measured at the zoom that can
(:func:`zoom_to_resolve`) rather than being guessed either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import fitz
import numpy as np

from .raster import ink_mask, mask_bbox, pixmap_array


#: A thresholded anti-aliased edge can fall anywhere inside its boundary pixel,
#: so a bounding-box extent is uncertain by up to this many pixels.
EDGE_QUANTISATION_PX = 1.0

#: Reference raster density. At 400 px per em an Arial digit is ~286 px tall,
#: so the reference's own +-1 px contributes ~0.6% to a two-digit run's aspect -
#: under a third of the 2% fidelity tolerance - on any point size.
REFERENCE_PX_PER_EM = 400.0

#: Hard ceiling on the reference raster. The previous fixed zoom of 8 turned the
#: 430 pt item number on the 900 x 200 mm owner front into a 47481 x 20644 px
#: page (980 MP) that MuPDF refused with "Overly large image", crashing the whole
#: audit. A seven-digit run at 400 px/em needs ~3.3 MP; 16 MP leaves room for
#: long runs and still bounds memory at ~48 MB of RGB.
REFERENCE_MAX_PIXELS = 16_000_000


@dataclass(frozen=True)
class RunShape:
    """Ink extent of a rendered text run, plus its per-glyph pieces.

    ``px_per_pt`` is the raster density the extents were measured at, when
    known; it lets two runs rendered at different zooms be compared in points.
    """

    width: float
    height: float
    pieces: tuple[tuple[float, float], ...] = ()
    px_per_pt: float = 0.0

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass
class GlyphFidelity:
    """How far a rendered run drifted from the font's own proportions."""

    text: str
    distortion: float
    declared: float
    horizontal_scale: float
    vertical_scale: float
    per_character: list[float] = field(default_factory=list)
    #: Worst-case interval of ``distortion`` given +-1 px on every extent.
    distortion_low: float = 0.0
    distortion_high: float = 0.0
    per_character_bounds: list[tuple[float, float]] = field(default_factory=list)
    measured_px: tuple[float, float] = (0.0, 0.0)
    reference_px: tuple[float, float] = (0.0, 0.0)
    narrowest_piece_px: tuple[float, float] = (0.0, 0.0)

    @property
    def per_character_spread(self) -> float:
        """Max-minus-min across glyphs, as a share of the median.

        Zero for any honest transform, uniform or not. Non-zero means different
        glyphs in one run were scaled by different amounts.
        """
        values = [v for v in self.per_character if v > 0]
        if len(values) < 2:
            return 0.0
        median = float(np.median(values))
        if median <= 0:
            return 0.0
        return float((max(values) - min(values)) / median)

    def _bounded_pieces(self) -> list[tuple[float, float]] | None:
        bounds = self.per_character_bounds
        if len(bounds) < 2 or len(bounds) != len(self.per_character):
            return None
        return bounds

    @property
    def per_character_spread_low(self) -> float:
        """The smallest spread the glyph pixels are consistent with.

        The largest glyph ratio is at least its own lower bound and the
        smallest is at most its own upper bound, so the spread can be no less
        than their gap. Positive only when the unevenness survives quantisation.
        """
        bounds = self._bounded_pieces()
        if bounds is None:
            return self.per_character_spread
        highs = [hi for _, hi in bounds]
        median = float(np.median(highs))
        if not math.isfinite(median) or median <= 0:
            return 0.0
        gap = max(lo for lo, _ in bounds) - min(highs)
        return max(0.0, float(gap / median))

    @property
    def per_character_spread_high(self) -> float:
        """The largest spread the glyph pixels are consistent with."""
        bounds = self._bounded_pieces()
        if bounds is None:
            return self.per_character_spread
        lows = [lo for lo, _ in bounds]
        median = float(np.median(lows))
        if median <= 0:
            return math.inf
        return float((max(hi for _, hi in bounds) - min(lows)) / median)

    @property
    def error(self) -> float:
        return abs(self.distortion - self.declared) / max(1e-9, self.declared)

    def to_dict(self) -> dict[str, Any]:
        def _r(value: float) -> float | None:
            return round(value, 4) if math.isfinite(value) else None

        return {
            "text": self.text,
            "distortion": round(self.distortion, 4),
            "distortion_interval": [_r(self.distortion_low), _r(self.distortion_high)],
            "declared": round(self.declared, 4),
            "error": round(self.error, 4),
            "horizontal_scale": round(self.horizontal_scale, 4),
            "vertical_scale": round(self.vertical_scale, 4),
            "per_character": [round(v, 4) for v in self.per_character],
            "per_character_spread": round(self.per_character_spread, 4),
            "per_character_spread_low": _r(self.per_character_spread_low),
            "measured_px": [round(v, 1) for v in self.measured_px],
            "reference_px": [round(v, 1) for v in self.reference_px],
        }


# --------------------------------------------------------------------------- #
# Reference rendering
# --------------------------------------------------------------------------- #


def natural_advance(font: fitz.Font, text: str, fontsize: float) -> float:
    """The font's own advance width for ``text``, in points.

    Deliberately ``fitz.Font.text_length`` - the exact call the production
    renderer uses to fit text - so a disagreement is a rendering fault and never
    a difference between two metric libraries.
    """
    return float(font.text_length(text, fontsize))


def _font_for(fontfile: str | None, fontname: str) -> fitz.Font:
    if fontfile:
        return fitz.Font(fontfile=fontfile)
    try:
        return fitz.Font(fontname)
    except Exception:  # noqa: BLE001 - only used to size the page, never to draw
        return fitz.Font("helv")


def render_reference(
    text: str,
    *,
    fontsize: float,
    fontfile: str | None = None,
    fontname: str = "helv",
    zoom: float | None = None,
    split: bool = True,
    px_per_em: float = REFERENCE_PX_PER_EM,
    max_pixels: int = REFERENCE_MAX_PIXELS,
) -> RunShape:
    """Draw ``text`` with no transform and measure the ink it makes.

    Distortion is a ratio of aspect ratios, so the reference needs enough pixels
    to make its own edge error small and no more. The zoom therefore follows a
    density per em (``zoom=None``) rather than multiplying the production point
    size, and the page is sized from the font's own advance - one em of margin
    each side covers overhang past the advance - instead of a per-character
    guess. ``max_pixels`` bounds the raster on any input, an explicit ``zoom``
    included; the returned ``px_per_pt`` says what was actually used.
    """
    fontsize = float(fontsize)
    if fontsize <= 0 or not text:
        return RunShape(0.0, 0.0, ())
    font = _font_for(fontfile, fontname)
    advance = max(float(font.text_length(text, fontsize)), fontsize * 0.5)
    margin = fontsize
    width = advance + 2.0 * margin
    # 2.25 em above the baseline and 1.25 em below hold accents and descenders.
    height = fontsize * 3.5
    baseline = fontsize * 2.25

    scale = float(zoom) if zoom else float(px_per_em) / fontsize
    pixels = width * height * scale * scale
    if pixels > max_pixels:
        scale *= math.sqrt(max_pixels / pixels)

    doc = fitz.open()
    try:
        page = doc.new_page(width=width, height=height)
        args: dict[str, Any] = {"fontname": fontname}
        if fontfile:
            args = {"fontname": "AuditRef", "fontfile": fontfile}
        page.insert_text(fitz.Point(margin, baseline), text, fontsize=fontsize, **args)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        mask = ink_mask(pixmap_array(pix))
    finally:
        doc.close()
    shape = _shape(mask, split=split)
    return RunShape(shape.width, shape.height, shape.pieces, px_per_pt=scale)


def _shape(mask: np.ndarray, *, split: bool) -> RunShape:
    box = mask_bbox(mask)
    if box is None:
        return RunShape(0.0, 0.0, ())
    x0, y0, x1, y1 = box
    pieces = _pieces(mask) if split else ()
    return RunShape(float(x1 - x0), float(y1 - y0), pieces)


def _pieces(mask: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Glyph-sized ink blobs, left to right, as ``(width, height)`` pairs.

    Column projection rather than connected components: it splits on the gaps
    the font's own spacing makes, needs no extra dependency, and does not merge
    two letters that happen to touch diagonally. Blobs narrower than a tenth of
    the tallest are dropped, which removes punctuation dots without removing
    thin letters like I.
    """
    columns = mask.any(axis=0)
    if not columns.any():
        return ()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, filled in enumerate(columns):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(columns)))

    out: list[tuple[float, float]] = []
    for left, right in spans:
        block = mask[:, left:right]
        rows = np.nonzero(block.any(axis=1))[0]
        if rows.size == 0:
            continue
        height = float(rows[-1] - rows[0] + 1)
        out.append((float(right - left), height))
    if not out:
        return ()
    tallest = max(height for _, height in out)
    return tuple(piece for piece in out if piece[1] >= tallest * 0.1)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def aspect_ratio_bounds(
    measured_w: float,
    measured_h: float,
    reference_w: float,
    reference_h: float,
    edge_px: float = EDGE_QUANTISATION_PX,
) -> tuple[float, float]:
    """Worst-case interval of ``(mw / mh) / (rw / rh)`` with +-``edge_px`` on each extent.

    Worst case rather than a statistical combination: the four errors are
    bounded, not Gaussian, and a hard check must not call a defect on a number
    its own pixels could have produced. ``(0, inf)`` when a pixel is the whole
    extent, i.e. nothing can be said at this resolution.
    """
    e = float(edge_px)
    if min(measured_w, measured_h, reference_w, reference_h) - e <= 0:
        return 0.0, math.inf
    low = ((measured_w - e) / (measured_h + e)) / ((reference_w + e) / (reference_h - e))
    high = ((measured_w + e) / (measured_h - e)) / ((reference_w - e) / (reference_h + e))
    return float(low), float(high)


def fidelity_of_run(
    actual: np.ndarray,
    text: str,
    *,
    fontsize: float,
    fontfile: str | None = None,
    fontname: str = "helv",
    declared: float = 1.0,
    zoom: float | None = None,
    measured_zoom: float | None = None,
    edge_px: float = EDGE_QUANTISATION_PX,
    px_per_em: float = REFERENCE_PX_PER_EM,
    max_pixels: int = REFERENCE_MAX_PIXELS,
) -> GlyphFidelity:
    """Compare a rendered run's ink against the same string drawn untransformed.

    ``actual`` is the ink mask of just that run - in practice one optional
    content layer, so nothing else can contaminate the bounding box.
    ``zoom`` forces the reference zoom (default: density per em);
    ``measured_zoom`` is the zoom ``actual`` was rasterised at, needed only to
    express the horizontal and vertical scales in points.
    """
    reference = render_reference(
        text,
        fontsize=fontsize,
        fontfile=fontfile,
        fontname=fontname,
        zoom=zoom,
        px_per_em=px_per_em,
        max_pixels=max_pixels,
    )
    measured = _shape(actual, split=True)
    if reference.width <= 0 or reference.height <= 0 or measured.height <= 0:
        return GlyphFidelity(text, 0.0, declared, 0.0, 0.0, [])

    distortion = measured.aspect / reference.aspect if reference.aspect else 0.0
    low, high = aspect_ratio_bounds(
        measured.width, measured.height, reference.width, reference.height, edge_px
    )
    if measured_zoom and reference.px_per_pt:
        horizontal = (measured.width / measured_zoom) / (reference.width / reference.px_per_pt)
        vertical = (measured.height / measured_zoom) / (reference.height / reference.px_per_pt)
    else:
        # Pixel ratios: only meaningful when both runs share a zoom.
        horizontal = measured.width / reference.width
        vertical = measured.height / reference.height
    narrowest = min(measured.pieces, key=lambda p: p[0]) if measured.pieces else (0.0, 0.0)
    return GlyphFidelity(
        text=text,
        distortion=distortion,
        declared=declared,
        horizontal_scale=horizontal,
        vertical_scale=vertical,
        per_character=per_character_fidelity(measured, reference),
        distortion_low=low,
        distortion_high=high,
        per_character_bounds=per_character_bounds(measured, reference, edge_px),
        measured_px=(measured.width, measured.height),
        reference_px=(reference.width, reference.height),
        narrowest_piece_px=narrowest,
    )


def per_character_fidelity(measured: RunShape, reference: RunShape) -> list[float]:
    """Per-glyph aspect ratio of measured to reference, in reading order.

    Returns an empty list when the two runs did not split into the same number
    of pieces - a mismatch means the comparison would pair up the wrong glyphs,
    and a wrong pairing is worse than no number.
    """
    if not measured.pieces or len(measured.pieces) != len(reference.pieces):
        return []
    out: list[float] = []
    for (mw, mh), (rw, rh) in zip(measured.pieces, reference.pieces):
        if mh <= 0 or rw <= 0 or rh <= 0:
            out.append(0.0)
            continue
        out.append((mw / mh) / (rw / rh))
    return out


def per_character_bounds(
    measured: RunShape, reference: RunShape, edge_px: float = EDGE_QUANTISATION_PX
) -> list[tuple[float, float]]:
    """The :func:`aspect_ratio_bounds` interval of every glyph pair, same pairing rule."""
    if not measured.pieces or len(measured.pieces) != len(reference.pieces):
        return []
    return [
        aspect_ratio_bounds(mw, mh, rw, rh, edge_px)
        for (mw, mh), (rw, rh) in zip(measured.pieces, reference.pieces)
    ]


def scale_ratio_bounds(
    rendered_w: float,
    source_w: float,
    rendered_h: float,
    source_h: float,
    edge_px: float = EDGE_QUANTISATION_PX,
) -> tuple[float, float]:
    """Worst-case interval of ``(rendered_w/source_w) / (rendered_h/source_h)``.

    The same bound arithmetic as :func:`aspect_ratio_bounds`, with the four ink
    extents of a static component in place of a text run's two. It is the same
    measurement problem: both the rendered layer and the source tile are
    bounding boxes of thresholded ink, so each of the four extents is uncertain
    by up to one pixel, and on a small component that is the whole tolerance.

    Measured on the half-scale owner front (450 x 100 mm, carton 45 x 18 x 10 cm)
    on 2026-09-15: ``origin_text`` read 0.9763 at the audit's zoom 3 - a 2.4%
    "stretch" past the 2% tolerance - while the same artwork re-measured at zoom
    6/12/24 read 0.9817/0.9909/1.0002. Nothing moved; the 25 px tall ink extent
    simply could not resolve 2%. This function is what says so.
    """
    e = float(edge_px)
    if min(rendered_w, rendered_h, source_w, source_h) - e <= 0:
        return 0.0, math.inf
    low = ((rendered_w - e) / (rendered_h + e)) / ((source_w + e) / (source_h - e))
    high = ((rendered_w + e) / (rendered_h - e)) / ((source_w - e) / (source_h + e))
    return float(low), float(high)


def zoom_to_resolve_scale(
    rendered_w: float,
    rendered_h: float,
    source_w: float,
    source_h: float,
    current_zoom: float,
    target_uncertainty: float,
    *,
    edge_px: float = EDGE_QUANTISATION_PX,
) -> float:
    """Zoom at which a component's scale ratio is uncertain by ``target_uncertainty``.

    Both sides are rasterised at the same zoom - the element at its planned
    point size, the source tile at its own - so every extent grows with zoom and
    the ratio's relative edge error falls in proportion to it. The uncertainty
    now is the sum of the four relative extents; dividing it by the budget gives
    the factor the zoom has to grow by.
    """
    if current_zoom <= 0 or min(rendered_w, rendered_h, source_w, source_h) <= 0:
        return current_zoom
    now = edge_px / rendered_w + edge_px / rendered_h + edge_px / source_w + edge_px / source_h
    return float(current_zoom * now / max(target_uncertainty, 1e-9))


def zoom_to_resolve(
    result: GlyphFidelity,
    current_zoom: float,
    target_uncertainty: float,
    *,
    edge_px: float = EDGE_QUANTISATION_PX,
) -> float:
    """Zoom at which the measurement's relative edge error fits ``target_uncertainty``.

    The measured side's error is ``edge/w + edge/h`` and falls in proportion to
    zoom; the narrowest glyph dominates the per-character bounds, so it counts
    too. The reference's own error does not shrink with the measured zoom and is
    taken off the budget first. The quarter-target floor only keeps the division
    finite: at :data:`REFERENCE_PX_PER_EM` the reference uses ~0.6% of a 1%
    target, never all of it.
    """
    mw, mh = result.measured_px
    if current_zoom <= 0 or min(mw, mh) <= 0:
        return current_zoom
    measured = edge_px / mw + edge_px / mh
    pw, ph = result.narrowest_piece_px
    if min(pw, ph) > 0:
        measured = max(measured, edge_px / pw + edge_px / ph)
    rw, rh = result.reference_px
    reference = edge_px / rw + edge_px / rh if min(rw, rh) > 0 else 0.0
    budget = max(target_uncertainty - reference, target_uncertainty / 4.0)
    return float(current_zoom * measured / budget)


def cap_height(mask: np.ndarray) -> float:
    """Ink height of a run, in pixels - the size a reader actually sees.

    Used for the same-class consistency check. Two labels set at the same point
    size must produce the same ink height; the plan's ``fontsize`` value cannot
    tell you whether they did.
    """
    box = mask_bbox(mask)
    return 0.0 if box is None else float(box[3] - box[1])


def consistency(values: Sequence[float]) -> float:
    """Spread of a same-class group, as a share of its median.

    0 for a healthy class. The 2026-09-02 evidence - label glyph heights of
    17.54 mm and 8.18 mm inside one group - lands near 0.74 here.
    """
    usable = [float(v) for v in values if v > 0]
    if len(usable) < 2:
        return 0.0
    median = float(np.median(usable))
    if median <= 0:
        return 0.0
    return (max(usable) - min(usable)) / median


__all__ = [
    "EDGE_QUANTISATION_PX",
    "GlyphFidelity",
    "REFERENCE_MAX_PIXELS",
    "REFERENCE_PX_PER_EM",
    "RunShape",
    "aspect_ratio_bounds",
    "cap_height",
    "consistency",
    "fidelity_of_run",
    "natural_advance",
    "per_character_bounds",
    "per_character_fidelity",
    "render_reference",
    "scale_ratio_bounds",
    "zoom_to_resolve",
    "zoom_to_resolve_scale",
]
