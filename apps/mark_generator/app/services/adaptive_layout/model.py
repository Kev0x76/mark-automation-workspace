"""Data model for carton-size adaptive mark layout.

Vocabulary, fixed by the owner's decision and confirmed against real factory
files: the front mark prints on the carton's ``length x height`` face and the
side mark on its ``width x height`` face, so both faces always share the same
height.

Everything here is geometry and provenance only - no drawing, no file IO - so
layout decisions stay inspectable and testable independently of PyMuPDF.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


MM_PER_PT = 25.4 / 72.0
PT_PER_MM = 72.0 / 25.4

FaceKind = Literal["front", "side"]
FitMode = Literal["uniform", "stretch", "stretch_x", "slice_x", "rotate90_uniform"]


#: Stretchable interiors measured out of base.pdf, keyed by the source tile
#: ``(source_page, clip)``. Filled by ``Atlas.slice_x`` - the only producer of a
#: measurement - and read by :class:`StaticPlacement` when a placement is rebuilt
#: from its five historical fields. ``pipeline._shift_static`` and older callers
#: copy a placement as ``StaticPlacement(name, clip, target, fit, source_page)``;
#: without this a shifted ``slice_x`` placement would silently lose the geometry
#: that keeps its labels undistorted. The value is a pure function of the tile,
#: so looking it up by the tile is the same fact, not a second source of truth.
_SOURCE_SLICES: dict[tuple[int, tuple[float, ...]], tuple[float, float]] = {}


def register_source_slice(source_page: int, clip: Any, slice_x: tuple[float, float]) -> None:
    key = (int(source_page), tuple(float(v) for v in clip))
    value = (float(slice_x[0]), float(slice_x[1]))
    known = _SOURCE_SLICES.get(key)
    if known is not None and any(abs(a - b) > 1e-9 for a, b in zip(known, value)):
        # Two different base.pdf files with an identical clip in one process.
        # Guessing which one a rebuilt placement meant would print the wrong
        # cell, so refuse instead.
        raise ValueError(f"conflicting stretchable interiors for tile {key}: {known} vs {value}")
    _SOURCE_SLICES[key] = value


def source_slice(source_page: int, clip: Any) -> tuple[float, float] | None:
    return _SOURCE_SLICES.get((int(source_page), tuple(float(v) for v in clip)))


def mm(value_pt: float) -> float:
    return value_pt * MM_PER_PT


def pt(value_mm: float) -> float:
    return value_mm * PT_PER_MM


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in panel-local points, origin top-left, y down."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def inset_by(self, amount: float) -> "Box":
        return Box(self.x0 + amount, self.y0 + amount, self.x1 - amount, self.y1 - amount)

    def fraction(self, fx0: float, fy0: float, fx1: float, fy1: float) -> "Box":
        """Sub-box addressed by fractions of this box."""
        return Box(
            self.x0 + self.width * fx0,
            self.y0 + self.height * fy0,
            self.x0 + self.width * fx1,
            self.y0 + self.height * fy1,
        )

    def region(self, x_pair: Any, y_pair: Any) -> "Box":
        """Sub-box from an ``{"x": [..], "y": [..]}`` style region spec.

        Separate from ``fraction`` on purpose: the family configs store x and y
        as pairs, and splatting them into ``fraction``'s interleaved signature
        silently produces inverted boxes.
        """
        fx0, fx1 = (float(v) for v in x_pair)
        fy0, fy1 = (float(v) for v in y_pair)
        return self.fraction(fx0, fy0, fx1, fy1)

    def fit_uniform(self, source_aspect: float, *, align_x: str = "center", align_y: str = "middle") -> "Box":
        """Largest box of ``source_aspect`` fitting inside, aspect preserved.

        This is the only sizing path allowed for logos, icons, line art and
        barcodes: it can never stretch one axis independently.
        """
        if source_aspect <= 0 or self.width <= 0 or self.height <= 0:
            return self
        target_aspect = self.width / self.height
        if source_aspect >= target_aspect:
            width = self.width
            height = width / source_aspect
        else:
            height = self.height
            width = height * source_aspect
        if align_x == "left":
            x0 = self.x0
        elif align_x == "right":
            x0 = self.x1 - width
        else:
            x0 = self.x0 + (self.width - width) / 2.0
        if align_y == "top":
            y0 = self.y0
        elif align_y == "bottom":
            y0 = self.y1 - height
        else:
            y0 = self.y0 + (self.height - height) / 2.0
        return Box(x0, y0, x0 + width, y0 + height)

    def overlaps(self, other: "Box", *, gap: float = 0.0) -> bool:
        return not (
            self.x1 + gap <= other.x0
            or other.x1 + gap <= self.x0
            or self.y1 + gap <= other.y0
            or other.y1 + gap <= self.y0
        )

    def contains(self, other: "Box", *, tolerance: float = 0.01) -> bool:
        return (
            other.x0 >= self.x0 - tolerance
            and other.y0 >= self.y0 - tolerance
            and other.x1 <= self.x1 + tolerance
            and other.y1 <= self.y1 + tolerance
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def as_mm(self) -> list[float]:
        return [round(mm(v), 3) for v in self.as_tuple()]


@dataclass
class StaticPlacement:
    """A vector tile lifted out of base.pdf and dropped somewhere new.

    ``slice_x`` belongs to ``fit="slice_x"`` only: the source-space x interval
    ``(ix0, ix1)`` of the tile's empty value interior. The tile is then drawn as
    a horizontal 3-slice - ``[clip.x0, ix0]`` and ``[ix1, clip.x1]`` at the one
    uniform scale ``target.height / clip height``, the interior stretched in x
    to take up the rest of ``target.width``. Every side table cell in base.pdf
    is ``[black label][empty interior][unit text][hairline border]`` and the
    interior carries only the frame's top and bottom rules, so this widens a
    cell without touching a glyph; ``stretch_x`` widened the label with it.

    ``slice_y`` (optional, ``slice_x`` cells only) makes a cell taller without
    touching a glyph either: ``(t0, t1, b0, b1)`` are the source-space rows above
    and below the glyphs that hold nothing but the cell's own fill (the black
    label block, the white interior, the vertical hairline). With
    ``glyph_scale`` set, every other row - the rules and the glyph band - is
    drawn at that one uniform scale and the extra height of the target is split
    equally between the two padding bands. This is how a factory draws a taller
    row (557099 photo, 2026-09-23): the frame grows, the words do not.

    ``label_pad`` (optional, ``slice_x`` cells only) keeps the black label block
    at its source share of the cell's width when the cell is widened:
    ``(l0, l1, r0, r1)`` are the source-space black columns left and right of
    the label word. The block is drawn ``target.width * (ix0 - cx0) / clip
    width`` wide (never narrower than at the uniform scale), the extra split
    equally between the two padding bands; the word is not touched. In base.pdf
    the block is a third of the cell, as on the factory's boxes (owner photos
    2026-09-29); without this only the white interior widened and the block
    shrank to about a tenth ("黑框框就一点点").

    ``label_word`` (optional, with ``label_pad``) grows the label word with its
    block: the source box of the word, drawn on its own at one uniform scale so
    that it keeps its source share of the block's width, but no taller than
    ``label_word_row_share`` of the row and never smaller than the tile's own
    scale. The block under it is then drawn from its black padding alone. Owner
    2026-09-29: "QTY 等标志也可以跟着变大".

    ``unit_word`` (optional, with ``label_word``) lifts the unit (PCS / cm / kg)
    out the same way: its columns are drawn blank from the interior and the unit
    is drawn over them, right-aligned where it was. ``text_growth`` is the
    factor, over the tile's own scale, at which this cell's words are drawn -
    set by the family from :meth:`text_growth_limit`; the unit takes the same
    factor unless that would make it taller than ``label_word_row_share`` of
    the row.
    """

    name: str
    clip: tuple[float, float, float, float]
    target: Box
    fit: FitMode
    source_page: int
    slice_x: tuple[float, float] | None = None
    slice_y: tuple[float, float, float, float] | None = None
    glyph_scale: float | None = None
    label_pad: tuple[float, float, float, float] | None = None
    label_word: tuple[float, float, float, float] | None = None
    label_word_row_share: float = 0.5
    unit_word: tuple[float, float, float, float] | None = None
    text_growth: float | None = None

    def __post_init__(self) -> None:
        if self.fit != "slice_x":
            if self.slice_x is not None:
                raise ValueError(f"{self.name}: slice_x geometry is only meaningful for fit='slice_x'")
            return
        if self.slice_x is None:
            self.slice_x = source_slice(self.source_page, self.clip)
        if self.slice_x is None:
            raise ValueError(
                f"{self.name}: fit='slice_x' without a measured interior for its source tile; "
                "drawing it any other way would stretch the label"
            )
        cx0, _, cx1, _ = self.clip
        ix0, ix1 = self.slice_x
        if not cx0 <= ix0 < ix1 <= cx1:
            raise ValueError(f"{self.name}: interior {self.slice_x} is not inside clip x {cx0}-{cx1}")
        if self._interior_target_width() < -1e-6:
            raise ValueError(
                f"{self.name}: target is {self.target.width:.2f} pt wide but its label and unit "
                f"slices alone need {self.target.width - self._interior_target_width():.2f} pt"
            )
        if self.label_pad is not None:
            l0, l1, r0, r1 = self.label_pad
            if not cx0 <= l0 <= l1 <= r0 <= r1 <= ix0:
                raise ValueError(f"{self.name}: label padding {self.label_pad} is not ordered inside the label {cx0}-{ix0}")
        if self.label_word is not None and self.label_pad is None:
            raise ValueError(f"{self.name}: label_word needs label_pad")
        if self.unit_word is not None and not ix1 <= self.unit_word[0] < self.unit_word[2] <= cx1:
            raise ValueError(f"{self.name}: unit word {self.unit_word} is not between the interior and the border")
        if (self.slice_y is None) != (self.glyph_scale is None):
            raise ValueError(f"{self.name}: slice_y and glyph_scale come together")
        if self.slice_y is not None:
            _, cy0, _, cy1 = self.clip
            t0, t1, b0, b1 = self.slice_y
            if not cy0 <= t0 <= t1 <= b0 <= b1 <= cy1:
                raise ValueError(f"{self.name}: padding rows {self.slice_y} are not ordered inside clip y {cy0}-{cy1}")
            if self._pad_extra() < -1e-6:
                raise ValueError(f"{self.name}: target is shorter than the tile at glyph_scale {self.glyph_scale}")

    # -- geometry of the transform, shared by renderer and audit -------------- #

    @property
    def uniform_scale(self) -> float:
        """Scale of the undistorted slices: the tile's height governs, unless glyph_scale is set."""
        if self.glyph_scale is not None:
            return self.glyph_scale
        height = self.clip[3] - self.clip[1]
        return self.target.height / height if height else 0.0

    def _pad_extra(self) -> float:
        return self.target.height - (self.clip[3] - self.clip[1]) * (self.glyph_scale or 0.0)

    def _y_bands(self) -> list[tuple[float, float, float, float, bool]]:
        """``(source y0, source y1, target y0, target y1, is_padding)`` top to bottom."""
        _, cy0, _, cy1 = self.clip
        if self.slice_y is None:
            return [(cy0, cy1, self.target.y0, self.target.y1, False)]
        t0, t1, b0, b1 = self.slice_y
        pads = [(t0, t1), (b0, b1)]
        live = [p for p in pads if p[1] > p[0]]
        extra = self._pad_extra() / len(live) if live else 0.0
        s = self.glyph_scale
        bands, y = [], self.target.y0
        for sy0, sy1, pad in ((cy0, t0, False), (t0, t1, True), (t1, b0, False), (b0, b1, True), (b1, cy1, False)):
            if sy1 <= sy0:
                continue
            height = (sy1 - sy0) * s + (extra if pad else 0.0)
            bands.append((sy0, sy1, y, y + height, pad))
            y += height
        return bands

    def _interior_target_width(self) -> float:
        cx0, _, cx1, _ = self.clip
        ix0, ix1 = self.slice_x or (cx0, cx1)
        return self.target.width - ((ix0 - cx0) + (cx1 - ix1)) * self.uniform_scale

    def _label_extra(self) -> float:
        """Width added to the label block's black padding, split over its two sides."""
        if self.label_pad is None:
            return 0.0
        cx0, _, cx1, _ = self.clip
        ix0 = self.slice_x[0]
        share = (ix0 - cx0) / (cx1 - cx0)
        wanted = self.target.width * share - (ix0 - cx0) * self.uniform_scale
        return max(0.0, min(wanted, self._interior_target_width()))

    def slice_segments(self) -> list[tuple[tuple[float, float, float, float], Box, bool]]:
        """``(source clip, target box, keep_proportion)`` for each draw.

        One entry for an ordinary placement, three for ``slice_x``. The outer
        two have exactly their source aspect; only the middle one is stretched.
        """
        if self.slice_x is None:
            return [(self.clip, self.target, self.fit not in ("stretch", "stretch_x"))]
        segments = []
        ix0, ix1 = self.slice_x
        for sx0, sx1, tx0, tx1, stretched_x in self._x_bands():
            if self.unit_word is not None and sx0 == ix1:
                mid = (ix0 + ix1) / 2.0
                sx0, sx1 = mid - 1.0, mid + 1.0  # blank where the unit was, from the interior's middle
            for sy0, sy1, ty0, ty1, pad in self._y_bands():
                segments.append(((sx0, sy0, sx1, sy1), Box(tx0, ty0, tx1, ty1), not (stretched_x or pad)))
        if self.label_word is not None:
            segments.append((self.label_word, self.label_word_box(), True))
        if self.unit_word is not None:
            segments.append((self.unit_word, self.unit_word_box(), True))
        return segments

    @property
    def text_scale(self) -> float:
        """Scale of the table's words: the tile's own scale times the table's text growth."""
        return self.uniform_scale * (self.text_growth or 1.0)

    def text_growth_limit(self) -> float:
        """Growth of this cell's words: the label keeps its template share of the block, no taller than the row share."""
        wx0, wy0, wx1, wy1 = self.label_word
        s = self.uniform_scale
        cap = self.label_word_row_share * self.target.height
        return max(1.0, min(self.label_block_box().width / ((self.slice_x[0] - self.clip[0]) * s), cap / ((wy1 - wy0) * s)))

    def unit_word_box(self) -> Box:
        """Target of the unit word at the text scale, right-aligned where it was, centred on its own row centre."""
        ux0, uy0, ux1, uy1 = self.unit_word
        cap = self.label_word_row_share * self.target.height / ((uy1 - uy0) * self.uniform_scale)
        k = self.uniform_scale * max(1.0, min(self.text_growth or 1.0, cap))
        right = self.map_source_x(ux1)
        cy = self.map_source_y((uy0 + uy1) / 2.0)
        w, h = (ux1 - ux0) * k, (uy1 - uy0) * k
        return Box(right - w, cy - h / 2.0, right, cy + h / 2.0)

    def label_block_box(self) -> Box:
        """Where the black label block lands: from the cell's left edge to the interior."""
        return Box(self.target.x0, self.target.y0, self.interior_box().x0, self.target.y1)

    def label_word_box(self) -> Box:
        """Target of the grown label word: its share of the block's width, capped by the row."""
        wx0, wy0, wx1, wy1 = self.label_word
        cx0 = self.clip[0]
        block = self.label_block_box()
        if self.text_growth is not None:
            k = self.text_scale
        else:
            by_width = block.width / (self.slice_x[0] - cx0)
            by_height = self.label_word_row_share * self.target.height / (wy1 - wy0)
            k = max(self.uniform_scale, min(by_width, by_height))
        w, h = (wx1 - wx0) * k, (wy1 - wy0) * k
        # Centred on the block across, and on the word's own mapped centre down,
        # so the word stays where the template put it in its row.
        cx = (block.x0 + block.x1) / 2.0
        cy = self.map_source_y((wy0 + wy1) / 2.0)
        return Box(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def interior_box(self) -> Box:
        """The stretched empty interior of a ``slice_x`` cell, over the cell's full height.

        Named instead of read as ``slice_segments()[1]``: with ``slice_y`` a cell
        is fifteen draws, and position 1 is no longer the interior.
        """
        ix0, ix1 = self.slice_x
        _, _, tx0, tx1, _ = next(band for band in self._x_bands() if band[0] == ix0 and band[1] == ix1)
        return Box(tx0, self.target.y0, tx1, self.target.y1)

    def _x_bands(self) -> list[tuple[float, float, float, float, bool]]:
        """``(source x0, source x1, target x0, target x1, is_stretched)`` left to right."""
        cx0, _, cx1, _ = self.clip
        ix0, ix1 = self.slice_x
        scale = self.uniform_scale
        right = self.target.x1 - (cx1 - ix1) * scale
        if self.label_pad is None:
            label = [(cx0, ix0, False)]
            extra = 0.0
        elif self.label_word is None:
            l0, l1, r0, r1 = self.label_pad
            label = [(cx0, l0, False), (l0, l1, True), (l1, r0, False), (r0, r1, True), (r1, ix0, False)]
            extra = self._label_extra() / 2.0
        else:
            # The block is drawn from its black padding alone (it covers where
            # the word was); the word is drawn over it by label_word_box.
            l0, l1, r0, r1 = self.label_pad
            label = [(cx0, l0, False), (l0, l1, True), (r1, ix0, False)]
            extra = (r1 - l1) * scale + self._label_extra()
        bands, x = [], self.target.x0
        for sx0, sx1, pad in label:
            if sx1 <= sx0:
                continue
            width = (sx1 - sx0) * scale + (extra if pad else 0.0)
            bands.append((sx0, sx1, x, x + width, pad))
            x += width
        bands.append((ix0, ix1, x, right, True))
        if self.unit_word is None:
            bands.append((ix1, cx1, right, self.target.x1, False))
        else:
            # The unit's columns are drawn blank (from the interior) and the
            # unit itself over them by unit_word_box.
            u1 = self.unit_word[2]
            edge = right + (u1 - ix1) * scale
            bands.append((ix1, u1, right, edge, True))
            bands.append((u1, cx1, edge, self.target.x1, False))
        return bands

    def map_source_x(self, x: float) -> float:
        """Where a source-space x lands in the target (piecewise linear)."""
        cx0, _, cx1, _ = self.clip
        if self.slice_x is None:
            width = cx1 - cx0
            return self.target.x0 + (x - cx0) * (self.target.width / width if width else 0.0)
        for sx0, sx1, tx0, tx1, _ in self._x_bands():
            if x <= sx1 or sx1 >= cx1:
                span = sx1 - sx0
                return tx0 + (x - sx0) * ((tx1 - tx0) / span if span else 0.0)
        return self.target.x1  # pragma: no cover - loop always returns

    def map_source_y(self, y: float) -> float:
        _, cy0, _, cy1 = self.clip
        if self.slice_y is None:
            height = cy1 - cy0
            return self.target.y0 + (y - cy0) * (self.target.height / height if height else 0.0)
        for sy0, sy1, ty0, ty1, _ in self._y_bands():
            if y <= sy1 or sy1 >= cy1:
                span = sy1 - sy0
                return ty0 + (y - sy0) * ((ty1 - ty0) / span if span else 0.0)
        return self.target.y1  # pragma: no cover - loop always returns

    def map_source_rect(self, rect: Any) -> Box:
        x0, y0, x1, y1 = (float(v) for v in rect)
        return Box(self.map_source_x(x0), self.map_source_y(y0), self.map_source_x(x1), self.map_source_y(y1))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "clip_pt": [round(v, 3) for v in self.clip],
            "target_mm": self.target.as_mm(),
            "fit": self.fit,
            "source_page": self.source_page,
        }
        # Only when present, so every plan without a sliced tile keeps the
        # exact dictionary - and therefore the fingerprint - it had before.
        if self.slice_x is not None:
            out["slice_x_pt"] = [round(v, 3) for v in self.slice_x]
        if self.slice_y is not None:
            out["slice_y_pt"] = [round(v, 3) for v in self.slice_y]
            out["glyph_scale"] = round(self.glyph_scale, 6)
        if self.label_pad is not None:
            out["label_pad_pt"] = [round(v, 3) for v in self.label_pad]
        if self.label_word is not None:
            out["label_word_pt"] = [round(v, 3) for v in self.label_word]
            out["label_word_row_share"] = self.label_word_row_share
        if self.unit_word is not None:
            out["unit_word_pt"] = [round(v, 3) for v in self.unit_word]
        if self.text_growth is not None:
            out["text_growth"] = round(self.text_growth, 6)
        return out


@dataclass
class BarcodePlacement:
    bars: Box
    rotation_deg: int
    x_dimension_mm: float
    bar_length_mm: float
    magnification: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bars_mm": self.bars.as_mm(),
            "rotation_deg": self.rotation_deg,
            "physical_rotation_deg": 270 if self.rotation_deg == 90 else 0,
            "x_dimension_mm": round(self.x_dimension_mm, 4),
            "bar_length_mm": round(self.bar_length_mm, 3),
            "magnification": round(self.magnification, 4),
        }


@dataclass
class FacePlan:
    """A complete, renderable layout for one printed face."""

    kind: FaceKind
    family: str
    panel: Box
    content: Box
    scale: float
    inset_mm: float
    statics: list[StaticPlacement] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    barcode: BarcodePlacement | None = None
    notes: list[str] = field(default_factory=list)
    safe_margin_mm: float = 2.0

    @property
    def aspect(self) -> float:
        return self.panel.aspect

    @property
    def safe_area(self) -> Box:
        """The hard printability limit, distinct from the design box.

        ``content`` is where the engine chooses to place things; ``safe_area``
        is the edge past which ink must not go.
        """
        return self.panel.inset_by(self.safe_margin_mm * PT_PER_MM)

    def occupied_boxes(self) -> list[tuple[str, Box]]:
        """Every box that must stay inside the content area and not collide."""
        boxes: list[tuple[str, Box]] = [(s.name, s.target) for s in self.statics]
        if self.barcode is not None:
            boxes.append(("barcode", self.barcode.bars))
        for name in ("lineart", "names", "cn_name", "qty", "es_name"):
            cfg = self.fields.get(name)
            if isinstance(cfg, dict) and "rect" in cfg:
                boxes.append((name, Box(*cfg["rect"])))
        return boxes

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family,
            "panel_mm": self.panel.as_mm(),
            "panel_size_mm": [round(mm(self.panel.width), 2), round(mm(self.panel.height), 2)],
            "aspect": round(self.aspect, 4),
            "content_mm": self.content.as_mm(),
            "scale": round(self.scale, 5),
            "inset_mm": round(self.inset_mm, 3),
            "statics": [s.to_dict() for s in self.statics],
            "barcode": self.barcode.to_dict() if self.barcode else None,
            "notes": list(self.notes),
        }


@dataclass
class LayoutPlan:
    """One SKU's front+side pair, plus how it was chosen."""

    item_no: str
    carton_mm: tuple[float, float, float]
    front: FacePlan
    side: FacePlan
    page_size_pt: tuple[float, float]
    gutter_pt: float
    score: float = 0.0
    runner_up_score: float | None = None
    selection: str = "auto_selected"
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[dict[str, Any]]:
        return [i for i in self.issues if i.get("severity") == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_no": self.item_no,
            "carton_mm": {
                "length": self.carton_mm[0],
                "width": self.carton_mm[1],
                "height": self.carton_mm[2],
            },
            "page_size_mm": [round(mm(self.page_size_pt[0]), 2), round(mm(self.page_size_pt[1]), 2)],
            "gutter_mm": round(mm(self.gutter_pt), 2),
            "front": self.front.to_dict(),
            "side": self.side.to_dict(),
            "score": round(self.score, 4),
            "runner_up_score": None if self.runner_up_score is None else round(self.runner_up_score, 4),
            "selection": self.selection,
            "issues": self.issues,
        }

    def fingerprint(self) -> str:
        """Stable hash of the geometry, so a revision can be audited."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LayoutRevision:
    """Versioned record of every plan produced in one render."""

    policy_id: str
    schema_version: str
    template_id: str
    plans: list[LayoutPlan] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        plans = [plan.to_dict() for plan in self.plans]
        return {
            "policy_id": self.policy_id,
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "plan_count": len(plans),
            "plans": plans,
            "plan_fingerprints": {plan.item_no: plan.fingerprint() for plan in self.plans},
            "issues": self.issues,
        }


__all__ = [
    "BarcodePlacement",
    "Box",
    "FaceKind",
    "FacePlan",
    "FitMode",
    "LayoutPlan",
    "LayoutRevision",
    "MM_PER_PT",
    "PT_PER_MM",
    "StaticPlacement",
    "mm",
    "pt",
    "register_source_slice",
    "source_slice",
]
