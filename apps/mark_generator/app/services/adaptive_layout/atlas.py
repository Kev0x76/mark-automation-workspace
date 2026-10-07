"""Component atlas: the bridge between the frozen Kmt Style v1 assets and the
adaptive layout engine.

Two hard rules shape this module.

1. ``template.json`` and ``base.pdf`` are read-only. The project handoff states
   that template.json must not be converted into an adaptive scheme, so the
   adaptive policy lives in its own asset file and the template is only ever
   *read*. That also means the standard-family geometry cannot drift from the
   template: it is derived from it at load time.

2. Detail is never redrawn. Every static element (logo, icon row, table cells,
   barcode brackets, black bar) is addressed as a clip rectangle into base.pdf
   and re-stamped as vector, so the adaptive output keeps byte-identical artwork
   to the current standard mark. An earlier attempt that simplified and redrew
   these elements was rejected, and an attempt that scaled whole panels as
   images produced collage-looking output; component clipping avoids both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .model import Box, register_source_slice


ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "templates" / "kmt_style_v1"


class AdaptiveAssetError(RuntimeError):
    pass


#: Raster resolution of the interior scan. At 6x one column is 1/6 pt, finer
#: than the 0.2-0.3 pt hairline borders of the cells, so a border cannot fall
#: between two sampled columns. It is the zoom of the 2026-09-15 manual ink scan
#: this measurement replaces, so the two can be compared column for column.
SLICE_SCAN_ZOOM = 6.0

#: Anything this far from white is ink for the scan. Deliberately near-white,
#: not the audit's mid-grey threshold: an anti-aliased fringe of a label or unit
#: glyph must count as ink, so it can never end up inside the stretched slice.
SLICE_SCAN_INK_BELOW = 250

#: A row belongs to the frame's top or bottom rule when ink covers this share of
#: the tile width. Not 1.0 because a clip may end a few columns short of or past
#: the hairline border, so a rule row is not guaranteed to be inked edge to edge.
SLICE_SCAN_RULE_ROW_SHARE = 0.95


@lru_cache(maxsize=64)
def measure_stretchable_interior(
    base_pdf: str, source_page: int, clip: tuple[float, ...]
) -> tuple[float, float]:
    """Source-space x interval of a cell tile that holds nothing but its rules.

    A column is stretchable when every inked pixel in it lies on a rule row -
    the frame's full-width top and bottom lines. The interior is the longest
    contiguous run of such columns, shrunk by one scan column on each side so
    the label's or unit's edge is never part of what gets stretched.

    On kmt_style_v1 this reproduces the lead engineer's 2026-09-15 manual scan
    to within 0.3 pt (cell_qty 95.0-181.1, cell_meas 94.6-204.0, cell_name
    319.4-451.6, cell_nw / cell_gw 319.4-412.8).
    """
    import fitz
    import numpy as np

    rect = fitz.Rect(*clip)
    source = fitz.open(base_pdf)
    probe = fitz.open()
    try:
        page = probe.new_page(width=max(rect.width, 1.0), height=max(rect.height, 1.0))
        page.show_pdf_page(page.rect, source, source_page, clip=rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(SLICE_SCAN_ZOOM, SLICE_SCAN_ZOOM), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        probe.close()
        source.close()

    ink = image[:, :, :3].min(axis=2) < SLICE_SCAN_INK_BELOW
    rule_rows = ink.mean(axis=1) >= SLICE_SCAN_RULE_ROW_SHARE
    if not rule_rows.any():
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no frame rules to stretch")
    stretchable = ~(ink & ~rule_rows[:, None]).any(axis=0)

    best = (0, 0)
    start = None
    for index, free in enumerate(list(stretchable) + [False]):
        if free and start is None:
            start = index
        elif not free and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    first, stop = best[0] + 1, best[1] - 1
    if stop - first < 2:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no empty interior to stretch")
    return (rect.x0 + first / SLICE_SCAN_ZOOM, rect.x0 + stop / SLICE_SCAN_ZOOM)


def measure_padding_rows(
    base_pdf: str, source_page: int, clip: tuple[float, ...]
) -> tuple[float, float, float, float]:
    """Source-space rows above and below a cell tile's glyphs that hold only its fill.

    Every non-rule row of a table cell is either a *fill* row - the black label
    block, the white interior and the hairline, identical from row to row - or a
    row that crosses a glyph (the white label word, the unit). The fill rows
    before the first glyph row and after the last one are the padding that may
    grow when a row is drawn taller; the glyph band and the full-width rules
    never are. Returned as ``(t0, t1, b0, b1)``, shrunk by one scan row on each
    side so no glyph edge or rule is ever part of what stretches.
    """
    import fitz
    import numpy as np

    rect = fitz.Rect(*clip)
    source = fitz.open(base_pdf)
    probe = fitz.open()
    try:
        page = probe.new_page(width=max(rect.width, 1.0), height=max(rect.height, 1.0))
        page.show_pdf_page(page.rect, source, source_page, clip=rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(SLICE_SCAN_ZOOM, SLICE_SCAN_ZOOM), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        probe.close()
        source.close()

    ink = image[:, :, :3].min(axis=2) < SLICE_SCAN_INK_BELOW
    rule_rows = ink.mean(axis=1) >= SLICE_SCAN_RULE_ROW_SHARE
    inner = [i for i in range(ink.shape[0]) if not rule_rows[i]]
    if len(inner) < 4:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no rows between its rules")
    # The fill signature is the commonest non-rule row; a row differing from it
    # in at most 0.5% of its columns is anti-aliasing, not a glyph.
    rows = {i: ink[i].tobytes() for i in inner}
    counts: dict[bytes, int] = {}
    for sig in rows.values():
        counts[sig] = counts.get(sig, 0) + 1
    fill = np.frombuffer(max(counts, key=counts.get), dtype=bool)
    tolerance = max(1, int(ink.shape[1] * 0.005))
    glyph = [i for i in inner if int((ink[i] != fill).sum()) > tolerance]
    if not glyph:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no glyph rows")
    first_inner, last_inner = inner[0], inner[-1]
    top = (first_inner + 1, glyph[0] - 1)
    bottom = (glyph[-1] + 1, last_inner - 1)

    def to_pt(index: int) -> float:
        return rect.y0 + index / SLICE_SCAN_ZOOM

    t0, t1 = (to_pt(top[0]), to_pt(top[1])) if top[1] - top[0] >= 2 else (to_pt(glyph[0]), to_pt(glyph[0]))
    b0, b1 = (to_pt(bottom[0]), to_pt(bottom[1])) if bottom[1] - bottom[0] >= 2 else (to_pt(glyph[-1]), to_pt(glyph[-1]))
    if t1 <= t0 and b1 <= b0:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no padding rows to grow")
    return (t0, t1, b0, b1)


def measure_label_padding(
    base_pdf: str, source_page: int, clip: tuple[float, ...], interior_x0: float
) -> tuple[float, float, float, float]:
    """Source-space columns left and right of a cell's label word that hold only black.

    Every side cell starts with a black label block (``[clip.x0, interior_x0]``)
    carrying a white word (Q.TY, MEAS, NAME, N.W., G.W.). In base.pdf that block
    is a third of the cell's width (0.335-0.343), as on the factory's printed
    boxes (0.33-0.37, owner photos 2026-09-29); widening only the white interior
    shrank it to about a tenth. The columns between the block's edge and the
    word hold nothing but black between the rules, so they may widen while the
    word keeps its shape. Returned as ``(l0, l1, r0, r1)``, shrunk by one scan
    column on each side so no glyph edge is ever part of what stretches.
    """
    import fitz
    import numpy as np

    rect = fitz.Rect(*clip)
    source = fitz.open(base_pdf)
    probe = fitz.open()
    try:
        page = probe.new_page(width=max(rect.width, 1.0), height=max(rect.height, 1.0))
        page.show_pdf_page(page.rect, source, source_page, clip=rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(SLICE_SCAN_ZOOM, SLICE_SCAN_ZOOM), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        probe.close()
        source.close()

    ink = image[:, :, :3].min(axis=2) < SLICE_SCAN_INK_BELOW
    rule_rows = ink.mean(axis=1) >= SLICE_SCAN_RULE_ROW_SHARE
    inner = ink[~rule_rows]
    stop = min(int(round((interior_x0 - rect.x0) * SLICE_SCAN_ZOOM)), inner.shape[1])
    solid = inner[:, :stop].all(axis=0)
    # The block's own edges are anti-aliased against the white interior, so the
    # padding is the solid run at each end of the block, and the word is what
    # lies between the two runs.
    solid_cols = [i for i in range(stop) if solid[i]]
    if not solid_cols:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no black label block")
    first, last = solid_cols[0], solid_cols[-1]
    word = [i for i in range(first, last + 1) if not solid[i]]
    if not word:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no label word in its black block")
    left = list(range(first, word[0]))
    right = list(range(word[-1] + 1, last + 1))
    if len(left) < 3 or len(right) < 3:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no black padding beside its label word")

    def to_pt(index: int) -> float:
        return rect.x0 + index / SLICE_SCAN_ZOOM

    return (to_pt(left[0] + 1), to_pt(left[-1] - 1), to_pt(right[0] + 1), to_pt(right[-1] - 1))


def measure_label_word(
    base_pdf: str, source_page: int, clip: tuple[float, ...], pad: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Source-space box of a cell's label word (white on the black block), between its padding.

    x spans the columns between the two black padding runs of
    :func:`measure_label_padding`; y spans the non-rule rows in which those
    columns hold any white, grown by one scan row so no glyph edge is cut. In
    base.pdf the word is 0.55-0.78 of the block's width and 0.32-0.38 of the
    row's height - as on the factory's boxes (owner photos 2026-09-29).
    """
    import fitz
    import numpy as np

    rect = fitz.Rect(*clip)
    source = fitz.open(base_pdf)
    probe = fitz.open()
    try:
        page = probe.new_page(width=max(rect.width, 1.0), height=max(rect.height, 1.0))
        page.show_pdf_page(page.rect, source, source_page, clip=rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(SLICE_SCAN_ZOOM, SLICE_SCAN_ZOOM), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        probe.close()
        source.close()

    ink = image[:, :, :3].min(axis=2) < SLICE_SCAN_INK_BELOW
    rule_rows = ink.mean(axis=1) >= SLICE_SCAN_RULE_ROW_SHARE
    c0 = int(round((pad[1] - rect.x0) * SLICE_SCAN_ZOOM))
    c1 = int(round((pad[2] - rect.x0) * SLICE_SCAN_ZOOM))
    rows = [i for i in np.where((~ink[:, c0:c1]).any(axis=1))[0] if not rule_rows[i]]
    if not rows:
        raise AdaptiveAssetError(f"tile {clip} on page {source_page} has no label word rows")

    def to_pt(index: int) -> float:
        return float(rect.y0 + int(index) / SLICE_SCAN_ZOOM)

    return (pad[1], to_pt(max(rows[0] - 1, 0)), pad[2], to_pt(min(rows[-1] + 2, ink.shape[0])))


def measure_unit_word(
    base_pdf: str, source_page: int, clip: tuple[float, ...], interior_x1: float
) -> tuple[float, float, float, float] | None:
    """Source-space box of a cell's unit word (PCS / cm / kg) right of the interior, or None.

    Between the interior's right end and the cell's right border sits the unit,
    black on white. Its columns are the ones with ink on some non-rule row but
    not on all of them (those are the border hairline); its rows are the
    non-rule rows with ink in those columns. Grown by one scan step so no glyph
    edge is cut. NAME has no unit and answers None.
    """
    import fitz
    import numpy as np

    rect = fitz.Rect(*clip)
    source = fitz.open(base_pdf)
    probe = fitz.open()
    try:
        page = probe.new_page(width=max(rect.width, 1.0), height=max(rect.height, 1.0))
        page.show_pdf_page(page.rect, source, source_page, clip=rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(SLICE_SCAN_ZOOM, SLICE_SCAN_ZOOM), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        probe.close()
        source.close()

    ink = image[:, :, :3].min(axis=2) < SLICE_SCAN_INK_BELOW
    rule_rows = ink.mean(axis=1) >= SLICE_SCAN_RULE_ROW_SHARE
    inner = ink[~rule_rows]
    start = int(round((interior_x1 - rect.x0) * SLICE_SCAN_ZOOM))
    cols = [i for i in range(start, inner.shape[1]) if inner[:, i].any() and not inner[:, i].all()]
    if not cols:
        return None
    c0, c1 = cols[0], cols[-1] + 1
    rows = [i for i in np.where(ink[:, c0:c1].any(axis=1))[0] if not rule_rows[i]]

    def x_pt(index: int) -> float:
        return float(rect.x0 + int(index) / SLICE_SCAN_ZOOM)

    def y_pt(index: int) -> float:
        return float(rect.y0 + int(index) / SLICE_SCAN_ZOOM)

    return (x_pt(max(c0 - 1, start)), y_pt(max(rows[0] - 1, 0)), x_pt(min(c1 + 1, inner.shape[1])),
            y_pt(min(rows[-1] + 2, ink.shape[0])))


@lru_cache(maxsize=None)
def _cached_unit_word(
    base_pdf: str, source_page: int, clip: tuple[float, ...], interior_x1: float
) -> tuple[float, float, float, float] | None:
    return measure_unit_word(base_pdf, source_page, clip, interior_x1)


@lru_cache(maxsize=None)
def _cached_label_word(
    base_pdf: str, source_page: int, clip: tuple[float, ...], pad: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return measure_label_word(base_pdf, source_page, clip, pad)


@lru_cache(maxsize=None)
def _cached_label_padding(
    base_pdf: str, source_page: int, clip: tuple[float, ...], interior_x0: float
) -> tuple[float, float, float, float]:
    return measure_label_padding(base_pdf, source_page, clip, interior_x0)


@lru_cache(maxsize=None)
def _cached_padding_rows(base_pdf: str, source_page: int, clip: tuple[float, ...]) -> tuple[float, float, float, float]:
    return measure_padding_rows(base_pdf, source_page, clip)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    clip: tuple[float, float, float, float]
    fit: str
    source_page: int

    @property
    def aspect(self) -> float:
        width = self.clip[2] - self.clip[0]
        height = self.clip[3] - self.clip[1]
        return width / height if height else 0.0

    @property
    def size(self) -> tuple[float, float]:
        return (self.clip[2] - self.clip[0], self.clip[3] - self.clip[1])


@dataclass(frozen=True)
class TemplateFrame:
    """The template's own panel, expressed so it can be re-projected.

    ``content`` is the design's live area: the inset implied by the item-number
    bar, mirrored on all four sides. Every standard-family placement is stored
    as an offset into this box so re-projection onto a different face is a pure
    coordinate transform.
    """

    panel: Box
    content: Box
    front: dict[str, Any]
    side: dict[str, Any]
    fonts: dict[str, Any]
    item_no_bar_height_ratio: float
    item_no_fontsize_ratio: float
    item_no_hscale_by_digits: dict[int, float]
    digit_advance_em: float
    #: The template's own inner margin between the digits and the bar edge,
    #: expressed against the bar height so it survives being applied to a face
    #: of a different size.
    #:
    #: ``_bar_padding`` below is an absolute rule and correctly so - it mirrors
    #: what ``KmtTemplateRenderer._draw_item_no`` really does on the template's
    #: one fixed panel, where its 8 pt ceiling binds. Generalising that absolute
    #: formula to arbitrary faces is what broke: a face twice the size got the
    #: same 8 pt margin instead of twice it, so the mark did not scale.
    item_no_bar_padding_ratio: float

    @property
    def content_height(self) -> float:
        return self.content.height

    @property
    def content_width(self) -> float:
        return self.content.width

    def scale_for(self, content: Box) -> float:
        """Uniform scale that fits the design into ``content`` on both axes.

        Face height governs on a normal or wide face - which is what the factory
        files show, product names holding a constant share of panel height. But
        a face narrower than the template's proportions must be governed by
        width, or the design overflows sideways. Taking the smaller ratio does
        both and returns exactly 1.0 at the template's own face.
        """
        return min(
            content.height / self.content_height,
            content.width / self.content_width,
        )


@dataclass(frozen=True)
class Atlas:
    policy: dict[str, Any]
    components: dict[str, dict[str, ComponentSpec]]
    frame: TemplateFrame
    template_id: str
    assets_dir: str = str(ASSETS_DIR)

    def slice_x(self, face: str, name: str) -> tuple[float, float]:
        """The stretchable empty interior of a table-cell tile, measured.

        Measured from base.pdf on first use and cached, never typed in: the
        breaks are a property of the artwork, and a hand-copied number would go
        stale the day the artwork is re-exported. See
        :func:`measure_stretchable_interior` for what counts as "empty".
        """
        spec = self.component(face, name)
        interior = measure_stretchable_interior(
            str(Path(self.assets_dir) / "base.pdf"), spec.source_page, tuple(spec.clip)
        )
        register_source_slice(spec.source_page, spec.clip, interior)
        return interior

    def slice_y(self, face: str, name: str) -> tuple[float, float, float, float]:
        """The padding rows above and below a table-cell tile's glyphs, measured.

        Measured from base.pdf like :meth:`slice_x`; see
        :func:`measure_padding_rows`.
        """
        spec = self.component(face, name)
        return _cached_padding_rows(str(Path(self.assets_dir) / "base.pdf"), spec.source_page, tuple(spec.clip))

    def label_pad(self, face: str, name: str) -> tuple[float, float, float, float]:
        """The black columns beside a table-cell tile's label word, measured.

        Measured from base.pdf like :meth:`slice_x`; see
        :func:`measure_label_padding`.
        """
        spec = self.component(face, name)
        return _cached_label_padding(
            str(Path(self.assets_dir) / "base.pdf"), spec.source_page, tuple(spec.clip), self.slice_x(face, name)[0]
        )

    def unit_word(self, face: str, name: str) -> tuple[float, float, float, float] | None:
        """The unit word's source box inside a table-cell tile, or None; see :func:`measure_unit_word`."""
        spec = self.component(face, name)
        return _cached_unit_word(
            str(Path(self.assets_dir) / "base.pdf"), spec.source_page, tuple(spec.clip), self.slice_x(face, name)[1]
        )

    def label_word(self, face: str, name: str) -> tuple[float, float, float, float]:
        """The label word's source box inside a table-cell tile; see :func:`measure_label_word`."""
        spec = self.component(face, name)
        return _cached_label_word(
            str(Path(self.assets_dir) / "base.pdf"), spec.source_page, tuple(spec.clip), self.label_pad(face, name)
        )

    def component(self, face: str, name: str) -> ComponentSpec:
        try:
            return self.components[face][name]
        except KeyError as exc:
            raise AdaptiveAssetError(f"Unknown component {face}/{name} in adaptive atlas.") from exc

    def has_component(self, face: str, name: str) -> bool:
        return name in self.components.get(face, {})

    @property
    def constraints(self) -> dict[str, Any]:
        return self.policy["constraints"]

    @property
    def families(self) -> dict[str, Any]:
        return self.policy["families"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self.policy["scoring"]


def _digit_advance_em(font_key: str, fonts_cfg: dict[str, Any]) -> float:
    """Advance width of one digit at fontsize 1, using the real font file.

    Falls back to Arial Bold's 0.5562 em only if no font can be loaded, so the
    item-number fitting rule degrades predictably instead of crashing.
    """
    import os

    import fitz

    spec = fonts_cfg.get(font_key, {})
    windir = os.environ.get("WINDIR", "C:/Windows")
    for candidate in spec.get("candidates", []):
        path = Path(str(candidate).replace("{WINDIR}", windir))
        if path.exists():
            try:
                return float(fitz.Font(fontfile=str(path)).text_length("0", 1.0))
            except Exception:  # pragma: no cover - defensive font loading
                continue
    fallback = spec.get("fallback")
    if fallback:
        try:
            return float(fitz.Font(fallback).text_length("0", 1.0))
        except Exception:  # pragma: no cover
            pass
    return 0.5562


def _bar_padding(bar_width: float) -> float:
    """Mirror KmtTemplateRenderer._draw_item_no's horizontal padding exactly."""
    return max(2.0, min(bar_width * 0.02, 8.0))


def _build_frame(template: dict[str, Any]) -> TemplateFrame:
    panel = Box(0.0, 0.0, float(template["panel"]["width"]), float(template["panel"]["height"]))
    bar = template["front"]["item_no"]["bar_rect"]
    inset_x = float(bar[0])
    inset_y = float(bar[1])
    content = Box(inset_x, inset_y, panel.x1 - inset_x, panel.y1 - inset_y)

    bar_height = float(bar[3]) - float(bar[1])
    fontsize = float(template["front"]["item_no"]["fontsize"])
    advance = _digit_advance_em(template["front"]["item_no"]["font"], template["fonts"])

    # The golden artwork condenses item numbers horizontally. That is part of
    # the current standard look, so it is measured and preserved rather than
    # "corrected": for each digit count, record the h-scale the template
    # itself produces. The adaptive engine then keeps that same condensation
    # instead of letting it get worse on a narrow face.
    bar_width = float(bar[2]) - float(bar[0])
    available = bar_width - 2 * _bar_padding(bar_width)
    hscale_by_digits: dict[int, float] = {}
    for digits in range(4, 13):
        natural = digits * advance * fontsize
        hscale_by_digits[digits] = min(1.0, available / natural) if natural else 1.0

    return TemplateFrame(
        panel=panel,
        content=content,
        front=template["front"],
        side=template["side"],
        fonts=template["fonts"],
        item_no_bar_height_ratio=bar_height / content.height,
        item_no_fontsize_ratio=fontsize / bar_height,
        item_no_hscale_by_digits=hscale_by_digits,
        digit_advance_em=advance,
        item_no_bar_padding_ratio=_bar_padding(bar_width) / bar_height,
    )


@lru_cache(maxsize=4)
def load_atlas(assets_dir: str | None = None) -> Atlas:
    directory = Path(assets_dir) if assets_dir else ASSETS_DIR
    template_path = directory / "template.json"
    policy_path = directory / "adaptive_layout.json"
    base_path = directory / "base.pdf"
    for path in (template_path, policy_path, base_path):
        if not path.is_file():
            raise AdaptiveAssetError(
                f"Adaptive layout asset missing: {path}. "
                "Run tools/build_template.py for the base template; adaptive_layout.json ships with the app."
            )

    template = json.loads(template_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    source_page = {"front": int(template["front"]["base_page"]), "side": int(template["side"]["base_page"])}
    components: dict[str, dict[str, ComponentSpec]] = {}
    for face, entries in policy["components"].items():
        components[face] = {
            name: ComponentSpec(
                name=name,
                clip=tuple(float(v) for v in entry["clip"]),  # type: ignore[arg-type]
                fit=entry.get("fit", "uniform"),
                source_page=source_page[face],
            )
            for name, entry in entries.items()
        }

    return Atlas(
        policy=policy,
        components=components,
        frame=_build_frame(template),
        template_id=str(template.get("template_id", "kmt_style_v1")),
        assets_dir=str(directory),
    )


def verify_atlas(assets_dir: str | None = None) -> list[dict[str, Any]]:
    """Check every component clip actually contains ink in base.pdf.

    A silently blank clip would drop a required element from production
    artwork, so this is a real gate rather than a convenience.
    """
    import fitz

    directory = Path(assets_dir) if assets_dir else ASSETS_DIR
    atlas = load_atlas(str(directory))
    source = fitz.open(directory / "base.pdf")
    findings: list[dict[str, Any]] = []
    try:
        for face, entries in atlas.components.items():
            for name, spec in entries.items():
                clip = fitz.Rect(*spec.clip)
                probe = fitz.open()
                page = probe.new_page(width=max(clip.width, 1.0), height=max(clip.height, 1.0))
                page.show_pdf_page(page.rect, source, spec.source_page, clip=clip)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                samples = pix.samples
                dark = sum(1 for i in range(0, len(samples), pix.n) if samples[i] < 240)
                coverage = dark / max(1, pix.width * pix.height)
                findings.append(
                    {
                        "face": face,
                        "component": name,
                        "clip_pt": [round(v, 2) for v in spec.clip],
                        "ink_coverage": round(coverage, 5),
                        "status": "ok" if coverage > 0.002 else "blank",
                    }
                )
                probe.close()
    finally:
        source.close()
    return findings


__all__ = [
    "ASSETS_DIR",
    "AdaptiveAssetError",
    "Atlas",
    "ComponentSpec",
    "TemplateFrame",
    "load_atlas",
    "measure_stretchable_interior",
    "verify_atlas",
]
