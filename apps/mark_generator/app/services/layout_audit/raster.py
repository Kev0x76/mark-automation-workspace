"""Panel-level measurements taken off the rendered raster.

Everything here works on ink, not on boxes, so a factory PDF and our own output
go through exactly the same code. That is the whole point: an L2 band is only
meaningful if the baseline and the candidate were measured the same way.

Sampling is pinned to the 2026-07-29 baseline run - ``zoom=4`` (288 DPI) with
4-pixel cells, i.e. a 0.353 mm cell - because the factory figures already in the
handoff (ink 32.5%, largest empty rectangle median 6.4%) were measured that way
and re-sampling would silently move them. L0 work that depends on print
resolution uses :data:`DECODE_DPI` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import fitz
import numpy as np


#: Render zoom for panel metrics. 4x72 = 288 DPI, matching the factory baseline.
METRIC_ZOOM = 4.0
#: Cell edge in pixels at :data:`METRIC_ZOOM`; 4 px = 0.353 mm.
METRIC_CELL_PX = 4
#: Resolution for anything whose truth depends on print detail (barcode decode,
#: glyph measurement). Two passes are required by L0, 300 and 600.
DECODE_DPI = 300


@dataclass(frozen=True)
class PanelMetrics:
    """The four L2 quantities, plus what they were measured on."""

    ink_coverage: float
    largest_empty_rect: float
    spread: float
    balance: float
    width_mm: float
    height_mm: float

    @property
    def aspect(self) -> float:
        return self.width_mm / self.height_mm if self.height_mm else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ink_coverage": round(self.ink_coverage, 5),
            "largest_empty_rect": round(self.largest_empty_rect, 5),
            "spread": round(self.spread, 5),
            "balance": round(self.balance, 5),
            "panel_mm": [round(self.width_mm, 2), round(self.height_mm, 2)],
            "aspect": round(self.aspect, 4),
        }


# --------------------------------------------------------------------------- #
# Ink
# --------------------------------------------------------------------------- #


def pixmap_array(pix: fitz.Pixmap) -> np.ndarray:
    """Pixmap samples as an ``(h, w, n)`` uint8 array."""
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def background_colour(image: np.ndarray, *, samples: int = 20000) -> tuple[int, int, int]:
    """The page's most common colour, sampled.

    Most common rather than "white" so kraft board, a tinted proof or an
    inverted panel all measure the same way.
    """
    flat = image[:, :, :3].reshape(-1, 3)
    step = max(1, len(flat) // samples)
    sample = flat[::step].astype(np.int32)
    packed = (sample[:, 0] << 16) | (sample[:, 1] << 8) | sample[:, 2]
    winner = int(np.bincount(packed).argmax())
    return ((winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF)


def ink_mask(
    image: np.ndarray,
    *,
    threshold: int = 90,
    background: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """``True`` where the pixel differs from the page's background colour.

    Pass ``background`` when masking many renders of the same page: detecting it
    per render is the expensive half of this call, and a layer rendered in
    isolation can have a *different* most-common colour from the composite,
    which would silently change what counts as ink between layers.
    """
    if background is None:
        background = background_colour(image)
    channels = image[:, :, :3]
    difference = np.abs(channels[:, :, 0].astype(np.int16) - background[0])
    difference += np.abs(channels[:, :, 1].astype(np.int16) - background[1])
    difference += np.abs(channels[:, :, 2].astype(np.int16) - background[2])
    return difference > threshold


def _cells(mask: np.ndarray, cell_px: int) -> np.ndarray:
    """Downsample to cells; a cell is inked if any pixel in it is."""
    height, width = mask.shape
    rows, cols = height // cell_px, width // cell_px
    if rows < 4 or cols < 4:
        return mask
    trimmed = mask[: rows * cell_px, : cols * cell_px]
    return trimmed.reshape(rows, cell_px, cols, cell_px).any(axis=(1, 3))


def ink_coverage(mask: np.ndarray, *, cell_px: int = METRIC_CELL_PX) -> float:
    """Share of the panel carrying ink, on the cell grid."""
    cells = _cells(mask, cell_px)
    return float(cells.mean())


def largest_empty_rect_ratio(mask: np.ndarray, *, cell_px: int = METRIC_CELL_PX) -> float:
    """Area of the largest all-empty axis-aligned rectangle, as a share of the panel.

    Same maximal-rectangle-in-histogram method as the 2026-07-29 baseline, so
    the factory medians recorded there stay directly comparable.
    """
    cells = _cells(mask, cell_px)
    rows, cols = cells.shape
    heights = np.zeros(cols, dtype=np.int32)
    best = 0
    for row in range(rows):
        heights = np.where(cells[row], 0, heights + 1)
        stack: list[tuple[int, int]] = []
        for col in range(cols + 1):
            current = int(heights[col]) if col < cols else 0
            start = col
            while stack and stack[-1][1] >= current:
                index, height = stack.pop()
                best = max(best, height * (col - index))
                start = index
            stack.append((start, current))
    return best / float(rows * cols) if rows and cols else 0.0


def uniformity(mask: np.ndarray, *, cell_px: int = METRIC_CELL_PX) -> tuple[float, float]:
    """``(spread, balance)`` - how the ink is distributed over the panel.

    ``spread`` is the ink's RMS distance from its own centroid divided by the
    RMS a uniformly inked panel of the same shape would have. Below 1 the ink is
    clumped; above 1 it is pushed out to the edges; at 1 it is laid out like a
    full sheet. ``balance`` is how far the ink centroid sits from the panel
    centre, as a share of the half-diagonal, so 0 is centred.

    This pairs with :func:`largest_empty_rect_ratio` rather than duplicating it:
    the empty rectangle finds one local hole, these two find global lopsidedness.

    The definition is this module's own. The 2026-09-02 note quotes a "U" from a
    scratchpad script that no longer exists, so its band cannot be reused as-is
    and has to be re-measured from the factory files.
    """
    cells = _cells(mask, cell_px)
    rows, cols = cells.shape
    if not cells.any():
        return 0.0, 0.0
    ys, xs = np.nonzero(cells)
    # Normalise both axes to [-1, 1] so the measure does not depend on aspect.
    nx = (xs + 0.5) / cols * 2.0 - 1.0
    ny = (ys + 0.5) / rows * 2.0 - 1.0
    cx, cy = float(nx.mean()), float(ny.mean())
    rms = float(np.sqrt(((nx - cx) ** 2 + (ny - cy) ** 2).mean()))
    # A uniform distribution over [-1,1]^2 has variance 1/3 per axis.
    reference = float(np.sqrt(2.0 / 3.0))
    balance = float(np.hypot(cx, cy) / np.sqrt(2.0))
    return rms / reference, balance


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` of the inked pixels, or ``None`` when empty."""
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def alignment_error(
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    width: float,
    height: float,
) -> float:
    """How far each element sits from its nearest alignment partner. Lower is better.

    Six axes, the ones a designer actually snaps to: left / centre / right and
    top / middle / bottom. For each element, take the smallest normalised
    distance to any other element on any axis, and average that over elements -
    the measure used in the graphic-layout generation literature, where it is
    called the alignment loss. Zero means every element shares an edge or a
    centre line with something.

    A tolerance-based "share of aligned elements" score was tried first and
    saturated at 1.0 on real marks: with six axes and a dozen elements, almost
    everything is within a millimetre of something. A continuous measure keeps
    reporting after a layout is already roughly aligned.

    Measured on ink bounding boxes rather than plan rectangles, so an element
    with a generous box but off-centre ink counts where the ink is.

    This is the counterweight to :func:`largest_empty_rect_ratio`. Spreading
    elements out to close a big empty rectangle necessarily breaks shared edges,
    which is the trade the 2026-07-29 spacing experiment ran into by eye.
    """
    if len(boxes) < 2 or width <= 0 or height <= 0:
        return 0.0

    axes = (
        (lambda b: b[0], width), (lambda b: (b[0] + b[2]) / 2.0, width),
        (lambda b: b[2], width), (lambda b: b[1], height),
        (lambda b: (b[1] + b[3]) / 2.0, height), (lambda b: b[3], height),
    )
    total = 0.0
    for index, box in enumerate(boxes):
        best = 1.0
        for extract, extent in axes:
            value = extract(box)
            for other_index, other in enumerate(boxes):
                if other_index == index:
                    continue
                best = min(best, abs(value - extract(other)) / extent)
        total += best
    return total / len(boxes)


# --------------------------------------------------------------------------- #
# Rendering a panel
# --------------------------------------------------------------------------- #


def render_panel(page: fitz.Page, clip: fitz.Rect, *, zoom: float = METRIC_ZOOM) -> np.ndarray:
    """Rasterise one face and return its ink mask."""
    pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return ink_mask(pixmap_array(pix))


def measure_panel(page: fitz.Page, clip: fitz.Rect, *, zoom: float = METRIC_ZOOM) -> PanelMetrics:
    """The four L2 quantities for one printed face."""
    mask = render_panel(page, clip, zoom=zoom)
    spread, balance = uniformity(mask)
    return PanelMetrics(
        ink_coverage=ink_coverage(mask),
        largest_empty_rect=largest_empty_rect_ratio(mask),
        spread=spread,
        balance=balance,
        width_mm=clip.width * 25.4 / 72.0,
        height_mm=clip.height * 25.4 / 72.0,
    )


def summarise(values: Iterable[float]) -> dict[str, float]:
    """Min / median / max / p10 / p90 of a sample, for band calibration."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {}
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 5),
        "p10": round(float(np.percentile(array, 10)), 5),
        "median": round(float(np.median(array)), 5),
        "p90": round(float(np.percentile(array, 90)), 5),
        "max": round(float(array.max()), 5),
    }


__all__ = [
    "DECODE_DPI",
    "METRIC_CELL_PX",
    "METRIC_ZOOM",
    "PanelMetrics",
    "alignment_error",
    "background_colour",
    "ink_coverage",
    "ink_mask",
    "largest_empty_rect_ratio",
    "mask_bbox",
    "measure_panel",
    "pixmap_array",
    "render_panel",
    "summarise",
    "uniformity",
]
