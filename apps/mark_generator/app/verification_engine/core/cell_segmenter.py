from __future__ import annotations

from statistics import median

from dataclasses import dataclass

from .models import BarcodeEvidence, Cell, PageData, TextSpan
from .normalize import normalize_item_number


KMT_PAGE_WIDTH = 3826.771728515625
KMT_MARGIN_LEFT = 188.0
KMT_MARGIN_TOP = 57.0
KMT_PAIR_WIDTH = 1134.0
KMT_PANEL_HEIGHT = 468.0
KMT_PAIR_PITCH_X = 1172.0
KMT_ROW_PITCH_Y = 511.0
KMT_COLUMNS = 3
KMT_ROWS = 8
KMT_ITEM_BAND_BOTTOM = 220.0


@dataclass
class Anchor:
    text: str
    bbox: tuple[float, float, float, float]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


def segment_page(page: PageData) -> list[Cell]:
    template_cells = _cells_from_kmt_grid(page)
    if template_cells:
        _attach_barcodes(template_cells, page.barcodes)
        return template_cells

    if page.visual_rects:
        cells = _cells_from_visual_rects(page)
        if cells:
            return cells

    if page.has_text_layer and page.spans:
        cells = _segment_text_grid(page)
        if cells:
            _attach_barcodes(cells, page.barcodes)
            return cells

    if page.barcodes:
        return _cells_from_barcodes(page)

    return [
        Cell(
            cell_id=f"{_file_tag(page.file_path)}:p{page.page + 1}:whole",
            file_path=page.file_path,
            page=page.page,
            bbox=(0, 0, page.width, page.height),
            spans=page.spans,
            barcodes=page.barcodes,
            segmentation_method="whole_page",
            needs_ai=not page.has_text_layer,
        )
    ]


def _cells_from_kmt_grid(page: PageData) -> list[Cell]:
    if abs(page.width - KMT_PAGE_WIDTH) > 2.0:
        return []
    expected_heights = {
        KMT_MARGIN_TOP
        + rows * KMT_ROW_PITCH_Y
        - (KMT_ROW_PITCH_Y - KMT_PANEL_HEIGHT)
        + 149.97
        for rows in range(1, KMT_ROWS + 1)
    }
    if min(abs(page.height - height) for height in expected_heights) > 2.0:
        return []

    slots: list[Cell] = []
    for column in range(KMT_COLUMNS):
        x0 = KMT_MARGIN_LEFT + column * KMT_PAIR_PITCH_X
        x1 = x0 + KMT_PAIR_WIDTH
        for row in range(KMT_ROWS):
            y0 = KMT_MARGIN_TOP + row * KMT_ROW_PITCH_Y
            y1 = y0 + KMT_PANEL_HEIGHT
            if y1 > page.height + 2.0:
                continue
            bbox = (x0, y0, x1, y1)
            spans = [span for span in page.spans if _inside_center(span, bbox)]
            header_spans = [
                span
                for span in spans
                if span.y0 <= y0 + KMT_ITEM_BAND_BOTTOM
                and float(span.size or 0) >= 40.0
            ]
            slot_barcodes = [
                barcode
                for barcode in page.barcodes
                if _barcode_inside(barcode, bbox)
            ]
            if not header_spans and not slot_barcodes:
                continue
            slots.append(
                Cell(
                    cell_id=f"{_file_tag(page.file_path)}:p{page.page + 1}:kmt-r{row + 1}c{column + 1}",
                    file_path=page.file_path,
                    page=page.page,
                    bbox=bbox,
                    spans=spans,
                    segmentation_method="kmt_style_v1_geometry",
                )
            )
    return slots


def _segment_text_grid(page: PageData) -> list[Cell]:
    anchors = _reconstruct_anchors(page.spans)
    # A single-SKU page is still a valid text-anchored cell. Falling back to
    # one cell per barcode here discards the surrounding item number, names
    # and quantity, especially when a front/side pair shares one wide page.
    if not anchors:
        return []

    row_groups = [
        _collapse_same_item_anchors(row)
        for row in _cluster_anchors_by_y(anchors)
    ]
    row_bounds = _row_bounds_from_anchor_groups(row_groups, page.height)

    # If the same item number appears twice in one row (front + side on one plane),
    # keep the whole row as a single cell. If there are multiple distinct anchors,
    # split that row into adaptive columns.
    cells: list[Cell] = []
    for ri, row in enumerate(row_groups):
        unique_items = sorted({a.text for a in row})
        if len(unique_items) <= 1:
            col_bounds = [(0, page.width)]
        else:
            col_centers = [a.cx for a in row]
            col_centers = [sum(c) / len(c) for c in _cluster_axis(col_centers, tolerance=max(45, page.width * 0.04))]
            col_bounds = _bounds_from_centers(col_centers, 0, page.width)

        for ci, cb in enumerate(col_bounds):
            rb = row_bounds[ri]
            bbox = (cb[0], rb[0], cb[1], rb[1])
            spans = [s for s in page.spans if _inside_center(s, bbox)]
            if len(spans) < 2:
                continue
            cells.append(
                Cell(
                    cell_id=f"{_file_tag(page.file_path)}:p{page.page + 1}:r{ri + 1}c{ci + 1}",
                    file_path=page.file_path,
                    page=page.page,
                    bbox=bbox,
                    spans=spans,
                    segmentation_method=f"text_anchor_rows_{len(row_groups)}",
                )
            )

    if len(cells) < 1:
        return []
    return cells


def _reconstruct_anchors(spans: list[TextSpan]) -> list[Anchor]:
    candidates = [s for s in spans if _anchor_fragment(s.text)]
    # Shipping marks often expose the human-readable EAN-13 digits as several
    # short vertical PDF spans (for example ``047912`` and ``447655``).  Those
    # fragments happen to look like item numbers, but they are much smaller
    # than the repeated front/side SKU header.  When display-size anchors are
    # present, use them as the segmentation authority instead of letting
    # barcode fragments manufacture extra rows and merge unrelated marks.
    if candidates:
        max_size = max(float(span.size or 0) for span in candidates)
        display_threshold = max(14.0, max_size * 0.45)
        display_candidates = [
            span
            for span in candidates
            if float(span.size or 0) >= display_threshold
            and _span_contains_anchor(span.text)
        ]
        if display_candidates:
            candidates = display_candidates
    candidates.sort(key=lambda s: (s.y0, s.x0))
    rows: list[list[TextSpan]] = []
    for span in candidates:
        placed = False
        for row in rows:
            if abs(row[0].y0 - span.y0) <= max(12, span.size * 0.3):
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    anchors: list[Anchor] = []
    for row in rows:
        row.sort(key=lambda s: s.x0)
        cur: list[TextSpan] = []
        for span in row:
            if not cur:
                cur = [span]
                continue
            gap = span.x0 - cur[-1].x1
            threshold = max(8, min(28, span.size * 0.2))
            if gap <= threshold:
                cur.append(span)
            else:
                _append_anchor(anchors, cur)
                cur = [span]
        _append_anchor(anchors, cur)
    return anchors


def _span_contains_anchor(text: str) -> bool:
    compact = normalize_item_number(text)
    return bool(_repeated_item_anchor(compact) or _looks_like_anchor(compact))


def _append_anchor(anchors: list[Anchor], group: list[TextSpan]) -> None:
    if not group:
        return
    text = normalize_item_number("".join(s.text for s in group))
    repeated = _repeated_item_anchor(text)
    if repeated:
        text = repeated
    if not _looks_like_anchor(text):
        return
    anchors.append(
        Anchor(
            text=text,
            bbox=(
                min(s.x0 for s in group),
                min(s.y0 for s in group),
                max(s.x1 for s in group),
                max(s.y1 for s in group),
            ),
        )
    )


def _repeated_item_anchor(text: str) -> str:
    """Collapse one PDF span containing the same front/side SKU twice."""

    if not text.isdigit() or len(text) % 2:
        return ""
    half = len(text) // 2
    if 3 <= half <= 8 and text[:half] == text[half:]:
        return text[:half]
    return ""


def _anchor_fragment(text: str) -> bool:
    import re

    compact = normalize_item_number(text)
    return bool(re.match(r"^[A-Z0-9-]{1,12}$", compact))


def _looks_like_anchor(text: str) -> bool:
    import re

    compact = normalize_item_number(text)
    return bool(re.match(r"^(?:\d{3,8}|[A-Z]{1,6}\d{2,8}|[A-Z0-9]{3,10}-[A-Z0-9]{1,3})$", compact))


def _large_text_threshold(spans: list[TextSpan]) -> float:
    sizes = sorted(s.size for s in spans if s.size)
    if not sizes:
        return 9999
    return sizes[int(len(sizes) * 0.75)]


def _axis_tolerance(spans: list[TextSpan], axis: str) -> float:
    if not spans:
        return 25
    vals = sorted(s.y0 if axis == "y" else s.x0 for s in spans)
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b - a > 2]
    if not diffs:
        return 35 if axis == "y" else 45
    base = median(diffs)
    return max(18, min(80 if axis == "x" else 55, base * 0.45))


def _collapse_same_item_anchors(row: list[Anchor]) -> list[Anchor]:
    pending = sorted(row, key=lambda anchor: anchor.x0)
    collapsed: list[Anchor] = []
    index = 0
    while index < len(pending):
        group = [pending[index]]
        index += 1
        while index < len(pending):
            previous = group[-1]
            candidate = pending[index]
            width = max(previous.x1 - previous.x0, candidate.x1 - candidate.x0)
            if candidate.text != group[0].text or candidate.x0 - previous.x1 > max(80, width * 0.35):
                break
            group.append(candidate)
            index += 1
        collapsed.append(
            Anchor(
                text=group[0].text,
                bbox=(
                    min(anchor.x0 for anchor in group),
                    min(anchor.y0 for anchor in group),
                    max(anchor.x1 for anchor in group),
                    max(anchor.y1 for anchor in group),
                ),
            )
        )
    return collapsed


def _cluster_anchors_by_y(anchors: list[Anchor]) -> list[list[Anchor]]:
    rows: list[list[Anchor]] = []
    for anchor in sorted(anchors, key=lambda a: (a.y0, a.x0)):
        placed = False
        for row in rows:
            if abs(row[0].y0 - anchor.y0) <= max(25, (anchor.y1 - anchor.y0) * 0.3):
                row.append(anchor)
                placed = True
                break
        if not placed:
            rows.append([anchor])
    for row in rows:
        row.sort(key=lambda a: a.x0)
    return rows


def _row_bounds_from_anchor_groups(
    rows: list[list[Anchor]],
    page_height: float,
) -> list[tuple[float, float]]:
    if not rows:
        return []
    ys = [min(a.y0 for a in row) for row in rows]
    heights = [max(a.y1 - a.y0 for a in row) for row in rows]
    top_margin = max(25, min(80, median(heights) * 0.28 if heights else 40))
    bounds: list[tuple[float, float]] = []
    for idx, y in enumerate(ys):
        start = 0 if idx == 0 else max(0, y - top_margin)
        if idx + 1 < len(ys):
            end = max(start + 10, ys[idx + 1] - top_margin)
        else:
            end = page_height
        bounds.append((start, end))
    return bounds


def _cluster_axis(values: list[float], tolerance: float) -> list[list[float]]:
    if not values:
        return []
    vals = sorted(values)
    clusters: list[list[float]] = [[vals[0]]]
    for val in vals[1:]:
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(val - center) <= tolerance:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return clusters


def _bounds_from_centers(centers: list[float], low: float, high: float) -> list[tuple[float, float]]:
    centers = sorted(centers)
    if len(centers) == 1:
        return [(low, high)]
    cuts = [(a + b) / 2 for a, b in zip(centers, centers[1:])]
    bounds = []
    start = low
    for cut in cuts:
        bounds.append((start, cut))
        start = cut
    bounds.append((start, high))
    return bounds


def _inside_center(span: TextSpan, bbox: tuple[float, float, float, float]) -> bool:
    cx = (span.x0 + span.x1) / 2
    cy = (span.y0 + span.y1) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _attach_barcodes(cells: list[Cell], barcodes: list[BarcodeEvidence]) -> None:
    for bc in barcodes:
        if bc.bbox is None:
            continue
        cx = (bc.bbox[0] + bc.bbox[2]) / 2
        cy = (bc.bbox[1] + bc.bbox[3]) / 2
        best = None
        for cell in cells:
            x0, y0, x1, y1 = cell.bbox
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                best = cell
                break
        if best is not None:
            best.barcodes.append(bc)


def _cells_from_barcodes(page: PageData) -> list[Cell]:
    cells: list[Cell] = []
    for idx, bc in enumerate(page.barcodes):
        bbox = _expand_bbox(bc.bbox, page.width, page.height) if bc.bbox else (0, 0, page.width, page.height)
        cells.append(
            Cell(
                cell_id=f"{_file_tag(page.file_path)}:p{page.page + 1}:bc{idx + 1}",
                file_path=page.file_path,
                page=page.page,
                bbox=bbox,
                spans=[],
                barcodes=[bc],
                segmentation_method="barcode_cells",
                needs_ai=True,
            )
        )
    return cells


def _cells_from_visual_rects(page: PageData) -> list[Cell]:
    cells: list[Cell] = []
    for idx, raw_rect in enumerate(page.visual_rects):
        bbox = _clamp_bbox(raw_rect, page.width, page.height)
        if bbox is None:
            continue
        spans = [span for span in page.spans if _inside_center(span, bbox)]
        barcodes = [barcode for barcode in page.barcodes if _barcode_inside(barcode, bbox)]
        if not spans and not barcodes:
            continue
        cells.append(
            Cell(
                cell_id=f"{_file_tag(page.file_path)}:p{page.page + 1}:vr{idx + 1}",
                file_path=page.file_path,
                page=page.page,
                bbox=bbox,
                spans=spans,
                barcodes=barcodes,
                segmentation_method="image_raster_rects",
                needs_ai=True,
            )
        )
    return cells


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = max(0.0, min(width, x0))
    y0 = max(0.0, min(height, y0))
    x1 = max(0.0, min(width, x1))
    y1 = max(0.0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _barcode_inside(
    barcode: BarcodeEvidence,
    bbox: tuple[float, float, float, float],
) -> bool:
    if barcode.bbox is None:
        return False
    x0, y0, x1, y1 = barcode.bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _expand_bbox(
    bbox: tuple[float, float, float, float] | None,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    if bbox is None:
        return (0, 0, width, height)
    x0, y0, x1, y1 = bbox
    pad_x = max((x1 - x0) * 2.5, width * 0.08)
    pad_y = max((y1 - y0) * 4.0, height * 0.08)
    return (max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y))


def _file_tag(path: str) -> str:
    from pathlib import Path

    return Path(path).stem
