"""Run every layer of the audit over a rendered mark PDF and report.

The entry point takes the artwork plus the revision that produced it. The
revision declares the requested faces, elements, source tiles and transforms
- never evidence that the ask was honoured. Every measurement comes off the
raster.

Findings carry the severity the layer implies, and nothing here decides what a
caller does with them:

* ``error``   - L0/L1. The artwork disagrees with a standard or with itself.
* ``warning`` - L2. Outside the range measured on factory marks; a person should look.
* ``info``    - L2. Inside the range but away from where factory marks cluster.

Two checks need sharper pixels than the face sweep's zoom gives them - small
type whose +-1 px edges swamp the 2% fidelity tolerance, and barcodes whose
decode-back is specified at 300 and 600 DPI. Neither lowers the sweep's zoom
nor raises it for the whole face (a 1250 x 170 mm owner front is already 15 MP
per layer at zoom 3); both re-rasterise only the few square millimetres they
need, from the same bytes, with the same layer isolation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from ..adaptive_layout.atlas import ASSETS_DIR, Atlas, load_atlas
from ..adaptive_layout.model import FacePlan, LayoutPlan, LayoutRevision, mm
from .bands import Band, check_metrics, load_bands
from .barcodes import audit_barcode
from .checks import (
    check_aspect_locked_components,
    check_expected_elements,
    check_ink_inside_safe_area,
    check_text_fidelity,
    check_uniform_scale_groups,
    load_audit_policy,
)
from .layers import (
    FaceLayers,
    LayeredDocument,
    collisions,
    occlusion,
    render_faces,
    split_layer_name,
    unattributed,
)
from .raster import alignment_error, mask_bbox

AUDIT_SCHEMA_VERSION = "1.0"
DEFAULT_ZOOM = 3.0
#: Ceiling for any targeted re-rasterisation (text remeasure, barcode group,
#: decode crop). 16 MP bounds one RGB render at ~48 MB.
DEFAULT_MAX_RASTER_PIXELS = 16_000_000


def _face_clip(plan: LayoutPlan, kind: str) -> fitz.Rect:
    face = plan.front if kind == "front" else plan.side
    return fitz.Rect(face.panel.x0, face.panel.y0, face.panel.x1, face.panel.y1)


def describe_runs(face_fields: dict[str, Any], item: dict[str, Any], renderer: Any) -> dict[str, Any]:
    """What text each element was asked to set, and in what font.

    Only single-line runs. A wrapped, balanced name block has no single natural
    width to compare against, so measuring its distortion against one reference
    string would invent a number rather than report one.
    """
    runs: dict[str, dict[str, Any]] = {}

    item_no_cfg = face_fields.get("item_no")
    item_no = str(item.get("item_no") or "")
    if isinstance(item_no_cfg, dict) and item_no:
        runs["item_no"] = {
            "text": item_no,
            "fontsize": float(item_no_cfg.get("fontsize", 10.0)),
            "fontfile": renderer.font_files.get(item_no_cfg.get("font")),
        }

    qty_cfg = face_fields.get("qty")
    qty = item.get("qty")
    if isinstance(qty_cfg, dict) and qty not in (None, ""):
        runs["qty"] = {
            "text": str(qty),
            "fontsize": float(qty_cfg.get("fontsize", 10.0)),
            "fontfile": renderer.font_files.get(qty_cfg.get("font")),
        }
    return runs


def expected_elements(face: FacePlan, item: dict[str, Any]) -> set[str]:
    """Requested artwork, including empty barcode output after invalid input.

    Text blocks are required only when their input contains text. A configured
    line-art slot always draws either an image or a placeholder; checking its
    presence does not approve a placeholder as production artwork.
    """
    expected = {placement.name for placement in face.statics}
    for name, cfg in face.fields.items():
        if not isinstance(cfg, dict):
            continue
        if name in {"barcode", "lineart"}:
            expected.add(name)
        elif name in {"names", "cn_name"}:
            if any(str(item.get(line.get("field")) or "").strip() for line in cfg.get("lines", [])):
                expected.add(name)
        elif name in {"item_no", "qty", "es_name"}:
            value = item.get(name)
            if value is not None and str(value).strip():
                expected.add(name)
    return expected


# --------------------------------------------------------------------------- #
# Targeted re-rasterisation
# --------------------------------------------------------------------------- #


def page_layer_xrefs(layered: LayeredDocument, page_no: int) -> dict[str, dict[str, int]]:
    """``{face: {element: ocg xref}}`` for the layers this page actually uses.

    The tagger adds one group per element per page under the same name, so a
    name alone does not identify the group on a multi-page document. The page's
    ``/Resources /Properties`` does.
    """
    doc = fitz.open("pdf", layered.buffer)
    try:
        page = doc[page_no]
        kind, value = doc.xref_get_key(page.xref, "Resources/Properties")
        if kind == "xref":
            value = doc.xref_object(int(str(value).split()[0]))
        used = {int(ref) for ref in re.findall(r"(\d+)\s+0\s+R", str(value or ""))}
    finally:
        doc.close()
    out: dict[str, dict[str, int]] = {}
    for xref, name in sorted(layered.ocgs.items()):
        if used and xref not in used:
            continue
        face, element = split_layer_name(name)
        out.setdefault(face, {}).setdefault(element, xref)
    return out


def _clip_around(face: FaceLayers, box_px: tuple[int, int, int, int], margin_pt: float) -> fitz.Rect:
    scale = 1.0 / face.zoom
    rect = fitz.Rect(
        face.clip.x0 + box_px[0] * scale - margin_pt,
        face.clip.y0 + box_px[1] * scale - margin_pt,
        face.clip.x0 + box_px[2] * scale + margin_pt,
        face.clip.y0 + box_px[3] * scale + margin_pt,
    )
    return rect & face.clip


def _bounded_zoom(clip: fitz.Rect, zoom: float, max_pixels: int) -> float:
    area = max(clip.width * clip.height, 1e-6)
    return float(min(zoom, math.sqrt(max_pixels / area)))


def make_isolator(
    layered: LayeredDocument,
    page_no: int,
    face: FaceLayers,
    xrefs: dict[str, int],
    *,
    max_pixels: int = DEFAULT_MAX_RASTER_PIXELS,
):
    """``isolate(element, box_px, zoom) -> (mask, zoom_used)`` for one face.

    Same isolation as :func:`render_faces` - own ink with unconditional ink
    removed, or the knockout's contribution for white-on-black type - just over
    a small clip at a higher zoom. The clip keeps three face pixels of margin:
    the face-zoom box is exact to one pixel, and ink edges only move inward or
    by under a face pixel when sampled more finely.
    """
    every = list(layered.ocgs)

    def isolate(element: str, box_px: tuple[int, int, int, int], zoom: float):
        xref = xrefs.get(element)
        if xref is None:
            return None
        clip = _clip_around(face, box_px, 3.0 / face.zoom)
        if clip.is_empty:
            return None
        used = _bounded_zoom(clip, zoom, max_pixels)
        others = [other for other in every if other != xref]
        if element not in face.layers and element in face.contributions:
            composite = layered.render(page_no, clip, used)
            without = layered.render(page_no, clip, used, on=others, off=[xref])
            return composite ^ without, used
        mask = layered.render(page_no, clip, used, on=[xref], off=others)
        x0, y0, x1, y1 = box_px
        near = face.unconditional[max(0, y0 - 3) : y1 + 3, max(0, x0 - 3) : x1 + 3]
        if near.any():
            mask &= ~layered.render(page_no, clip, used, on=[], off=every)
        return mask, used

    return isolate


def make_gray_renderer(
    layered: LayeredDocument,
    page_no: int,
    face: FaceLayers,
    *,
    max_pixels: int = DEFAULT_MAX_RASTER_PIXELS,
):
    """``render_gray(box_px, margin_pt, dpi) -> (gray, dpi_used)``: the finished page, 8-bit.

    Every layer in its default (visible) state - what prints, neighbours
    included, because a scanner sees them too. The DPI actually used is
    returned so a capped render is reported, never silently substituted.
    """

    def render_gray(box_px: tuple[int, int, int, int], margin_pt: float, dpi: float):
        clip = _clip_around(face, box_px, margin_pt)
        zoom = _bounded_zoom(clip, dpi / 72.0, max_pixels)
        doc = fitz.open("pdf", layered.buffer)
        try:
            pix = doc[page_no].get_pixmap(
                clip=clip, matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False
            )
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        finally:
            doc.close()
        return gray, zoom * 72.0

    return render_gray


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def audit_revision(
    pdf_path: Path | str,
    revision: LayoutRevision,
    items_by_no: dict[str, dict[str, Any]],
    *,
    atlas: Atlas | None = None,
    assets_dir: Path | str | None = None,
    bands_path: Path | str | None = None,
    zoom: float = DEFAULT_ZOOM,
) -> dict[str, Any]:
    """Audit every face of every plan in ``revision``."""
    atlas = atlas or load_atlas(str(assets_dir) if assets_dir else None)
    directory = Path(assets_dir) if assets_dir else ASSETS_DIR
    policy = load_audit_policy(directory)

    bands: dict[str, dict[str, Band]] = {}
    band_file = Path(bands_path) if bands_path else directory / "layout_bands.json"
    if band_file.exists():
        bands = load_bands(band_file)

    from ..template_renderer import KmtTemplateRenderer

    renderer = KmtTemplateRenderer()
    layered = LayeredDocument.load(Path(pdf_path))
    nested = [frozenset(pair) for pair in policy.get("nested_pairs", [])]
    tolerances = policy.get("tolerances", {})
    max_pixels = int(
        policy.get("text_fidelity_measurement", {}).get("remeasure_max_pixels", DEFAULT_MAX_RASTER_PIXELS)
    )

    face_reports: list[dict[str, Any]] = []
    for page_no, plan in enumerate(revision.plans):
        item = items_by_no.get(plan.item_no, {"item_no": plan.item_no})
        clips = {kind: _face_clip(plan, kind) for kind in ("front", "side")}
        rendered = render_faces(layered, page_no, clips, zoom=zoom)
        xrefs = page_layer_xrefs(layered, page_no)

        for kind, face in rendered.items():
            plan_face = plan.front if kind == "front" else plan.side
            findings: list[dict[str, Any]] = []
            isolate = make_isolator(
                layered, page_no, face, xrefs.get(kind, {}), max_pixels=max_pixels
            )

            expected = expected_elements(plan_face, item)
            findings.extend(check_expected_elements(face, expected))

            findings.extend(
                check_ink_inside_safe_area(
                    face,
                    margin_mm=plan_face.safe_margin_mm,
                    tolerance_mm=float(tolerances.get("safe_area_mm", 0.5)),
                )
            )
            findings.extend(
                collisions(
                    face,
                    nested=nested,
                    tolerance_px=int(tolerances.get("collision_px", 1)),
                )
            )
            findings.extend(
                occlusion(
                    face,
                    allowed=policy.get("occlusion_allowed", []),
                    tolerance_px=int(tolerances.get("occlusion_px", 4)),
                )
            )
            findings.extend(
                unattributed(face, tolerance_px=int(tolerances.get("unattributed_px", 8)))
            )
            findings.extend(
                check_uniform_scale_groups(
                    face, atlas, policy, assets_dir=directory, placements=plan_face.statics
                )
            )

            def remeasure(element: str, wanted: float, _face=face, _isolate=isolate):
                box = mask_bbox(_face.shape_of(element))
                return None if box is None else _isolate(element, box, wanted)

            aspect_measurements: dict[str, Any] = {}
            findings.extend(
                check_aspect_locked_components(
                    face,
                    atlas,
                    policy,
                    assets_dir=directory,
                    placements=plan_face.statics,
                    remeasure=remeasure,
                    measurements=aspect_measurements,
                )
            )

            text_measurements: dict[str, Any] = {}
            findings.extend(
                check_text_fidelity(
                    face,
                    atlas,
                    policy,
                    describe_runs(plan_face.fields, item, renderer),
                    remeasure=remeasure,
                    measurements=text_measurements,
                )
            )

            barcode_findings, barcode_report = audit_barcode(
                face,
                plan_face,
                item,
                isolate=isolate,
                render_gray=make_gray_renderer(layered, page_no, face, max_pixels=max_pixels),
                renderer=renderer,
                atlas=atlas,
                policy=policy,
            )
            findings.extend(barcode_findings)

            metrics = measure_panel_from_layers(face)
            if kind in bands:
                findings.extend(
                    check_metrics(metrics, bands[kind], face=kind, item_no=plan.item_no)
                )

            boxes = [
                box
                for box in (mask_bbox(mask) for mask in face.layers.values())
                if box is not None
            ]
            alignment = alignment_error(
                [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes],
                width=float(face.composite.shape[1]),
                height=float(face.composite.shape[0]),
            )

            for finding in findings:
                finding.setdefault("item_no", plan.item_no)
            face_reports.append(
                {
                    "item_no": plan.item_no,
                    "face": kind,
                    "family": plan_face.family,
                    "panel_mm": [round(mm(plan_face.panel.width), 2), round(mm(plan_face.panel.height), 2)],
                    "aspect": round(plan_face.aspect, 4),
                    "metrics": metrics.to_dict(),
                    "alignment_error": round(alignment, 5),
                    "elements": sorted(set(face.layers) | set(face.contributions)),
                    "expected_elements": sorted(expected),
                    "component_scales": aspect_measurements,
                    "text_fidelity": text_measurements,
                    "barcode": barcode_report,
                    "findings": findings,
                }
            )

    flattened = [finding for report in face_reports for finding in report["findings"]]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "policy_id": policy.get("policy_id"),
        "source_pdf": str(Path(pdf_path).name),
        "zoom": zoom,
        "band_source": str(band_file.name) if bands else None,
        "faces": face_reports,
        "summary": {
            "faces": len(face_reports),
            "errors": sum(1 for f in flattened if f.get("severity") == "error"),
            "warnings": sum(1 for f in flattened if f.get("severity") == "warning"),
            "info": sum(1 for f in flattened if f.get("severity") == "info"),
        },
        "findings": flattened,
    }


def measure_panel_from_layers(face: Any) -> Any:
    """Panel metrics from the composite already rendered for the layer sweep.

    Re-rendering at the baseline zoom would be more faithful to the factory
    calibration, but the composite is the same artwork and re-rasterising it a
    second time doubles the slowest part of the audit for no new information.
    """
    from .raster import PanelMetrics, ink_coverage, largest_empty_rect_ratio, uniformity

    # Cell size is defined in millimetres by the baseline (4 px at zoom 4), so
    # keep the physical cell constant when the audit runs at another zoom.
    cell_px = max(1, round(4 * face.zoom / 4.0))
    spread, balance = uniformity(face.composite, cell_px=cell_px)
    scale = 25.4 / 72.0 / face.zoom
    return PanelMetrics(
        ink_coverage=ink_coverage(face.composite, cell_px=cell_px),
        largest_empty_rect=largest_empty_rect_ratio(face.composite, cell_px=cell_px),
        spread=spread,
        balance=balance,
        width_mm=face.composite.shape[1] * scale,
        height_mm=face.composite.shape[0] * scale,
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "audit_revision",
    "describe_runs",
    "expected_elements",
    "make_gray_renderer",
    "make_isolator",
    "measure_panel_from_layers",
    "page_layer_xrefs",
]
