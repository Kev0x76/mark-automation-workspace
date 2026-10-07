"""L1 self-consistency, measured on the rendered face.

Nothing here needs a factory sample, a threshold anyone had to argue about, or
an owner decision. Every rule is of the form "the artwork disagrees with itself"
and the reference is either the source artwork in ``base.pdf`` or the font's own
metrics - both already in the tree.

The four checks correspond one-to-one with the four defects the 2026-09-02
review found by eye, none of which any existing check reported:

============================  ===========================================
review finding                check
============================  ===========================================
five cells scaled unequally   :func:`check_uniform_scale_groups`
Q.TY stretched 3.64x          :func:`check_aspect_locked_components`
labels squeezed to 0.46       :func:`check_text_fidelity`
barcode digits over the bars  ``layers.collisions``
============================  ===========================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import fitz
import numpy as np

from ..adaptive_layout.atlas import ASSETS_DIR, Atlas
from ..adaptive_layout.model import StaticPlacement
from .glyphs import (
    EDGE_QUANTISATION_PX,
    REFERENCE_MAX_PIXELS,
    REFERENCE_PX_PER_EM,
    GlyphFidelity,
    fidelity_of_run,
    scale_ratio_bounds,
    zoom_to_resolve,
    zoom_to_resolve_scale,
)
from .layers import FaceLayers
from .raster import ink_mask, mask_bbox, pixmap_array


def load_audit_policy(assets_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(assets_dir) if assets_dir else ASSETS_DIR
    return json.loads((directory / "layout_audit.json").read_text(encoding="utf-8"))


def _issue(code: str, message: str, *, severity: str = "error", **extra: Any) -> dict[str, Any]:
    payload = {"severity": severity, "code": code, "message": message}
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


# --------------------------------------------------------------------------- #
# The source artwork, measured the same way as the output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceShape:
    width: float
    height: float

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


@lru_cache(maxsize=256)
def _source_shape(assets_dir: str, face: str, name: str, zoom: float) -> SourceShape | None:
    """Ink extent of one component as it exists in ``base.pdf``.

    Measured from ink rather than from the clip rectangle: a clip usually has
    margin around the artwork, and comparing an inked extent against a padded
    rectangle would report a scale that no element actually has.
    """
    from ..adaptive_layout.atlas import load_atlas

    atlas = load_atlas(assets_dir)
    if not atlas.has_component(face, name):
        return None
    spec = atlas.component(face, name)
    return _source_clip_shape(assets_dir, spec.source_page, tuple(spec.clip), zoom)


@lru_cache(maxsize=256)
def _source_clip_shape(
    assets_dir: str, source_page: int, source_clip: tuple[float, ...], zoom: float
) -> SourceShape | None:
    """Measure the actual source tile requested by a static placement."""
    clip = fitz.Rect(*source_clip)
    source = fitz.open(Path(assets_dir) / "base.pdf")
    probe = fitz.open()
    page = probe.new_page(width=max(clip.width, 1.0), height=max(clip.height, 1.0))
    page.show_pdf_page(page.rect, source, source_page, clip=clip)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    box = mask_bbox(ink_mask(pixmap_array(pix)))
    probe.close()
    source.close()
    if box is None:
        return None
    return SourceShape(float(box[2] - box[0]), float(box[3] - box[1]))


def component_scales(
    face: FaceLayers,
    atlas: Atlas,
    *,
    assets_dir: str | Path | None = None,
    placements: Iterable[StaticPlacement] | None = None,
) -> dict[str, tuple[float, float]]:
    """Measured scales expressed in the original artwork's coordinate axes.

    A quarter-turn exchanges the rendered width and height. The placement tells
    us which axes to compare and which source tile was requested; its target
    dimensions never substitute for the final PDF's measured ink.
    """
    return {
        name: (size.scale_x, size.scale_y)
        for name, size in component_measurements(
            face, atlas, assets_dir=assets_dir, placements=placements
        ).items()
    }


@dataclass(frozen=True)
class ComponentScale:
    """One component's rendered ink against its source tile, with pixel extents.

    The extents are kept, not just the ratio, because both sides are thresholded
    ink boxes: a verdict that cannot survive +-1 px on each of them is a verdict
    about the raster, not about the artwork. See
    :func:`~.glyphs.scale_ratio_bounds`.
    """

    name: str
    rendered_px: tuple[float, float]
    source_px: tuple[float, float]

    @property
    def scale_x(self) -> float:
        return self.rendered_px[0] / self.source_px[0] if self.source_px[0] > 0 else 0.0

    @property
    def scale_y(self) -> float:
        return self.rendered_px[1] / self.source_px[1] if self.source_px[1] > 0 else 0.0

    @property
    def ratio(self) -> float:
        return self.scale_x / self.scale_y if self.scale_y > 0 else 0.0

    def ratio_bounds(self, edge_px: float = EDGE_QUANTISATION_PX) -> tuple[float, float]:
        return scale_ratio_bounds(
            self.rendered_px[0], self.source_px[0], self.rendered_px[1], self.source_px[1], edge_px
        )

    def zoom_to_resolve(
        self, current_zoom: float, target_uncertainty: float, edge_px: float = EDGE_QUANTISATION_PX
    ) -> float:
        return zoom_to_resolve_scale(
            self.rendered_px[0], self.rendered_px[1], self.source_px[0], self.source_px[1],
            current_zoom, target_uncertainty, edge_px=edge_px,
        )


def component_measurements(
    face: FaceLayers,
    atlas: Atlas,
    *,
    assets_dir: str | Path | None = None,
    placements: Iterable[StaticPlacement] | None = None,
) -> dict[str, ComponentScale]:
    """Every component layer that can be compared with its own source tile."""
    directory = str(Path(assets_dir) if assets_dir else ASSETS_DIR)
    by_name = {placement.name: placement for placement in placements or ()}
    out: dict[str, ComponentScale] = {}
    for name in sorted(face.layers):
        if not atlas.has_component(face.face, name):
            continue
        placement = by_name.get(name)
        source = (
            _source_clip_shape(directory, placement.source_page, tuple(placement.clip), face.zoom)
            if placement is not None
            else _source_shape(directory, face.face, name, face.zoom)
        )
        if source is None or source.width <= 0 or source.height <= 0:
            continue
        box = mask_bbox(face.layers[name])
        if box is None:
            continue
        width, height = float(box[2] - box[0]), float(box[3] - box[1])
        if placement is not None and placement.fit == "rotate90_uniform":
            width, height = height, width
        out[name] = ComponentScale(name, (width, height), (source.width, source.height))
    return out


def _ratio_verdict(low: float, high: float, tolerance: float) -> str:
    """``pass`` / ``fail`` / ``unresolved`` for an interval against a 1.0 +- tol band.

    ``unresolved`` is the honest answer when +-1 px on the four extents straddles
    a limit: the pixels allow both verdicts, so neither may be reported.
    """
    if low >= 1.0 - tolerance and high <= 1.0 + tolerance:
        return "pass"
    if high < 1.0 - tolerance or low > 1.0 + tolerance:
        return "fail"
    return "unresolved"


def check_expected_elements(face: FaceLayers, expected: Iterable[str]) -> list[dict[str, Any]]:
    """Every requested element must leave a measurable mark in the final face.

    A missing optional-content group must not turn an absent component into a
    vacuously passing aspect/collision check. White knockout text is measured
    through its visible contribution, just as it is for text fidelity.
    Presence alone does not establish the element's content or authenticity.
    """
    return [
        _issue(
            "expected_element_missing",
            f"{name} was requested but has no measurable layer on the {face.face} face.",
            face=face.face,
            component=name,
        )
        for name in sorted(set(expected))
        if not face.shape_of(name).any()
    ]


# --------------------------------------------------------------------------- #
# L1-1 same-class consistency
# --------------------------------------------------------------------------- #


def check_uniform_scale_groups(
    face: FaceLayers, atlas: Atlas, policy: dict[str, Any], **kwargs: Any
) -> list[dict[str, Any]]:
    """Elements declared to share a scale must actually share it.

    This is the group relation ``adaptive_layout.json`` has no way to express:
    ``fit`` says how one component may deform, never that five of them have to
    deform together.

    Two things have to agree for a group to pass. The members' *drawn* scales
    must be uniform - that is the format check, and it is what the 2026-09-10
    ruling made an approved parameter. And if the group declares a ``pattern``,
    its geometry may carry structure, in which case the structure must be the
    declared one instead of a loose spread.

    The two are measured differently on purpose. Scale comes from ink, because
    that is what a reader sees. Structure comes from the placement targets,
    because the five cells are five *different* tiles - 128.5 / 66.1 / 83.8 /
    70.7 / 70.6 pt tall in base.pdf - so equal row heights and equal ink scales
    are not the same statement, and the factory reference that justifies the
    taller ``Q.TY`` row states its 1.489 as a height ratio. Measuring structure
    from ink here would compare vectors that were never meant to be equal.
    """
    scales = component_scales(face, atlas, **kwargs)
    targets = _placement_heights(kwargs.get("placements"))
    source_heights = {
        name: measured.source_px[1]
        for name, measured in component_measurements(
            face, atlas, assets_dir=kwargs.get("assets_dir"), placements=kwargs.get("placements")
        ).items()
    }
    findings: list[dict[str, Any]] = []
    for group_name, group in policy.get("uniform_scale_groups", {}).items():
        axis = 1 if str(group.get("axis", "y")) == "y" else 0
        tolerance = float(group.get("tolerance", 0.02))
        members = {
            name: scales[name][axis] for name in group["elements"] if name in scales
        }
        if len(members) < 2:
            continue
        values = list(members.values())
        median = float(np.median(values))
        spread = (max(values) - min(values)) / median if median > 0 else 0.0
        if spread <= tolerance:
            continue
        # The group is not uniform as measured. Before reporting, ask whether
        # the artwork says it should not be: a declared pattern whose geometry
        # really shows that structure is a design decision, not a torn table.
        if _pattern_is_in_effect(
            group, members, targets, source_heights, tolerance, axis
        ):
            continue
        worst = max(members, key=lambda n: abs(members[n] - median))
        findings.append(
            _issue(
                "group_scale_inconsistent",
                f"{group_name} is meant to scale as one block on the {face.face} face, "
                f"but its members render at {min(values):.3f} to {max(values):.3f} of the "
                f"source artwork - a spread of {spread * 100:.1f}%, worst offender {worst}.",
                face=face.face,
                group=group_name,
                axis="y" if axis else "x",
                spread=round(spread, 4),
                tolerance=tolerance,
                scales={name: round(value, 4) for name, value in sorted(members.items())},
            )
        )
    return findings


def _placement_heights(placements: Any) -> dict[str, float]:
    """Target heights of the placements, keyed by name.

    The drawn size of a component, before any ink is thresholded. Missing or
    unnamed placements simply do not appear, so a caller that has no placements
    (the unit tests) falls back to the flat uniform rule.
    """
    out: dict[str, float] = {}
    for placement in placements or ():
        name = getattr(placement, "name", None)
        target = getattr(placement, "target", None)
        if name and target is not None:
            out[str(name)] = float(target.height)
    return out


def _pattern_is_in_effect(
    group: dict[str, Any],
    members: dict[str, float],
    targets: dict[str, float],
    source_heights: dict[str, float],
    tolerance: float,
    axis: int,
) -> bool:
    """Whether a declared group ``pattern`` explains the measured spread.

    ``False`` means the spread stands and should be reported. Two statements are
    accepted, and they disagree by construction, so the pattern has to be read
    as a *shape* rather than a number:

    ``reference`` - the factory mark's shape. The members named in ``pattern``
    are taller than the rest and the rest are equal to each other, judged on the
    placement geometry, with the drawing confirmed against the plan.

    ``artwork`` - base.pdf's own shape. The five tiles are five different
    heights (128.5 / 66.1 / 83.8 / 70.7 / 70.6 pt), and this path draws each
    cell at its own tile's height at one shared table scale. So the drawn scales
    are uniform - they have to be, that is what one scale means - while the row
    heights are the tiles' 1.765 with a 1.268 spread. A uniform ink scale across
    unequal tiles is not a torn table; it is the artwork being reproduced.

    The verdict on the reference path is taken on the placement targets, not on
    the ink scales, and that is not a shortcut. ``cell_qty`` has the tallest row
    (83.56 pt) and the *smallest* drawn scale (0.6537), because its source tile
    is the tallest one (128.5 pt against 56-84 pt for the others): drawn scale
    measures a tile against itself, so with unequal tiles it says nothing about
    which row is tall - reading structure off it would invert the answer.
    Geometry is the quantity the factory reference measured when it got 1.489.

    What neither path excuses is a table drawn at *neither* shape: heights that
    answer to no statement. That is still the defect this group exists to catch.
    """
    if axis != 1:
        # Only a vertical pattern is expressed; a horizontal one would need its
        # own evidence before it could excuse anything.
        return False
    pattern = group.get("pattern") or {}
    taller = [name for name in (pattern.get("taller_members") or []) if name in members]
    if not taller:
        return False
    rest = [name for name in members if name not in taller]
    if len(rest) < 2:
        return False
    required = set(members)
    if not required <= set(targets) or not required <= set(source_heights):
        return False

    # 1) Geometry: the declared taller rows really are taller, and the rest
    # really do agree with each other.
    geo_rest = [targets[name] for name in rest]
    geo_rest_median = float(np.median(geo_rest))
    if not geo_rest_median > 0:
        return False
    if (max(geo_rest) - min(geo_rest)) / geo_rest_median > tolerance:
        return False
    minimum = float(pattern.get("min_taller_ratio", 1.0 + tolerance))
    if not all(targets[name] / geo_rest_median > minimum for name in taller):
        return False

    # 2) Ink agrees with the geometry: for every member, the drawn scale must be
    # the row height over the source tile height - the drawing did what the plan
    # said. Compared per member against its own tile, never across members,
    # because the tiles are not the same size. ``source_heights`` is measured in
    # the raster's pixels and ``targets`` is in points, so the comparison is on
    # each member's ratio to itself; the zoom cancels.
    drawn = []
    for name in members:
        source = source_heights[name]
        if not source > 0:
            return False
        planned = targets[name] / source
        if planned <= 0:
            return False
        drawn.append(members[name] / planned)
    drawn_median = float(np.median(drawn))
    if not drawn_median > 0:
        return False
    if (max(drawn) - min(drawn)) / drawn_median <= tolerance:
        return True

    # 3) The geometry did not show the reference's shape. Before the spread
    # stands, ask whether it shows the *other* statement instead: base.pdf's own
    # tiles drawn at one scale. That is legible precisely because it is uniform
    # in ink while the rows differ - the opposite signature to a torn table,
    # where it is the ink that scatters around a geometry nobody planned.
    alt = pattern.get("alt_pattern") or {}
    if not alt:
        return False
    own = [source_heights[name] for name in members]
    own_median = float(np.median(own))
    if not own_median > 0:
        return False
    if (max(own) - min(own)) / own_median <= tolerance:
        # Every tile is the same height, so "own heights" and "equal rows" are
        # the same statement, and it has already been rejected above.
        return False
    drawn_median_all = float(np.median(list(members.values())))
    if not drawn_median_all > 0:
        return False
    return (max(members.values()) - min(members.values())) / drawn_median_all <= tolerance


# --------------------------------------------------------------------------- #
# L1-2 aspect fidelity of locked components
# --------------------------------------------------------------------------- #


def check_aspect_locked_components(
    face: FaceLayers,
    atlas: Atlas,
    policy: dict[str, Any],
    *,
    remeasure: Remeasure | None = None,
    measurements: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """A component declared aspect-locked must render with one scale, not two.

    Judged on the interval the pixels allow, not on the point estimate. A
    quarter-turn on a small component makes +-1 px per extent worth more than
    the tolerance, so an interval spanning the limit is re-measured at the zoom
    that resolves it (``remeasure``, the same hook the glyph check uses). If it
    still spans the limit - no hook, or the zoom ceiling reached - the finding
    says a stretch *may* be present instead of asserting one.

    Measured on 2026-09-15: ``origin_text`` on the 450 x 100 mm owner front read
    0.9763 at zoom 3 and 1.0002 at zoom 24. Reporting 2.4% of stretch there was
    reporting the raster.
    """
    locked = set(policy.get("aspect_locked_fit_modes", ["uniform"]))
    tolerance = float(policy.get("aspect_tolerance", 0.02))
    measuring = policy.get("text_fidelity_measurement", {})
    edge_px = float(measuring.get("edge_quantisation_px", EDGE_QUANTISATION_PX))
    target = float(measuring.get("remeasure_target_uncertainty", tolerance / 2.0))
    max_zoom = float(measuring.get("remeasure_max_zoom", 48.0))
    assets_dir = kwargs.get("assets_dir")
    placements = kwargs.get("placements")
    by_name = {placement.name: placement for placement in placements or ()}
    sizes = component_measurements(face, atlas, **kwargs)

    findings: list[dict[str, Any]] = []
    for name, size in sorted(sizes.items()):
        spec = atlas.component(face.face, name)
        if spec.fit not in locked or size.scale_y <= 0:
            continue
        zoom = float(face.zoom)
        ratio, (low, high) = size.ratio, size.ratio_bounds(edge_px)
        verdict = _ratio_verdict(low, high, tolerance)

        if verdict == "unresolved" and remeasure is not None:
            wanted = min(max_zoom, size.zoom_to_resolve(zoom, target, edge_px))
            if wanted > zoom * 1.05:
                sharper = remeasure(name, wanted)
                if sharper is not None and sharper[0].any():
                    box = mask_bbox(sharper[0])
                    used = float(sharper[1])
                    placement = by_name.get(name)
                    directory = str(Path(assets_dir) if assets_dir else ASSETS_DIR)
                    source = (
                        _source_clip_shape(directory, placement.source_page, tuple(placement.clip), used)
                        if placement is not None
                        else _source_shape(directory, face.face, name, used)
                    )
                    if box is not None and source is not None:
                        width, height = float(box[2] - box[0]), float(box[3] - box[1])
                        if placement is not None and placement.fit == "rotate90_uniform":
                            width, height = height, width
                        sharper_size = ComponentScale(
                            name, (width, height), (source.width, source.height)
                        )
                        if sharper_size.scale_y > 0:
                            size, zoom = sharper_size, used
                            ratio, (low, high) = size.ratio, size.ratio_bounds(edge_px)
                            verdict = _ratio_verdict(low, high, tolerance)

        if measurements is not None:
            # Passing components are evidence too: the report shows what was
            # measured, at which zoom, and how tightly the pixels pin it.
            measurements[name] = {
                "measured_zoom": round(zoom, 3),
                "scale_x": round(size.scale_x, 4),
                "scale_y": round(size.scale_y, 4),
                "stretch_ratio": round(ratio, 4),
                "stretch_ratio_interval": [round(low, 4), round(high, 4)],
                "rendered_px": [round(v, 1) for v in size.rendered_px],
                "source_px": [round(v, 1) for v in size.source_px],
            }

        detail = {
            "face": face.face,
            "component": name,
            "measured_zoom": round(zoom, 3),
            "tolerance": tolerance,
            "stretch_ratio": round(ratio, 4),
            "stretch_ratio_interval": [round(low, 4), round(high, 4)],
            "scale_x": round(size.scale_x, 4),
            "scale_y": round(size.scale_y, 4),
        }
        interval = f"[{low:.3f}, {high:.3f}]"
        if verdict == "fail":
            findings.append(
                _issue(
                    "aspect_locked_component_stretched",
                    f"{name} is declared '{spec.fit}' but rendered {ratio:.3f} {interval} times wider "
                    f"than tall relative to the artwork in base.pdf on the {face.face} face.",
                    **detail,
                )
            )
        elif verdict == "unresolved":
            findings.append(
                _issue(
                    "aspect_locked_component_unresolved_at_audit_resolution",
                    f"{name} on the {face.face} face measured {ratio:.3f} outside the "
                    f"{1.0 - tolerance:.2f}-{1.0 + tolerance:.2f} band, but its pixels allow "
                    f"{interval} at zoom {zoom:.1f}; the measurement cannot tell a stretch from "
                    "quantisation.",
                    severity="warning",
                    **detail,
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# L1-3 glyph fidelity
# --------------------------------------------------------------------------- #


def _condense_floor(policy: dict[str, Any], atlas: Atlas, element: str, text: str) -> float:
    rule = policy.get("text_fidelity", {}).get(element, {})
    floor = rule.get("min_condense", 1.0)
    if floor == "atlas:item_no_hscale_by_digits":
        digits = max(1, len(text))
        table = atlas.frame.item_no_hscale_by_digits
        return float(table.get(digits, min(table.values(), default=1.0)))
    return float(floor)


#: ``remeasure(element, zoom) -> (mask, zoom_used)`` - the element's ink again,
#: rasterised at (up to) ``zoom``. Supplied by the audit, which owns the PDF.
Remeasure = Callable[[str, float], "tuple[np.ndarray, float] | None"]


def _verdict_is_settled(
    result: GlyphFidelity, floor: float, tolerance: float, spread_tolerance: float
) -> bool:
    """True when every value the pixels allow gives the same verdict."""
    for bound in (1.0 + tolerance, floor - tolerance):
        if result.distortion_low <= bound <= result.distortion_high:
            return False
    return not (
        result.per_character_spread_low <= spread_tolerance < result.per_character_spread_high
    )


def check_text_fidelity(
    face: FaceLayers,
    atlas: Atlas,
    policy: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    *,
    remeasure: Remeasure | None = None,
    measurements: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rendered type must keep the proportions the font drew.

    ``runs`` maps an element name to ``{"text", "fontsize", "fontfile"}`` - what
    was asked for. The check does not trust that the ask was honoured; it draws
    the same string untransformed and compares the two shapes.

    Verdicts are taken on the interval the pixels allow, not on the point
    estimate (see :mod:`.glyphs`, "Measurement resolution"). When the interval
    straddles a limit, ``remeasure`` fetches the run at the zoom that resolves
    it. If it still straddles - no hook, or the zoom ceiling reached - a point
    estimate past the limit is reported as a warning that says so, never as a
    defect the measurement cannot support.
    """
    tolerance = float(policy.get("fidelity_tolerance", 0.02))
    spread_tolerance = float(policy.get("per_character_spread_tolerance", 0.05))
    measuring = policy.get("text_fidelity_measurement", {})
    edge_px = float(measuring.get("edge_quantisation_px", EDGE_QUANTISATION_PX))
    target = float(measuring.get("remeasure_target_uncertainty", tolerance / 2.0))
    max_zoom = float(measuring.get("remeasure_max_zoom", 48.0))
    reference = {
        "px_per_em": float(measuring.get("reference_px_per_em", REFERENCE_PX_PER_EM)),
        "max_pixels": int(measuring.get("reference_max_pixels", REFERENCE_MAX_PIXELS)),
    }
    findings: list[dict[str, Any]] = []

    for element, spec in sorted(runs.items()):
        text = str(spec.get("text") or "")
        mask = face.shape_of(element)
        if not text or not mask.any():
            continue
        floor = _condense_floor(policy, atlas, element, text)
        measure: dict[str, Any] = {
            "fontsize": float(spec.get("fontsize") or 10.0),
            "fontfile": spec.get("fontfile"),
            "fontname": str(spec.get("fontname") or "helv"),
            "declared": 1.0,
            "edge_px": edge_px,
            **reference,
        }
        zoom = float(face.zoom)
        result: GlyphFidelity = fidelity_of_run(mask, text, measured_zoom=zoom, **measure)
        if result.distortion <= 0:
            continue

        if remeasure is not None and not _verdict_is_settled(
            result, floor, tolerance, spread_tolerance
        ):
            wanted = min(max_zoom, zoom_to_resolve(result, zoom, target, edge_px=edge_px))
            if wanted > zoom * 1.05:
                sharper = remeasure(element, wanted)
                if sharper is not None and sharper[0].any():
                    candidate = fidelity_of_run(
                        sharper[0], text, measured_zoom=float(sharper[1]), **measure
                    )
                    if candidate.distortion > 0:
                        result, zoom = candidate, float(sharper[1])

        if measurements is not None:
            # Passing runs are evidence too: the report must show what was measured
            # and at which zoom, not only what failed.
            measurements[element] = {"measured_zoom": round(zoom, 3), **result.to_dict()}
        detail = {
            "face": face.face,
            "component": element,
            "measured_zoom": round(zoom, 3),
            "tolerance": tolerance,
            **result.to_dict(),
        }
        interval = f"[{result.distortion_low:.3f}, {result.distortion_high:.3f}]"
        if result.distortion_high < floor - tolerance:
            findings.append(
                _issue(
                    "text_condensed_past_declared_floor",
                    f"{element} rendered at {result.distortion:.3f} {interval} of the font's "
                    f"natural proportions on the {face.face} face; the floor for this element "
                    f"is {floor:.3f}.",
                    floor=round(floor, 4),
                    **detail,
                )
            )
        elif result.distortion_low > 1.0 + tolerance:
            findings.append(
                _issue(
                    "text_stretched_beyond_natural_width",
                    f"{element} rendered {result.distortion:.3f} {interval} times wider than "
                    f"the font draws it on the {face.face} face.",
                    **detail,
                )
            )
        elif result.distortion < floor - tolerance or result.distortion > 1.0 + tolerance:
            findings.append(
                _issue(
                    "text_fidelity_unresolved_at_audit_resolution",
                    f"{element} on the {face.face} face measured {result.distortion:.3f}, past "
                    f"the limit, but its pixels allow {interval} at zoom {zoom:.1f}; the "
                    "measurement cannot tell a distortion from quantisation.",
                    severity="warning",
                    quantity="distortion",
                    floor=round(floor, 4),
                    **detail,
                )
            )

        if result.per_character_spread_low > spread_tolerance:
            findings.append(
                _issue(
                    "glyphs_distorted_unevenly",
                    f"{element}'s glyphs on the {face.face} face were not scaled by the same "
                    f"amount ({result.per_character_spread * 100:.1f}% spread across "
                    f"{len(result.per_character)} glyphs, at least "
                    f"{result.per_character_spread_low * 100:.1f}% after pixel uncertainty). "
                    "No font and no single transform produces that.",
                    **detail,
                )
            )
        elif result.per_character_spread > spread_tolerance:
            findings.append(
                _issue(
                    "text_fidelity_unresolved_at_audit_resolution",
                    f"{element}'s per-glyph spread on the {face.face} face measured "
                    f"{result.per_character_spread * 100:.1f}%, but quantisation alone could "
                    f"produce it at zoom {zoom:.1f}.",
                    severity="warning",
                    quantity="per_character_spread",
                    **detail,
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# L0 safe area, measured on ink
# --------------------------------------------------------------------------- #


def check_ink_inside_safe_area(
    face: FaceLayers, *, margin_mm: float, tolerance_mm: float = 0.5
) -> list[dict[str, Any]]:
    """No ink past the printable margin - judged on marks, not on boxes."""
    scale = 25.4 / 72.0 / face.zoom
    height_px, width_px = face.composite.shape
    limit = (margin_mm - tolerance_mm) / scale

    findings: list[dict[str, Any]] = []
    for name, mask in sorted(face.layers.items()):
        box = mask_bbox(mask)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        overshoot = max(limit - x0, limit - y0, x1 - (width_px - limit), y1 - (height_px - limit))
        if overshoot > 0:
            findings.append(
                _issue(
                    "ink_outside_safe_area",
                    f"{name} prints {overshoot * scale:.2f} mm past the {margin_mm} mm safe "
                    f"margin on the {face.face} face.",
                    face=face.face,
                    component=name,
                    overshoot_mm=round(overshoot * scale, 3),
                    margin_mm=margin_mm,
                )
            )
    return findings


__all__ = [
    "ComponentScale",
    "SourceShape",
    "check_aspect_locked_components",
    "check_expected_elements",
    "check_ink_inside_safe_area",
    "check_text_fidelity",
    "check_uniform_scale_groups",
    "component_measurements",
    "component_scales",
    "load_audit_policy",
]
