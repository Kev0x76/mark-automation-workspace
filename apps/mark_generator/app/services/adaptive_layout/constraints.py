"""Hard constraints and scoring for candidate face layouts.

Hard constraints are pass/fail and exist to stop the engine quietly producing a
mark that cannot be printed or scanned: content outside the safe area, elements
colliding, text below a legible size, a barcode that has been stretched or
shrunk past its GS1 floor. When every candidate fails, the caller must surface
``manual_layout_required`` rather than shipping the least-bad option.

Scoring only ever ranks candidates that already passed. It deliberately favours
staying close to the current standard mark, because the owner's rule is that
detail comes from the existing automatic mark and only the large-panel
arrangement adapts.
"""

from __future__ import annotations

from typing import Any

from .atlas import Atlas
from .model import Box, FacePlan, mm


def _issue(code: str, message: str, *, severity: str = "warning", **extra: Any) -> dict[str, Any]:
    payload = {"severity": severity, "code": code, "message": message}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


# Elements that legitimately nest inside one another; their overlap is by design
# and must not be reported as a collision.
NESTED_PAIRS = {
    frozenset({"barcode", "barcode_brackets"}),
    frozenset({"cell_qty", "qty"}),
    frozenset({"cell_name", "es_name"}),
    frozenset({"names", "cn_name"}),
}


def validate_face(atlas: Atlas, plan: FacePlan, item: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    limits = atlas.constraints
    item_no = str(item.get("item_no") or "")

    if plan.scale <= 0:
        return [
            _issue(
                "layout_family_unavailable",
                f"{plan.family} could not produce a layout for this face.",
                severity="error",
                face=plan.kind,
                item_no=item_no or None,
            )
        ]

    boxes = plan.occupied_boxes()

    # 1. everything inside the hard printable safe area
    safe = plan.safe_area
    for name, box in boxes:
        if not safe.contains(box, tolerance=0.5):
            issues.append(
                _issue(
                    "component_outside_safe_area",
                    f"{name} extends outside the printable safe area of the {plan.kind} face.",
                    severity="error",
                    face=plan.kind,
                    component=name,
                    item_no=item_no or None,
                    box_mm=box.as_mm(),
                    safe_area_mm=safe.as_mm(),
                )
            )

    # 2. Real overlap is an error. Merely tight clearance is a warning: the
    #    approved standard mark itself sets some elements only ~1.2 mm apart,
    #    so treating the clearance target as hard would reject today's design.
    min_gap_pt = mm_to_pt(float(limits.get("min_component_gap_mm", 1.5)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            name_a, box_a = boxes[i]
            name_b, box_b = boxes[j]
            if frozenset({name_a, name_b}) in NESTED_PAIRS:
                continue
            if box_a.contains(box_b) or box_b.contains(box_a):
                continue
            if box_a.overlaps(box_b):
                issues.append(
                    _issue(
                        "components_overlap",
                        f"{name_a} and {name_b} overlap on the {plan.kind} face.",
                        severity="error",
                        face=plan.kind,
                        components=[name_a, name_b],
                        item_no=item_no or None,
                    )
                )
            elif box_a.overlaps(box_b, gap=min_gap_pt):
                issues.append(
                    _issue(
                        "components_clearance_tight",
                        f"{name_a} and {name_b} sit closer than "
                        f"{limits.get('min_component_gap_mm')} mm on the {plan.kind} face.",
                        face=plan.kind,
                        components=[name_a, name_b],
                        item_no=item_no or None,
                    )
                )

    # 3. barcode physics
    if plan.barcode is not None:
        bc_limits = limits["barcode"]
        x_mm = plan.barcode.x_dimension_mm
        if x_mm < float(bc_limits["fail_below_x_mm"]):
            issues.append(
                _issue(
                    "barcode_below_minimum_x_dimension",
                    f"Barcode X-dimension {x_mm:.3f} mm is below the GS1 floor "
                    f"{bc_limits['fail_below_x_mm']} mm; this face cannot carry a scannable EAN-13 "
                    "at this layout.",
                    severity="error",
                    face=plan.kind,
                    item_no=item_no or None,
                    x_dimension_mm=round(x_mm, 4),
                    magnification=round(plan.barcode.magnification, 3),
                )
            )
        elif x_mm < float(bc_limits["warn_below_x_mm"]):
            issues.append(
                _issue(
                    "barcode_below_distribution_magnification",
                    f"Barcode X-dimension {x_mm:.3f} mm ({plan.barcode.magnification * 100:.0f}% "
                    "magnification) is under the 150% GS1 recommends for general distribution "
                    "scanning; it is still above the absolute minimum.",
                    face=plan.kind,
                    item_no=item_no or None,
                    x_dimension_mm=round(x_mm, 4),
                    magnification=round(plan.barcode.magnification, 3),
                )
            )
        # Magnification is uncapped by owner decision (2026-08-26). A symbol above the
        # GS1 ceiling still prints and still decodes from the artwork, but a scan has to
        # cross the whole symbol plus both quiet zones in one pass, so an oversized one
        # can fall outside a scanner's field of view. Report it, never shrink it.
        _above = bc_limits.get("warn_above_x_mm")
        if _above and x_mm > float(_above):
            issues.append(
                _issue(
                    "barcode_above_gs1_ceiling",
                    f"Barcode X-dimension {x_mm:.3f} mm ({plan.barcode.magnification * 100:.0f}% "
                    f"magnification) is above the GS1 EAN/UPC ceiling of {float(_above):.3f} mm "
                    "(200%). Magnification is uncapped by owner decision, so the symbol was not "
                    "shrunk; confirm it still scans on a printed sample before release.",
                    face=plan.kind,
                    item_no=item_no or None,
                    x_dimension_mm=round(x_mm, 4),
                    magnification=round(plan.barcode.magnification, 3),
                )
            )
        if int(plan.barcode.rotation_deg) not in [int(v) for v in bc_limits["allowed_rotation_deg"]]:
            issues.append(
                _issue(
                    "barcode_rotation_not_allowed",
                    f"Barcode rotation {plan.barcode.rotation_deg} is not permitted.",
                    severity="error",
                    face=plan.kind,
                )
            )
        issues.extend(_validate_quiet_zones(plan, bc_limits))

    # 4. legibility floors
    issues.extend(_validate_text_sizes(plan, limits, item_no))

    return issues


def mm_to_pt(value_mm: float) -> float:
    return value_mm * 72.0 / 25.4


def _validate_quiet_zones(plan: FacePlan, bc_limits: dict[str, Any]) -> list[dict[str, Any]]:
    """Quiet zones must be clear of other ink, measured along the module axis."""
    assert plan.barcode is not None
    bars = plan.barcode.bars
    x_pt = mm_to_pt(plan.barcode.x_dimension_mm)
    lead = float(bc_limits["quiet_zone_leading_x"]) * x_pt
    trail = float(bc_limits["quiet_zone_trailing_x"]) * x_pt
    if plan.barcode.rotation_deg == 90:
        zone = Box(bars.x0, bars.y0 - trail, bars.x1, bars.y1 + lead)
    else:
        zone = Box(bars.x0 - lead, bars.y0, bars.x1 + trail, bars.y1)

    issues: list[dict[str, Any]] = []
    if not plan.safe_area.contains(zone, tolerance=0.5):
        issues.append(
            _issue(
                "barcode_quiet_zone_clipped",
                "The barcode quiet zone runs past the printable safe area.",
                severity="error",
                face=plan.kind,
                zone_mm=zone.as_mm(),
            )
        )
    for name, box in plan.occupied_boxes():
        if name in {"barcode", "barcode_brackets"}:
            continue
        if box.overlaps(zone):
            issues.append(
                _issue(
                    "barcode_quiet_zone_intruded",
                    f"{name} intrudes into the barcode quiet zone.",
                    severity="error",
                    face=plan.kind,
                    component=name,
                )
            )
    return issues


TEXT_LIMIT_KEYS = {
    "item_no": "item_no",
    "qty": "table_value",
    "es_name": "table_value",
}


def _validate_text_sizes(plan: FacePlan, limits: dict[str, Any], item_no: str) -> list[dict[str, Any]]:
    floors = limits["min_font_mm"]
    issues: list[dict[str, Any]] = []

    for field_name, limit_key in TEXT_LIMIT_KEYS.items():
        cfg = plan.fields.get(field_name)
        if not isinstance(cfg, dict):
            continue
        size_mm = mm(float(cfg.get("fontsize", 0.0)))
        floor = float(floors.get(limit_key, 0.0))
        if size_mm and size_mm < floor:
            issues.append(
                _issue(
                    "text_below_minimum_size",
                    f"{field_name} would print at {size_mm:.2f} mm, below the {floor} mm floor.",
                    severity="error",
                    face=plan.kind,
                    component=field_name,
                    item_no=item_no or None,
                    size_mm=round(size_mm, 3),
                )
            )

    # The names block's shrink floor is clamped to the physical minimum when the
    # plan is built, so the meaningful question is not "could it go too small"
    # but "does the block still fit in its slot at that floor". If it does not,
    # the face genuinely cannot carry the names and must be reported.
    for block in ("names", "cn_name"):
        cfg = plan.fields.get(block)
        if not isinstance(cfg, dict):
            continue
        lines = cfg.get("lines", [])
        if not lines:
            continue
        floor_pt = float(cfg.get("min_fontsize", 0.0))
        leading = float(cfg.get("leading_ratio", 1.15))
        block_gap = float(cfg.get("block_gap", 0.0))
        slot = float(cfg.get("bottom_y", 0.0)) - float(cfg.get("top_y", 0.0))
        needed = len(lines) * floor_pt * leading + max(0, len(lines) - 1) * block_gap
        if slot > 0 and needed > slot + 0.5:
            issues.append(
                _issue(
                    "names_block_cannot_fit_at_minimum_size",
                    f"The {len(lines)}-line name block needs {mm(needed):.1f} mm at the "
                    f"{mm(floor_pt):.2f} mm legibility floor but only {mm(slot):.1f} mm is "
                    f"available on the {plan.kind} face.",
                    severity="error",
                    face=plan.kind,
                    component=block,
                    item_no=item_no or None,
                    needed_mm=round(mm(needed), 2),
                    available_mm=round(mm(slot), 2),
                )
            )
        for line in lines:
            key = "cn_name" if line.get("field") == "cn_name" else "latin_name"
            floor = float(floors.get(key, 0.0))
            size_mm = mm(floor_pt)
            if size_mm and size_mm < floor - 0.01:
                issues.append(
                    _issue(
                        "text_below_minimum_size",
                        f"{line.get('field')} would print at {size_mm:.2f} mm, "
                        f"below the {floor} mm floor.",
                        severity="error",
                        face=plan.kind,
                        component=str(line.get("field")),
                        item_no=item_no or None,
                        size_mm=round(size_mm, 3),
                    )
                )
    return issues


def score_face(atlas: Atlas, plan: FacePlan) -> tuple[float, dict[str, float]]:
    """Rank a passing candidate. Higher is better; components are 0..1."""
    weights = atlas.scoring["weights"]
    frame = atlas.frame

    used = 0.0
    for _, box in plan.occupied_boxes():
        used += box.width * box.height
    area = max(1e-6, plan.content.width * plan.content.height)
    # Real marks are not densely packed; ~45% ink-box coverage is the healthy
    # middle, so reward proximity to that rather than raw maximisation.
    utilisation = max(0.0, 1.0 - abs(min(used / area, 1.0) - 0.45) / 0.45)

    if plan.barcode is not None:
        target = float(atlas.constraints["barcode"]["warn_below_x_mm"])
        ratio = plan.barcode.x_dimension_mm / target if target else 0.0
        barcode_score = max(0.0, min(1.0, ratio))
    else:
        barcode_score = 0.0

    item_cfg = plan.fields.get("item_no", {})
    readability = max(0.0, min(1.0, mm(float(item_cfg.get("fontsize", 0.0))) / 20.0))

    edges: list[float] = []
    for _, box in plan.occupied_boxes():
        edges.extend([box.x0 - plan.content.x0, plan.content.x1 - box.x1])
    alignment = _alignment_score(edges, plan.content.width)

    # Closeness of this face's aspect to the template's: how far the design had
    # to deviate from the approved standard. The preference for the standard
    # family only applies while the face is inside the band that family was
    # measured to work in - otherwise it would keep winning on faces the factory
    # visibly reflows.
    similarity = 1.0 / (1.0 + abs(plan.aspect - frame.panel.aspect))
    band = atlas.families.get(plan.family, {}).get("applies_when", {})
    in_band = (
        float(band.get("aspect_min", 0.0)) <= plan.aspect <= float(band.get("aspect_max", 1e9))
    )
    if not in_band:
        similarity *= 0.5
    elif plan.family.endswith("_standard"):
        similarity = min(1.0, similarity + 0.25)

    parts = {
        "area_utilisation": utilisation,
        "readability": readability,
        "barcode_magnification": barcode_score,
        "alignment": alignment,
        "standard_similarity": similarity,
    }
    total_weight = sum(float(weights[key]) for key in parts)
    total = sum(float(weights[key]) * value for key, value in parts.items()) / total_weight
    return total, parts


def _alignment_score(edges: list[float], width: float) -> float:
    """Reward layouts whose elements share edges, as designers do."""
    if not edges or width <= 0:
        return 0.0
    tolerance = max(1.0, width * 0.01)
    clusters: list[list[float]] = []
    for value in sorted(edges):
        if clusters and abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    shared = sum(len(c) for c in clusters if len(c) > 1)
    return min(1.0, shared / len(edges))


def select_face(
    atlas: Atlas, candidates: list[FacePlan], item: dict[str, Any]
) -> tuple[FacePlan | None, list[dict[str, Any]], float, float | None]:
    """Validate then rank. Returns (winner, issues, score, runner_up_score)."""
    evaluated: list[tuple[float, FacePlan, list[dict[str, Any]], bool]] = []
    rejected: list[dict[str, Any]] = []
    for plan in candidates:
        issues = validate_face(atlas, plan, item)
        blocking = [i for i in issues if i.get("severity") == "error"]
        if blocking:
            rejected.extend(
                {**issue, "rejected_family": plan.family} for issue in blocking
            )
            continue
        score, parts = score_face(atlas, plan)
        plan.notes.append(
            "score " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(parts.items()))
        )
        band = atlas.families.get(plan.family, {}).get("applies_when", {})
        in_band = (
            float(band.get("aspect_min", 0.0))
            <= plan.aspect
            <= float(band.get("aspect_max", 1e9))
        )
        evaluated.append((score, plan, issues, in_band))

    if not evaluated:
        return None, rejected, 0.0, None

    # The measured aspect bands decide which family applies; scoring only ranks
    # families that are equally applicable. Candidates are still built outside
    # their band so a face whose proper family fails validation can fall back to
    # a neighbour instead of failing outright - but a neighbour never outscores
    # the family the factory evidence actually supports for this aspect.
    in_band_only = [entry for entry in evaluated if entry[3]]
    if in_band_only:
        for entry in evaluated:
            if not entry[3]:
                rejected.append(
                    {
                        "severity": "info",
                        "code": "family_outside_declared_aspect_band",
                        "rejected_family": entry[1].family,
                        "aspect": round(entry[1].aspect, 3),
                    }
                )
        evaluated = in_band_only

    evaluated.sort(key=lambda entry: entry[0], reverse=True)
    best_score, best_plan, best_issues, _ = evaluated[0]
    runner_up = evaluated[1][0] if len(evaluated) > 1 else None
    # Keep why the alternatives lost. Without this, a family silently rejected
    # on every SKU looks identical to a family that was never a contender, and
    # the thresholds become impossible to tune against real data.
    for issue_entry in rejected:
        best_plan.notes.append(
            "rejected %s: %s" % (issue_entry.get("rejected_family"), issue_entry.get("code"))
        )
    return best_plan, best_issues, best_score, runner_up


__all__ = ["score_face", "select_face", "validate_face"]
