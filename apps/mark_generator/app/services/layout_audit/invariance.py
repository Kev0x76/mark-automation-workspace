"""Scale invariance and monotonicity: the two properties a layout owes you.

If a layout is genuinely a function of content and container, then scaling the
container by ``k`` must scale every length it produces by ``k``:

    f(content, k * container) == k * f(content, container)

That is not a style opinion and it needs no factory sample, no threshold and no
owner decision - it follows from the layout being a function of the container at
all. Any absolute constant baked into the code breaks it, which makes this the
cheapest possible detector for a whole class of defect the project has now hit
twice: "the constant did not follow the size".

Two rules legitimately break the equality, and both are *declared*, not implied:

* the design inset is ``ratio * panel height`` clamped into ``[min_mm, max_mm]``
  (``adaptive_layout.json -> inset``), measured from factory files. Once the
  clamp engages, content grows faster than the panel and every length keyed to
  content follows. :func:`inset_regime` says which side of the clamp a panel is
  on so a caller can compare inside one regime.
* a hard physical bound - the barcode's X-dimension floor, a legibility floor -
  binds at one size and not the other.

Everything else that fails to scale is an undeclared constant. The report says
so rather than guessing, because the fix is not always "multiply it by scale":
sometimes the constant is right and belongs in the policy file with its
evidence, which is where the project already decided measured constants live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from ..adaptive_layout.atlas import Atlas
from ..adaptive_layout.families import build_candidates
from ..adaptive_layout.model import Box, FaceKind, FacePlan, pt


# Numbers that are ratios, counts or colour channels: they must stay *identical*
# when the panel scales. Everything else in a plan is a length in points and
# must scale with it.
DIMENSIONLESS_KEYS = frozenset(
    {
        "leading_ratio",
        "readout_xscale",
        "color",
        "max_lines",
        "max_lines_per_name",
        "preferred_max_lines",
        "source_page",
        "rotation_deg",
        "physical_rotation_deg",
    }
)

# Keys whose value is a name, not a measurement.
_TEXT_KEYS = frozenset(
    {"fit", "font", "field", "digits_side", "hri_grouping", "hri_layout", "reading",
     "orientation", "prefix", "name"}
)


@dataclass(frozen=True)
class Deviation:
    """One number that did not follow the container."""

    path: str
    key: str
    base: float
    scaled: float
    expected_ratio: float

    @property
    def ratio(self) -> float:
        return self.scaled / self.base if abs(self.base) > 1e-9 else float("inf")

    @property
    def error(self) -> float:
        """How far off, as a fraction of what was expected."""
        return abs(self.ratio - self.expected_ratio) / max(1e-9, self.expected_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "key": self.key,
            "base": round(self.base, 5),
            "scaled": round(self.scaled, 5),
            "ratio": round(self.ratio, 5),
            "expected_ratio": round(self.expected_ratio, 5),
            "error": round(self.error, 5),
            "reads_as": "constant" if abs(self.ratio - 1.0) < 1e-3 else "partly_scaled",
        }


@dataclass
class InvarianceReport:
    kind: str
    family: str
    base_panel_mm: tuple[float, float]
    factor: float
    deviations: list[Deviation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.deviations

    def to_dict(self) -> dict[str, Any]:
        return {
            "face": self.kind,
            "family": self.family,
            "base_panel_mm": [round(v, 2) for v in self.base_panel_mm],
            "factor": self.factor,
            "deviation_count": len(self.deviations),
            "deviations": [d.to_dict() for d in self.deviations],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Walking a plan's geometry
# --------------------------------------------------------------------------- #


def _walk(node: Any, path: str, key: str | None) -> Iterator[tuple[str, str, float]]:
    if isinstance(node, dict):
        for name, value in node.items():
            yield from _walk(value, f"{path}/{name}", name)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}/[{index}]", key)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        if key in _TEXT_KEYS:
            return
        yield path, key or "", float(node)


def geometry_leaves(plan: FacePlan) -> dict[str, tuple[str, float]]:
    """Every measurable number in a plan, addressed by a stable path.

    Static placements contribute their *target* only. The clip rectangle is an
    address inside ``base.pdf`` and is meant to be identical at every size, so
    including it would report the artwork source as a scaling failure.
    """
    out: dict[str, tuple[str, float]] = {}
    for path, key, value in _walk(plan.fields, "fields", None):
        out[path] = (key, value)
    for placement in plan.statics:
        base = f"statics/{placement.name}"
        for index, value in enumerate(placement.target.as_tuple()):
            out[f"{base}/target/[{index}]"] = ("target", float(value))
    if plan.barcode is not None:
        out["barcode/bars/[0]"] = ("bars", plan.barcode.bars.x0)
        out["barcode/bars/[1]"] = ("bars", plan.barcode.bars.y0)
        out["barcode/bars/[2]"] = ("bars", plan.barcode.bars.x1)
        out["barcode/bars/[3]"] = ("bars", plan.barcode.bars.y1)
        out["barcode/x_dimension_mm"] = ("x_dimension_mm", plan.barcode.x_dimension_mm)
        out["barcode/bar_length_mm"] = ("bar_length_mm", plan.barcode.bar_length_mm)
    out["content/[0]"] = ("content", plan.content.x0)
    out["content/[1]"] = ("content", plan.content.y0)
    out["content/[2]"] = ("content", plan.content.x1)
    out["content/[3]"] = ("content", plan.content.y1)
    return out


# --------------------------------------------------------------------------- #
# The declared exception: the inset clamp
# --------------------------------------------------------------------------- #


def inset_regime(atlas: Atlas, panel_height_mm: float) -> str:
    """``"min"``, ``"proportional"`` or ``"max"`` for this panel height.

    Comparing two panels only proves something about the layout when both sit in
    the same regime; across a clamp boundary the content box legitimately grows
    at a different rate than the panel and every honest length follows it.
    """
    rule = atlas.policy["inset"]
    raw = panel_height_mm * float(rule["ratio"])
    if raw <= float(rule["min_mm"]):
        return "min"
    if raw >= float(rule["max_mm"]):
        return "max"
    return "proportional"


def proportional_inset_span_mm(atlas: Atlas) -> tuple[float, float]:
    """Panel heights over which the inset is a pure ratio of height."""
    rule = atlas.policy["inset"]
    ratio = float(rule["ratio"])
    return float(rule["min_mm"]) / ratio, float(rule["max_mm"]) / ratio


# --------------------------------------------------------------------------- #
# The properties
# --------------------------------------------------------------------------- #


def _plans_by_family(
    atlas: Atlas, kind: FaceKind, panel_mm: tuple[float, float], item: dict[str, Any]
) -> dict[str, FacePlan]:
    panel = Box(0.0, 0.0, pt(panel_mm[0]), pt(panel_mm[1]))
    return {plan.family: plan for plan in build_candidates(atlas, kind, panel, item)}


def _barcode_at_bound(atlas: Atlas, plan: FacePlan) -> bool:
    """True when a physical barcode bound is binding, so scaling cannot hold."""
    if plan.barcode is None:
        return False
    limits = atlas.constraints["barcode"]
    x_mm = plan.barcode.x_dimension_mm
    floor = float(limits.get("fail_below_x_mm", 0.0))
    ceiling = limits.get("max_x_mm")
    if floor and x_mm <= floor * 1.001:
        return True
    if ceiling and x_mm >= float(ceiling) * 0.999:
        return True
    return False


def _legibility_clamp_explains(atlas: Atlas, path: str, base: float, scaled: float, factor: float) -> bool:
    """Only exempt the declared side-name clamp, within its possible range.

    A value merely matching some unrelated floor must not hide a size defect.
    For max(floor, k * raw), a clamped base can grow anywhere between the floor
    and k * floor; a decrease or excessive increase cannot come from that clamp.
    """
    if path != "fields/es_name/min_fontsize" or factor < 1.0:
        return False
    floor = pt(float(atlas.constraints["min_font_mm"]["table_value"]))
    return abs(base - floor) < 1e-6 and floor - 1e-6 <= scaled <= floor * factor + 1e-6


def scale_invariance(
    atlas: Atlas,
    kind: FaceKind,
    panel_mm: tuple[float, float],
    item: dict[str, Any],
    *,
    factor: float = 1.25,
    tolerance: float = 0.005,
) -> list[InvarianceReport]:
    """Build the same face at two sizes and report every length that disagreed.

    ``tolerance`` is a fraction of the expected ratio. 0.5% is far looser than
    float error and far tighter than any real baked constant, which shows up as
    a ratio of exactly 1.0.
    """
    base = _plans_by_family(atlas, kind, panel_mm, item)
    scaled_mm = (panel_mm[0] * factor, panel_mm[1] * factor)
    scaled = _plans_by_family(atlas, kind, scaled_mm, item)

    reports: list[InvarianceReport] = []
    for family in sorted(set(base) & set(scaled)):
        plan_a, plan_b = base[family], scaled[family]
        report = InvarianceReport(
            kind=str(kind), family=family, base_panel_mm=panel_mm, factor=factor
        )

        regime_a = inset_regime(atlas, panel_mm[1])
        regime_b = inset_regime(atlas, scaled_mm[1])
        if regime_a != regime_b:
            report.notes.append(
                f"inset clamp regime changes between the two sizes "
                f"({regime_a} -> {regime_b}); comparison is not conclusive"
            )
        if _barcode_at_bound(atlas, plan_a) or _barcode_at_bound(atlas, plan_b):
            report.notes.append(
                "a barcode X-dimension bound is binding at one of the two sizes"
            )

        leaves_a = geometry_leaves(plan_a)
        for path, (key, value_b) in geometry_leaves(plan_b).items():
            entry = leaves_a.get(path)
            if entry is None:
                report.notes.append(f"{path} exists only at the larger size")
                continue
            _, value_a = entry
            expected = 1.0 if key in DIMENSIONLESS_KEYS else factor
            if abs(value_a) < 1e-9 and abs(value_b) < 1e-9:
                continue
            deviation = Deviation(path, key, value_a, value_b, expected)
            if deviation.error > tolerance:
                if _legibility_clamp_explains(atlas, path, value_a, value_b, factor):
                    # A declared physical clamp binding on the smaller face, like
                    # a barcode bound: noted, not a scaling defect.
                    report.notes.append(f"{path} sits on the legibility floor at the smaller size")
                    continue
                report.deviations.append(deviation)

        if plan_a.scale > 0:
            scale_dev = Deviation("scale", "scale", plan_a.scale, plan_b.scale, factor)
            if scale_dev.error > tolerance:
                report.deviations.append(scale_dev)

        report.deviations.sort(key=lambda d: d.error, reverse=True)
        reports.append(report)
    return reports


def monotonicity(
    atlas: Atlas,
    kind: FaceKind,
    panel_mm: tuple[float, float],
    item: dict[str, Any],
    *,
    factors: tuple[float, ...] = (1.0, 1.15, 1.3),
    tolerance_pt: float = 0.25,
) -> list[dict[str, Any]]:
    """A bigger face may not make any element smaller.

    Weaker than scale invariance and true across clamp boundaries, so this one
    holds everywhere - including the sizes where the inset saturates.
    """
    series: list[tuple[float, dict[str, tuple[str, float]]]] = []
    for factor in factors:
        plans = _plans_by_family(
            atlas, kind, (panel_mm[0] * factor, panel_mm[1] * factor), item
        )
        for family, plan in plans.items():
            series.append((factor, {f"{family}::{k}": v for k, v in geometry_leaves(plan).items()}))

    # Sizes are the only thing that must not shrink; positions may move freely.
    size_paths = ("fontsize", "min_fontsize", "text_fontsize", "x_dimension_mm",
                  "bar_length_mm", "max_width")
    findings: list[dict[str, Any]] = []
    for index in range(1, len(series)):
        prev_factor, prev = series[index - 1]
        curr_factor, curr = series[index]
        if curr_factor <= prev_factor:
            continue
        for path, (key, value) in curr.items():
            if key not in size_paths:
                continue
            entry = prev.get(path)
            if entry is None:
                continue
            _, previous = entry
            if value < previous - tolerance_pt:
                findings.append(
                    {
                        "severity": "error",
                        "code": "element_shrinks_on_a_larger_face",
                        "face": str(kind),
                        "path": path,
                        "message": (
                            f"{path} is {value:.3f} pt on the "
                            f"{panel_mm[0] * curr_factor:.0f} x {panel_mm[1] * curr_factor:.0f} mm face "
                            f"but {previous:.3f} pt on the smaller "
                            f"{panel_mm[0] * prev_factor:.0f} x {panel_mm[1] * prev_factor:.0f} mm one."
                        ),
                        "smaller_face_value_pt": round(previous, 4),
                        "larger_face_value_pt": round(value, 4),
                    }
                )
    return findings


def sweep(
    atlas: Atlas,
    item: dict[str, Any],
    *,
    factor: float = 1.25,
    tolerance: float = 0.005,
) -> list[InvarianceReport]:
    """Run scale invariance over one panel per measured aspect band.

    Heights are chosen inside the proportional inset span at both sizes so the
    declared clamp never confuses the result; aspects come from the bands in
    ``adaptive_layout.json``, so every family the policy claims to cover is
    exercised.
    """
    low, high = proportional_inset_span_mm(atlas)
    height = round(min(high / factor, max(low * 1.05, 150.0)), 1)

    aspects: dict[str, list[float]] = {"front": [], "side": []}
    for name, cfg in atlas.families.items():
        band = cfg.get("applies_when", {})
        kind = "side" if name.startswith("side") else "front"
        lo = float(band.get("aspect_min", 0.5))
        hi = float(band.get("aspect_max", 4.0))
        hi = min(hi, 12.0)
        aspects[kind].append(round((lo + hi) / 2.0, 3))

    reports: list[InvarianceReport] = []
    for kind, values in aspects.items():
        for aspect in sorted(set(values)):
            reports.extend(
                scale_invariance(
                    atlas,
                    kind,  # type: ignore[arg-type]
                    (round(height * aspect, 1), height),
                    item,
                    factor=factor,
                    tolerance=tolerance,
                )
            )
    return reports


__all__ = [
    "DIMENSIONLESS_KEYS",
    "Deviation",
    "InvarianceReport",
    "geometry_leaves",
    "inset_regime",
    "monotonicity",
    "proportional_inset_span_mm",
    "scale_invariance",
    "sweep",
]
