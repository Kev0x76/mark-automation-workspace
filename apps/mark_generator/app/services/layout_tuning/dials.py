"""One dial, one visible change.

Each dial is a pure function ``FacePlan -> FacePlan`` parameterised by a single
number, where 1.0 always means "what the engine does today". That gives every
comparison a defined origin: whatever the owner picks can be read as a
multiplier on current behaviour, which is what has to go back into the policy
file at the end.

Two of them - ``inset`` and ``spread`` - are deliberately a pair. The 2026-07-29
experiment measured that pushing elements apart improved the whitespace figure
while looking visibly worse, so "how much of the panel is used" and "how far
apart things sit" have to be calibrated against each other rather than one at a
time. They are kept separate here so the owner can be asked about each without
the other moving.

``scale_plan`` stays although no dial uses it. Why it is not a dial is itself a
measurement, recorded at :data:`REBUILD_DIALS`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from ..adaptive_layout.model import Box, FacePlan

# Numbers in a field config that are ratios or counts. Everything else is a
# length in points. Same split ``layout_audit.invariance`` uses; kept in one
# place here so a dial and the invariance check never disagree about what a key
# means.
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
    }
)

# Keys that hold coordinates rather than sizes, addressed by shape.
_RECT_KEYS = ("rect", "bar_rect", "bars_rect")
_POINT_KEYS = ("baseline", "readout_origin")
_X_KEYS = ("x0",)
_Y_KEYS = ("baseline_y", "top_y", "bottom_y")

# Field subtrees that belong to a single drawn element, so a dial can move one
# element without touching its neighbours.
ELEMENT_FIELDS = ("item_no", "names", "cn_name", "lineart", "qty", "es_name", "barcode")


@dataclass(frozen=True)
class Dial:
    """A single adjustable property of a layout."""

    key: str
    label: str
    question: str
    low: float
    high: float
    apply: Callable[[FacePlan, float], FacePlan]
    unit: str = "x"

    def ladder(self, steps: int) -> list[float]:
        """Evenly spaced values across the dial's range, always including 1.0."""
        span = [self.low + (self.high - self.low) * i / (steps - 1) for i in range(steps)]
        if not any(abs(v - 1.0) < 1e-6 for v in span):
            span.append(1.0)
        return sorted(round(v, 4) for v in span)


# --------------------------------------------------------------------------- #
# Generic geometry transforms
# --------------------------------------------------------------------------- #


def _scale_value(value: float, centre: float, factor: float) -> float:
    return centre + (value - centre) * factor


def _scale_box(box: Box, cx: float, cy: float, factor: float) -> Box:
    return Box(
        _scale_value(box.x0, cx, factor),
        _scale_value(box.y0, cy, factor),
        _scale_value(box.x1, cx, factor),
        _scale_value(box.y1, cy, factor),
    )


def _scale_cfg(cfg: dict[str, Any], cx: float, cy: float, factor: float) -> dict[str, Any]:
    """Scale one field config about a centre, coordinates and sizes together."""
    out = dict(cfg)
    for key in _RECT_KEYS:
        if key in out:
            x0, y0, x1, y1 = (float(v) for v in out[key])
            out[key] = [
                _scale_value(x0, cx, factor), _scale_value(y0, cy, factor),
                _scale_value(x1, cx, factor), _scale_value(y1, cy, factor),
            ]
    for key in _X_KEYS:
        if key in out:
            out[key] = _scale_value(float(out[key]), cx, factor)
    for key in _Y_KEYS:
        if key in out:
            out[key] = _scale_value(float(out[key]), cy, factor)
    for key in _POINT_KEYS:
        if key in out:
            px, py = (float(v) for v in out[key])
            out[key] = [_scale_value(px, cx, factor), _scale_value(py, cy, factor)]
    if "lines" in out:
        lines = []
        for line in out["lines"]:
            entry = dict(line)
            bx, by = (float(v) for v in entry["baseline"])
            entry["baseline"] = [_scale_value(bx, cx, factor), _scale_value(by, cy, factor)]
            entry["fontsize"] = float(entry["fontsize"]) * factor
            lines.append(entry)
        out["lines"] = lines
    for key, value in list(out.items()):
        if key in DIMENSIONLESS_KEYS or key in _RECT_KEYS or key in _POINT_KEYS:
            continue
        if key in _X_KEYS or key in _Y_KEYS or key == "lines":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = float(value) * factor
    return out


def _translate_cfg(cfg: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Move one field config without changing any size."""
    out = dict(cfg)
    for key in _RECT_KEYS:
        if key in out:
            x0, y0, x1, y1 = (float(v) for v in out[key])
            out[key] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
    for key in _X_KEYS:
        if key in out:
            out[key] = float(out[key]) + dx
    for key in _Y_KEYS:
        if key in out:
            out[key] = float(out[key]) + dy
    for key in _POINT_KEYS:
        if key in out:
            px, py = (float(v) for v in out[key])
            out[key] = [px + dx, py + dy]
    if "lines" in out:
        lines = []
        for line in out["lines"]:
            entry = dict(line)
            bx, by = (float(v) for v in entry["baseline"])
            entry["baseline"] = [bx + dx, by + dy]
            lines.append(entry)
        out["lines"] = lines
    return out


def _cfg_box(cfg: dict[str, Any]) -> Box | None:
    for key in _RECT_KEYS:
        if key in cfg:
            return Box(*(float(v) for v in cfg[key]))
    return None


# --------------------------------------------------------------------------- #
# The dials
# --------------------------------------------------------------------------- #


def scale_plan(plan: FacePlan, factor: float) -> FacePlan:
    """Grow or shrink the whole design about the content centre.

    Sizes and positions move together, so the arrangement is unchanged and only
    how much of the panel it occupies differs. This is the "fill the panel or
    leave air around it" question with nothing else attached to it.
    """
    out = copy.deepcopy(plan)
    cx = (plan.content.x0 + plan.content.x1) / 2.0
    cy = (plan.content.y0 + plan.content.y1) / 2.0

    for placement in out.statics:
        placement.target = _scale_box(placement.target, cx, cy, factor)
    if out.barcode is not None:
        out.barcode.bars = _scale_box(out.barcode.bars, cx, cy, factor)
        out.barcode.x_dimension_mm *= factor
        out.barcode.bar_length_mm *= factor
        out.barcode.magnification *= factor
    out.fields = {
        name: (_scale_cfg(cfg, cx, cy, factor) if isinstance(cfg, dict) else cfg)
        for name, cfg in out.fields.items()
    }
    out.notes = list(plan.notes) + [f"dial fill={factor:.3f}"]
    return out


def spread_plan(plan: FacePlan, factor: float) -> FacePlan:
    """Push elements apart or pull them together, without resizing anything.

    The counterweight to :func:`scale_plan`. Spreading raises the ink's distance
    from its own centre and breaks shared edges while leaving every element the
    size it was, which is exactly the change the 2026-07-29 spacing experiment
    made - and the one that improved the whitespace number while looking worse.
    """
    out = copy.deepcopy(plan)
    cx = (plan.content.x0 + plan.content.x1) / 2.0
    cy = (plan.content.y0 + plan.content.y1) / 2.0

    def offset(box: Box) -> tuple[float, float]:
        bx = (box.x0 + box.x1) / 2.0
        by = (box.y0 + box.y1) / 2.0
        return (bx - cx) * (factor - 1.0), (by - cy) * (factor - 1.0)

    for placement in out.statics:
        dx, dy = offset(placement.target)
        placement.target = Box(
            placement.target.x0 + dx, placement.target.y0 + dy,
            placement.target.x1 + dx, placement.target.y1 + dy,
        )
    if out.barcode is not None:
        dx, dy = offset(out.barcode.bars)
        out.barcode.bars = Box(
            out.barcode.bars.x0 + dx, out.barcode.bars.y0 + dy,
            out.barcode.bars.x1 + dx, out.barcode.bars.y1 + dy,
        )

    fields: dict[str, Any] = {}
    for name, cfg in out.fields.items():
        if not isinstance(cfg, dict):
            fields[name] = cfg
            continue
        box = _cfg_box(cfg)
        if box is None and "top_y" in cfg:
            # A names block carries no rect and, on the standard families, no
            # x0 either - its horizontal position lives in each line's
            # baseline. Round 1 fell back to the content centre here, which put
            # the synthesised centre at 472.6 pt where the block really sits at
            # 152.3 pt, so the dial pushed the names the wrong way on every
            # front_standard face.
            left = cfg.get("x0")
            if left is None:
                baselines = [float(line["baseline"][0]) for line in cfg.get("lines", [])]
                left = min(baselines) if baselines else cx
            left = float(left)
            box = Box(
                left, float(cfg["top_y"]),
                left + float(cfg.get("max_width", 0.0)),
                float(cfg["bottom_y"]),
            )
        if box is None:
            fields[name] = cfg
            continue
        dx, dy = offset(box)
        fields[name] = _translate_cfg(cfg, dx, dy)
    out.fields = fields
    out.notes = list(plan.notes) + [f"dial spread={factor:.3f}"]
    return out


def cn_name_plan(plan: FacePlan, factor: float) -> FacePlan:
    """Resize the Chinese product name relative to the three Latin ones.

    Only the Chinese line's point size changes. The renderer re-flows the whole
    name block from the per-line sizes, so the rest of the block follows without
    anything else being touched here.

    The 2026-09-02 review measured our Chinese name at 1.38 times the trilingual
    names where the factory sits at 0.88, and called it "Chinese too big" - so
    this is the dial that complaint lands on.
    """
    out = copy.deepcopy(plan)
    for block in ("names", "cn_name"):
        cfg = out.fields.get(block)
        if not isinstance(cfg, dict) or "lines" not in cfg:
            continue
        lines = []
        for line in cfg["lines"]:
            entry = dict(line)
            if entry.get("field") == "cn_name":
                entry["fontsize"] = float(entry["fontsize"]) * factor
            lines.append(entry)
        cfg = dict(cfg)
        cfg["lines"] = lines
        out.fields[block] = cfg
    out.notes = list(plan.notes) + [f"dial cn_name={factor:.3f}"]
    return out


def item_bar_plan(plan: FacePlan, factor: float) -> FacePlan:
    """Change how much of the panel width the item-number bar takes.

    Width only: the bar keeps its height and its top edge, so nothing below it
    moves. Digits stay centred and are capped so they cannot outgrow a narrowed
    bar.

    The factory does not use one value here - measured at 92-95% of face width
    below aspect 2.36, 68-86% above 2.62, and 49.4% at 7.35 - so what the right
    value is on a given face is exactly a preference question.
    """
    out = copy.deepcopy(plan)
    cfg = out.fields.get("item_no")
    if not isinstance(cfg, dict) or "bar_rect" not in cfg:
        return out

    x0, y0, x1, y1 = (float(v) for v in cfg["bar_rect"])
    new_width = (x1 - x0) * factor
    new_rect = [x0, y0, x0 + new_width, y1]

    cfg = dict(cfg)
    cfg["bar_rect"] = new_rect
    cfg["max_width"] = new_width
    # The point size is left alone. Round 1 scaled it with the bar on every
    # value below 1.0, so the "longer bar" option was always also a larger item
    # number - the audit measured digit cap height climbing 358 -> 426 px across
    # that ladder while the bar height never moved, which makes the two
    # explanations of the answers inseparable.
    #
    # Nothing needs to replace it: KmtTemplateRenderer._draw_item_no already
    # condenses the digits horizontally when they exceed max_width, so a
    # narrowed bar still holds its number. Cap height - the legibility cue -
    # then stays constant across the whole ladder, which is what makes this a
    # question about bar length.
    out.fields["item_no"] = cfg

    for placement in out.statics:
        if placement.name == "item_no_bar":
            placement.target = Box(*new_rect)
    out.notes = list(plan.notes) + [f"dial item_bar={factor:.3f}"]
    return out


def lineart_plan(plan: FacePlan, factor: float) -> FacePlan:
    """Resize the line art about its own centre, nothing else moves."""
    out = copy.deepcopy(plan)
    cfg = out.fields.get("lineart")
    if not isinstance(cfg, dict) or "rect" not in cfg:
        return out
    box = Box(*(float(v) for v in cfg["rect"]))
    cx = (box.x0 + box.x1) / 2.0
    cy = (box.y0 + box.y1) / 2.0
    cfg = dict(cfg)
    cfg["rect"] = list(_scale_box(box, cx, cy, factor).as_tuple())
    out.fields["lineart"] = cfg
    out.notes = list(plan.notes) + [f"dial lineart={factor:.3f}"]
    return out


#: Dials that cannot be applied to a finished plan, because they change what the
#: builder would have produced in the first place.
#:
#: ``inset`` is the real "how full is the panel" control. The design inset sets
#: the live area and every element is sized from it, so shrinking the inset
#: grows the whole mark the way the engine would grow it.
#:
#: A post-hoc uniform scale was tried first and hit a wall: on the three study
#: faces it could only reach 1.01-1.06 before ink crossed the printable margin,
#: because the design already sits close to it. That is a fact about the panel,
#: not a preference, and a dial that can only move one way asks nothing.
REBUILD_DIALS: dict[str, dict[str, Any]] = {
    "inset": {
        "label": "留白多少",
        "question": "唛头四周该留多宽的空白",
        "low": 0.45,
        "high": 2.0,
        "evidence": (
            "工厂实测内缩：面高 >=120 mm 用 10 mm，<120 mm 用 7.5 mm；"
            "模板自身 6.21 mm / 165.1 mm = 3.76%"
        ),
    }
}


def patched_inset_policy(policy: dict[str, Any], factor: float) -> dict[str, Any]:
    """A copy of the layout policy with the design inset scaled by ``factor``.

    All three numbers move together so the clamp keeps its shape. Scaling only
    the ratio would quietly move a face from one clamp regime to another, and
    the comparison would then be about two different rules rather than one
    value.
    """
    out = copy.deepcopy(policy)
    rule = out["inset"]
    for key in ("ratio", "min_mm", "max_mm"):
        rule[key] = float(rule[key]) * factor
    return out


DIALS: dict[str, Dial] = {
    "spread": Dial(
        key="spread",
        label="松紧",
        question="元素之间是抱团还是摊开",
        low=0.88,
        high=1.12,
        apply=spread_plan,
    ),
    "cn_name": Dial(
        key="cn_name",
        label="中文名大小",
        question="中文名相对三语名该多大",
        low=0.65,
        high=1.5,
        apply=cn_name_plan,
    ),
    "item_bar": Dial(
        key="item_bar",
        label="货号黑条长度",
        question="货号黑条该占面宽多少",
        # Round 1 ran 0.55-1.0, which put the status quo AT the ceiling: no
        # value longer than today's bar was ever shown, so "prefers longer"
        # could only ever report a direction, never locate an optimum - and
        # 0.925, the answer it returned, is the largest number that ladder's
        # estimator can produce. A dial has to bracket 1.0 to say anything
        # about it. Values above 1.0 overflow the design box on some faces and
        # are dropped by the feasibility probe, which is the correct behaviour:
        # the ladder then honestly reports that this face has no headroom.
        low=0.60,
        high=1.35,
        apply=item_bar_plan,
    ),
    "lineart": Dial(
        key="lineart",
        label="线稿大小",
        question="产品线稿该多大",
        low=0.7,
        high=1.35,
        apply=lineart_plan,
    ),
}


def apply_dial(plan: FacePlan, key: str, value: float) -> FacePlan:
    return DIALS[key].apply(plan, value)


def describe_dial(key: str) -> dict[str, Any]:
    dial = DIALS[key]
    return {
        "key": dial.key,
        "label": dial.label,
        "question": dial.question,
        "range": [dial.low, dial.high],
        "unit": dial.unit,
    }


__all__ = [
    "DIALS",
    "REBUILD_DIALS",
    "DIMENSIONLESS_KEYS",
    "Dial",
    "apply_dial",
    "cn_name_plan",
    "describe_dial",
    "item_bar_plan",
    "lineart_plan",
    "patched_inset_policy",
    "scale_plan",
    "spread_plan",
]
