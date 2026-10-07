"""Tunable dials for a built layout, for owner preference calibration.

The layout engine has no adjustable parameters: each family produces exactly one
arrangement for a given face. That is why the existing scorer never chooses
anything - measured over 20 faces, the aspect band picks the family and there is
never a runner-up to rank.

To learn what the owner actually prefers we need candidates that differ, and
differ in a way a person can answer about. A pair of layouts that differs in five
places teaches nothing: "the left one is better" cannot be attributed. So each
dial here changes exactly one thing and leaves everything else alone.

Dials operate on a finished ``FacePlan`` rather than inside the builders. That
keeps the engine untouched, makes each change trivially reversible, and means a
dial can be applied to any family - including ones written later.

Generated variants are filtered by the existing plan-level ``validate_face``
before anyone is shown them, so no comparison ever asks the owner to choose
between two marks one of which cannot be printed.
"""

from .dials import (
    DIALS,
    REBUILD_DIALS,
    Dial,
    apply_dial,
    describe_dial,
    patched_inset_policy,
    scale_plan,
    spread_plan,
)

__all__ = [
    "DIALS",
    "REBUILD_DIALS",
    "Dial",
    "apply_dial",
    "describe_dial",
    "patched_inset_policy",
    "scale_plan",
    "spread_plan",
]
