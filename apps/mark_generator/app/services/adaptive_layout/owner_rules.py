"""Structural rules confirmed in the owner's 2026-09-15 calibration.

The two wide-front examples establish adjacency and ordering, not a curve of
optimal coordinates for every carton. These allocations are a first production
candidate for that structure: all artwork still fits uniformly and every face
must pass the normal hard constraints and final-PDF checks.
"""

from __future__ import annotations

import math


OWNER_RULE_REVISION = "owner-20260915"


def wants_owner_column(front_aspect: float | None, side_aspect: float | None) -> bool:
    """Both exact owner conditions must hold; a missing face is not evidence."""
    if front_aspect is None or side_aspect is None:
        return False
    if not math.isfinite(front_aspect) or not math.isfinite(side_aspect) or side_aspect <= 0:
        return False
    # Ratios are derived from millimetres converted to points, so accept only
    # floating-point rounding at the inclusive boundary, not a design band.
    return front_aspect >= 3.90 - 1e-12 and front_aspect / side_aspect >= 2.50 - 1e-12


# Representative allocation from the confirmed four-column relationship.
# 55% is deliberately a prototype allocation inside the observed 54-58% band,
# not a fit through the two samples or a claim of a universal optimum.
#
# "names" holds the Chinese name and the IT/ES/EN names as one block (owner,
# 2026-09-16: 贴紧). With two regions the Chinese name sat at the top of its own
# slot and printed 7.5-28.9 mm above the Latin names; one block stacks them at
# the template's block gap, as front_standard does.
FRONT_COLUMN_REGIONS = {
    "item_no_bar": {"x": [0.00, 0.55], "y": [0.03, 0.97]},
    "kmt_logo": {"x": [0.565, 0.75], "y": [0.06, 0.30]},
    "names": {"x": [0.565, 0.75], "y": [0.35, 0.95]},
    "lineart": {"x": [0.765, 0.915], "y": [0.06, 0.59]},
    "origin_text": {"x": [0.765, 0.915], "y": [0.64, 0.73]},
    "icon_row": {"x": [0.765, 0.915], "y": [0.80, 0.95]},
    "barcode": {"x": [0.93, 1.00], "y": [0.04, 0.96]},
}


# Owner side structure, calibration doc section 5, identical on the 4.50 side
# (360 x 200 mm) and the 7.35 side (170 x 170 mm): item-number bar across the
# top, then a left column QTY over MEAS, a middle column NAME / N.W. / G.W.,
# and the vertical barcode at the right edge. Only order and membership are
# taken from the owner; sizes come from base.pdf and the face.
OWNER_SIDE_COLUMNS: tuple[tuple[str, ...], ...] = (
    ("cell_qty", "cell_meas"),
    ("cell_name", "cell_nw", "cell_gw"),
)

#: Bar height as a share of the content height on the owner's two side faces
#: (4.50 side: 0.419, 7.35 side: 0.383). Evidence for a band, not a target: the
#: grid takes template.json's own side bar share, which falls inside it.
OWNER_SIDE_BAR_SHARE_OBSERVED = (0.383, 0.419)

