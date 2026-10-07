"""Acceptance criteria for generated marks: read the artwork, not the plan.

``adaptive_layout.constraints`` validates a ``FacePlan`` - the rectangles the
engine *intends* to draw. Every defect introduced by the renderer itself is
structurally invisible there: a glyph squeezed inside its box, five labels that
disagree on size, barcode digits printed over the bars, a font-size cliff, a
point/millimetre mix-up that is self-consistent inside the wrong unit system.

This package checks the other side of that boundary, in four layers:

``L0`` physical / standards
    Truth comes from GS1 and print physics. Safe area, X-dimension, quiet zones,
    module grid, decode-back, legibility floors.

``L1`` self-consistency
    Truth comes from the artwork itself, with no external reference at all:
    same-class elements must render at the same size, glyphs must keep their
    natural proportions, ink must not collide, and geometry must be invariant
    under a pure change of scale (:mod:`invariance`).

``L2`` style conformance
    Truth comes from the distribution measured on real factory marks
    (:mod:`bands`). Out-of-band is a review trigger, never a silent failure.

``L3`` regression
    Truth comes from a frozen, owner-approved golden.

The layers are deliberately kept apart from any layout engine. The same
measurements have to run against factory PDFs to calibrate L2, there is more
than one engine in the tree, and ``docs/integration_v5.md`` already sets the
precedent that a checking domain reads generated files only.
"""

from .barcodes import (
    ALLOWED_PHYSICAL_ORIENTATIONS,
    DecodedSymbol,
    DigitSide,
    GroupRelation,
    audit_barcode,
    decode_ean13,
    group_relation,
    locate_digits,
    orientation_findings,
    uniform_group_findings,
)
from .glyphs import (
    GlyphFidelity,
    aspect_ratio_bounds,
    fidelity_of_run,
    natural_advance,
    per_character_bounds,
    per_character_fidelity,
    render_reference,
    zoom_to_resolve,
)
from .invariance import (
    Deviation,
    geometry_leaves,
    inset_regime,
    monotonicity,
    scale_invariance,
)
from .raster import (
    PanelMetrics,
    alignment_error,
    ink_coverage,
    ink_mask,
    largest_empty_rect_ratio,
    measure_panel,
    uniformity,
)

__all__ = [
    "ALLOWED_PHYSICAL_ORIENTATIONS",
    "DecodedSymbol",
    "Deviation",
    "DigitSide",
    "GlyphFidelity",
    "GroupRelation",
    "PanelMetrics",
    "alignment_error",
    "aspect_ratio_bounds",
    "audit_barcode",
    "decode_ean13",
    "fidelity_of_run",
    "geometry_leaves",
    "group_relation",
    "ink_coverage",
    "ink_mask",
    "inset_regime",
    "largest_empty_rect_ratio",
    "locate_digits",
    "measure_panel",
    "monotonicity",
    "natural_advance",
    "orientation_findings",
    "per_character_bounds",
    "per_character_fidelity",
    "render_reference",
    "scale_invariance",
    "uniform_group_findings",
    "uniformity",
    "zoom_to_resolve",
]
