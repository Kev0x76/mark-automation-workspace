"""Carton-size adaptive layout for Kmt Style v1 shipping marks.

The mark's *detail* stays exactly what the current automatic standard mark
produces - static artwork is re-stamped as vector from base.pdf and all text and
barcode drawing is delegated to the existing KmtTemplateRenderer. Only the
large-panel arrangement adapts to the real printed face:

    front face = carton length x height
    side face  = carton width  x height

Two families per face, measured from real factory files: the standard
arrangement (re-proportioned, used across face aspect ~0.9-2.4) and a reflowed
arrangement for very wide, short faces (~2.5 and above).
"""

from .atlas import AdaptiveAssetError, Atlas, load_atlas, verify_atlas
from .constraints import score_face, select_face, validate_face
from .families import build_candidates, candidate_families
from .model import Box, FacePlan, LayoutPlan, LayoutRevision, mm, pt
from .pipeline import build_revision, plan_item, render_adaptive_marks
from .renderer import AdaptiveRenderer, page_geometry

__all__ = [
    "AdaptiveAssetError",
    "AdaptiveRenderer",
    "Atlas",
    "Box",
    "FacePlan",
    "LayoutPlan",
    "LayoutRevision",
    "build_candidates",
    "build_revision",
    "candidate_families",
    "load_atlas",
    "mm",
    "page_geometry",
    "plan_item",
    "pt",
    "render_adaptive_marks",
    "score_face",
    "select_face",
    "validate_face",
    "verify_atlas",
]
