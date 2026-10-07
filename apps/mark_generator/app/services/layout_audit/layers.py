"""Per-element ink, taken from the rendered page rather than from the plan.

Each element is drawn inside its own PDF optional content group, so the audit
can switch one on at a time and rasterise exactly the ink that element produced
- including whatever the renderer got wrong about it. This is still reading the
artwork: a layer is not the box the engine intended, it is the marks that
actually landed.

Being the generator has one advantage no downstream checker has: the
segmentation is exact. Nothing is inferred by computer vision, so a collision
result is a fact about the page rather than a fact about a segmentation model.

Tagging is done by :class:`LayerTagger`, which wraps whatever a draw call
appended to the content stream. That keeps the audit off the layout engine's
back: no drawing method has to grow an ``oc`` parameter, the existing text and
barcode code is untouched, and with tagging off the output is byte-for-byte what
it was.

Isolation renders a fresh copy of the document per layer. Toggling visibility on
a live document was measured to return stale ink once a page had already been
rasterised, and a checker that is silently wrong is worse than a slow one.

Two derived facts fall out of the same render and are worth as much as the
layers themselves:

``occlusion``
    ink present when a layer is isolated but absent from the composite - one
    element painted over another. Legitimate for a white value box on a black
    bar, a defect everywhere else, so it is reported against a whitelist.

``unattributed``
    ink in the composite that no layer explains. Either something was drawn
    without a layer, or the layer set is stale; either way every other
    per-element check is incomplete until it is empty.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import fitz
import numpy as np

from .raster import METRIC_ZOOM, background_colour, ink_mask, mask_bbox, pixmap_array


def layer_name(face: str, element: str) -> str:
    """Layer names are ``"<face>:<element>"``.

    The face prefix keeps a page's front and side elements apart, which matters
    because one page carries both faces of a SKU.
    """
    return f"{face}:{element}"


def split_layer_name(name: str) -> tuple[str, str]:
    face, _, element = name.partition(":")
    return (face, element) if element else ("", name)


# --------------------------------------------------------------------------- #
# Tagging, on the way out
# --------------------------------------------------------------------------- #


class LayerTagger:
    """Wrap each element's drawing operators in its own optional content group.

    Usage::

        tagger = LayerTagger(doc, enabled=audit_layers)
        with tagger.element(page, "front", "item_no"):
            renderer.draw_item_no(...)

    Disabled, ``element`` does nothing at all - no OCG, no content rewrite - so
    production output is unaffected by the audit existing.
    """

    def __init__(self, doc: fitz.Document, *, enabled: bool = True) -> None:
        self.doc = doc
        self.enabled = enabled
        self.names: list[str] = []

    @contextmanager
    def element(self, page: fitz.Page, face: str, element: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        before = set(page.get_contents())
        yield
        added = [xref for xref in page.get_contents() if xref not in before]
        if not added:
            return
        name = layer_name(face, element)
        ocg = self.doc.add_ocg(name)
        key = f"OC{ocg}"
        self.doc.xref_set_key(_resources_xref(self.doc, page), f"Properties/{key}", f"{ocg} 0 R")
        prefix = f"/OC /{key} BDC\n".encode("latin-1")
        for xref in added:
            self.doc.update_stream(xref, prefix + self.doc.xref_stream(xref) + b"\nEMC\n")
        self.names.append(name)


def _resources_xref(doc: fitz.Document, page: fitz.Page) -> int:
    """The xref that actually holds the page's resource dictionary.

    ``/Resources`` is usually an indirect reference, and setting a key through
    the page object fails on that path.
    """
    kind, value = doc.xref_get_key(page.xref, "Resources")
    if kind == "xref":
        return int(str(value).split()[0])
    return page.xref


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #


@dataclass
class LayeredDocument:
    """A tagged PDF held in memory, with its optional content groups indexed."""

    buffer: bytes
    ocgs: dict[int, str]
    background: tuple[int, int, int] | None = None

    @classmethod
    def load(cls, source: Path | str | bytes) -> "LayeredDocument":
        if isinstance(source, (bytes, bytearray)):
            buffer = bytes(source)
        else:
            buffer = Path(source).read_bytes()
        doc = fitz.open("pdf", buffer)
        ocgs = {xref: str(info["name"]) for xref, info in doc.get_ocgs().items()}
        doc.close()
        return cls(buffer=buffer, ocgs=ocgs)

    def faces(self) -> set[str]:
        return {split_layer_name(name)[0] for name in self.ocgs.values()}

    @contextmanager
    def _visible(self, on: list[int] | None, off: list[int] | None) -> Iterator[fitz.Document]:
        """A document with the requested layer visibility, ready to rasterise.

        The save/reopen is not optional. Calling ``set_layer`` on a live document
        and rendering it again was measured to return the previous state's ink,
        so every state gets a freshly parsed document. Yielding it lets a caller
        take several clips out of one parse, which is where the time goes.
        """
        doc = fitz.open("pdf", self.buffer)
        try:
            if on is not None or off is not None:
                doc.set_layer(-1, on=on or [], off=off or [])
                reopened = fitz.open("pdf", doc.tobytes())
                doc.close()
                doc = reopened
            yield doc
        finally:
            doc.close()

    def _mask(self, doc: fitz.Document, page_no: int, clip: fitz.Rect, zoom: float) -> np.ndarray:
        pix = doc[page_no].get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = pixmap_array(pix)
        if self.background is None:
            self.background = background_colour(image)
        return ink_mask(image, background=self.background)

    def render(
        self,
        page_no: int,
        clip: fitz.Rect,
        zoom: float,
        *,
        on: list[int] | None = None,
        off: list[int] | None = None,
    ) -> np.ndarray:
        """Ink mask of one clip with a given layer visibility state."""
        with self._visible(on, off) as doc:
            return self._mask(doc, page_no, clip, zoom)

    def render_clips(
        self,
        page_no: int,
        clips: dict[str, fitz.Rect],
        zoom: float,
        *,
        on: list[int] | None = None,
        off: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Every clip from one visibility state, so the parse is paid once."""
        with self._visible(on, off) as doc:
            return {name: self._mask(doc, page_no, clip, zoom) for name, clip in clips.items()}

    def page_rect(self, page_no: int) -> fitz.Rect:
        doc = fitz.open("pdf", self.buffer)
        rect = fitz.Rect(doc[page_no].rect)
        doc.close()
        return rect


@dataclass
class FaceLayers:
    """One face, rendered whole and rendered one element at a time.

    Two masks per element, because one is not enough to describe a printed mark:

    ``layers``
        what the element draws on its own. This is the mask collisions are
        computed on - two elements' marks landing on the same paper.
    ``contributions``
        what the element visibly changes in the finished page, measured as the
        difference the page shows when that element is removed. White knockout
        text has no ink of its own but a very definite contribution, and this is
        the only mask that captures its shape.
    """

    face: str
    clip: fitz.Rect
    zoom: float
    composite: np.ndarray
    unconditional: np.ndarray
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    contributions: dict[str, np.ndarray] = field(default_factory=dict)

    def shape_of(self, name: str) -> np.ndarray:
        """The mask to measure this element's rendered form with.

        Its own ink when it has any, otherwise the hole it knocks out of what is
        underneath it.
        """
        own = self.layers.get(name)
        if own is not None and own.any():
            return own
        return self.contributions.get(name, np.zeros_like(self.composite))

    @property
    def union(self) -> np.ndarray:
        total = np.zeros_like(self.composite)
        for mask in self.layers.values():
            total |= mask
        return total | self.unconditional

    def bbox_px(self, name: str) -> tuple[int, int, int, int] | None:
        return mask_bbox(self.layers[name])

    def bbox_mm(self, name: str) -> tuple[float, float, float, float] | None:
        box = self.bbox_px(name)
        if box is None:
            return None
        scale = 25.4 / 72.0 / self.zoom
        return (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "face": self.face,
            "zoom": self.zoom,
            "clip_mm": [round(v * 25.4 / 72.0, 2) for v in self.clip],
            "layer_ink_px": {name: int(mask.sum()) for name, mask in sorted(self.layers.items())},
        }


def render_faces(
    layered: LayeredDocument,
    page_no: int,
    clips: dict[str, fitz.Rect],
    *,
    zoom: float = METRIC_ZOOM,
) -> dict[str, FaceLayers]:
    """Isolate every element of every face on one page.

    All faces come out of the same visibility states. Doing one face at a time
    doubles the work for a page that carries two, and the page is the unit the
    renderer actually produces.
    """
    every = list(layered.ocgs)
    composite = layered.render_clips(page_no, clips, zoom)
    unconditional = layered.render_clips(page_no, clips, zoom, on=[], off=every)

    faces = {
        face: FaceLayers(
            face=face,
            clip=clip,
            zoom=zoom,
            composite=composite[face],
            unconditional=unconditional[face],
        )
        for face, clip in clips.items()
    }

    for xref, name in sorted(layered.ocgs.items(), key=lambda kv: kv[1]):
        face, element = split_layer_name(name)
        if face not in faces:
            continue
        others = [other for other in every if other != xref]
        target = {face: clips[face]}

        isolated = layered.render_clips(page_no, target, zoom, on=[xref], off=others)[face]
        own = isolated & ~faces[face].unconditional
        if own.any():
            faces[face].layers[element] = own
            continue

        # No ink of its own means the element is a knockout - white type on a
        # dark ground, which is invisible against the page when isolated. Its
        # shape is the difference the page shows without it. Only these elements
        # pay for the second render; on a standard mark that is one of twenty.
        without = layered.render_clips(page_no, target, zoom, on=others, off=[xref])[face]
        changed = faces[face].composite ^ without
        if changed.any():
            faces[face].contributions[element] = changed

    return faces


def render_face_layers(
    layered: LayeredDocument,
    page_no: int,
    clip: fitz.Rect,
    *,
    face: str,
    zoom: float = METRIC_ZOOM,
) -> FaceLayers:
    """One face on its own. Prefer :func:`render_faces` when a page has two."""
    return render_faces(layered, page_no, {face: clip}, zoom=zoom)[face]


# --------------------------------------------------------------------------- #
# What the layers prove
# --------------------------------------------------------------------------- #


def _issue(code: str, message: str, *, severity: str = "error", **extra: Any) -> dict[str, Any]:
    payload = {"severity": severity, "code": code, "message": message}
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def collisions(
    face: FaceLayers,
    *,
    nested: Iterable[frozenset[str]] = (),
    tolerance_px: int = 1,
) -> list[dict[str, Any]]:
    """Element pairs whose *ink* overlaps.

    Not box intersection. Two elements may share a rectangle happily as long as
    their marks miss each other, and two elements with disjoint boxes can still
    collide once one of them renders outside the box it was given - which is
    exactly what barcode digits printed over the bars look like.
    """
    allowed = {frozenset(pair) for pair in nested}
    names = sorted(face.layers)
    findings: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first, second = names[i], names[j]
            if frozenset({first, second}) in allowed:
                continue
            shared = int((face.layers[first] & face.layers[second]).sum())
            if shared > tolerance_px:
                findings.append(
                    _issue(
                        "ink_collision",
                        f"{first} and {second} print on top of each other on the "
                        f"{face.face} face ({shared} shared pixels).",
                        face=face.face,
                        components=[first, second],
                        shared_px=shared,
                    )
                )
    return findings


def occlusion(
    face: FaceLayers,
    *,
    allowed: Iterable[str] = (),
    tolerance_px: int = 4,
) -> list[dict[str, Any]]:
    """Ink that exists in isolation but is painted over in the finished page.

    Whitelisting is by the element that loses its ink, because that is the
    design decision being approved - a value box over the black item-number bar
    is intended. Anything else disappearing under another element is a defect
    nobody notices by eye until the wrong thing has gone missing.
    """
    permitted = set(allowed)
    hidden = face.union & ~face.composite
    if int(hidden.sum()) <= tolerance_px:
        return []

    findings: list[dict[str, Any]] = []
    for name, mask in sorted(face.layers.items()):
        lost = int((mask & hidden).sum())
        if lost <= tolerance_px:
            continue
        share = lost / max(1, int(mask.sum()))
        findings.append(
            _issue(
                "element_ink_occluded",
                f"{share * 100:.1f}% of {name}'s ink is painted over by another "
                f"element on the {face.face} face."
                + (" This one is declared intentional." if name in permitted else ""),
                severity="info" if name in permitted else "error",
                face=face.face,
                component=name,
                occluded_px=lost,
                occluded_share=round(share, 4),
                whitelisted=name in permitted,
            )
        )
    return findings


def unattributed(face: FaceLayers, *, tolerance_px: int = 8) -> list[dict[str, Any]]:
    """Composite ink that no layer accounts for.

    A non-empty result means the audit is blind to part of the page. That
    matters more than any single check, because every other per-element result
    is only as complete as the layer set.
    """
    stray = int((face.composite & ~face.union).sum())
    if stray <= tolerance_px:
        return []
    return [
        _issue(
            "ink_not_attributed_to_any_layer",
            f"{stray} inked pixels on the {face.face} face belong to no optional "
            "content group, so no per-element check covers them.",
            severity="warning",
            face=face.face,
            stray_px=stray,
        )
    ]


__all__ = [
    "FaceLayers",
    "LayerTagger",
    "LayeredDocument",
    "collisions",
    "layer_name",
    "occlusion",
    "render_face_layers",
    "render_faces",
    "split_layer_name",
    "unattributed",
]
