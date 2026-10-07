"""Font handling MuPDF cannot do on its own for CJK CFF fonts.

On Linux servers the template's Chinese font is ``NotoSansCJK-Bold.ttc``:

* ``fitz.Font(fontfile=...)`` always opens face 0 of a collection, which is
  "Noto Sans CJK JP". :func:`collection_face_buffer` extracts the face whose
  family the template prefers (``Noto Sans CJK SC``).
* ``Document.subset_fonts()`` fails on these CID-keyed CFF fonts
  ("MuPDF error: format error: ...") and silently embeds the whole ~14 MB
  font. :func:`subset_large_embedded_fonts` subsets such font programs with
  fontTools, keeping glyph ids so content streams stay valid, and keeps the
  result only if every page renders pixel-identically.

TrueType fonts (Windows ``msyhbd.ttc``) are subset by MuPDF and never reach
the fontTools path.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from fontTools import subset as ft_subset
from fontTools.ttLib import TTCollection, TTFont

# Any embedded font program at or above this size was not subset.
LARGE_FONT_BYTES = 1_000_000
COLLECTION_SUFFIXES = {".ttc", ".otc"}
_RENDER_CHECK_DPI = 72

_face_cache: dict[tuple[str, int, int, str], bytes] = {}


def preferred_families(fontconfig_families: list[str]) -> list[str]:
    """``"Noto Sans CJK SC:style=Bold"`` -> ``"Noto Sans CJK SC"``."""

    return [value.split(":", 1)[0].strip() for value in fontconfig_families if value.strip()]


def collection_face_buffer(path: Path, families: list[str]) -> bytes | None:
    """Return the first face of a font collection whose family is preferred.

    ``None`` when ``path`` is not a collection or no face matches, so callers
    keep their previous behaviour (face 0 via ``fontfile=``).
    """

    if path.suffix.lower() not in COLLECTION_SUFFIXES or not families:
        return None
    stat = path.stat()
    for family in families:
        key = (str(path), stat.st_mtime_ns, stat.st_size, family)
        if key in _face_cache:
            return _face_cache[key]
    collection = TTCollection(str(path), lazy=True)
    index_by_family: dict[str, int] = {}
    for index, face in enumerate(collection.fonts):
        index_by_family.setdefault(face["name"].getBestFamilyName(), index)
    for family in families:
        index = index_by_family.get(family)
        if index is None:
            continue
        face = TTCollection(str(path)).fonts[index]
        buffer = io.BytesIO()
        face.save(buffer)
        data = buffer.getvalue()
        _face_cache[(str(path), stat.st_mtime_ns, stat.st_size, family)] = data
        return data
    return None


def _render_pages(doc: fitz.Document) -> list[bytes]:
    scale = _RENDER_CHECK_DPI / 72.0
    return [
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, annots=False).samples
        for page in doc
    ]


def _used_glyph_ids(doc: fitz.Document) -> set[int]:
    glyphs: set[int] = {0}
    for page in doc:
        for span in page.get_texttrace():
            glyphs.update(int(char[1]) for char in span.get("chars", ()) if char[1] >= 0)
    return glyphs


def _large_opentype_font_files(doc: fitz.Document) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "Type")[1] != "/FontDescriptor":
            continue
        for key in ("FontFile3", "FontFile2"):
            kind, value = doc.xref_get_key(xref, key)
            if kind != "xref":
                continue
            file_xref = int(value.split()[0])
            if len(doc.xref_stream_raw(file_xref)) >= LARGE_FONT_BYTES:
                found.append((file_xref, doc.xref_get_key(xref, "FontName")[1]))
    return found


def _subset_font_program(data: bytes, glyph_ids: set[int]) -> bytes:
    font = TTFont(io.BytesIO(data))
    options = ft_subset.Options()
    options.retain_gids = True  # content streams address glyphs by id
    options.notdef_outline = True
    options.layout_features = []  # TextWriter places cmap glyphs; no shaping
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.legacy_kern = False
    subsetter = ft_subset.Subsetter(options)
    subsetter.populate(gids=sorted(glyph_ids))
    subsetter.subset(font)
    buffer = io.BytesIO()
    font.save(buffer)
    return buffer.getvalue()


def subset_large_embedded_fonts(doc: fitz.Document) -> list[dict[str, Any]]:
    """Subset embedded font programs MuPDF left whole. Returns issue dicts.

    The replacement is kept only if every page renders to identical pixels;
    otherwise the original programs are restored and a warning is returned.
    """

    targets = _large_opentype_font_files(doc)
    if not targets:
        return []
    issues: list[dict[str, Any]] = []
    glyph_ids = _used_glyph_ids(doc)
    before = _render_pages(doc)
    originals: dict[int, bytes] = {}
    for file_xref, name in targets:
        original = doc.xref_stream(file_xref)
        try:
            replacement = _subset_font_program(original, glyph_ids)
        except Exception as exc:  # fontTools rejects the program: keep it whole
            issues.append(_issue("font_subset_failed", f"Could not subset embedded font {name}: {exc}"))
            continue
        originals[file_xref] = original
        doc.update_stream(file_xref, replacement)
    if not originals:
        return issues
    # Re-open from bytes so no cached font program can mask a broken subset.
    check = fitz.open("pdf", doc.tobytes(garbage=0))
    try:
        identical = _render_pages(check) == before
    finally:
        check.close()
    if not identical:
        for file_xref, original in originals.items():
            doc.update_stream(file_xref, original)
        issues.append(
            _issue(
                "font_subset_failed",
                "Subset embedded fonts changed the rendered page; kept the full fonts.",
            )
        )
    return issues


def large_embedded_fonts(doc: fitz.Document) -> list[str]:
    """Names of embedded font programs that are still not subset."""

    return [name for _xref, name in _large_opentype_font_files(doc)]


def _issue(code: str, message: str) -> dict[str, Any]:
    return {"severity": "warning", "code": code, "message": message}
