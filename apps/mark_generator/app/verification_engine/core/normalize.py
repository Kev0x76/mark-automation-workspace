from __future__ import annotations

import re
import unicodedata


HYPHEN_EQUIVALENTS = str.maketrans(
    {
        "­": "-",  # PDF text layers commonly expose a visible hyphen this way.
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "−": "-",
        "﹣": "-",
        "－": "-",
    }
)


def _canonical_unicode(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def normalize_text(value: object) -> str:
    text = _canonical_unicode(value).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_compact(value: object) -> str:
    return re.sub(r"\s+", "", normalize_text(value)).upper()


def normalize_barcode(value: object) -> str:
    text = normalize_text(value)
    if re.match(r"^\d+\.0$", text):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def normalize_item_number(value: object) -> str:
    text = normalize_text(str(value).translate(HYPHEN_EQUIVALENTS)).upper()
    if re.match(r"^[A-Z0-9-]+\.0$", text):
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    return text


def normalize_qty(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if re.match(r"^\d+\.0$", text):
        return str(int(float(text)))
    m = re.match(r"^\d+\s*[*xX×]\s*(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"\d+", text)
    return m.group(0) if m else ""


def is_dispimg(value: object) -> bool:
    text = normalize_text(value).upper()
    return "DISPIMG" in text or text.startswith("=_XLFN.DISPIMG")


def looks_like_barcode(value: object) -> bool:
    digits = normalize_barcode(value)
    return 12 <= len(digits) <= 14


def looks_like_item_number(value: object) -> bool:
    text = normalize_item_number(value)
    if not text:
        return False
    return bool(re.match(r"^(?:\d{5,8}|[A-Z]{1,6}\d{3,8}|\d{4,8}-[A-Z0-9]{1,3})$", text))
