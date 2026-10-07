from __future__ import annotations

from dataclasses import dataclass

import fitz

from .ean13 import is_valid_ean13


EAN13_MODULES = 95
EAN13_LEFT_ODD = {
    "0": "0001101",
    "1": "0011001",
    "2": "0010011",
    "3": "0111101",
    "4": "0100011",
    "5": "0110001",
    "6": "0101111",
    "7": "0111011",
    "8": "0110111",
    "9": "0001011",
}
EAN13_LEFT_EVEN = {
    "0": "0100111",
    "1": "0110011",
    "2": "0011011",
    "3": "0100001",
    "4": "0011101",
    "5": "0111001",
    "6": "0000101",
    "7": "0010001",
    "8": "0001001",
    "9": "0010111",
}
EAN13_RIGHT = {
    "0": "1110010",
    "1": "1100110",
    "2": "1101100",
    "3": "1000010",
    "4": "1011100",
    "5": "1001110",
    "6": "1010000",
    "7": "1000100",
    "8": "1001000",
    "9": "1110100",
}
EAN13_PARITY = {
    "0": "LLLLLL",
    "1": "LLGLGG",
    "2": "LLGGLG",
    "3": "LLGGGL",
    "4": "LGLLGG",
    "5": "LGGLLG",
    "6": "LGGGLL",
    "7": "LGLGLG",
    "8": "LGLGGL",
    "9": "LGGLGL",
}
EAN13_GUARD_BLACK_MODULES = {0, 2, 46, 48, 92, 94}


class BarcodeRenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ean13Bar:
    rect: fitz.Rect
    is_guard: bool
    start_module: int
    end_module: int


def ean13_module_bits(value: str) -> str:
    """Return the canonical 95-module EAN-13 bit pattern, left-to-right."""
    if not is_valid_ean13(value):
        raise BarcodeRenderError(f"Invalid EAN-13 value {value!r}.")
    pattern = "101"
    for parity, digit in zip(EAN13_PARITY[value[0]], value[1:7]):
        pattern += (EAN13_LEFT_ODD if parity == "L" else EAN13_LEFT_EVEN)[digit]
    pattern += "01010"
    for digit in value[7:]:
        pattern += EAN13_RIGHT[digit]
    pattern += "101"
    if len(pattern) != EAN13_MODULES:
        raise BarcodeRenderError(f"EAN-13 encoder produced {len(pattern)} modules for {value!r}.")
    return pattern


def format_ean13_hri(value: str, grouping: str = "golden_1_6_6") -> str:
    """Format human-readable EAN-13 digits.

    The default mirrors the golden Kmt samples: first digit, six digits, six
    digits. It remains one text span when drawn so PDF text segmentation stays
    predictable.
    """
    if not is_valid_ean13(value):
        raise BarcodeRenderError(f"Invalid EAN-13 value {value!r}.")
    if grouping == "continuous":
        return value
    if grouping in {"golden_1_6_6", "ean13_1_6_6"}:
        return f"{value[0]} {value[1:7]} {value[7:]}"
    if grouping == "compact_7_6":
        return f"{value[:7]} {value[7:]}"
    raise BarcodeRenderError(f"Unsupported EAN-13 HRI grouping: {grouping}")


def ean13_bars(value: str, bars_rect: fitz.Rect, regular_bar_inset: float) -> list[Ean13Bar]:
    """Build vertical EAN-13 bars in page coordinates.

    The bars are drawn bottom-to-top because the Kmt template rotates the
    standard horizontal EAN-13 symbol 90 degrees to match the golden sheet.
    Guard bars extend through the human-readable notch; data bars stop short.
    """
    modules = ean13_module_bits(value)
    module_height = bars_rect.height / EAN13_MODULES
    bars: list[Ean13Bar] = []
    index = 0
    while index < EAN13_MODULES:
        if modules[index] == "0":
            index += 1
            continue
        end = index + 1
        while end < EAN13_MODULES and modules[end] == "1":
            end += 1
        is_guard = any(module in EAN13_GUARD_BLACK_MODULES for module in range(index, end))
        x1 = bars_rect.x1 if is_guard else bars_rect.x1 - regular_bar_inset
        y0 = bars_rect.y1 - end * module_height
        y1 = bars_rect.y1 - index * module_height
        bars.append(Ean13Bar(fitz.Rect(bars_rect.x0, y0, x1, y1), is_guard, index, end))
        index = end
    return bars


def barcode_bar_rects(value: str, bars_rect: fitz.Rect, regular_bar_inset: float) -> list[fitz.Rect]:
    return [bar.rect for bar in ean13_bars(value, bars_rect, regular_bar_inset)]


def draw_ean13_bars(page: fitz.Page, bars_rect: fitz.Rect, value: str, regular_bar_inset: float) -> None:
    shape = page.new_shape()
    for bar in ean13_bars(value, bars_rect, regular_bar_inset):
        shape.draw_rect(bar.rect)
    shape.finish(color=None, fill=(0, 0, 0))
    shape.commit()


def ean13_bars_horizontal(
    value: str, bars_rect: fitz.Rect, regular_bar_inset: float
) -> list[Ean13Bar]:
    """Build EAN-13 bars in the unrotated orientation.

    The Kmt template rotates the symbol 90 degrees, which suits a tall panel.
    A very wide, short face cannot fit 95 modules along its height, so the
    adaptive layout needs the upright symbol too: modules run left-to-right and
    guard bars extend downward through the human-readable notch.
    """
    modules = ean13_module_bits(value)
    module_width = bars_rect.width / EAN13_MODULES
    bars: list[Ean13Bar] = []
    index = 0
    while index < EAN13_MODULES:
        if modules[index] == "0":
            index += 1
            continue
        end = index + 1
        while end < EAN13_MODULES and modules[end] == "1":
            end += 1
        is_guard = any(module in EAN13_GUARD_BLACK_MODULES for module in range(index, end))
        y1 = bars_rect.y1 if is_guard else bars_rect.y1 - regular_bar_inset
        x0 = bars_rect.x0 + index * module_width
        x1 = bars_rect.x0 + end * module_width
        bars.append(Ean13Bar(fitz.Rect(x0, bars_rect.y0, x1, y1), is_guard, index, end))
        index = end
    return bars


def draw_ean13_bars_horizontal(
    page: fitz.Page, bars_rect: fitz.Rect, value: str, regular_bar_inset: float
) -> None:
    shape = page.new_shape()
    for bar in ean13_bars_horizontal(value, bars_rect, regular_bar_inset):
        shape.draw_rect(bar.rect)
    shape.finish(color=None, fill=(0, 0, 0))
    shape.commit()
