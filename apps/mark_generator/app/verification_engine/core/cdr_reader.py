from __future__ import annotations

import re
import zipfile
from html import unescape
from pathlib import Path

from .models import BarcodeEvidence, PageData, TextSpan
from .normalize import normalize_barcode, normalize_item_number, normalize_text


def read_cdr(path: str | Path) -> list[PageData]:
    """Read text and barcode evidence from zip-based CorelDRAW CDR files.

    CorelDRAW 2020+ CDR files are zip containers. They often expose text objects
    in META-INF/textinfo.xml and barcode payloads in content/data/data1.dat.
    This reader turns those source-specific hints into the same PageData model
    used by PDF extraction, so downstream segmentation/matching stays generic.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Unsupported CDR container: {path}")

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        text_xml = zf.read("META-INF/textinfo.xml").decode("utf-8", errors="replace") if "META-INF/textinfo.xml" in names else ""
        data1 = zf.read("content/data/data1.dat") if "content/data/data1.dat" in names else b""

    streams = _extract_text_streams(text_xml)
    item_numbers = _ordered_unique(
        normalize_item_number(s)
        for s in streams
        if _looks_like_item_number(normalize_item_number(s))
    )
    barcodes = _extract_ean_payloads(data1)
    barcode_by_item = _associate_barcodes_to_items(item_numbers, barcodes)

    page = PageData(
        file_path=str(path),
        page=0,
        width=2000.0,
        height=max(2000.0, float(max(1, len(item_numbers)) * 220)),
        has_text_layer=True,
        text_lines=streams,
    )

    spans: list[TextSpan] = []
    bcs: list[BarcodeEvidence] = []
    for row_idx, item_no in enumerate(item_numbers):
        lines = _lines_for_item(streams, item_no)
        barcode = barcode_by_item.get(item_no, "")
        if barcode and barcode not in lines:
            lines.insert(1, barcode)

        y = 80.0 + row_idx * 220.0
        for line_idx, line in enumerate(_ordered_unique(lines)):
            spans.append(
                TextSpan(
                    text=line,
                    bbox=(80.0, y + line_idx * 24.0, 80.0 + max(60.0, len(line) * 9.0), y + line_idx * 24.0 + 18.0),
                    page=0,
                    size=14.0 if line != item_no else 20.0,
                    source="cdr_textinfo",
                )
            )
        if barcode:
            bcs.append(
                BarcodeEvidence(
                    value=barcode,
                    bbox=(500.0, y + 20.0, 720.0, y + 80.0),
                    page=0,
                    source="cdr_data1_payload",
                )
            )

    page.spans = spans
    page.barcodes = bcs
    return [page]


def _extract_text_streams(xml_text: str) -> list[str]:
    if not xml_text:
        return []
    streams: list[str] = []
    for block in re.findall(r"<TextStream\b[^>]*>(.*?)</TextStream>", xml_text, flags=re.S | re.I):
        parts: list[str] = []
        for attrs, raw in re.findall(r"<TextRun\b([^>]*)>(.*?)</TextRun>", block, flags=re.S | re.I):
            text = normalize_text(unescape(re.sub(r"<[^>]+>", "", raw)))
            if not text:
                continue
            break_kind = ""
            m = re.search(r'break="([^"]+)"', attrs)
            if m:
                break_kind = m.group(1).lower()
            if break_kind in {"para", "line"} and parts:
                parts.append("\n")
            elif break_kind == "none" and parts and parts[-1] != "\n":
                parts.append(" ")
            parts.append(text)
        stream = normalize_text(" ".join("".join(parts).splitlines()))
        if stream:
            streams.append(stream)
    return streams


def _extract_ean_payloads(data: bytes) -> list[str]:
    values: list[str] = []
    for raw in re.findall(rb"(?<!\d)(\d{12,13})(?!\d)", data):
        digits = raw.decode("ascii", errors="ignore")
        if len(digits) == 12:
            digits = _complete_ean13(digits)
        if _valid_ean13(digits) and digits not in values:
            values.append(digits)
    return values


def _complete_ean13(first_12: str) -> str:
    total = 0
    for idx, ch in enumerate(first_12):
        total += int(ch) * (1 if idx % 2 == 0 else 3)
    check = (10 - (total % 10)) % 10
    return first_12 + str(check)


def _valid_ean13(digits: str) -> bool:
    digits = normalize_barcode(digits)
    return len(digits) == 13 and _complete_ean13(digits[:12]) == digits


def _associate_barcodes_to_items(items: list[str], barcodes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    unused = list(barcodes)
    for item in items:
        suffixes = [item[-n:] for n in range(min(5, len(item)), 2, -1)]
        match = next((bc for bc in unused if any(bc[:-1].endswith(suffix) or bc.endswith(suffix) for suffix in suffixes)), "")
        if match:
            mapping[item] = match
            unused.remove(match)

    remaining_items = [item for item in items if item not in mapping]
    for item, bc in zip(sorted(remaining_items), sorted(unused)):
        mapping[item] = bc
    return mapping


def _lines_for_item(streams: list[str], item_no: str) -> list[str]:
    lines: list[str] = [item_no]
    positions = [idx for idx, text in enumerate(streams) if normalize_item_number(text) == item_no]
    for pos in positions:
        start = max(0, pos - 8)
        end = min(len(streams), pos + 8)
        for text in streams[start:end]:
            compact = normalize_item_number(text)
            if compact == item_no:
                continue
            if _looks_like_item_number(compact):
                continue
            if _is_low_value_structural_text(text):
                continue
            lines.append(text)
    return lines


def _is_low_value_structural_text(text: str) -> bool:
    upper = normalize_text(text).upper()
    if upper in {"KG", "CM"}:
        return True
    return False


def _looks_like_item_number(text: str) -> bool:
    return bool(re.match(r"^(?:\d{5,8}|[A-Z]{1,6}\d{3,8}|[A-Z0-9]{3,10}-[A-Z0-9]{1,3})$", text))


def _ordered_unique(values) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
