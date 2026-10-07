from __future__ import annotations

import re
from statistics import median

from .models import Cell, CellEvidence, TextSpan
from .normalize import normalize_barcode, normalize_item_number


NON_PRODUCT_PATTERNS = [
    r"^MADE\s+IN\s+CHINA",
    r"^HECHO\s+EN\s+CHINA",
    r"^FATTO\s+IN\s+CINA",
    r"^FABRIQU",
]

IT_MARKERS = {"FLACONE", "VIAGGIO", "PEZZI", "TAPPO", "COSTUME", "MAIALE", "CONIGLIO", "GRUCCIA", "PIGIAMA", "ANIMALE", "MOCIO", "TESSUTO", "RICAMBIO", "PIATTO", "ROTANTE", "FUNZIONE"}
ES_MARKERS = {"BOTE", "BOTES", "PARA", "VIAJE", "DISFRAZ", "CERDO", "CONEJO", "PERCHA", "PIJAMA", "PLASTICO", "PLÁSTICO", "MOPA", "PALO", "RECAMBIO", "FREGONA", "GIRATORIA", "CUBO", "PULVERIZADOR", "JUEGO"}
EN_MARKERS = {"TRAVEL", "BOTTLE", "SPRAY", "COSTUME", "PIG", "RABBIT", "PLASTIC", "HANGER", "PAJAMAS", "ADULT"}
PT_MARKERS = {"FRASCO", "VIAGEM", "PORCO", "COELHO", "CABIDE", "PULVERIZADOR", "FANTASIA"}


def extract_cell_evidence(cell: Cell) -> CellEvidence:
    evidence = CellEvidence(cell=cell, needs_ai=cell.needs_ai)
    evidence.raw_lines = _reconstruct_lines(cell.spans)
    span_source = _cell_text_source(cell.spans)

    pre_items = set(_reconstruct_item_numbers(cell.spans))
    for bc in cell.barcodes:
        norm = normalize_barcode(bc.value)
        if any(norm == item + item or norm.startswith(item) for item in pre_items):
            continue
        if not norm:
            continue
        if _is_scanned_barcode_source(bc.source):
            if norm not in evidence.barcode_scans:
                evidence.barcode_scans.append(norm)
                evidence.evidence_sources.setdefault("barcode_scan", bc.source)
        elif _is_text_barcode_source(bc.source):
            if norm not in evidence.barcode_texts:
                evidence.barcode_texts.append(norm)
                evidence.evidence_sources.setdefault("barcode_text", bc.source)
        if norm and norm not in evidence.barcodes:
            evidence.barcodes.append(norm)
            evidence.evidence_sources.setdefault("barcode", bc.source)

    for value in pre_items:
        if value not in evidence.item_numbers:
            evidence.item_numbers.append(value)
            evidence.evidence_sources.setdefault("item_number", span_source)

    known_items = set(evidence.item_numbers)
    for value in _find_barcodes_from_spans(cell.spans, known_items):
        if value not in evidence.barcode_texts:
            evidence.barcode_texts.append(value)
            evidence.evidence_sources.setdefault("barcode_text", _digits_source(span_source))
        if value not in evidence.barcodes:
            evidence.barcodes.append(value)
            evidence.evidence_sources.setdefault("barcode", _digits_source(span_source))

    qty = _find_packing_qty(cell.spans, evidence.raw_lines)
    if qty:
        evidence.packing_qty = qty
        evidence.evidence_sources.setdefault("packing_qty", span_source)

    for line in evidence.raw_lines:
        line = _strip_noise_phrases(line)
        if _is_noise_line(line):
            continue
        if _is_structural_number(line):
            continue
        lang_hint = _line_language_hint(line)
        for clean in _split_name_candidates(_clean_name_line(line)):
            if _is_junk_name_candidate(clean):
                continue
            _add_name_candidate(evidence, clean, span_source, lang_hint)

    _dedupe_name_fragments(evidence)
    return evidence


def _cell_text_source(spans: list[TextSpan]) -> str:
    source_text = " ".join(str(span.source).lower() for span in spans)
    if "mark_image_ocr" in source_text:
        return "mark_image_ocr"
    if "visual_candidate" in source_text:
        return "visual_candidate_ocr"
    if "ocr" in source_text:
        return "ocr_text"
    return "text_layer"


def _digits_source(source: str) -> str:
    if source == "text_layer":
        return "text_layer_digits"
    return f"{source}_digits"


def _is_scanned_barcode_source(source: str) -> bool:
    return source.startswith("scan_")


def _is_text_barcode_source(source: str) -> bool:
    return source.startswith("text_")


def _add_name_candidate(
    evidence: CellEvidence,
    clean: str,
    source: str,
    lang_hint: str = "",
) -> None:
    lang = lang_hint or detect_language(clean)
    if clean and lang != "cn" and clean not in evidence.names.setdefault("latin", []):
        evidence.names["latin"].append(clean)
        evidence.evidence_sources.setdefault("name_latin", source)
    if clean and clean not in evidence.names.setdefault(lang, []):
        evidence.names[lang].append(clean)
        evidence.evidence_sources.setdefault(f"name_{lang}", source)


def detect_language(text: str) -> str:
    text = text.strip()
    upper = text.upper()
    if re.search(r"[\u4e00-\u9fff]", text):
        return "cn"
    scores = {
        "it": sum(1 for m in IT_MARKERS if m in upper),
        "es": sum(1 for m in ES_MARKERS if m in upper),
        "en": sum(1 for m in EN_MARKERS if m in upper),
        "pt": sum(1 for m in PT_MARKERS if m in upper),
    }
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    if scores[best] > 0 and (len(ordered) == 1 or ordered[0] > ordered[1]):
        return best
    return "unknown"


def _reconstruct_lines(spans: list[TextSpan]) -> list[str]:
    if not spans:
        return []
    rows: list[list[TextSpan]] = []
    sorted_spans = sorted(spans, key=lambda s: (s.y0, s.x0))
    y_tol = _row_tolerance(spans)
    for span in sorted_spans:
        placed = False
        for row in rows:
            if abs(row[0].y0 - span.y0) <= y_tol:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])
    lines = []
    for row in rows:
        row.sort(key=lambda s: s.x0)
        parts: list[str] = []
        prev = None
        for span in row:
            if prev is not None:
                gap = span.x0 - prev.x1
                if gap > max(80, span.size * 3.0):
                    line = re.sub(r"\s+", " ", "".join(parts)).strip()
                    if line:
                        lines.append(line)
                    parts = []
                elif gap > max(5, span.size * 0.4):
                    parts.append(" ")
            parts.append(span.text.strip())
            prev = span
        line = re.sub(r"\s+", " ", "".join(parts)).strip()
        if line:
            lines.append(line)
    return lines


def _row_tolerance(spans: list[TextSpan]) -> float:
    sizes = [s.size for s in spans if s.size]
    if not sizes:
        return 8
    return max(8, median(sizes) * 0.65)


def _reconstruct_item_numbers(spans: list[TextSpan]) -> list[str]:
    results: list[str] = []
    for group in _merge_adjacent_spans(spans, allow_letters=True):
        if _looks_like_vertical_ocr_barcode_fragment(group):
            continue
        text = normalize_item_number("".join(s.text for s in group))
        candidates = [_repeated_item_number(text), text]
        for span in group:
            if _looks_like_vertical_ocr_barcode_fragment([span]):
                continue
            candidates.append(normalize_item_number(span.text))
        for candidate in candidates:
            if _looks_like_item_number(candidate) and candidate not in results:
                results.append(candidate)
    return results


def _repeated_item_number(text: str) -> str:
    if not text.isdigit():
        return ""
    for size in range(5, 9):
        if len(text) == size * 2 and text[:size] == text[size:]:
            return text[:size]
    return ""


def _find_barcodes_from_spans(spans: list[TextSpan], known_items: set[str] | None = None) -> list[str]:
    known_items = known_items or set()
    results: list[str] = []
    for group in _merge_adjacent_spans(spans, allow_letters=False):
        raw = "".join(s.text for s in group)
        if " " not in raw and len(group) <= 4:
            # Large item-number fragments can concatenate to 12-14 digits.
            continue
        text = _canonical_barcode_text(normalize_barcode("".join(s.text for s in group)))
        if any(text == item + item or text.startswith(item) for item in known_items):
            continue
        if 12 <= len(text) <= 14 and text not in results:
            results.append(text)
    return results


def _canonical_barcode_text(digits: str) -> str:
    if len(digits) == 14 and _valid_ean13(digits[:13]):
        return digits[:13]
    return digits


def _valid_ean13(digits: str) -> bool:
    digits = normalize_barcode(digits)
    if len(digits) != 13:
        return False
    total = sum(int(ch) * (1 if idx % 2 == 0 else 3) for idx, ch in enumerate(digits[:12]))
    check = (10 - (total % 10)) % 10
    return check == int(digits[-1])


def _merge_adjacent_spans(spans: list[TextSpan], allow_letters: bool) -> list[list[TextSpan]]:
    pattern = r"^[A-Z0-9\-\s]+$" if allow_letters else r"^[\d\s]+$"
    candidates = [
        span
        for span in spans
        if re.match(
            pattern,
            normalize_item_number(span.text) if allow_letters else normalize_barcode(span.text),
        )
    ]
    rows: list[list[TextSpan]] = []
    y_tol = _row_tolerance(candidates)
    for span in sorted(candidates, key=lambda s: (s.y0, s.x0)):
        placed = False
        for row in rows:
            if abs(row[0].y0 - span.y0) <= y_tol:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    groups: list[list[TextSpan]] = []
    for row in rows:
        row.sort(key=lambda s: s.x0)
        cur: list[TextSpan] = []
        for span in row:
            if not cur:
                cur = [span]
                continue
            gap = span.x0 - cur[-1].x1
            # CorelDRAW often splits digits with tiny or zero gaps.
            if gap <= max(24, min(42, span.size * 0.25)):
                cur.append(span)
            else:
                groups.append(cur)
                cur = [span]
        if cur:
            groups.append(cur)
    return groups


def _looks_like_item_number(text: str) -> bool:
    return bool(re.match(r"^(?:\d{5,8}|[A-Z]{1,6}\d{3,8}|[A-Z0-9]{3,10}-[A-Z0-9]{1,3})$", text))


def _looks_like_vertical_ocr_barcode_fragment(group: list[TextSpan]) -> bool:
    if len(group) != 1:
        return False
    span = group[0]
    width = max(0.0, span.x1 - span.x0)
    height = max(0.0, span.y1 - span.y0)
    # Rotated human-readable barcode digits can come from a PDF text layer as
    # well as OCR.  Their tall, narrow geometry is the reliable signal; source
    # labels are not.  Treating a six-digit fragment as a SKU can associate a
    # whole grid row with the wrong contract item.
    return height > max(width * 2.0, 40.0) and bool(re.fullmatch(r"\d{5,14}", span.text.strip()))


def _find_packing_qty(spans: list[TextSpan], lines: list[str]) -> str:
    if spans:
        large_threshold = _large_text_threshold(spans)
        max_size = max((s.size for s in spans), default=0)
        for span in sorted(spans, key=lambda s: -s.size):
            text = span.text.strip()
            if (
                span.size >= large_threshold
                and (not max_size or span.size <= max_size * 0.65)
                and re.fullmatch(r"\d{1,3}", text)
            ):
                n = int(text)
                if 1 <= n <= 500:
                    return text
    for line in lines:
        m = re.search(r"(\d{1,3})\s*(PCS|PC|PZ|PZS|UN|CTN)", line, re.I)
        if m and not re.search(r"(ML|CM|MM|KG|TRAVEL|BOTTLE|BOTE|FLACONE|瓶|套|香水)", line, re.I):
            return m.group(1)
    for line in lines:
        if re.fullmatch(r"\d{1,3}", line.strip()):
            n = int(line.strip())
            if 1 <= n <= 500:
                return line.strip()
    for line in lines:
        digit_groups = re.findall(r"\d+", line)
        if len(normalize_barcode(line)) >= 13 and digit_groups:
            # Human-readable barcode lines often look like "8 447655 108088 36".
            # The final short token is the side-mark packing quantity.
            tail = digit_groups[-1]
            if len(tail) <= 3:
                n = int(tail)
                if n in {10, 12, 18, 24, 30, 36, 48, 60, 72, 96, 100, 120, 144, 200, 240, 500}:
                    return tail
    return ""


def _large_text_threshold(spans: list[TextSpan]) -> float:
    sizes = sorted(s.size for s in spans if s.size)
    if not sizes:
        return 9999
    return max(sizes[int(len(sizes) * 0.75)], median(sizes) * 1.25)


def _is_noise_line(line: str) -> bool:
    upper = line.strip().upper()
    if not upper:
        return True
    return any(re.search(p, upper) for p in NON_PRODUCT_PATTERNS)


def _is_structural_number(line: str) -> bool:
    # Soft hyphens / fullwidth digits are presentation noise; structural
    # classification must use the same item-number normalization as association.
    compact = normalize_item_number(line)
    if _looks_like_item_number(compact):
        return True
    # Front + side headers are often concatenated on one visual line.
    if compact and len(compact) % 2 == 0:
        half = compact[: len(compact) // 2]
        if half == compact[len(compact) // 2 :] and _looks_like_item_number(half):
            return True
    if 12 <= len(normalize_barcode(line)) <= 14 and not re.search(r"[A-Za-z\u4e00-\u9fff]", line):
        return True
    if re.fullmatch(r"\d{1,3}", compact):
        return True
    if re.fullmatch(r"(CM|KG|G|ML|MM|M|L|PCS|PC|GR)", compact.upper()):
        return True
    return False


def _is_junk_name_candidate(text: str) -> bool:
    """Reject residual SKU fragments that are not product-name evidence.

    PDF soft-hyphen item headers like ``3213878\u00adA`` leave bare suffix
    leftovers after digit stripping. Those leftovers are not product names and
    can falsely dominate EU fuzzy matching via partial_ratio.
    """

    raw = str(text or "").strip()
    if not raw:
        return True
    compact = normalize_item_number(raw)
    if not compact:
        return True
    if _looks_like_item_number(compact):
        return True
    if re.fullmatch(r"[\-A-Z0-9]{1,3}", compact):
        return True
    letters = re.sub(r"[^A-Z\u4e00-\u9fff]", "", compact)
    digits = re.sub(r"\D", "", compact)
    if letters and len(letters) <= 2 and not re.search(r"[\u4e00-\u9fff]", letters):
        return True
    if digits and not letters and len(digits) <= 4:
        return True
    return False


def _clean_name_line(line: str) -> str:
    clean = line.strip()
    clean = re.sub(
        r"^(?:NAME|NOME|NOMBRE)?\s*[:：]?\s*(?:ES|IT|EN|PT|CN|SP)\.?\s*",
        "",
        clean,
        flags=re.I,
    ).strip()
    clean = re.sub(r"^(?:NAME|NOME|NOMBRE)\s*[:：]?\s*", "", clean, flags=re.I).strip()
    clean = _strip_name_structural_noise(clean)
    clean = _strip_noise_phrases(clean)
    return clean


def _line_language_hint(line: str) -> str:
    match = re.match(
        r"^\s*(?:NAME|NOME|NOMBRE)?\s*[:：]?\s*(ES|IT|EN|PT|CN|SP)\.?\b",
        line,
        flags=re.I,
    )
    if not match:
        return ""
    value = match.group(1).lower()
    if value == "sp":
        return "es"
    return value


def _strip_name_structural_noise(line: str) -> str:
    clean = line.strip()
    clean = re.sub(r"\bQ\.?\s*T\.?\s*Y\.?\b\s*\d{0,3}\s*(?:PCS|PC|PZ|PZS|UN)?", " ", clean, flags=re.I)
    clean = re.sub(r"\bQTY\b\s*\d{0,3}\s*(?:PCS|PC|PZ|PZS|UN)?", " ", clean, flags=re.I)
    clean = re.sub(r"\b\d{1,3}\s*(?:PCS|PC|PZ|PZS|UN)\b", " ", clean, flags=re.I)
    # Drop full item numbers first, including soft-hyphen variants, before bare digit runs.
    clean = re.sub(
        r"(?<![A-Z0-9])(?:[A-Z]{1,6}\d{3,8}|\d{4,10})[\u00ad\-][A-Z0-9]{1,3}(?![A-Z0-9])",
        " ",
        clean,
        flags=re.I,
    )
    clean = re.sub(r"(?<!\d)\d{5,14}(?!\d)", " ", clean)
    # Residual soft-hyphen / hyphen SKU suffixes left after digit stripping.
    clean = re.sub(r"[\u00ad\-]\s*[A-Z0-9]{1,3}\b", " ", clean, flags=re.I)
    clean = re.sub(r"\b(?:N\.?\s*W\.?|G\.?\s*W\.?|MEAS|CM|KG)\b", " ", clean, flags=re.I)
    clean = re.sub(r"\b(?:NAME|NOME|NOMBRE)\b", " ", clean, flags=re.I)
    return re.sub(r"\s+", " ", clean).strip(" ;,/|")


def _split_name_candidates(line: str) -> list[str]:
    clean = line.strip()
    if not clean:
        return []
    upper = clean.upper()
    markers = ["DECORAZIONE", "ADORNO", "PARTY", "MULTI-PURPOSE", "MULTIPURPOSE"]
    hits = sorted((upper.find(marker), marker) for marker in markers if upper.find(marker) > 0)
    if not hits:
        return [clean]

    starts = [0] + [pos for pos, _marker in hits]
    parts: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(clean)
        part = clean[start:end].strip(" ;,/|")
        if part:
            parts.append(part)
    parts.append(clean)
    return list(dict.fromkeys(parts))


def _strip_noise_phrases(line: str) -> str:
    clean = line.strip()
    clean = re.sub(r"\bMADE\s+IN\s+CHINA\b", " ", clean, flags=re.I)
    clean = re.sub(r"\bHECHO\s+EN\s+CHINA\b", " ", clean, flags=re.I)
    clean = re.sub(r"\bFATTO\s+IN\s+CINA\b", " ", clean, flags=re.I)
    return re.sub(r"\s+", " ", clean).strip()


def _dedupe_name_fragments(evidence: CellEvidence) -> None:
    for lang, values in list(evidence.names.items()):
        deduped: list[str] = []
        for val in values:
            compact = re.sub(r"\s+", "", val).upper()
            if not compact:
                continue
            if any(compact in re.sub(r"\s+", "", old).upper() for old in deduped):
                continue
            deduped.append(val)
        evidence.names[lang] = deduped
