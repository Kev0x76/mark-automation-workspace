from __future__ import annotations

from .models import Association, CellEvidence, ContractItem, FieldCheck, VerificationResult
from .normalize import HYPHEN_EQUIVALENTS, normalize_barcode, normalize_compact, normalize_item_number, normalize_qty, normalize_text
from .line_art_checker import line_art_semantic_field
from .semantic_matcher import name_similarity


def verify_item(
    item: ContractItem,
    evidence: CellEvidence | None,
    association: Association | None,
    line_art_override: FieldCheck | None = None,
) -> VerificationResult:
    if evidence is None or association is None:
        return VerificationResult(
            item_number=item.item_number,
            association_confidence=0.0,
            error="未找到可关联的唛头 cell",
        )

    fields: list[FieldCheck] = []
    mark = evidence.to_mark_fields()

    fields.append(_item_number_field(item, evidence))
    fields.append(_barcode_field(item, evidence))
    fields.append(_barcode_scan_field(item, evidence))
    fields.append(_barcode_text_field_v2(item, evidence))
    fields.append(_barcode_scan_text_field_v2(evidence))
    fields.append(_qty_field(item.packing_qty, mark["packing_qty"], evidence.evidence_sources.get("packing_qty", "")))

    for lang, label in [("cn", "中文品名"), ("it", "意文品名"), ("es", "西文品名"), ("en", "英文品名"), ("pt", "葡文品名")]:
        contract_name = item.names.get(lang, "")
        mark_name, source = _best_name_candidate(contract_name, evidence, lang)
        if lang == "cn" and contract_name and not mark_name:
            fields.append(FieldCheck(
                field="name_cn",
                contract_value=contract_name,
                mark_value="",
                status="PASS",
                similarity=1.0,
                note="唛头未出现中文品名，按可选字段跳过",
                source=source,
            ))
            continue
        if contract_name or mark_name:
            status, sim, note = name_similarity(contract_name, mark_name, "cn" if lang == "cn" else lang)
            if status == "MISMATCH" and _is_visual_ocr_source(source):
                status = "NEEDS_REVIEW"
                note = f"OCR候选与合同不一致，需复核：{note}"
            fields.append(FieldCheck(
                field=f"name_{lang}",
                contract_value=contract_name,
                mark_value=mark_name,
                status=status,
                similarity=sim,
                note=note,
                source=source,
            ))

    line_art_field = line_art_override or line_art_semantic_field(item, evidence)
    if line_art_field is not None:
        fields.append(line_art_field)

    return VerificationResult(
        item_number=item.item_number,
        cell_id=evidence.cell.cell_id,
        association_confidence=association.confidence,
        association_reason=association.reason,
        fields=fields,
    )


def _best_name_candidate(contract_name: str, evidence: CellEvidence, lang: str) -> tuple[str, str]:
    if contract_name and lang != "cn":
        spatial = _exact_spatial_name(contract_name, evidence)
        if spatial:
            return spatial, "text_layer"
    candidates = list(evidence.names.get(lang, []))
    source = evidence.evidence_sources.get(f"name_{lang}", "")
    if contract_name:
        if lang != "cn":
            candidates.extend(evidence.names.get("latin", []))
            candidates.extend(evidence.names.get("unknown", []))
        candidates = _with_wrapped_name_candidates(candidates)
    if not candidates:
        return "", ""
    if not contract_name:
        return " ".join(dict.fromkeys(candidates)), source

    best_text = ""
    best_rank: tuple[int, int, float, float, float] | None = None
    best_status = ""
    for candidate in dict.fromkeys(candidates):
        status, sim, _note = name_similarity(contract_name, candidate, "cn" if lang == "cn" else lang)
        contract_compact = normalize_compact(contract_name)
        candidate_compact = normalize_compact(candidate)
        exact = int(bool(contract_compact) and candidate_compact == contract_compact)
        status_rank = {
            "PASS": 3,
            "NEEDS_REVIEW": 2,
            "MISSING": 1,
            "MISMATCH": 0,
        }.get(status, 0)
        length_coverage = min(len(candidate_compact), len(contract_compact)) / max(len(contract_compact), 1)
        length_distance = abs(len(candidate_compact) - len(contract_compact)) / max(len(contract_compact), 1)
        # Exact normalized evidence must beat a short substring that happens to
        # receive the same fuzzy score.  The remaining terms prefer complete,
        # contract-sized evidence without allowing length alone to outrank the
        # verifier status.
        rank = (exact, status_rank, sim, length_coverage, -length_distance)
        if best_rank is None or rank > best_rank:
            best_text = candidate
            best_rank = rank
            best_status = status

    if lang != "cn" and best_text in evidence.names.get("latin", []) and best_status in {"PASS", "NEEDS_REVIEW"}:
        return best_text, source or "contract_guided_latin"
    return best_text, source


def _exact_spatial_name(contract_name: str, evidence: CellEvidence) -> str:
    """Recover exact text in adjacent same-column PDF spans, never skip tokens.

    Front and side text can interleave in the page-wide y ordering. Only an
    exact complete match from real text-layer spans may bypass that ordering;
    OCR and fuzzy name decisions continue through the existing verifier.
    """
    expected = normalize_compact(contract_name.translate(HYPHEN_EQUIVALENTS))
    spans = sorted((s for s in evidence.cell.spans if s.source == "text_layer" and s.text.strip()),
                   key=lambda s: (s.y0, s.x0))
    for first in spans:
        current = first
        parts = [first.text.strip().translate(HYPHEN_EQUIVALENTS)]
        for _ in range(4):
            observed = " ".join(parts)
            if normalize_compact(observed) == expected:
                return normalize_text(observed)
            if not expected.startswith(normalize_compact(observed)) or first.size <= 0:
                break
            below = [s for s in spans if s.y0 > current.y0 + first.size * .5
                     and abs(s.x0 - first.x0) <= max(.5, first.size * .15)]
            if not below:
                break
            following = min(below, key=lambda s: s.y0)
            if following.y0 - current.y0 > first.size * 1.8 or abs(following.size - first.size) > first.size * .05:
                break
            parts.append(following.text.strip().translate(HYPHEN_EQUIVALENTS))
            current = following
    return ""


def _with_wrapped_name_candidates(candidates: list[str]) -> list[str]:
    """Add adjacent two/three-fragment candidates for wrapped product names.

    A standard mark may split ``MARCO PARA PHOTOCALL`` into ``MARCO`` and
    ``PARA PHOTOCALL`` in the side table or into ``MARCO PARA`` and
    ``PHOTOCALL`` on the front.  Both fragments are real evidence in the same
    already-associated SKU cell, so joining nearby ordered candidates avoids a
    false review without inventing any token.
    """

    unique = list(dict.fromkeys(value.strip() for value in candidates if value.strip()))
    combined = list(unique)
    for index in range(len(unique)):
        for fragment_count in (2, 3):
            fragment_group = unique[index : index + fragment_count]
            if len(fragment_group) == fragment_count:
                combined.append(" ".join(fragment_group))
    return list(dict.fromkeys(combined))


def _item_number_field(item: ContractItem, evidence: CellEvidence) -> FieldCheck:
    source = evidence.evidence_sources.get("item_number", "")
    expected = normalize_item_number(item.item_number)
    candidates = [normalize_item_number(x) for x in evidence.item_numbers if normalize_item_number(x)]
    if expected and expected in candidates:
        return FieldCheck("item_number", item.item_number, expected, "PASS", 1.0, "货号一致", source)
    mark = candidates[0] if candidates else ""
    check = _exact_field("item_number", item.item_number, mark, "货号", source)
    if check.status == "MISMATCH" and _is_visual_ocr_source(source):
        check.status = "NEEDS_REVIEW"
        check.note = "OCR候选货号与合同不一致，需复核"
    return check


def _best_latin_candidate(contract_name: str, evidence: CellEvidence, lang: str) -> tuple[str, str]:
    candidates = []
    for bucket in ("latin", "unknown"):
        candidates.extend(evidence.names.get(bucket, []))
    best_text = ""
    best_score = 0.0
    for candidate in dict.fromkeys(candidates):
        status, sim, _note = name_similarity(contract_name, candidate, lang)
        score = sim if status in {"PASS", "NEEDS_REVIEW"} else sim * 0.5
        if score > best_score:
            best_text = candidate
            best_score = score
    if best_score >= 0.45:
        return best_text, "contract_guided_latin"
    return "", ""


def _exact_field(field: str, contract: str, mark: str, label: str, source: str) -> FieldCheck:
    c = normalize_item_number(contract)
    m = normalize_item_number(mark)
    if c and not m:
        return FieldCheck(field=field, contract_value=contract, mark_value=mark, status="MISSING", note=f"唛头缺少{label}", source=source)
    if not c:
        return FieldCheck(field=field, contract_value=contract, mark_value=mark, status="NEEDS_REVIEW", note=f"合同无{label}", source=source)
    if c == m:
        return FieldCheck(field=field, contract_value=contract, mark_value=mark, status="PASS", similarity=1.0, note=f"{label}一致", source=source)
    return FieldCheck(field=field, contract_value=contract, mark_value=mark, status="MISMATCH", note=f"{label}不一致", source=source)


def _barcode_field(item: ContractItem, evidence: CellEvidence) -> FieldCheck:
    expected = normalize_barcode(item.barcode)
    observed_values = [normalize_barcode(x) for x in evidence.barcode_scans + evidence.barcode_texts + evidence.barcodes]
    observed = [x for x in dict.fromkeys(observed_values) if x]
    source = evidence.evidence_sources.get("barcode", "")
    if not expected:
        return FieldCheck("barcode", item.barcode, ", ".join(observed), "NEEDS_REVIEW", note="合同无条码", source=source)
    if not observed:
        return FieldCheck("barcode", item.barcode, "", "MISSING", note="唛头未识别到条码", source=source)
    if expected in observed:
        extras = [x for x in observed if x != expected]
        benign = all(x.startswith(expected) or expected.startswith(x) for x in extras)
        if len(set(observed)) > 1 and not benign:
            return FieldCheck("barcode", item.barcode, ", ".join(observed), "NEEDS_REVIEW", 1.0, "本 cell 内有多个条码，其中包含合同条码", source)
        return FieldCheck("barcode", item.barcode, observed[0], "PASS", 1.0, "条码一致", source)
    return FieldCheck("barcode", item.barcode, ", ".join(observed), "MISMATCH", 0.0, "已关联到该货号，但条码不一致", source)


def _barcode_scan_field(item: ContractItem, evidence: CellEvidence) -> FieldCheck:
    expected = normalize_barcode(item.barcode)
    observed = _dedupe_barcodes(evidence.barcode_scans)
    source = evidence.evidence_sources.get("barcode_scan", "")
    if not expected:
        return FieldCheck("barcode_scan", item.barcode, ", ".join(observed), "NEEDS_REVIEW", note="合同无条码", source=source)
    if not observed:
        return FieldCheck("barcode_scan", item.barcode, "", "MISSING", note="未扫描到真实条码图形", source=source)
    if expected in observed:
        extras = [x for x in observed if x != expected]
        if extras:
            return FieldCheck("barcode_scan", item.barcode, ", ".join(observed), "NEEDS_REVIEW", 1.0, "真实扫码包含合同条码，但同一单元还有其他扫码值", source)
        return FieldCheck("barcode_scan", item.barcode, observed[0], "PASS", 1.0, "真实条码图形扫码一致", source)
    return FieldCheck("barcode_scan", item.barcode, ", ".join(observed), "MISMATCH", 0.0, "真实条码图形扫码值与合同不一致", source)


def _barcode_text_field(item: ContractItem, evidence: CellEvidence) -> FieldCheck:
    expected = normalize_barcode(item.barcode)
    observed = _dedupe_barcodes(evidence.barcode_texts)
    source = evidence.evidence_sources.get("barcode_text", "")
    if not expected:
        return FieldCheck("barcode_text", item.barcode, ", ".join(observed), "NEEDS_REVIEW", note="合同无条码", source=source)
    if not observed:
        return FieldCheck("barcode_text", item.barcode, "", "MISSING", note="未识别到条码下方数字", source=source)
    if expected in observed:
        extras = [x for x in observed if x != expected]
        benign = all(x.startswith(expected) or expected.startswith(x) for x in extras)
        if extras and not benign:
            return FieldCheck("barcode_text", item.barcode, ", ".join(observed), "NEEDS_REVIEW", 1.0, "条码数字包含合同条码，但同一单元还有其他数字候选", source)
        return FieldCheck("barcode_text", item.barcode, expected, "PASS", 1.0, "条码下方数字一致", source)
    return FieldCheck("barcode_text", item.barcode, ", ".join(observed), "MISMATCH", 0.0, "条码下方数字与合同不一致", source)


def _barcode_scan_text_field(evidence: CellEvidence) -> FieldCheck:
    scans = _dedupe_barcodes(evidence.barcode_scans)
    texts = _dedupe_barcodes(evidence.barcode_texts)
    if not scans and not texts:
        return FieldCheck("barcode_scan_vs_text", "", "", "MISSING", note="没有扫码值和条码数字")
    if not scans:
        return FieldCheck("barcode_scan_vs_text", "", ", ".join(texts), "MISSING", note="有条码数字，但未扫描到真实条码图形")
    if not texts:
        return FieldCheck(
            "barcode_scan_vs_text",
            ", ".join(scans),
            "",
            "PASS",
            1.0,
            "scan barcode decoded; human-readable barcode OCR missing is not blocking",
        )
        return FieldCheck("barcode_scan_vs_text", ", ".join(scans), "", "NEEDS_REVIEW", note="有真实扫码值，但未识别到条码下方数字")
    if set(scans) & set(texts):
        extras = sorted((set(scans) | set(texts)) - (set(scans) & set(texts)))
        if extras:
            return FieldCheck("barcode_scan_vs_text", ", ".join(scans), ", ".join(texts), "NEEDS_REVIEW", 1.0, "扫码值与下方数字有交集，但存在额外候选")
        return FieldCheck("barcode_scan_vs_text", ", ".join(scans), ", ".join(texts), "PASS", 1.0, "真实扫码值与下方数字一致")
    return FieldCheck("barcode_scan_vs_text", ", ".join(scans), ", ".join(texts), "MISMATCH", 0.0, "真实扫码值与条码下方数字不一致")


def _barcode_text_field_v2(item: ContractItem, evidence: CellEvidence) -> FieldCheck:
    expected = normalize_barcode(item.barcode)
    observed = _dedupe_barcodes(evidence.barcode_texts)
    scanned = _dedupe_barcodes(evidence.barcode_scans)
    source = evidence.evidence_sources.get("barcode_text", "")
    if not expected:
        return FieldCheck("barcode_text", item.barcode, ", ".join(observed), "NEEDS_REVIEW", note="contract has no barcode", source=source)
    if not observed:
        if expected in scanned:
            return FieldCheck(
                "barcode_text",
                item.barcode,
                "",
                "PASS",
                1.0,
                "scan barcode matches contract; human-readable barcode OCR missing is not blocking",
                source or evidence.evidence_sources.get("barcode_scan", ""),
            )
        return FieldCheck("barcode_text", item.barcode, "", "MISSING", note="barcode text was not recognized", source=source)
    if expected in observed:
        extras = [x for x in observed if x != expected]
        benign = all(x.startswith(expected) or expected.startswith(x) for x in extras)
        if extras and not benign:
            return FieldCheck("barcode_text", item.barcode, ", ".join(observed), "NEEDS_REVIEW", 1.0, "barcode text includes contract barcode plus extra values", source)
        return FieldCheck("barcode_text", item.barcode, expected, "PASS", 1.0, "barcode text matches contract", source)
    return FieldCheck("barcode_text", item.barcode, ", ".join(observed), "MISMATCH", 0.0, "barcode text does not match contract", source)


def _barcode_scan_text_field_v2(evidence: CellEvidence) -> FieldCheck:
    scans = _dedupe_barcodes(evidence.barcode_scans)
    texts = _dedupe_barcodes(evidence.barcode_texts)
    if not scans and not texts:
        return FieldCheck("barcode_scan_vs_text", "", "", "MISSING", note="no scanned barcode or barcode text")
    if not scans:
        return FieldCheck("barcode_scan_vs_text", "", ", ".join(texts), "MISSING", note="barcode text exists but barcode graphic was not scanned")
    if not texts:
        return FieldCheck(
            "barcode_scan_vs_text",
            ", ".join(scans),
            "",
            "PASS",
            1.0,
            "scan barcode decoded; human-readable barcode OCR missing is not blocking",
        )
    if set(scans) & set(texts):
        extras = sorted((set(scans) | set(texts)) - (set(scans) & set(texts)))
        if extras:
            return FieldCheck("barcode_scan_vs_text", ", ".join(scans), ", ".join(texts), "NEEDS_REVIEW", 1.0, "scan barcode and barcode text overlap but include extra values")
        return FieldCheck("barcode_scan_vs_text", ", ".join(scans), ", ".join(texts), "PASS", 1.0, "scan barcode and barcode text match")
    return FieldCheck("barcode_scan_vs_text", ", ".join(scans), ", ".join(texts), "MISMATCH", 0.0, "scan barcode and barcode text conflict")


def _dedupe_barcodes(values: list[str]) -> list[str]:
    return [x for x in dict.fromkeys(normalize_barcode(v) for v in values) if x]


def _qty_field(contract: str, mark: str, source: str) -> FieldCheck:
    c = normalize_qty(contract)
    m = normalize_qty(mark)
    if c and not m:
        return FieldCheck("packing_qty", contract, mark, "MISSING", note="唛头缺少装箱数", source=source)
    if not c:
        return FieldCheck("packing_qty", contract, mark, "NEEDS_REVIEW", note="合同无装箱数", source=source)
    if c == m:
        return FieldCheck("packing_qty", contract, mark, "PASS", 1.0, "装箱数一致", source=source)
    if _is_visual_ocr_source(source):
        return FieldCheck("packing_qty", contract, mark, "NEEDS_REVIEW", 0.0, "OCR装箱数与合同不一致，需复核", source=source)
    return FieldCheck("packing_qty", contract, mark, "MISMATCH", 0.0, "装箱数不一致", source=source)


def _is_visual_ocr_source(source: str) -> bool:
    source = str(source)
    return source.startswith("visual_candidate_ocr") or source.startswith("mark_image_ocr")
