from __future__ import annotations

from .models import Association, CellEvidence, ContractItem
from .normalize import normalize_barcode, normalize_item_number, normalize_qty
from .semantic_matcher import name_similarity


def resolve_cells(
    evidences: list[CellEvidence],
    contract_items: list[ContractItem],
) -> dict[str, Association]:
    associations: dict[str, Association] = {}
    used_cells: set[str] = set()

    for item in contract_items:
        best_ev: CellEvidence | None = None
        best_detail: dict[str, float] = {}
        best_score = -1.0
        for ev in evidences:
            if ev.cell.cell_id in used_cells:
                # 如果同一 cell 有多个货号会漏，但这是比整页混合更保守的第一版。
                continue
            score, detail = _score_item_cell(item, ev)
            if score > best_score:
                best_score = score
                best_detail = detail
                best_ev = ev
        if best_ev is None:
            continue
        if best_detail.get("item_number", 0) <= 0 and best_detail.get("barcode", 0) <= 0:
            continue
        confidence = min(1.0, best_score / 100.0)
        if confidence < 0.35:
            continue
        reason = _reason_from_detail(best_detail)
        associations[item.item_number] = Association(
            cell_id=best_ev.cell.cell_id,
            item_number=item.item_number,
            confidence=confidence,
            score_detail=best_detail,
            reason=reason,
        )
        if confidence >= 0.60:
            used_cells.add(best_ev.cell.cell_id)
        _promote_unknown_names(item, best_ev)
    return associations


def _score_item_cell(item: ContractItem, ev: CellEvidence) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    item_no = normalize_item_number(item.item_number)
    cell_items = [normalize_item_number(x) for x in ev.item_numbers]
    if item_no and item_no in cell_items:
        detail["item_number"] = 70
    elif item_no and any(item_no in x or x in item_no for x in cell_items if len(x) >= 4):
        detail["item_number"] = 35
    else:
        detail["item_number"] = 0

    contract_barcodes = {normalize_barcode(item.barcode), normalize_barcode(item.outer_barcode)} - {""}
    cell_barcodes = {normalize_barcode(x) for x in ev.barcodes + ev.barcode_texts} - {""}
    if contract_barcodes and contract_barcodes & cell_barcodes:
        detail["barcode"] = 45
    elif contract_barcodes and cell_barcodes:
        detail["barcode"] = -8
    else:
        detail["barcode"] = 0

    name_score = 0.0
    name_count = 0
    for lang, contract_name in item.names.items():
        if not contract_name:
            continue
        mark_name = ev.best_name(lang)
        if not mark_name:
            continue
        status, sim, _note = name_similarity(contract_name, mark_name, "cn" if lang == "cn" else lang)
        if status == "PASS":
            name_score += 16 * sim
        elif status == "NEEDS_REVIEW":
            name_score += 8 * sim
        name_count += 1
    detail["name"] = min(35, name_score if name_count else 0)

    if item.packing_qty and ev.packing_qty:
        detail["packing_qty"] = 8 if normalize_qty(item.packing_qty) == normalize_qty(ev.packing_qty) else -2
    else:
        detail["packing_qty"] = 0

    score = sum(detail.values())
    return score, detail


def _reason_from_detail(detail: dict[str, float]) -> str:
    parts = [f"{k}:{v:g}" for k, v in detail.items() if v]
    return ", ".join(parts) if parts else "无强关联证据"


def _promote_unknown_names(item: ContractItem, ev: CellEvidence | None) -> None:
    if ev is None:
        return
    unknown = list(ev.names.get("unknown", []))
    if not unknown:
        return
    for lang, contract_name in item.names.items():
        if lang == "cn":
            continue
        if not contract_name or ev.names.get(lang):
            continue
        best_text = ""
        best_sim = 0.0
        for text in unknown:
            status, sim, _ = name_similarity(contract_name, text, lang)
            if status in {"PASS", "NEEDS_REVIEW"} and sim > best_sim:
                best_text = text
                best_sim = sim
        if best_text and best_sim >= 0.45:
            ev.names.setdefault(lang, []).append(best_text)
            ev.evidence_sources.setdefault(f"name_{lang}", "contract_guided_unknown")
