from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...services.contract_snapshot import read_contract_snapshot
from .models import ContractItem
from .normalize import (
    is_dispimg,
    looks_like_barcode,
    looks_like_item_number,
    normalize_barcode,
    normalize_item_number,
    normalize_qty,
    normalize_text,
)

logger = logging.getLogger(__name__)


FIELD_ALIASES: dict[str, list[str]] = {
    "item_number": ["电脑编号", "电脑  编号", "货号", "产品编号", "Item No.", "item no"],
    "barcode": ["条形码", "条码", "barcode", "ean", "upc"],
    "outer_barcode": ["外箱条码", "外箱条形码", "外箱码", "箱码", "carton barcode", "outer barcode"],
    "name_cn": ["品名", "中文品名", "产品名称", "商品名称"],
    "name_it": ["意文品名", "意大利语品名", "意大利文品名", "Italian Name"],
    "name_es": ["西文品名", "西班牙语品名", "西班牙文品名", "Spanish Name"],
    "name_en": ["英语品名", "英文品名", "英文", "English Name"],
    "name_pt": ["葡文品名", "葡萄牙语品名", "Portuguese Name"],
    "packing_qty": ["装箱数", "装箱数量", "装箱量", "Packing Qty", "Qty/CTN", "pcs/ctn"],
}


def load_contract_snapshot(
    snapshot_path: str | Path,
    *,
    contract_path: str | Path | None = None,
) -> list[ContractItem]:
    """Load verified contract facts without reopening the Excel workbook.

    ``contract_path`` is still supplied for project verification so the
    snapshot's source hash is checked against the archived uploaded file.
    """

    payload = read_contract_snapshot(snapshot_path, contract_path=contract_path)
    items: list[ContractItem] = []
    for entry in payload["items"]:
        if not isinstance(entry, dict):
            raise ValueError("Contract snapshot contains an invalid item entry.")
        names = entry.get("names") or {}
        source = entry.get("source") or {}
        item = ContractItem(
            item_number=normalize_item_number(entry.get("item_number")),
            barcode=normalize_barcode(entry.get("barcode")),
            name_cn=normalize_text(names.get("cn")),
            name_it=normalize_text(names.get("it")),
            name_es=normalize_text(names.get("es")),
            name_en=normalize_text(names.get("en")),
            name_pt=normalize_text(names.get("pt")),
            packing_qty=normalize_qty(entry.get("packing_qty")),
            raw={
                "snapshot_source_row": source.get("row"),
                "snapshot_source_cells": source.get("cells") or {},
            },
        )
        if not item.is_valid:
            raise ValueError(
                f"Contract snapshot item is incomplete: {entry.get('item_number') or '(missing item number)'}"
            )
        items.append(item)
    return items


def load_contract(path: str | Path, sheet: str | int = 0) -> list[ContractItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = _read_rows(path, sheet)
    if not rows:
        return []

    header_idx = _detect_header_row(rows)
    header = rows[header_idx]
    col_map = _map_columns(header)
    col_map = _repair_columns_by_content(rows, header_idx, col_map)

    if "item_number" not in col_map:
        raise ValueError(f"无法识别货号列: {path}")

    items: list[ContractItem] = []
    for row in rows[header_idx + 1:]:
        if _row_empty(row):
            continue
        item = _parse_item(row, col_map)
        if item and item.is_valid:
            items.append(item)
        elif item and item.item_number and not any(item.names.values()):
            # 遇到合计/说明行时通常不会再有有效品名，继续容错而不是中断。
            continue

    logger.info("loaded %d contract items from %s", len(items), path.name)
    return items


def _read_rows(path: Path, sheet: str | int) -> list[list[Any]]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
        ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
        rows = [list(row) for row in ws.iter_rows(values_only=True)]
        wb.close()
        return rows
    if suffix == ".xls":
        import xlrd

        wb = xlrd.open_workbook(str(path))
        ws = wb.sheet_by_index(sheet) if isinstance(sheet, int) else wb.sheet_by_name(sheet)
        return [[ws.cell_value(r, c) for c in range(ws.ncols)] for r in range(ws.nrows)]
    raise ValueError(f"不支持的合同格式: {path.suffix}")


def _clean_header(value: Any) -> str:
    return normalize_text(value).replace(" ", "").lower()


def _detect_header_row(rows: list[list[Any]]) -> int:
    alias_set = {_clean_header(alias) for aliases in FIELD_ALIASES.values() for alias in aliases}
    best_idx = 0
    best_score = 0.0
    for idx, row in enumerate(rows[:60]):
        cells = [_clean_header(v) for v in row if normalize_text(v)]
        if len(cells) < 2:
            continue
        exact = sum(1 for c in cells if c in alias_set)
        fuzzy = sum(1 for c in cells for a in alias_set if a and (a in c or c in a))
        score = exact * 2 + fuzzy * 0.3
        if any("条" in c and "码" in c for c in cells):
            score += 2
        if any("品名" in c or "名称" in c for c in cells):
            score += 2
        if any("编号" in c or c == "货号" for c in cells):
            score += 2
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _map_columns(header: list[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for idx, cell in enumerate(header):
        text = _clean_header(cell)
        if not text:
            continue
        for field, aliases in FIELD_ALIASES.items():
            if field in result:
                continue
            cleaned_aliases = [_clean_header(alias) for alias in aliases]
            if text in cleaned_aliases or any(a and (a in text or text in a) for a in cleaned_aliases):
                result[field] = idx
                break
    return result


def _repair_columns_by_content(
    rows: list[list[Any]],
    header_idx: int,
    col_map: dict[str, int],
) -> dict[str, int]:
    max_cols = max((len(r) for r in rows), default=0)
    data_rows = rows[header_idx + 1: header_idx + 30]

    def values_for(col: int) -> list[Any]:
        return [r[col] for r in data_rows if col < len(r) and normalize_text(r[col])]

    def score_barcode_col(col: int) -> float:
        vals = values_for(col)
        if not vals:
            return 0.0
        score = 0.0
        for v in vals:
            if is_dispimg(v):
                score -= 2.5
            if looks_like_barcode(v):
                score += 3.0
            elif normalize_barcode(v):
                score += min(len(normalize_barcode(v)) / 14, 1)
        return score / max(len(vals), 1)

    def score_item_col(col: int) -> float:
        vals = values_for(col)
        if not vals:
            return 0.0
        return sum(1 for v in vals if looks_like_item_number(v)) / len(vals)

    if "barcode" in col_map:
        current = col_map["barcode"]
        candidates = [c for c in range(max(0, current - 2), min(max_cols, current + 4))]
        best = max(candidates, key=score_barcode_col)
        if score_barcode_col(best) > max(score_barcode_col(current), 0.5):
            col_map["barcode"] = best
    else:
        best = max(range(max_cols), key=score_barcode_col, default=-1)
        if best >= 0 and score_barcode_col(best) >= 1.5:
            col_map["barcode"] = best

    if "item_number" in col_map:
        current = col_map["item_number"]
        candidates = [c for c in range(max(0, current - 1), min(max_cols, current + 3))]
        best = max(candidates, key=score_item_col)
        if score_item_col(best) > score_item_col(current):
            col_map["item_number"] = best
    else:
        best = max(range(max_cols), key=score_item_col, default=-1)
        if best >= 0 and score_item_col(best) >= 0.5:
            col_map["item_number"] = best

    return col_map


def _parse_item(row: list[Any], col_map: dict[str, int]) -> ContractItem | None:
    def get(field: str) -> Any:
        idx = col_map.get(field)
        if idx is None or idx >= len(row):
            return ""
        return row[idx]

    item_number = normalize_item_number(get("item_number"))
    if not item_number:
        return None

    raw_barcode = get("barcode")
    barcode = "" if is_dispimg(raw_barcode) else normalize_barcode(raw_barcode)
    outer_raw = get("outer_barcode")
    outer_barcode = "" if is_dispimg(outer_raw) else normalize_barcode(outer_raw)

    item = ContractItem(
        item_number=item_number,
        barcode=barcode,
        outer_barcode=outer_barcode,
        name_cn=normalize_text(get("name_cn")),
        name_it=normalize_text(get("name_it")),
        name_es=normalize_text(get("name_es")),
        name_en=normalize_text(get("name_en")),
        name_pt=normalize_text(get("name_pt")),
        packing_qty=normalize_qty(get("packing_qty")),
        raw={field: get(field) for field in col_map},
    )
    return item


def _row_empty(row: list[Any]) -> bool:
    return not any(normalize_text(v) and not is_dispimg(v) for v in row)
