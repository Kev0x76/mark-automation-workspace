"""Evidence-based product-row and product-region recognition for contracts.

A populated cell in the item-number column is not enough to prove that a row
is a product.  Contracts often reuse the same columns for notes, signatures,
or numbered terms after the product table.  This module therefore requires
independent business signals and keeps three outcomes separate:

* product: item number + strict 13-digit barcode + numeric packing quantity;
* review: enough product evidence exists, but one required signal is invalid;
* non-product: the row does not contain enough mutually supporting evidence.

The rules are deterministic and intentionally do not depend on a blacklist of
words such as "备注" or "100%".
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
import re
from typing import Any, Iterable, Mapping
import unicodedata


CONTRACT_ITEM_REGION_METHOD = "multi_field_contract_item_region_v1"
MAX_REPORTED_IGNORED_ROWS = 100
MAX_REPORTED_REVIEW_ROWS = 20


class ContractExtractionError(ValueError):
    """An actionable contract-ingestion error with a stable API payload."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def as_detail(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class BarcodeCellEvidence:
    raw_text: str
    digits: str
    kind: str

    @property
    def is_candidate(self) -> bool:
        return self.kind in {"numeric", "digits_with_spacing"} and bool(self.digits)

    @property
    def is_valid_format(self) -> bool:
        return self.is_candidate and len(self.digits) == 13


@dataclass(frozen=True)
class ContractRowEvidence:
    row: int
    item_no: str
    item_no_plausible: bool
    barcode: BarcodeCellEvidence
    qty: int | float | None
    names: Mapping[str, str]

    @property
    def has_name(self) -> bool:
        return any(str(value or "").strip() for value in self.names.values())

    @property
    def is_product(self) -> bool:
        return bool(self.item_no) and self.barcode.is_valid_format and self.qty is not None

    @property
    def has_mapped_content(self) -> bool:
        return bool(
            self.item_no
            or self.barcode.raw_text
            or self.qty is not None
            or self.has_name
        )

    def review_reason(self, *, first_product_row: int, last_product_row: int) -> str | None:
        if self.is_product:
            return None
        inside_product_bounds = first_product_row <= self.row <= last_product_row

        if self.barcode.is_valid_format:
            if not self.item_no:
                return "条码有效但货号为空"
            if self.qty is None:
                return "条码有效但装箱数为空或不是数字"

        if self.item_no and self.qty is not None and self.has_name:
            if not self.barcode.is_candidate:
                return "货号、品名和装箱数存在，但条码不是纯数字单元格"
            if not self.barcode.is_valid_format:
                return "货号、品名和装箱数存在，但条码不是13位"

        if self.item_no_plausible and self.barcode.is_candidate:
            if not self.barcode.is_valid_format:
                return "货号与数字条码候选存在，但条码不是13位"
            if self.qty is None:
                return "货号与条码存在，但装箱数为空或不是数字"

        if self.item_no_plausible and self.has_name:
            if inside_product_bounds or self.qty is not None or self.barcode.is_candidate:
                return "货号与品名存在，但商品关键字段不完整"

        if inside_product_bounds and self.barcode.is_candidate and self.qty is not None:
            return "商品区域内存在条码和装箱数，但货号无效或为空"
        return None

    def ignored_record(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "reason": "insufficient_product_evidence",
            "signals": {
                "item_no_present": bool(self.item_no),
                "item_no_plausible": self.item_no_plausible,
                "barcode_cell_kind": self.barcode.kind,
                "barcode_format_valid": self.barcode.is_valid_format,
                "qty_numeric": self.qty is not None,
                "name_present": self.has_name,
            },
            "values": {
                "item_no": _truncate(self.item_no),
                "barcode_cell": _truncate(self.barcode.raw_text),
            },
        }

    def review_record(self, reason: str) -> dict[str, Any]:
        return {
            "row": self.row,
            "item_no": self.item_no,
            "reason": reason,
            "values": {
                "barcode_cell": _truncate(self.barcode.raw_text),
                "barcode_digits": self.barcode.digits,
                "qty": self.qty,
                "cn_name": _truncate(str(self.names.get("cn_name") or "")),
            },
        }


@dataclass(frozen=True)
class ContractItemRegionResult:
    product_rows: tuple[ContractRowEvidence, ...]
    ignored_rows: tuple[dict[str, Any], ...]
    ignored_rows_count: int
    spans: tuple[dict[str, int], ...]

    @property
    def first_product_row(self) -> int:
        return self.product_rows[0].row

    @property
    def last_product_row(self) -> int:
        return self.product_rows[-1].row

    def report(self, *, sheet: str, header_rows: list[int]) -> dict[str, Any]:
        return {
            "method": CONTRACT_ITEM_REGION_METHOD,
            "sheet": sheet,
            "header_rows": list(header_rows),
            "first_product_row": self.first_product_row,
            "last_product_row": self.last_product_row,
            "item_rows": [entry.row for entry in self.product_rows],
            "spans": [dict(span) for span in self.spans],
            "ignored_non_product_rows_count": self.ignored_rows_count,
            "ignored_non_product_rows": [dict(entry) for entry in self.ignored_rows],
        }


def inspect_contract_row(
    *,
    row: int,
    item_no: str,
    barcode_value: Any,
    qty: int | float | None,
    names: Mapping[str, str],
) -> ContractRowEvidence:
    return ContractRowEvidence(
        row=int(row),
        item_no=str(item_no or "").strip(),
        item_no_plausible=is_plausible_item_no(item_no),
        barcode=inspect_barcode_cell(barcode_value),
        qty=qty,
        names=dict(names),
    )


def inspect_barcode_cell(value: Any) -> BarcodeCellEvidence:
    """Read a barcode without extracting digits from arbitrary prose."""

    if value is None or isinstance(value, bool):
        return BarcodeCellEvidence(raw_text="", digits="", kind="empty")
    if isinstance(value, Number):
        number = float(value)
        raw_text = str(int(number)) if number.is_integer() else str(value)
        digits = raw_text if re.fullmatch(r"\d+", raw_text) else ""
        return BarcodeCellEvidence(raw_text=raw_text, digits=digits, kind="numeric")

    raw_text = unicodedata.normalize("NFKC", str(value)).strip()
    if not raw_text:
        return BarcodeCellEvidence(raw_text="", digits="", kind="empty")
    if re.fullmatch(r"[0-9\s\u00a0]+", raw_text):
        digits = re.sub(r"\s+", "", raw_text)
        return BarcodeCellEvidence(
            raw_text=raw_text,
            digits=digits,
            kind="digits_with_spacing",
        )
    return BarcodeCellEvidence(raw_text=raw_text, digits="", kind="prose_or_mixed")


def is_plausible_item_no(value: Any) -> bool:
    text = re.sub(r"\s+", "", str(value or "").strip())
    if not text:
        return False
    digits = re.sub(r"\D", "", text)
    if digits == text and len(digits) in {12, 13, 14}:
        return False
    return bool(
        re.fullmatch(
            r"[A-Za-z]{0,8}[-_]?\d{3,10}(?:[-_][A-Za-z0-9]+)?",
            text,
        )
    )


def identify_contract_item_region(
    rows: Iterable[ContractRowEvidence],
    *,
    sheet: str,
    header_rows: list[int],
) -> ContractItemRegionResult:
    evidence = tuple(sorted(rows, key=lambda entry: entry.row))
    product_rows = tuple(entry for entry in evidence if entry.is_product)
    if not product_rows:
        raise ContractExtractionError(
            "contract_item_region_not_found",
            f"工作表“{sheet}”没有找到可证明的商品行；商品行必须同时具有货号、13位条码和数字装箱数。",
            details={"sheet": sheet, "header_rows": header_rows},
        )

    first_product_row = product_rows[0].row
    last_product_row = product_rows[-1].row
    review_rows = []
    ignored_all = []
    for entry in evidence:
        if entry.is_product or not entry.has_mapped_content:
            continue
        reason = entry.review_reason(
            first_product_row=first_product_row,
            last_product_row=last_product_row,
        )
        if reason:
            review_rows.append(entry.review_record(reason))
        else:
            ignored_all.append(entry.ignored_record())

    if review_rows:
        reported = review_rows[:MAX_REPORTED_REVIEW_ROWS]
        summary = "；".join(
            f"第{entry['row']}行"
            + (f"（货号 {entry['item_no']}）" if entry.get("item_no") else "")
            + f"：{entry['reason']}"
            for entry in reported[:5]
        )
        suffix = "" if len(review_rows) <= 5 else f"；另有 {len(review_rows) - 5} 行"
        raise ContractExtractionError(
            "contract_row_review_required",
            f"工作表“{sheet}”存在可能属于商品区域但字段不完整的行：{summary}{suffix}。"
            "系统已停止解析，避免静默漏掉真实商品；请修正合同后重新上传。",
            details={
                "sheet": sheet,
                "header_rows": header_rows,
                "rows": reported,
                "total_review_rows": len(review_rows),
            },
        )

    return ContractItemRegionResult(
        product_rows=product_rows,
        ignored_rows=tuple(ignored_all[:MAX_REPORTED_IGNORED_ROWS]),
        ignored_rows_count=len(ignored_all),
        spans=tuple(_contiguous_spans(entry.row for entry in product_rows)),
    )


def _contiguous_spans(rows: Iterable[int]) -> list[dict[str, int]]:
    ordered = sorted(set(int(row) for row in rows))
    if not ordered:
        return []
    spans: list[dict[str, int]] = []
    start = previous = ordered[0]
    count = 1
    for row in ordered[1:]:
        if row == previous + 1:
            previous = row
            count += 1
            continue
        spans.append({"start_row": start, "end_row": previous, "item_count": count})
        start = previous = row
        count = 1
    spans.append({"start_row": start, "end_row": previous, "item_count": count})
    return spans


def _truncate(value: str, limit: int = 120) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[: limit - 1]}…"
