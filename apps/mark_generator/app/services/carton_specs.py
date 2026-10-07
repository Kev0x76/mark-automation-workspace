"""Carton specification facts (L/W/H and gross weight) per SKU.

The factory supplies these as a small table, either as an extra sheet inside the
contract workbook or as a standalone workbook:

    电脑编号 | 品名 | 长 | 宽 | 高 | 毛重 | 体积
    3210808  | ...  | 25 | 12 | 25 | 0.7  | 0.0075

Dimensions are centimetres, gross weight is kilograms. There is no net weight
column, so ``N.W.`` stays unset and is never derived - the factory owns that
value (see business rule 10 in PROJECT_CONTEXT.md).

This module is deliberately optional and failure-closed: a contract with no
spec table yields no carton facts, and the existing fixed-template pipeline
keeps working untouched. Ambiguity is reported, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .excel_contract import load_contract_workbook, normalize_label


CM_TO_MM = 10.0

# Longest alias first so "外箱长" wins over "长".
DIMENSION_ALIASES: dict[str, list[str]] = {
    "length": ["外箱长", "箱长", "长度", "长", "length", "l"],
    "width": ["外箱宽", "箱宽", "宽度", "宽", "width", "w"],
    "height": ["外箱高", "箱高", "高度", "高", "height", "h"],
}
GROSS_WEIGHT_ALIASES = ["毛重", "gross weight", "g.w.", "gw", "gross"]
NET_WEIGHT_ALIASES = ["净重", "net weight", "n.w.", "nw", "net"]
ITEM_NO_ALIASES = [
    "电脑编号", "电脑  编号", "电脑编码", "货号", "商品编号", "产品编号",
    "item no", "item number", "sku", "article no",
]

# A carton edge below 20 mm or above 2000 mm is not a shipping carton; treat it
# as a mis-read column rather than silently producing an absurd layout.
MIN_EDGE_MM = 20.0
MAX_EDGE_MM = 2000.0
MAX_GROSS_WEIGHT_KG = 200.0


class CartonSpecError(ValueError):
    """Raised only for unusable input, never for a merely absent spec table."""


@dataclass(frozen=True)
class CartonFacts:
    """One SKU's carton, in millimetres and kilograms."""

    item_no: str
    length_mm: float
    width_mm: float
    height_mm: float
    gross_weight_kg: float | None = None
    net_weight_kg: float | None = None
    source_sheet: str | None = None
    source_row: int | None = None

    @property
    def front_face_mm(self) -> tuple[float, float]:
        """Front mark face = carton length x height."""
        return (self.length_mm, self.height_mm)

    @property
    def side_face_mm(self) -> tuple[float, float]:
        """Side mark face = carton width x height."""
        return (self.width_mm, self.height_mm)

    def meas_display_cm(self) -> str:
        """The MEAS value exactly as the factory writes it, in cm."""
        return "%s×%s×%s" % (
            _trim_number(self.length_mm / CM_TO_MM),
            _trim_number(self.width_mm / CM_TO_MM),
            _trim_number(self.height_mm / CM_TO_MM),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_no": self.item_no,
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "gross_weight_kg": self.gross_weight_kg,
            "net_weight_kg": self.net_weight_kg,
            "meas_display_cm": self.meas_display_cm(),
            "front_face_mm": list(self.front_face_mm),
            "side_face_mm": list(self.side_face_mm),
            "source_sheet": self.source_sheet,
            "source_row": self.source_row,
        }


@dataclass
class CartonSpecResult:
    facts: dict[str, CartonFacts] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    source_sheet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sheet": self.source_sheet,
            "item_count": len(self.facts),
            "items": {key: value.to_dict() for key, value in self.facts.items()},
            "issues": self.issues,
        }


def _trim_number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _issue(code: str, message: str, *, severity: str = "warning", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def _to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _match_column(label: str, aliases: list[str]) -> bool:
    """Exact-or-contains match on the normalized label, longest alias first."""
    normalized = normalize_label(label)
    if not normalized:
        return False
    for alias in aliases:
        target = normalize_label(alias)
        if not target:
            continue
        if normalized == target:
            return True
    # Fall back to containment only for multi-character aliases, so a bare "l"
    # never swallows an unrelated header.
    for alias in aliases:
        target = normalize_label(alias)
        if len(target) >= 2 and target in normalized:
            return True
    return False


def _resolve_header(worksheet: Any, max_scan_rows: int = 12) -> tuple[int, dict[str, int]] | None:
    """Find the row that carries item-no plus all three dimension columns."""
    max_row = min(int(getattr(worksheet, "max_row", 0) or 0), max_scan_rows)
    max_column = min(int(getattr(worksheet, "max_column", 0) or 0), 40)
    for row in range(1, max_row + 1):
        columns: dict[str, int] = {}
        for column in range(1, max_column + 1):
            raw = worksheet.cell(row, column).value
            if raw is None:
                continue
            label = str(raw)
            if "item_no" not in columns and _match_column(label, ITEM_NO_ALIASES):
                columns["item_no"] = column
                continue
            for field_name, aliases in DIMENSION_ALIASES.items():
                if field_name not in columns and _match_column(label, aliases):
                    columns[field_name] = column
                    break
            else:
                if "gross_weight" not in columns and _match_column(label, GROSS_WEIGHT_ALIASES):
                    columns["gross_weight"] = column
                elif "net_weight" not in columns and _match_column(label, NET_WEIGHT_ALIASES):
                    columns["net_weight"] = column
        if {"item_no", "length", "width", "height"} <= columns.keys():
            return row, columns
    return None


def _normalize_item_no(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def parse_carton_specs(source: Path | str) -> CartonSpecResult:
    """Read carton facts from a workbook. Absent table -> empty result."""
    path = Path(source)
    if not path.is_file():
        raise CartonSpecError(f"Carton spec workbook not found: {path}")

    workbook = load_contract_workbook(path, data_only=True)
    worksheets = list(getattr(workbook, "worksheets", []))
    result = CartonSpecResult()

    best: tuple[int, Any, int, dict[str, int]] | None = None
    for worksheet in worksheets:
        resolved = _resolve_header(worksheet)
        if not resolved:
            continue
        header_row, columns = resolved
        score = len(columns)
        if best is None or score > best[0]:
            best = (score, worksheet, header_row, columns)

    if best is None:
        result.issues.append(
            _issue(
                "carton_spec_table_absent",
                "No carton specification table (item number plus 长/宽/高) was found; "
                "carton-adaptive layout is unavailable for this contract.",
                severity="info",
            )
        )
        return result

    _, worksheet, header_row, columns = best
    result.source_sheet = worksheet.title

    seen_rows: dict[str, int] = {}
    for row in range(header_row + 1, int(worksheet.max_row) + 1):
        item_no = _normalize_item_no(worksheet.cell(row, columns["item_no"]).value)
        if not item_no:
            continue
        raw_dims = {
            name: _to_number(worksheet.cell(row, columns[name]).value)
            for name in ("length", "width", "height")
        }
        if any(value is None for value in raw_dims.values()):
            missing = [name for name, value in raw_dims.items() if value is None]
            result.issues.append(
                _issue(
                    "carton_spec_row_incomplete",
                    f"Carton row for {item_no} is missing {', '.join(missing)}; row skipped.",
                    item_no=item_no,
                    sheet=worksheet.title,
                    row=row,
                )
            )
            continue

        dims_mm = {name: value * CM_TO_MM for name, value in raw_dims.items()}
        out_of_range = {
            name: value for name, value in dims_mm.items()
            if not (MIN_EDGE_MM <= value <= MAX_EDGE_MM)
        }
        if out_of_range:
            result.issues.append(
                _issue(
                    "carton_spec_dimension_out_of_range",
                    f"Carton dimensions for {item_no} are outside {MIN_EDGE_MM:.0f}-{MAX_EDGE_MM:.0f} mm "
                    f"({out_of_range}); the column mapping or unit is likely wrong, so the row is skipped.",
                    severity="error",
                    item_no=item_no,
                    sheet=worksheet.title,
                    row=row,
                )
            )
            continue

        gross = _to_number(worksheet.cell(row, columns["gross_weight"]).value) if "gross_weight" in columns else None
        if gross is not None and not (0 < gross <= MAX_GROSS_WEIGHT_KG):
            result.issues.append(
                _issue(
                    "carton_spec_gross_weight_out_of_range",
                    f"Gross weight {gross} for {item_no} is not a plausible kg value; left unset.",
                    item_no=item_no,
                    sheet=worksheet.title,
                    row=row,
                )
            )
            gross = None
        net = _to_number(worksheet.cell(row, columns["net_weight"]).value) if "net_weight" in columns else None

        facts = CartonFacts(
            item_no=item_no,
            length_mm=round(dims_mm["length"], 3),
            width_mm=round(dims_mm["width"], 3),
            height_mm=round(dims_mm["height"], 3),
            gross_weight_kg=gross,
            net_weight_kg=net,
            source_sheet=worksheet.title,
            source_row=row,
        )

        if item_no in result.facts:
            previous = result.facts[item_no]
            if (previous.length_mm, previous.width_mm, previous.height_mm) != (
                facts.length_mm, facts.width_mm, facts.height_mm
            ):
                # Conflicting duplicates are ambiguous: fail closed for this SKU
                # rather than picking a winner.
                result.issues.append(
                    _issue(
                        "carton_spec_conflicting_duplicate",
                        f"{item_no} appears more than once with different carton dimensions "
                        f"(rows {seen_rows.get(item_no)} and {row}); both dropped.",
                        severity="error",
                        item_no=item_no,
                        sheet=worksheet.title,
                        row=row,
                    )
                )
                result.facts.pop(item_no, None)
                continue
            continue

        result.facts[item_no] = facts
        seen_rows[item_no] = row

    if not result.facts:
        result.issues.append(
            _issue(
                "carton_spec_table_empty",
                f"Sheet {worksheet.title!r} looked like a carton table but produced no usable rows.",
                severity="error",
                sheet=worksheet.title,
            )
        )
    return result


def carton_facts_for_items(
    spec: CartonSpecResult,
    items: list[dict[str, Any]],
) -> tuple[dict[str, CartonFacts], list[dict[str, Any]]]:
    """Align carton facts to contract items, reporting SKUs with no carton."""
    matched: dict[str, CartonFacts] = {}
    issues: list[dict[str, Any]] = []
    for item in items:
        item_no = _normalize_item_no(item.get("item_no"))
        if not item_no:
            continue
        facts = spec.facts.get(item_no)
        if facts is None:
            issues.append(
                _issue(
                    "carton_spec_missing_for_item",
                    f"No carton dimensions for {item_no}; it cannot use carton-adaptive layout.",
                    item_no=item_no,
                )
            )
            continue
        matched[item_no] = facts
    return matched, issues


CARTON_SPEC_CANDIDATES = (
    "carton_specs.xlsx",
    "carton_specs.xls",
    "箱规.xlsx",
    "箱规.xls",
)


def resolve_project_carton_specs(project_dir: Path) -> CartonSpecResult:
    """Find carton facts for a project.

    Two accepted sources, in order:

    1. a dedicated spec workbook dropped in ``input/`` - the explicit case;
    2. an extra sheet inside the contract workbook itself, which is how the
       factory usually sends it (the real A2 sample carries the order sheet and
       the carton sheet in one file).

    Nothing here mutates the contract snapshot: carton facts are layout input,
    not contract truth.
    """
    input_dir = project_dir / "input"
    searched: list[str] = []

    for name in CARTON_SPEC_CANDIDATES:
        candidate = input_dir / name
        if candidate.is_file():
            searched.append(candidate.name)
            result = parse_carton_specs(candidate)
            if result.facts:
                return result

    for pattern in ("contract.xlsx", "contract.xls"):
        candidate = input_dir / pattern
        if candidate.is_file():
            searched.append(candidate.name)
            try:
                result = parse_carton_specs(candidate)
            except CartonSpecError:
                continue
            if result.facts:
                return result

    empty = CartonSpecResult()
    empty.issues.append(
        _issue(
            "carton_spec_source_not_found",
            "No carton dimension table was found. Add a carton spec workbook to the "
            "project's input/ directory, or include a 长/宽/高 sheet in the contract. "
            f"Looked at: {', '.join(searched) or 'nothing'}.",
            severity="error",
        )
    )
    return empty


__all__ = [
    "CARTON_SPEC_CANDIDATES",
    "CartonFacts",
    "CartonSpecError",
    "CartonSpecResult",
    "carton_facts_for_items",
    "parse_carton_specs",
    "resolve_project_carton_specs",
]
