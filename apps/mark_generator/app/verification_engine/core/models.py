from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContractItem:
    item_number: str = ""
    barcode: str = ""
    outer_barcode: str = ""
    name_cn: str = ""
    name_it: str = ""
    name_es: str = ""
    name_en: str = ""
    name_pt: str = ""
    packing_qty: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> dict[str, str]:
        return {
            "cn": self.name_cn,
            "it": self.name_it,
            "es": self.name_es,
            "en": self.name_en,
            "pt": self.name_pt,
        }

    @property
    def is_valid(self) -> bool:
        return bool(self.item_number or self.barcode) and any(self.names.values())


@dataclass
class TextSpan:
    text: str
    bbox: tuple[float, float, float, float]
    page: int = 0
    size: float = 0.0
    source: str = "text_layer"

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass
class BarcodeEvidence:
    value: str
    bbox: tuple[float, float, float, float] | None = None
    page: int = 0
    source: str = "unknown"
    confidence: float = 1.0


@dataclass
class PageData:
    file_path: str
    page: int
    width: float
    height: float
    has_text_layer: bool
    spans: list[TextSpan] = field(default_factory=list)
    text_lines: list[str] = field(default_factory=list)
    barcodes: list[BarcodeEvidence] = field(default_factory=list)
    render_image: Any = None
    errors: list[str] = field(default_factory=list)
    visual_rects: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class Cell:
    cell_id: str
    file_path: str
    page: int
    bbox: tuple[float, float, float, float]
    spans: list[TextSpan] = field(default_factory=list)
    barcodes: list[BarcodeEvidence] = field(default_factory=list)
    segmentation_method: str = "unknown"
    needs_ai: bool = False


@dataclass
class CellEvidence:
    cell: Cell
    item_numbers: list[str] = field(default_factory=list)
    barcodes: list[str] = field(default_factory=list)
    barcode_scans: list[str] = field(default_factory=list)
    barcode_texts: list[str] = field(default_factory=list)
    names: dict[str, list[str]] = field(default_factory=lambda: {
        "cn": [], "it": [], "es": [], "en": [], "pt": [], "latin": [], "unknown": []
    })
    packing_qty: str = ""
    raw_lines: list[str] = field(default_factory=list)
    evidence_sources: dict[str, str] = field(default_factory=dict)
    needs_ai: bool = False

    def best_name(self, lang: str) -> str:
        vals = self.names.get(lang, [])
        return " ".join(dict.fromkeys(v for v in vals if v))

    def to_mark_fields(self) -> dict[str, str]:
        return {
            "item_number": self.item_numbers[0] if self.item_numbers else "",
            "barcode": self.barcodes[0] if self.barcodes else "",
            "barcode_scan": self.barcode_scans[0] if self.barcode_scans else "",
            "barcode_text": self.barcode_texts[0] if self.barcode_texts else "",
            "name_cn": self.best_name("cn"),
            "name_it": self.best_name("it"),
            "name_es": self.best_name("es"),
            "name_en": self.best_name("en"),
            "name_pt": self.best_name("pt"),
            "packing_qty": self.packing_qty,
        }


@dataclass
class Association:
    cell_id: str
    item_number: str
    confidence: float
    score_detail: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class FieldCheck:
    field: str
    contract_value: str
    mark_value: str
    status: str
    similarity: float = 0.0
    note: str = ""
    source: str = ""


@dataclass
class VerificationResult:
    item_number: str
    cell_id: str = ""
    association_confidence: float = 0.0
    association_reason: str = ""
    fields: list[FieldCheck] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    error: str = ""
