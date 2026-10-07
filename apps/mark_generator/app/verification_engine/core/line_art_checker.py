from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import CellEvidence, ContractItem, FieldCheck
from .normalize import normalize_compact, normalize_text


@dataclass(frozen=True)
class ProductFamily:
    code: str
    label: str
    terms: tuple[str, ...]


# Generic product-family guardrails. These are category words, not supplier,
# item-number, or sample-specific rules.
PRODUCT_FAMILIES: tuple[ProductFamily, ...] = (
    ProductFamily("cleaning_tool", "清洁工具", ("broom", "mop", "brush", "duster", "sweeper", "扫把", "扫帚", "拖把", "刷子", "掸子", "fregona", "scopa", "escoba")),
    ProductFamily("furniture", "家具", ("table", "desk", "chair", "stool", "shelf", "cabinet", "sofa", "桌子", "桌", "椅子", "椅", "凳", "架子", "柜", "tavolo", "sedia")),
    ProductFamily("footwear", "鞋类", ("shoe", "shoes", "insole", "slipper", "sandal", "鞋", "鞋垫", "拖鞋", "鞋子", "plantilla")),
    ProductFamily("apparel", "服饰", ("skirt", "dress", "shirt", "pants", "pajamas", "costume", "裙", "裙子", "衣服", "睡衣", "服装", "falda", "vestito")),
    ProductFamily("tableware", "餐具", ("fork", "spoon", "knife", "plate", "cup", "straw", "叉", "勺", "刀", "盘", "杯", "吸管", "tenedor", "cuchara", "plato")),
    ProductFamily("stationery", "文具", ("pen", "pencil", "notebook", "eraser", "ruler", "笔", "铅笔", "本子", "橡皮", "尺子")),
    ProductFamily("toy", "玩具", ("toy", "doll", "puzzle", "ball", "car", "玩具", "娃娃", "拼图", "球")),
    ProductFamily("beauty_container", "美妆容器", ("bottle", "jar", "tube", "flacone", "frasco", "瓶", "罐", "管")),
)


INCOMPATIBLE_GROUPS: tuple[set[str], ...] = (
    {"cleaning_tool", "furniture", "footwear", "apparel", "tableware", "stationery", "toy"},
    {"footwear", "tableware", "furniture", "stationery", "toy"},
)


def line_art_semantic_field(item: ContractItem, evidence: CellEvidence) -> FieldCheck | None:
    """Return a line-art semantic field when there is category evidence.

    This is not a full visual classifier. It catches only high-signal semantic
    conflicts exposed by mark text/OCR around the line-art area. Image-only
    recognition is intentionally not attempted: the current production policy
    trusts the generator's AI lineart and the visual OC path is disabled.
    """

    contract_texts = [value for value in item.names.values() if value]
    mark_texts = _mark_texts(evidence)
    expected_signals = detect_product_signals(contract_texts)
    observed_signals = detect_product_signals(mark_texts)
    expected = set(expected_signals)
    observed = set(observed_signals)
    if not expected or not observed:
        return None

    conflicts = sorted(
        (left, right)
        for left in expected
        for right in observed
        if left != right and _families_conflict(left, right)
    )
    shared = expected & observed
    expected_label = _family_summary(expected, expected_signals)
    observed_label = _family_summary(observed, observed_signals)
    if conflicts and not shared:
        return FieldCheck(
            field="line_art",
            contract_value=expected_label,
            mark_value=observed_label,
            status="MISMATCH",
            similarity=0.0,
            note="线稿/产品图语义疑似跨品类冲突，不能自动放行",
            source="line_art_text_semantic_guard",
        )
    if conflicts and shared:
        return FieldCheck(
            field="line_art",
            contract_value=expected_label,
            mark_value=observed_label,
            status="NEEDS_REVIEW",
            similarity=0.5,
            note="线稿/产品图同时出现合同品类和冲突品类信号，需要确认是否为 OCR 噪声或混入其他产品",
            source="line_art_text_semantic_guard",
        )
    if shared:
        return FieldCheck(
            field="line_art",
            contract_value=expected_label,
            mark_value=observed_label,
            status="NEEDS_REVIEW",
            similarity=0.5,
            note="唛头文字品类一致，但文字不能证明线稿视觉一致",
            source="line_art_text_semantic_guard",
        )
    if False and shared:  # Retained unreachable legacy block for source compatibility.
        return FieldCheck(
            field="line_art",
            contract_value=expected_label,
            mark_value=observed_label,
            status="NEEDS_REVIEW",
            similarity=0.5,
            note="线稿/产品图语义品类一致",
            source="line_art_text_semantic_guard",
        )
    return FieldCheck(
        field="line_art",
        contract_value=expected_label,
        mark_value=observed_label,
        status="NEEDS_REVIEW",
        similarity=0.3,
        note="线稿/产品图有品类信号，但与合同品类未形成同类匹配",
        source="line_art_text_semantic_guard",
    )


def detect_product_families(texts: list[str]) -> set[str]:
    return set(detect_product_signals(texts))


def detect_product_signals(texts: list[str]) -> dict[str, list[str]]:
    joined = " ".join(normalize_text(text) for text in texts if normalize_text(text))
    if not joined:
        return {}
    found: dict[str, list[str]] = {}
    for family in PRODUCT_FAMILIES:
        hits = [term for term in family.terms if _term_matches(joined, term)]
        if hits:
            found[family.code] = hits[:4]
    return found


def _mark_texts(evidence: CellEvidence) -> list[str]:
    values: list[str] = []
    values.extend(evidence.raw_lines)
    for names in evidence.names.values():
        values.extend(names)
    return list(dict.fromkeys(value for value in values if normalize_text(value)))


def _term_matches(text: str, term: str) -> bool:
    text_norm = normalize_text(text)
    term_norm = normalize_text(term)
    if not text_norm or not term_norm:
        return False
    if re.search(r"[\u4e00-\u9fff]", term_norm):
        return term_norm in text_norm

    compact_term = normalize_compact(term_norm)
    tokens = re.findall(r"[A-Z0-9]+", text_norm.upper())
    if compact_term in tokens:
        return True
    if len(compact_term) <= 5:
        return False
    compact_text = normalize_compact(text_norm)
    if compact_term in compact_text:
        return True
    return any(fuzz.ratio(compact_term, token) >= 92 for token in tokens)


def _families_conflict(left: str, right: str) -> bool:
    return any(left in group and right in group for group in INCOMPATIBLE_GROUPS)


def _family_summary(codes: set[str], signals: dict[str, list[str]] | None = None) -> str:
    labels = {family.code: family.label for family in PRODUCT_FAMILIES}
    signals = signals or {}
    parts = []
    for code in sorted(codes):
        terms = signals.get(code, [])
        suffix = f"({', '.join(terms[:3])})" if terms else ""
        parts.append(f"{code}:{labels.get(code, code)}{suffix}")
    return ", ".join(parts)
