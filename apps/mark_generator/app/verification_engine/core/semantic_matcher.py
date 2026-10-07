from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz

from .normalize import normalize_compact, normalize_text


CN_SPEC_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:ML|G|KG|CM|MM|M|L|PCS|只|个|件|厘米|毫升|克)", re.I)
EU_SPEC_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:ML|G|KG|CM|MM|M|L|PCS|PC|PZ|PZS|OZ)", re.I)


EU_NOISE = {
    "DA", "DI", "DE", "DEL", "CON", "PARA", "PER", "SET", "OF", "THE", "AND",
    "VIAGGIO", "VIAJE", "TRAVEL", "BOTTLE", "BOTTLES", "BOTE", "BOTES",
    "FLACONE", "FLACONI", "FRASCO", "KIT", "PACK", "PEZZI", "TAPPO",
}


def name_similarity(contract: str, mark: str, lang: str = "") -> tuple[str, float, str]:
    c = normalize_text(contract)
    m = normalize_text(mark)
    if not c and not m:
        return "PASS", 1.0, "双方为空"
    if c and not m:
        return "MISSING", 0.0, "唛头缺少品名"
    if not c and m:
        return "NEEDS_REVIEW", 0.0, "合同无该语言品名但唛头存在"

    if lang == "cn" or re.search(r"[\u4e00-\u9fff]", c):
        return _cn_similarity(c, m)
    return _eu_similarity(c, m)


def _cn_similarity(contract: str, mark: str) -> tuple[str, float, str]:
    c_core = re.sub(r"[\(（][^)）]*[\)）]", "", contract)
    c_specs = [normalize_compact(x) for x in CN_SPEC_RE.findall(contract)]
    c_tokens = _cn_tokens(c_core)
    c_compact = _cjk_compact(c_core)
    m_compact_cjk = _cjk_compact(mark)
    if len(m_compact_cjk) >= 2 and m_compact_cjk in c_compact:
        coverage = len(m_compact_cjk) / max(len(c_compact), 1)
        return "PASS", max(0.72, min(1.0, coverage)), "中文缩写/上位词可接受"
    if len(m_compact_cjk) >= 2 and c_compact and m_compact_cjk[-1] in c_compact:
        return "PASS", 0.68, "中文类目缩写可接受"
    if not c_tokens:
        score = fuzz.partial_ratio(normalize_compact(contract), normalize_compact(mark)) / 100
        return _status_from_score(score), score, f"中文 fuzzy={score:.2f}"

    m_compact = normalize_compact(mark)
    found = [t for t in c_tokens if normalize_compact(t) in m_compact]
    coverage = len(found) / max(len(c_tokens), 1)
    spec_found = [s for s in c_specs if s in m_compact]

    if coverage >= 0.75:
        if c_specs and not spec_found:
            return "PASS", coverage, "核心品名匹配，规格未完整出现"
        return "PASS", min(1.0, coverage), f"核心词匹配 {len(found)}/{len(c_tokens)}"
    if coverage >= 0.45:
        return "NEEDS_REVIEW", coverage, f"中文核心词部分匹配 {len(found)}/{len(c_tokens)}"

    fuzzy = fuzz.partial_ratio(normalize_compact(contract), normalize_compact(mark)) / 100
    if fuzzy >= 0.82:
        return "PASS", fuzzy, "中文 fuzzy 通过"
    if fuzzy >= 0.62:
        return "NEEDS_REVIEW", fuzzy, "中文 fuzzy 相似，需确认"
    return "MISMATCH", max(coverage, fuzzy), "中文核心品名不匹配"


def _cn_tokens(text: str) -> list[str]:
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens: list[str] = []
    for chunk in chunks:
        if len(chunk) <= 4:
            tokens.append(chunk)
        else:
            tokens.extend([chunk[i:i + 2] for i in range(0, len(chunk) - 1, 2)])
    return list(dict.fromkeys(tokens))


def _cjk_compact(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", normalize_text(text)))


def _eu_similarity(contract: str, mark: str) -> tuple[str, float, str]:
    if normalize_compact(contract) and normalize_compact(contract) == normalize_compact(mark):
        return "PASS", 1.0, "EU OCR_SPACING_COLLAPSED"

    c_words = _eu_keywords(contract)
    m_words = _eu_keywords(mark)
    contract_cmp = normalize_text(contract).upper()
    mark_cmp = normalize_text(mark).upper()
    contract_compact = normalize_compact(contract)
    mark_compact = normalize_compact(mark)
    length_coverage = min(len(mark_compact), len(contract_compact)) / max(len(contract_compact), 1)
    symmetric_fuzzy = max(
        fuzz.ratio(contract_cmp, mark_cmp),
        fuzz.token_sort_ratio(contract_cmp, mark_cmp),
    ) / 100
    partial_fuzzy = max(
        fuzz.partial_ratio(contract_cmp, mark_cmp),
        fuzz.token_set_ratio(contract_cmp, mark_cmp),
    ) / 100
    if not c_words:
        if symmetric_fuzzy >= 0.86 and length_coverage >= 0.75:
            return "PASS", symmetric_fuzzy, f"EU fuzzy={symmetric_fuzzy:.2f}"
        if max(symmetric_fuzzy, partial_fuzzy) >= 0.65:
            return "NEEDS_REVIEW", symmetric_fuzzy, "EU fuzzy 相似但完整性不足，需确认"
        return "MISMATCH", symmetric_fuzzy, "EU 品名不匹配"
    overlap = set(c_words) & set(m_words)
    coverage = len(overlap) / len(c_words)

    c_specs = set(_spec_tokens(contract))
    m_specs = set(_spec_tokens(mark))
    missing_specs = sorted(c_specs - m_specs)
    if (
        0.45 <= coverage < 0.70
        and symmetric_fuzzy >= 0.88
        and length_coverage >= 0.60
    ):
        return "PASS", symmetric_fuzzy, "EU fuzzy OCR_SPACING"

    if coverage >= 0.70:
        if missing_specs:
            return "PASS", coverage, "核心品名匹配，规格缺失: " + ", ".join(missing_specs[:3])
        return "PASS", coverage, f"关键词匹配 {len(overlap)}/{len(c_words)}"
    if coverage >= 0.45:
        return "NEEDS_REVIEW", coverage, f"关键词部分匹配 {len(overlap)}/{len(c_words)}"

    fuzzy = max(symmetric_fuzzy, partial_fuzzy)
    # A substring can score 100 with partial_ratio/token_set_ratio while proving
    # only a small part of the contract name.  Below the keyword PASS threshold,
    # fuzzy evidence may request review but must never create a production PASS.
    if fuzzy >= 0.88:
        if len(mark_compact) < 4:
            return "MISMATCH", max(coverage, fuzzy * 0.3), "EU 唛头品名证据过短"
        return "NEEDS_REVIEW", max(coverage, symmetric_fuzzy), "EU fuzzy 相似但合同关键词覆盖不足"
    if fuzzy >= 0.65:
        if len(mark_compact) < 4:
            return "MISMATCH", max(coverage, fuzzy * 0.3), "EU 唛头品名证据过短"
        return "NEEDS_REVIEW", fuzzy, "EU fuzzy 相似，需确认"
    return "MISMATCH", max(coverage, fuzzy), "EU 核心品名不匹配"


def _eu_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Z0-9]{2,}", normalize_text(text).upper())
    keywords = [w for w in words if w not in EU_NOISE and not re.fullmatch(r"\d+", w)]
    return list(dict.fromkeys(keywords))


def _spec_tokens(text: str) -> Iterable[str]:
    return [normalize_compact(x) for x in EU_SPEC_RE.findall(text)]


def _status_from_score(score: float) -> str:
    if score >= 0.86:
        return "PASS"
    if score >= 0.65:
        return "NEEDS_REVIEW"
    return "MISMATCH"
