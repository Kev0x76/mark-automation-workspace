from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .models import CellEvidence, VerificationResult


@dataclass(frozen=True)
class EvidenceGrade:
    grade: str
    label: str
    risk: str
    reason: str
    signals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GRADE_DEFINITIONS = {
    "A_STRONG_SCAN": {
        "label": "A 强扫描证据",
        "risk": "low",
        "reason": "货号、真实扫码和字段核对均通过，且裁剪没有明显边界风险。",
    },
    "B_STRUCTURAL_TEXT": {
        "label": "B 结构/文本证据",
        "risk": "medium_low",
        "reason": "规则核对通过，但主要依赖文本层、结构定位或缺少真实扫码确认。",
    },
    "C_OCR_PROMOTED": {
        "label": "C OCR/裁剪回读晋级",
        "risk": "medium",
        "reason": "规则核对通过，但依赖 OCR、裁剪后回读或唯一残余框晋级，应纳入抽检和回归重点。",
    },
    "D_REVIEW_REQUIRED": {
        "label": "D 需要复核",
        "risk": "high",
        "reason": "存在字段异常、关联不足、兜底整页或视觉候选证据，不能自动放行。",
    },
}


def apply_evidence_grades(
    results: list[VerificationResult],
    evidences: list[CellEvidence],
    crop_manifest: dict[str, Any],
) -> dict[str, Any]:
    evidence_by_cell = {e.cell.cell_id: e for e in evidences}
    crop_by_item = {
        str(item.get("item_number", "")): item
        for item in crop_manifest.get("items", [])
        if item.get("item_number")
    }
    counts: Counter[str] = Counter()

    for result in results:
        crop_review_reasons = _crop_review_reasons(crop_by_item.get(result.item_number, {}))
        if crop_review_reasons:
            _lock_review_status(result, crop_review_reasons)
        grade = grade_result(
            result,
            evidence_by_cell.get(result.cell_id),
            crop_by_item.get(result.item_number, {}),
        )
        result.review = dict(result.review or {})
        result.review["evidence_grade"] = grade.to_dict()
        counts[grade.grade] += 1

    return {
        "grade_counts": dict(counts),
        "definitions": GRADE_DEFINITIONS,
    }


def grade_result(
    result: VerificationResult,
    evidence: CellEvidence | None,
    crop_item: dict[str, Any] | None,
) -> EvidenceGrade:
    crop_item = crop_item or {}
    signals = _signals(result, evidence, crop_item)

    if _needs_review(result) or _is_fallback_or_visual(crop_item):
        return _grade("D_REVIEW_REQUIRED", signals)

    if _has_ocr_or_residual_dependency(evidence, crop_item):
        return _grade("C_OCR_PROMOTED", signals)

    if _has_real_scan_pass(result) and result.association_confidence >= 0.90 and _crop_is_clean(crop_item):
        return _grade("A_STRONG_SCAN", signals)

    return _grade("B_STRUCTURAL_TEXT", signals)


def _grade(code: str, signals: list[str]) -> EvidenceGrade:
    meta = GRADE_DEFINITIONS[code]
    return EvidenceGrade(
        grade=code,
        label=meta["label"],
        risk=meta["risk"],
        reason=meta["reason"],
        signals=signals,
    )


def _needs_review(result: VerificationResult) -> bool:
    if result.error:
        return True
    reviewer_status = (result.review or {}).get("reviewer_status", "")
    if reviewer_status in {"BLOCKING_ISSUE", "NEEDS_MANUAL_REVIEW"}:
        return True
    return any(field.status in {"MISMATCH", "MISSING", "NEEDS_REVIEW"} for field in result.fields)


def _is_fallback_or_visual(crop_item: dict[str, Any]) -> bool:
    status = str(crop_item.get("status", ""))
    method = str(crop_item.get("crop_method", ""))
    return status in {"PAGE_FALLBACK", "VISUAL_CELL_CANDIDATE"} or method == "visual_candidate"


def _crop_review_reasons(crop_item: dict[str, Any] | None) -> list[str]:
    crop_item = crop_item or {}
    reasons: list[str] = []
    status = str(crop_item.get("status", ""))
    method = str(crop_item.get("crop_method", ""))
    quality = crop_item.get("crop_quality", {}) or {}

    if status == "PAGE_FALLBACK":
        reasons.append("整页预览只是兜底图，不是单个唛头识别成功")
    if status == "VISUAL_CELL_CANDIDATE" or method == "visual_candidate":
        reasons.append("视觉候选切块未完成自动核对")
    if quality.get("association_unverified"):
        reasons.append("唯一残余框只证明还有一个未分配外框，未证明与该货号强关联")
    return reasons


def _lock_review_status(result: VerificationResult, reasons: list[str]) -> None:
    review = dict(result.review or {})
    if review.get("reviewer_status") != "BLOCKING_ISSUE":
        review["reviewer_status"] = "NEEDS_MANUAL_REVIEW"
        review["confidence"] = min(float(review.get("confidence") or 0.86), 0.86)
    existing_reason = str(review.get("reason_cn", "")).strip()
    reason_text = "；".join(reasons)
    if reason_text and reason_text not in existing_reason:
        review["reason_cn"] = "；".join(x for x in [existing_reason, reason_text] if x)
    if review.get("reason_cn"):
        review["reason_summary"] = _compact_text(str(review["reason_cn"]))
    existing_action = str(review.get("suggested_action", "")).strip()
    action = "先确认裁剪边界与货号关联，再按字段核对结果判断是否放行。"
    if action not in existing_action:
        review["suggested_action"] = "；".join(x for x in [existing_action, action] if x)
    review["safety_lock"] = "CROP_REVIEW_REQUIRED"
    review["safety_reasons"] = reasons
    result.review = review


def _compact_text(value: str, max_chars: int = 180) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip("；,， ") + "..."


def _has_ocr_or_residual_dependency(
    evidence: CellEvidence | None,
    crop_item: dict[str, Any],
) -> bool:
    method = str(crop_item.get("crop_method", ""))
    quality = crop_item.get("crop_quality", {}) or {}
    if method == "raster_residual_rect_bbox" or quality.get("association_unverified"):
        return True
    if evidence is None:
        return False
    source_text = " ".join(str(value) for value in evidence.evidence_sources.values()).lower()
    return "ocr" in source_text or "mark_image" in source_text


def _has_real_scan_pass(result: VerificationResult) -> bool:
    return any(
        field.field == "barcode_scan" and field.status == "PASS" and bool(field.mark_value)
        for field in result.fields
    )


def _crop_is_clean(crop_item: dict[str, Any]) -> bool:
    if crop_item.get("status") != "OK":
        return False
    quality = crop_item.get("crop_quality", {}) or {}
    if quality.get("touches_page_edge") or quality.get("possible_other_items"):
        return False
    if quality.get("foreign_item_anchors"):
        return False
    if float(quality.get("edge_cut_risk") or 0.0) > 0.5:
        return False
    return True


def _signals(
    result: VerificationResult,
    evidence: CellEvidence | None,
    crop_item: dict[str, Any],
) -> list[str]:
    values: list[str] = []
    if result.error:
        values.append("result_error")
    if result.association_confidence:
        values.append(f"association:{result.association_confidence:.3f}")
    field_counts = Counter(field.status for field in result.fields)
    for status, count in sorted(field_counts.items()):
        values.append(f"fields:{status}:{count}")
    if crop_item:
        values.append(f"crop_status:{crop_item.get('status', '')}")
        if crop_item.get("crop_method"):
            values.append(f"crop_method:{crop_item.get('crop_method')}")
        quality = crop_item.get("crop_quality", {}) or {}
        if quality.get("association_unverified"):
            values.append("crop_association_unverified")
        if quality.get("possible_other_items") or quality.get("foreign_item_anchors"):
            values.append("crop_possible_other_items")
        if quality.get("touches_page_edge"):
            values.append("crop_touches_page_edge")
        if float(quality.get("edge_cut_risk") or 0.0) > 0.5:
            values.append(f"edge_cut_risk:{float(quality.get('edge_cut_risk')):.3f}")
    else:
        values.append("crop_missing")
    if evidence:
        method = evidence.cell.segmentation_method
        if method:
            values.append(f"segmentation:{method}")
        sources = sorted({str(value) for value in evidence.evidence_sources.values() if value})
        if sources:
            values.append("sources:" + ",".join(sources[:6]))
    else:
        values.append("evidence_missing")
    return list(dict.fromkeys(values))
