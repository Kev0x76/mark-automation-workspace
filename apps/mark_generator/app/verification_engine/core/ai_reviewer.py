from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .circuit_breaker import CircuitBreaker
from .models import CellEvidence, ContractItem, FieldCheck, VerificationResult


@dataclass(frozen=True)
class ReviewLoopConfig:
    max_rounds: int = 3
    max_flags: int = 24
    max_reason_parts: int = 12
    max_repeated_state: int = 2


_BREAKER = CircuitBreaker(max_seconds=20.0, max_errors=3)
_CONFIG = ReviewLoopConfig()


def review_result(
    item: ContractItem,
    evidence: CellEvidence | None,
    result: VerificationResult,
) -> VerificationResult:
    """Attach controlled reviewer advice without changing hard statuses.

    The loop records a compact audit trace. It deliberately avoids exposing
    unbounded chain-of-thought; the report contains decision summaries only.
    """
    result.review = _BREAKER.run(
        "review_loop",
        lambda: _review_loop(item, evidence, result, _CONFIG),
        fallback={
            "reviewer_status": "NEEDS_MANUAL_REVIEW",
            "reason_cn": "复核循环生成失败，保留规则核对结果。",
            "suggested_action": "按字段状态人工复核异常项。",
            "confidence": 0.0,
            "loop_stop_reason": "circuit_breaker_fallback",
            "events": list(_BREAKER.events),
            "hard_rule_locked": True,
        },
    )
    if _BREAKER.events:
        result.review.setdefault("events", list(_BREAKER.events))
    return result


def _review_loop(
    item: ContractItem,
    evidence: CellEvidence | None,
    result: VerificationResult,
    config: ReviewLoopConfig,
) -> dict[str, Any]:
    audit_trace: list[dict[str, Any]] = []
    seen_states: Counter[str] = Counter()
    last_review: dict[str, Any] = {}

    for round_no in range(1, config.max_rounds + 1):
        review = _build_review(item, evidence, result, config)
        signature = _state_signature(review)
        seen_states[signature] += 1

        audit_trace.append({
            "round": round_no,
            "status": review["reviewer_status"],
            "flags": review.get("flags", [])[:8],
            "summary": review["reason_cn"],
            "action": review["suggested_action"],
        })
        last_review = review

        stop_reason = _loop_stop_reason(review, seen_states[signature], round_no, config)
        if stop_reason:
            last_review["loop_stop_reason"] = stop_reason
            break
    else:
        last_review["loop_stop_reason"] = "max_rounds_reached"

    last_review["review_iterations"] = len(audit_trace)
    last_review["audit_trace"] = audit_trace
    last_review["reason_summary"] = _compact_reason(last_review.get("reason_cn", ""))
    last_review["cot_policy"] = "audit_summary_only"
    return last_review


def _build_review(
    item: ContractItem,
    evidence: CellEvidence | None,
    result: VerificationResult,
    config: ReviewLoopConfig,
) -> dict[str, Any]:
    if result.error:
        return {
            "reviewer_status": "NEEDS_MANUAL_REVIEW",
            "reason_cn": "未能把合同货号稳定关联到某个唛头单元。",
            "suggested_action": "先人工定位对应唛头，再检查货号、条码、品名和装箱数。",
            "confidence": 0.95,
            "flags": ["association_missing"],
            "status_counts": {},
            "hard_rule_locked": True,
        }

    fields = result.fields
    statuses = Counter(f.status for f in fields)
    flags = _bounded(_collect_flags(fields, evidence), config.max_flags)
    reason_parts = _bounded(
        _reason_parts(fields, evidence, result.association_confidence),
        config.max_reason_parts,
    )
    action = _suggest_action(fields, flags)

    if statuses.get("MISMATCH"):
        reviewer_status = "BLOCKING_ISSUE"
        confidence = 0.95
    elif statuses.get("MISSING") or statuses.get("NEEDS_REVIEW"):
        reviewer_status = "NEEDS_MANUAL_REVIEW"
        confidence = 0.86
    else:
        reviewer_status = "OK"
        confidence = 0.98

    return {
        "reviewer_status": reviewer_status,
        "reason_cn": "；".join(reason_parts) if reason_parts else "规则核对未发现异常。",
        "suggested_action": action,
        "confidence": confidence,
        "status_counts": dict(statuses),
        "flags": flags,
        "hard_rule_locked": True,
    }


def _collect_flags(fields: list[FieldCheck], evidence: CellEvidence | None) -> list[str]:
    flags: list[str] = []
    by_field = {f.field: f for f in fields}

    for field in fields:
        if field.status == "MISMATCH":
            flags.append(f"mismatch:{field.field}")
        elif field.status == "MISSING":
            flags.append(f"missing:{field.field}")
        elif field.status == "NEEDS_REVIEW":
            flags.append(f"review:{field.field}")

    if by_field.get("barcode_scan", FieldCheck("", "", "", "")).status == "MISSING":
        flags.append("barcode_graphic_not_scanned")
    if by_field.get("barcode_scan_vs_text", FieldCheck("", "", "", "")).status == "MISMATCH":
        flags.append("barcode_scan_text_conflict")
    if evidence and evidence.needs_ai:
        flags.append("weak_extraction_evidence")
    if evidence and not evidence.cell.spans and evidence.cell.barcodes:
        flags.append("image_only_or_no_text_layer")
    if evidence and any("cdr" in b.source.lower() for b in evidence.cell.barcodes):
        flags.append("cdr_payload_barcode_not_render_scan")
    return list(dict.fromkeys(flags))


def _reason_parts(
    fields: list[FieldCheck],
    evidence: CellEvidence | None,
    association_confidence: float,
) -> list[str]:
    parts: list[str] = []
    if association_confidence and association_confidence < 0.75:
        parts.append(f"关联置信度偏低({association_confidence:.2f})，需要先确认唛头单元是否对应正确货号")

    for field in fields:
        if field.status == "PASS":
            if field.field == "name_cn" and not field.mark_value:
                parts.append("中文品名未出现，已按可选字段跳过")
            continue
        if field.field == "barcode_scan" and field.status == "MISSING":
            parts.append("未扫描到真实条码图形，不能只凭条码下方数字当作最终条码核对")
        elif field.field == "barcode_scan_vs_text" and field.status == "MISMATCH":
            parts.append("真实扫码值与条码下方数字不一致，这是高风险异常")
        elif field.field.startswith("name_") and field.status in {"MISSING", "NEEDS_REVIEW"}:
            parts.append(f"{field.field} 需要人工判断品名与规格是否为可接受省略")
        elif field.status == "MISMATCH":
            parts.append(f"{field.field} 与合同不一致")
        elif field.status == "MISSING":
            parts.append(f"{field.field} 未从唛头中识别到")
        elif field.status == "NEEDS_REVIEW":
            parts.append(f"{field.field} 需要复核：{field.note}")

    if evidence and not evidence.raw_lines and evidence.cell.barcodes:
        parts.append("该单元主要依靠条码证据，文字字段可能需要 OCR 或人工补看")
    if evidence and any("cdr" in b.source.lower() for b in evidence.cell.barcodes):
        parts.append("CDR 条码来自源文件数据载荷，不等同于渲染后真实扫码")
    return list(dict.fromkeys(parts))


def _suggest_action(fields: list[FieldCheck], flags: list[str]) -> str:
    if any(flag.startswith("mismatch:barcode_scan") for flag in flags):
        return "暂停放行；以真实扫码值为准核对条码源文件，必要时要求工厂重出唛头。"
    if any(flag.startswith("mismatch:") for flag in flags):
        return "不要自动放行；逐项核对合同值和唛头值，确认是否需要改版。"
    if "image_only_or_no_text_layer" in flags:
        return "条码可作为关联线索，但文字层证据不足；补跑 OCR 或人工查看原图后，再确认货号、品名和装箱数。"
    if "cdr_payload_barcode_not_render_scan" in flags:
        return "CDR 数据可用于预检；最终印刷稿仍建议导出 PDF 后做一次真实条码图形扫描。"
    if "barcode_graphic_not_scanned" in flags:
        return "检查原文件是否为 CDR 数据、低清图片或图片型 PDF；优先补充可扫码图形或走 OCR/人工复核。"
    if any(flag.startswith("missing:name_") or flag.startswith("review:name_") for flag in flags):
        return "人工确认品名主体是否一致；规格缺失但不矛盾时可按业务规则放行。"
    if any(f.status != "PASS" for f in fields):
        return "按异常字段逐项复核，保留规则核对状态作为最终依据。"
    return "可按规则结果放行；抽检时重点看真实扫码值、货号和装箱数。"


def _loop_stop_reason(
    review: dict[str, Any],
    repeated_state_count: int,
    round_no: int,
    config: ReviewLoopConfig,
) -> str:
    if review["reviewer_status"] == "BLOCKING_ISSUE":
        return "hard_mismatch_locked"
    if review["reviewer_status"] == "OK":
        return "no_exception"
    if len(review.get("flags", [])) >= config.max_flags:
        return "too_many_flags"
    if repeated_state_count >= config.max_repeated_state:
        return "stable_repeated_state"
    if round_no >= config.max_rounds:
        return "max_rounds_reached"
    return ""


def _state_signature(review: dict[str, Any]) -> str:
    flags = ",".join(review.get("flags", []))
    counts = ",".join(f"{k}:{v}" for k, v in sorted(review.get("status_counts", {}).items()))
    return f"{review.get('reviewer_status')}|{counts}|{flags}"


def _compact_reason(reason: str, max_chars: int = 180) -> str:
    if len(reason) <= max_chars:
        return reason
    return reason[: max_chars - 3].rstrip("；,， ") + "..."


def _bounded(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    return values[:limit] + [f"TRUNCATED_{len(values) - limit}_MORE"]
