from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .models import CellEvidence, VerificationResult


def write_json(
    path: str | Path,
    evidences: list[CellEvidence],
    results: list[VerificationResult],
) -> None:
    payload = {
        "evidence_cells": [_evidence_to_dict(e) for e in evidences],
        "results": [asdict(r) for r in results],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: str | Path, results: list[VerificationResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "item_number", "cell_id", "association_confidence", "association_reason",
        "field", "status", "contract_value", "mark_value", "similarity", "note", "source",
        "evidence_grade", "evidence_grade_label", "evidence_grade_reason",
        "reviewer_status", "review_reason", "suggested_action", "review_confidence",
        "review_iterations", "loop_stop_reason", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            if not r.fields:
                review = r.review or {}
                evidence_grade = review.get("evidence_grade", {})
                writer.writerow([
                    r.item_number, r.cell_id, f"{r.association_confidence:.3f}", r.association_reason,
                    "", "", "", "", "", "", "",
                    evidence_grade.get("grade", ""),
                    evidence_grade.get("label", ""),
                    evidence_grade.get("reason", ""),
                    review.get("reviewer_status", ""),
                    review.get("reason_cn", ""),
                    review.get("suggested_action", ""),
                    _format_confidence(review.get("confidence")),
                    review.get("review_iterations", ""),
                    review.get("loop_stop_reason", ""),
                    r.error,
                ])
                continue
            review = r.review or {}
            evidence_grade = review.get("evidence_grade", {})
            for field in r.fields:
                writer.writerow([
                    r.item_number,
                    r.cell_id,
                    f"{r.association_confidence:.3f}",
                    r.association_reason,
                    field.field,
                    field.status,
                    field.contract_value,
                    field.mark_value,
                    f"{field.similarity:.3f}",
                    field.note,
                    field.source,
                    evidence_grade.get("grade", ""),
                    evidence_grade.get("label", ""),
                    evidence_grade.get("reason", ""),
                    review.get("reviewer_status", ""),
                    review.get("reason_cn", ""),
                    review.get("suggested_action", ""),
                    _format_confidence(review.get("confidence")),
                    review.get("review_iterations", ""),
                    review.get("loop_stop_reason", ""),
                    r.error,
                ])


def write_item_summary_csv(path: str | Path, results: list[VerificationResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "item_number", "overall_status", "evidence_grade", "evidence_grade_label",
        "evidence_grade_reason", "cell_id", "association_confidence",
        "failed_fields", "review_fields", "missing_fields",
        "review_reason", "suggested_action", "loop_stop_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for result in results:
            review = result.review or {}
            evidence_grade = review.get("evidence_grade", {})
            fields = result.fields or []
            mismatches = [x.field for x in fields if x.status == "MISMATCH"]
            missing = [x.field for x in fields if x.status == "MISSING"]
            needs_review = [x.field for x in fields if x.status == "NEEDS_REVIEW"]
            writer.writerow([
                result.item_number,
                _overall_status(result, mismatches, missing, needs_review),
                evidence_grade.get("grade", ""),
                evidence_grade.get("label", ""),
                evidence_grade.get("reason", ""),
                result.cell_id,
                f"{result.association_confidence:.3f}",
                ";".join(mismatches),
                ";".join(needs_review),
                ";".join(missing),
                review.get("reason_summary") or review.get("reason_cn", ""),
                review.get("suggested_action", ""),
                review.get("loop_stop_reason", ""),
            ])


def _evidence_to_dict(e: CellEvidence) -> dict:
    d = asdict(e)
    d["cell"]["spans"] = [
        {"text": s.text, "bbox": s.bbox, "size": s.size, "source": s.source}
        for s in e.cell.spans
    ]
    return d


def _format_confidence(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return ""


def _overall_status(
    result: VerificationResult,
    mismatches: list[str],
    missing: list[str],
    needs_review: list[str],
) -> str:
    reviewer_status = (result.review or {}).get("reviewer_status", "")
    if result.error:
        return "NEEDS_MANUAL_REVIEW"
    if mismatches:
        return "BLOCKING_ISSUE"
    if reviewer_status == "BLOCKING_ISSUE":
        return "BLOCKING_ISSUE"
    if missing or needs_review:
        return "NEEDS_MANUAL_REVIEW"
    if reviewer_status == "NEEDS_MANUAL_REVIEW":
        return "NEEDS_MANUAL_REVIEW"
    return "OK"
