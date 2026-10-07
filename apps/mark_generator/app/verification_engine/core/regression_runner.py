from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..pipeline import run_pipeline
from .capability_probe import image_paths_from_manifest, run_probe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "regression_cases.json"


def run_regressions(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _read_json(config_path)
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "runs" / f"regression_{stamp}"
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_reports: list[dict[str, Any]] = []
    for case in config.get("cases", []):
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            continue
        if case_ids and case_id not in case_ids:
            continue
        if case.get("enabled", True) is False:
            case_reports.append({
                "id": case_id,
                "status": "SKIPPED",
                "failures": [],
                "metrics": {},
            })
            continue
        case_reports.append(_run_case(case, output_dir / _safe_name(case_id)))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "status": "PASS" if all(case["status"] in {"PASS", "SKIPPED"} for case in case_reports) else "FAIL",
        "cases": case_reports,
    }
    (output_dir / "regression_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "regression_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _run_case(case: dict[str, Any], case_output_dir: Path) -> dict[str, Any]:
    case_id = str(case.get("id", "case")).strip()
    failures: list[str] = []
    try:
        contract = _resolve_path(case.get("contract", ""))
        marks = [_resolve_path(path) for path in case.get("marks", [])]
        if not contract.exists():
            failures.append(f"missing_contract:{contract}")
        for mark in marks:
            if not mark.exists():
                failures.append(f"missing_mark:{mark}")
        if failures:
            return {"id": case_id, "status": "FAIL", "failures": failures, "metrics": {}}

        summary = run_pipeline(contract, marks, case_output_dir)
        probe_report = _run_capability_probe(case, case_output_dir)
        metrics = _collect_metrics(case_output_dir, summary, probe_report)
        failures.extend(_check_expectations(metrics, case.get("expect", {})))
        failures.extend(_check_probe_expectations(metrics, case.get("capability_probe", {})))
        return {
            "id": case_id,
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "output_dir": str(case_output_dir),
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "id": case_id,
            "status": "FAIL",
            "failures": [f"exception:{type(exc).__name__}:{exc}"],
            "output_dir": str(case_output_dir),
            "metrics": {},
        }


def _run_capability_probe(case: dict[str, Any], case_output_dir: Path) -> dict[str, Any]:
    config = case.get("capability_probe", {})
    if not config or config.get("enabled", False) is False:
        return {}

    manifest_path = case_output_dir / "mark_images_manifest.json"
    if not manifest_path.exists():
        return {
            "status": "SKIPPED",
            "summary": {},
            "breaker_events": ["PROBE_SKIPPED:missing_mark_images_manifest"],
        }

    max_images = int(config.get("max_images", 2))
    images = image_paths_from_manifest(manifest_path, max_images=max_images)
    return run_probe(
        image_paths=images,
        output_dir=case_output_dir / "capability_probe",
        manifest_path=manifest_path,
        ocr_engines=list(config.get("ocr_engines", ["rapidocr"])),
        max_images=max_images,
        max_total_seconds=float(config.get("max_total_seconds", 60.0)),
        ocr_timeout_seconds=float(config.get("ocr_timeout_seconds", 15.0)),
    )


def _collect_metrics(
    case_output_dir: Path,
    summary: dict[str, Any],
    probe_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_json = _read_json(case_output_dir / "v5_result.json")
    item_rows = _read_csv(case_output_dir / "v5_item_summary.csv")
    crop_manifest = _read_json(case_output_dir / "mark_images_manifest.json")

    status_counts = Counter(row.get("overall_status", "UNKNOWN") or "UNKNOWN" for row in item_rows)
    grade_counts = Counter(row.get("evidence_grade", "UNSET") or "UNSET" for row in item_rows)
    crop_status_by_item = {
        str(item.get("item_number", "")): str(item.get("status", ""))
        for item in crop_manifest.get("items", [])
    }
    crop_quality_by_item = {
        str(item.get("item_number", "")): item.get("crop_quality", {}) or {}
        for item in crop_manifest.get("items", [])
    }
    result_cell_by_item = {
        str(result.get("item_number", "")): str(result.get("cell_id", ""))
        for result in result_json.get("results", [])
    }

    page_fallback_ok_count = sum(
        1 for row in item_rows
        if row.get("overall_status") == "OK" and crop_status_by_item.get(row.get("item_number", "")) == "PAGE_FALLBACK"
    )
    visual_candidate_ok_count = sum(
        1 for row in item_rows
        if row.get("overall_status") == "OK" and "visual_candidate" in result_cell_by_item.get(row.get("item_number", ""))
    )
    unverified_crop_ok_count = sum(
        1 for row in item_rows
        if row.get("overall_status") == "OK"
        and crop_quality_by_item.get(row.get("item_number", ""), {}).get("association_unverified")
    )

    required_files = [
        "v5_result.json",
        "v5_result.csv",
        "v5_item_summary.csv",
        "v5_summary.json",
        "mark_images_manifest.json",
        "visual_report.html",
    ]
    missing_outputs = [name for name in required_files if not (case_output_dir / name).exists()]

    metrics = {
        "contract_items": int(summary.get("contract_items", 0)),
        "result_items": len(item_rows),
        "evidence_cells": int(summary.get("evidence_cells", 0)),
        "mark_images_count": int(summary.get("mark_images_count", 0)),
        "visual_cells_count": int(summary.get("visual_cells_count", 0)),
        "status_counts": dict(status_counts),
        "evidence_grade_counts": dict(grade_counts),
        "summary_evidence_grade_counts": summary.get("evidence_grade_counts", {}),
        "page_fallback_ok_count": page_fallback_ok_count,
        "visual_candidate_ok_count": visual_candidate_ok_count,
        "unverified_crop_ok_count": unverified_crop_ok_count,
        "missing_outputs": missing_outputs,
    }
    if probe_report:
        probe_summary = probe_report.get("summary", {})
        metrics["capability_probe"] = {
            "status": probe_report.get("status", ""),
            "report_json": str(case_output_dir / "capability_probe" / "capability_probe_report.json"),
            "report_md": str(case_output_dir / "capability_probe" / "capability_probe_report.md"),
            "images_tested": probe_summary.get("images_tested", 0),
            "barcode_hit_rate": probe_summary.get("barcode_hit_rate", 0.0),
            "ocr_hit_rate": probe_summary.get("ocr_hit_rate", {}),
            "ocr_avg_seconds": probe_summary.get("ocr_avg_seconds", {}),
            "crop_methods": probe_summary.get("crop_methods", {}),
            "crop_risks": probe_summary.get("crop_risks", {}),
            "disabled_engines": probe_summary.get("disabled_engines", []),
            "engine_errors": probe_summary.get("engine_errors", {}),
            "engine_timeouts": probe_summary.get("engine_timeouts", {}),
        }
    return metrics


def _check_expectations(metrics: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if metrics.get("contract_items") != metrics.get("result_items"):
        failures.append(f"result_count_mismatch:{metrics.get('result_items')}!={metrics.get('contract_items')}")
    if metrics.get("missing_outputs"):
        failures.append("missing_outputs:" + ",".join(metrics["missing_outputs"]))
    if sum(metrics.get("evidence_grade_counts", {}).values()) != metrics.get("result_items"):
        failures.append("evidence_grade_count_mismatch")
    if metrics.get("page_fallback_ok_count"):
        failures.append(f"page_fallback_marked_ok:{metrics['page_fallback_ok_count']}")
    if metrics.get("visual_candidate_ok_count"):
        failures.append(f"visual_candidate_marked_ok:{metrics['visual_candidate_ok_count']}")
    if metrics.get("unverified_crop_ok_count"):
        failures.append(f"unverified_crop_marked_ok:{metrics['unverified_crop_ok_count']}")

    expected_status_counts = expect.get("status_counts")
    if isinstance(expected_status_counts, dict):
        actual = {
            str(status): int(count)
            for status, count in metrics.get("status_counts", {}).items()
            if int(count)
        }
        expected = {
            str(status): int(count)
            for status, count in expected_status_counts.items()
            if int(count)
        }
        if actual != expected:
            failures.append(f"status_counts:{actual}!={expected}")

    exact_contract_items = expect.get("contract_items")
    if exact_contract_items is not None and metrics.get("contract_items") != int(exact_contract_items):
        failures.append(f"contract_items:{metrics.get('contract_items')}!={exact_contract_items}")
    min_contract_items = expect.get("min_contract_items")
    if min_contract_items is not None and metrics.get("contract_items", 0) < int(min_contract_items):
        failures.append(f"contract_items_below_min:{metrics.get('contract_items')}<{min_contract_items}")
    min_mark_images = expect.get("min_mark_images_count")
    if min_mark_images is not None and metrics.get("mark_images_count", 0) < int(min_mark_images):
        failures.append(f"mark_images_below_min:{metrics.get('mark_images_count')}<{min_mark_images}")
    max_blocking = expect.get("max_blocking")
    if max_blocking is not None:
        blocking = int(metrics.get("status_counts", {}).get("BLOCKING_ISSUE", 0))
        if blocking > int(max_blocking):
            failures.append(f"blocking_above_max:{blocking}>{max_blocking}")
    max_review = expect.get("max_manual_review")
    if max_review is not None:
        review = int(metrics.get("status_counts", {}).get("NEEDS_MANUAL_REVIEW", 0))
        if review > int(max_review):
            failures.append(f"manual_review_above_max:{review}>{max_review}")
    min_ok = expect.get("min_ok")
    if min_ok is not None:
        ok = int(metrics.get("status_counts", {}).get("OK", 0))
        if ok < int(min_ok):
            failures.append(f"ok_below_min:{ok}<{min_ok}")
    return failures


def _check_probe_expectations(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if not config or config.get("enabled", False) is False:
        return []
    failures: list[str] = []
    probe = metrics.get("capability_probe", {})
    if not probe:
        return ["capability_probe_missing"]
    if probe.get("status") in {"NO_IMAGES", "NO_SIGNAL"}:
        failures.append(f"capability_probe_status:{probe.get('status')}")

    expect = config.get("expect", {})
    min_barcode_hit_rate = expect.get("min_barcode_hit_rate")
    if min_barcode_hit_rate is not None:
        actual = float(probe.get("barcode_hit_rate") or 0.0)
        if actual < float(min_barcode_hit_rate):
            failures.append(f"barcode_hit_rate_below_min:{actual}<{min_barcode_hit_rate}")

    min_ocr_hit_rate = expect.get("min_ocr_hit_rate", {})
    for engine, minimum in min_ocr_hit_rate.items():
        actual = float((probe.get("ocr_hit_rate") or {}).get(engine, 0.0))
        if actual < float(minimum):
            failures.append(f"ocr_hit_rate_below_min:{engine}:{actual}<{minimum}")

    allowed_disabled = set(config.get("allowed_disabled_engines", []))
    unexpected_disabled = [
        engine for engine in probe.get("disabled_engines", [])
        if engine not in allowed_disabled
    ]
    if unexpected_disabled:
        failures.append("unexpected_disabled_engines:" + ",".join(unexpected_disabled))
    return failures


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _safe_name(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value]
    return "".join(keep).strip("_") or "case"


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# V5 Regression Report",
        "",
        f"- status: {report.get('status')}",
        f"- generated_at: {report.get('generated_at')}",
        f"- output_dir: {report.get('output_dir')}",
        "",
        "| case | status | OK | review | blocking | grades | probe | failures |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for case in report.get("cases", []):
        metrics = case.get("metrics", {})
        statuses = metrics.get("status_counts", {})
        grades = ", ".join(f"{k}:{v}" for k, v in sorted(metrics.get("evidence_grade_counts", {}).items()))
        probe = metrics.get("capability_probe", {})
        probe_text = ""
        if probe:
            ocr = ", ".join(f"{k}:{v}" for k, v in sorted((probe.get("ocr_hit_rate") or {}).items()))
            disabled = ",".join(probe.get("disabled_engines", [])) or "none"
            probe_text = f"barcode:{probe.get('barcode_hit_rate', 0)}; ocr:{ocr}; disabled:{disabled}"
        failures = "; ".join(case.get("failures", []))
        lines.append(
            "| {id} | {status} | {ok} | {review} | {blocking} | {grades} | {probe} | {failures} |".format(
                id=case.get("id", ""),
                status=case.get("status", ""),
                ok=statuses.get("OK", 0),
                review=statuses.get("NEEDS_MANUAL_REVIEW", 0),
                blocking=statuses.get("BLOCKING_ISSUE", 0),
                grades=grades,
                probe=probe_text,
                failures=failures,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Run minimal V5 regression checks")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--case", action="append", default=[], help="case id to run; repeatable")
    args = parser.parse_args()

    report = run_regressions(
        config_path=args.config,
        output_dir=args.output_dir or None,
        case_ids=set(args.case) if args.case else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
