from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .core.ai_extractor import enrich_cells_with_ai
from .core.cell_segmenter import segment_page
from .core.contract_loader import load_contract, load_contract_snapshot
from .core.cdr_reader import read_cdr
from .core.document_reader import read_pdf
from .core.evidence_grade import apply_evidence_grades
from .core.evidence_extractor import extract_cell_evidence
from .core.image_reader import IMAGE_EXTENSIONS, read_image
from .core.ai_reviewer import review_result
from .core.mark_cropper import crop_mark_images
from .core.line_art_visual_checker import build_line_art_fields
from .core.mark_image_evidence import extract_mark_image_evidences
from .core.report import write_csv, write_item_summary_csv, write_json
from .core.resolver import resolve_cells
from .core.verifier import verify_item
from .core.normalize import normalize_item_number
from .core.visual_candidate_evidence import (
    VisualCandidateBreaker,
    extract_visual_candidate_evidences,
)
from .core.visual_report import write_visual_report
from .core.visual_segmenter import generate_visual_candidates
from .core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


def run_pipeline(
    contract_path: str | Path,
    mark_paths: list[str | Path],
    output_dir: str | Path,
    lineart_evidence_dir: str | Path | None = None,
    contract_snapshot_path: str | Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_items = (
        load_contract_snapshot(contract_snapshot_path, contract_path=contract_path)
        if contract_snapshot_path is not None
        else load_contract(contract_path)
    )
    line_art_fields = build_line_art_fields(
        contract_items,
        Path(lineart_evidence_dir) if lineart_evidence_dir else None,
        output_dir,
    )
    all_evidences = []
    page_summaries = []
    all_pages = []  # 累积所有页面，供视觉候选切块（增量步骤）使用

    for mark_path in mark_paths:
        pages = read_mark(mark_path, output_dir)
        all_pages.extend(pages)
        for page in pages:
            cells = segment_page(page)
            evidences = merge_same_item_evidence([extract_cell_evidence(cell) for cell in cells])
            evidences = enrich_cells_with_ai(evidences)
            all_evidences.extend(evidences)
            page_summaries.append({
                "file": str(mark_path),
                "page": page.page + 1,
                "has_text_layer": page.has_text_layer,
                "cells": len(cells),
                "barcodes": [b.value for b in page.barcodes],
                "segmentation": cells[0].segmentation_method if cells else "",
                "events": page.errors,
            })

    # 视觉候选切块（增量、可熔断）：只对图片型页面运行，只新增产物，
    # 任何异常都不会影响主流程与既有产物。
    visual_cells_manifest = {"candidate_count": 0, "pages_processed": 0}
    visual_candidate_evidence_manifest = {"evidence_count": 0, "events": []}
    try:
        vb = CircuitBreaker(max_seconds=60, max_errors=3)
        visual_cells_manifest = generate_visual_candidates(output_dir, all_pages, vb)
        _visual_evidences, visual_candidate_evidence_manifest = extract_visual_candidate_evidences(
            output_dir,
            visual_cells_manifest,
            VisualCandidateBreaker(max_total_seconds=45, max_ocr_candidates=8, ocr_timeout_seconds=8),
        )
        # Visual candidates are audit evidence only until their crop boundaries
        # are strong enough to avoid multi-item association pollution.
    except Exception as exc:  # 最后一道防线，保证主流程不被影响
        logger.warning("visual candidate stage failed: %s", exc)

    results = _verify_results(all_evidences, contract_items, line_art_fields)
    crop_manifest = crop_mark_images(output_dir, all_evidences, results)
    mark_image_evidences, mark_image_evidence_manifest = extract_mark_image_evidences(
        output_dir,
        crop_manifest,
        contract_items,
    )
    final_evidences = all_evidences
    if mark_image_evidences:
        final_evidences = all_evidences + mark_image_evidences
        results = _verify_results(final_evidences, contract_items, line_art_fields)

    evidence_grade_manifest = apply_evidence_grades(results, final_evidences, crop_manifest)
    write_json(output_dir / "v5_result.json", final_evidences, results)
    write_csv(output_dir / "v5_result.csv", results)
    write_item_summary_csv(output_dir / "v5_item_summary.csv", results)
    visual_manifest = write_visual_report(output_dir, results, crop_manifest)

    summary = {
        "contract": str(contract_path),
        "contract_facts_source": "snapshot" if contract_snapshot_path is not None else "workbook_parse",
        "contract_snapshot": str(contract_snapshot_path) if contract_snapshot_path is not None else None,
        "marks": [str(p) for p in mark_paths],
        "contract_items": len(contract_items),
        "evidence_cells": len(final_evidences),
        "pages": page_summaries,
        "csv": str(output_dir / "v5_result.csv"),
        "item_summary_csv": str(output_dir / "v5_item_summary.csv"),
        "json": str(output_dir / "v5_result.json"),
        "mark_images": str(output_dir / "mark_images"),
        "mark_images_manifest": str(output_dir / "mark_images_manifest.json"),
        "mark_images_count": len([x for x in crop_manifest.get("items", []) if x.get("status") == "OK"]),
        "mark_image_evidence_manifest": str(output_dir / "mark_image_evidence_manifest.json"),
        "mark_image_evidence_count": mark_image_evidence_manifest.get("item_count", 0),
        "mark_image_evidence_recognition_counts": mark_image_evidence_manifest.get("recognition_counts", {}),
        "mark_image_evidence_promotion_counts": mark_image_evidence_manifest.get("promotion_counts", {}),
        "evidence_grade_counts": evidence_grade_manifest.get("grade_counts", {}),
        "evidence_grade_definitions": evidence_grade_manifest.get("definitions", {}),
        "visual_report": str(output_dir / "visual_report.html"),
        "visual_report_manifest": str(output_dir / "visual_report_manifest.json"),
        "visual_report_cards": visual_manifest.get("cards", 0),
        "visual_cells_manifest": str(output_dir / "visual_cells_manifest.json"),
        "visual_cells_count": visual_cells_manifest.get("candidate_count", 0),
        "visual_cells_pages": visual_cells_manifest.get("pages_processed", 0),
        "visual_candidate_evidence_manifest": str(output_dir / "visual_candidate_evidence_manifest.json"),
        "visual_candidate_evidence_count": visual_candidate_evidence_manifest.get("evidence_count", 0),
        "visual_candidate_evidence_events": visual_candidate_evidence_manifest.get("events", []),
        "line_art_verification": str(output_dir / "line_art_verification.json"),
        # Kept as an explicit fact in the HTTP/report contract so callers do
        # not mistake an old environment variable for an active vision call.
        "line_art_visual_oc_enabled": False,
        "line_art_ai_enabled": False,  # legacy response key; permanently off
    }
    (output_dir / "v5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_mark(mark_path: str | Path, output_dir: str | Path | None = None):
    path = Path(mark_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".cdr":
        return read_cdr(path)
    if suffix in IMAGE_EXTENSIONS:
        return read_image(path, output_dir)
    raise RuntimeError(f"Unsupported mark file type: {path.suffix}")


def _verify_results(evidences, contract_items, line_art_fields=None):
    associations = resolve_cells(evidences, contract_items)
    evidence_by_cell = {e.cell.cell_id: e for e in evidences}
    results = []
    for item in contract_items:
        assoc = associations.get(item.item_number)
        evidence = evidence_by_cell.get(assoc.cell_id) if assoc else None
        result = verify_item(item, evidence, assoc, (line_art_fields or {}).get(item.item_number))
        results.append(review_result(item, evidence, result))
    return results


def merge_same_item_evidence(evidences):
    merged = []
    used = set()
    for idx, ev in enumerate(evidences):
        if idx in used:
            continue
        key = normalize_item_number(ev.item_numbers[0]) if len(ev.item_numbers) == 1 else ""
        if not key:
            merged.append(ev)
            continue
        group = [ev]
        used.add(idx)
        for j, other in enumerate(evidences[idx + 1:], start=idx + 1):
            if j in used:
                continue
            other_key = (
                normalize_item_number(other.item_numbers[0])
                if len(other.item_numbers) == 1
                else ""
            )
            if (
                other.cell.file_path == ev.cell.file_path
                and other.cell.page == ev.cell.page
                and other_key == key
                and _evidence_barcodes_compatible(ev, other)
            ):
                group.append(other)
                used.add(j)
        if len(group) == 1:
            merged.append(ev)
        else:
            merged.append(_merge_group(group))
    return merged


def _evidence_barcodes_compatible(left, right):
    left_values = set(left.barcode_scans or left.barcode_texts or left.barcodes)
    right_values = set(right.barcode_scans or right.barcode_texts or right.barcodes)
    return not left_values or not right_values or bool(left_values & right_values)


def _merge_group(group):
    base = group[0]
    base.cell.cell_id = base.cell.cell_id + "+merged"
    base.cell.segmentation_method = base.cell.segmentation_method + "_merged_same_item"
    base.cell.bbox = _union_bboxes([item.cell.bbox for item in group])
    for other in group[1:]:
        base.cell.spans.extend(other.cell.spans)
        base.cell.barcodes.extend(other.cell.barcodes)
        for value in other.item_numbers:
            if value not in base.item_numbers:
                base.item_numbers.append(value)
        for value in other.barcodes:
            if value not in base.barcodes:
                base.barcodes.append(value)
        for value in other.barcode_scans:
            if value not in base.barcode_scans:
                base.barcode_scans.append(value)
        for value in other.barcode_texts:
            if value not in base.barcode_texts:
                base.barcode_texts.append(value)
        if not base.packing_qty and other.packing_qty:
            base.packing_qty = other.packing_qty
        base.raw_lines.extend(x for x in other.raw_lines if x not in base.raw_lines)
        for lang, values in other.names.items():
            base.names.setdefault(lang, [])
            for value in values:
                if value not in base.names[lang]:
                    base.names[lang].append(value)
        base.evidence_sources.update({k: v for k, v in other.evidence_sources.items() if k not in base.evidence_sources})
    return base


def _union_bboxes(bboxes):
    valid = []
    for bbox in bboxes:
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 > x0 and y1 > y0:
            valid.append((x0, y0, x1, y1))
    if not valid:
        return (0, 0, 0, 0)
    return (
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    )


def discover_sample_categories(root: str | Path) -> list[tuple[str, Path, list[Path]]]:
    root = Path(root)
    categories = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        contracts = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".xlsx", ".xls"} and not p.name.startswith("~$")])
        marks = sorted([p for p in folder.iterdir() if p.suffix.lower() in ({".pdf", ".cdr"} | IMAGE_EXTENSIONS)])
        if contracts and marks:
            # Prefer xlsx over xls unless xls is the only option.
            contract = next((p for p in contracts if p.suffix.lower() == ".xlsx"), contracts[0])
            categories.append((folder.name, contract, marks))
    return categories


def run_samples(samples_root: str | Path, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    all_summaries = {}
    for category, contract, marks in discover_sample_categories(samples_root):
        cat_dir = output_dir / category
        summary = run_pipeline(contract, marks, cat_dir)
        all_summaries[category] = summary
    (output_dir / "all_samples_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="V5 contract-mark cell verification CLI")
    parser.add_argument("--contract", help="contract Excel file")
    parser.add_argument("--marks", nargs="+", help="mark PDF/CDR/image files")
    parser.add_argument("--samples-root", help="samples root dir, batch run all categories")
    parser.add_argument("--output-dir", default="output", help="output dir")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    if args.samples_root:
        summary = run_samples(args.samples_root, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.contract or not args.marks:
        parser.error("--contract/--marks or --samples-root required")
    summary = run_pipeline(args.contract, args.marks, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
