from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .barcode_engines import decode_barcodes
from .evidence_extractor import extract_cell_evidence
from .models import Cell, CellEvidence, TextSpan

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional image dependency fallback
    Image = None  # type: ignore


@dataclass
class VisualCandidateBreaker:
    """Circuit breaker for turning visual crops into evidence cells.

    The visual candidate stage can fan out quickly: one image-only PDF page may
    produce dozens or hundreds of crops, and OCR/barcode engines may be slow or
    noisy. This breaker keeps the future evidence-extraction loop bounded.
    """

    max_candidates: int = 300
    max_total_seconds: float = 120.0
    max_errors: int = 3
    max_empty_streak: int = 8
    max_ocr_candidates: int = 8
    ocr_timeout_seconds: float = 8.0
    min_quality: float = 0.05
    max_area_ratio: float = 0.65
    events: list[str] = field(default_factory=list)
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    ocr_attempts: int = 0
    empty_streak: int = 0
    disabled: bool = False
    _started_at: float = field(default_factory=time.monotonic)

    def allow(self, item: dict[str, Any]) -> bool:
        if self.disabled:
            self.skipped += 1
            return False
        if self.processed >= self.max_candidates:
            self.disable(f"max_candidates:{self.max_candidates}")
            self.skipped += 1
            return False
        elapsed = time.monotonic() - self._started_at
        if elapsed > self.max_total_seconds:
            self.disable(f"max_total_seconds:{elapsed:.1f}")
            self.skipped += 1
            return False
        if item.get("status") != "VISUAL_CELL_CANDIDATE":
            self.skipped += 1
            self.events.append("VISUAL_EVIDENCE_SKIP:status_not_candidate")
            return False

        quality = _to_float(item.get("quality"))
        area_ratio = _to_float(item.get("area_ratio"))
        if quality < self.min_quality:
            self.skipped += 1
            self.events.append(
                f"VISUAL_EVIDENCE_SKIP:low_quality:{quality:.3f}:cell_{item.get('cell_index', '')}"
            )
            return False
        if area_ratio > self.max_area_ratio:
            self.skipped += 1
            self.events.append(
                f"VISUAL_EVIDENCE_SKIP:oversized:{area_ratio:.3f}:cell_{item.get('cell_index', '')}"
            )
            return False
        return True

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_total_seconds - (time.monotonic() - self._started_at))

    def allow_ocr(self, item: dict[str, Any]) -> bool:
        if self.disabled:
            return False
        if self.ocr_attempts >= self.max_ocr_candidates:
            self.events.append(
                f"VISUAL_OCR_SKIP:max_ocr_candidates:{self.max_ocr_candidates}:cell_{item.get('cell_index', '')}"
            )
            return False
        remaining = self.remaining_seconds()
        if remaining <= 1.0:
            self.disable(f"max_total_seconds:{self.max_total_seconds:.1f}")
            return False
        self.ocr_attempts += 1
        return True

    def record_processed(self, has_evidence: bool) -> None:
        self.processed += 1
        if has_evidence:
            self.empty_streak = 0
            return
        self.empty_streak += 1
        if self.empty_streak >= self.max_empty_streak:
            self.disable(f"empty_evidence_streak:{self.empty_streak}")

    def record_error(self, item: dict[str, Any], exc: Exception) -> None:
        self.errors += 1
        self.events.append(
            "VISUAL_EVIDENCE_ERROR:"
            f"cell_{item.get('cell_index', '')}:{type(exc).__name__}:{exc}"
        )
        if self.errors >= self.max_errors:
            self.disable(f"errors:{self.errors}")

    def disable(self, reason: str) -> None:
        if not self.disabled:
            self.disabled = True
            self.events.append(f"VISUAL_EVIDENCE_DISABLED:{reason}")


def write_visual_candidate_evidence_manifest(
    output_dir: str | Path,
    visual_cells_manifest: dict[str, Any],
    breaker: VisualCandidateBreaker | None = None,
) -> dict[str, Any]:
    """Write a bounded preflight manifest for visual-candidate evidence extraction.

    This does not create verification evidence yet. It records which visual
    candidates are safe to send into the future barcode/OCR/AI-Vision stage, and
    which candidates were skipped by circuit-breaker rules.
    """

    output_dir = Path(output_dir)
    breaker = breaker or VisualCandidateBreaker()
    eligible: list[dict[str, Any]] = []
    items = visual_cells_manifest.get("items", [])

    for item in items:
        if not breaker.allow(item):
            continue
        image_path = Path(item.get("image", ""))
        eligible.append({
            "source": item.get("source", ""),
            "page": item.get("page", ""),
            "cell_index": item.get("cell_index", ""),
            "status": "VISUAL_CELL_EVIDENCE_PENDING",
            "image": str(image_path),
            "bbox_px": item.get("bbox_px", []),
            "quality": item.get("quality", ""),
            "area_ratio": item.get("area_ratio", ""),
            "message": "视觉候选已通过证据化熔断预检，等待条码/OCR/AI Vision 抽取。",
        })
        # Preflight only: no evidence extraction has run yet.
        breaker.record_processed(has_evidence=True)

    manifest = {
        "stage": "visual_candidate_evidence_preflight",
        "candidate_count": len(items),
        "eligible_count": len(eligible),
        "processed_count": breaker.processed,
        "skipped_count": breaker.skipped,
        "disabled": breaker.disabled,
        "events": breaker.events,
        "items": eligible,
    }
    (output_dir / "visual_candidate_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def extract_visual_candidate_evidences(
    output_dir: str | Path,
    visual_cells_manifest: dict[str, Any],
    breaker: VisualCandidateBreaker | None = None,
) -> tuple[list[CellEvidence], dict[str, Any]]:
    """Extract hard evidence from visual candidate crops.

    This first implementation only promotes candidates with real decoded
    barcodes. OCR/AI Vision can be added behind the same breaker later.
    Empty candidates remain visual-only and cannot affect verification.
    """

    output_dir = Path(output_dir)
    breaker = breaker or VisualCandidateBreaker()
    evidences: list[CellEvidence] = []
    items_out: list[dict[str, Any]] = []

    if Image is None:
        breaker.disable("PIL_UNAVAILABLE")

    for item in visual_cells_manifest.get("items", []):
        if not breaker.allow(item):
            continue
        try:
            image_path = _resolve_candidate_path(output_dir, item.get("image", ""))
            if Image is None or image_path is None or not image_path.exists():
                breaker.events.append(
                    f"VISUAL_EVIDENCE_SKIP:image_missing:cell_{item.get('cell_index', '')}"
                )
                breaker.record_processed(has_evidence=False)
                continue

            image = Image.open(image_path).convert("RGB")
            barcode_result = decode_barcodes(
                image=image,
                page_index=int(item.get("page") or 1) - 1,
                page_width=float(image.width),
                page_height=float(image.height),
            )
            ocr_spans: list[TextSpan] = []
            ocr_events: list[str] = []
            if breaker.allow_ocr(item):
                ocr_timeout = min(breaker.ocr_timeout_seconds, max(1.0, breaker.remaining_seconds() - 0.5))
                ocr_spans, ocr_events = _run_ocr_service(image_path, item, ocr_timeout)
            else:
                ocr_events.append(f"VISUAL_CELL_{item.get('cell_index', '')}:ocr.service:SKIPPED_BY_BUDGET")
            breaker.events.extend(ocr_events)
            has_hard_evidence = bool(barcode_result.evidences)
            has_evidence = bool(barcode_result.evidences or ocr_spans)
            breaker.events.extend(_prefix_events(barcode_result.events, item))
            breaker.record_processed(has_evidence=has_evidence)
            if not has_hard_evidence:
                status = "VISUAL_CANDIDATE_OCR_ONLY_UNPROMOTED" if ocr_spans else "NO_BARCODE_EVIDENCE"
                items_out.append(_extraction_item(item, image_path, "NO_BARCODE_EVIDENCE", []))
                items_out[-1]["status"] = status
                if ocr_spans:
                    items_out[-1]["ocr_span_count"] = len(ocr_spans)
                    items_out[-1]["message"] = (
                        "视觉候选只抽取到 OCR 文本，缺少条码强证据；不进入自动关联核对。"
                    )
                continue

            cell = Cell(
                cell_id=_candidate_cell_id(item),
                file_path=str(image_path),
                page=int(item.get("page") or 1) - 1,
                bbox=(0.0, 0.0, float(image.width), float(image.height)),
                spans=ocr_spans,
                barcodes=barcode_result.evidences,
                segmentation_method="visual_cell_candidate_evidence",
                needs_ai=True,
            )
            evidence = extract_cell_evidence(cell)
            evidence.needs_ai = True
            evidence.evidence_sources.setdefault("visual_candidate", "barcode_scan")
            if ocr_spans:
                _mark_visual_ocr_sources(evidence)
            evidences.append(evidence)
            items_out.append(
                _extraction_item(
                    item,
                    image_path,
                    "VISUAL_CANDIDATE_EVIDENCE",
                    [bc.value for bc in barcode_result.evidences],
                )
            )
        except Exception as exc:
            breaker.record_error(item, exc)

    manifest = {
        "stage": "visual_candidate_barcode_evidence",
        "candidate_count": len(visual_cells_manifest.get("items", [])),
        "evidence_count": len(evidences),
        "processed_count": breaker.processed,
        "skipped_count": breaker.skipped,
        "ocr_attempts": breaker.ocr_attempts,
        "disabled": breaker.disabled,
        "events": breaker.events,
        "items": items_out,
    }
    (output_dir / "visual_candidate_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return evidences, manifest


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_candidate_path(output_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [
        path,
        output_dir / path,
        output_dir.parent / path,
    ]
    if len(output_dir.parents) >= 2:
        candidates.append(output_dir.parents[1] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _candidate_cell_id(item: dict[str, Any]) -> str:
    source = Path(str(item.get("source", "visual"))).stem or "visual"
    page = item.get("page", "")
    cell_index = item.get("cell_index", "")
    return f"{source}:p{page}:visual_cell_{cell_index}"


def _prefix_events(events: list[str], item: dict[str, Any]) -> list[str]:
    cell = item.get("cell_index", "")
    return [f"VISUAL_CELL_{cell}:{event}" for event in events]


def _extraction_item(
    item: dict[str, Any],
    image_path: Path,
    status: str,
    barcodes: list[str],
) -> dict[str, Any]:
    return {
        "source": item.get("source", ""),
        "page": item.get("page", ""),
        "cell_index": item.get("cell_index", ""),
        "status": status,
        "image": str(image_path),
        "bbox_px": item.get("bbox_px", []),
        "barcodes": barcodes,
        "message": (
            "视觉候选已抽取到真实条码/OCR证据，已进入自动关联核对。"
            if barcodes
            else "视觉候选未抽取到条码/OCR证据，未进入自动核对。"
        ),
    }


def _run_ocr_service(
    image_path: Path,
    item: dict[str, Any],
    timeout_seconds: float,
) -> tuple[list[TextSpan], list[str]]:
    root = _project_root()
    cli = root / "ocr" / "ocr_cli.py"
    if not cli.exists():
        return [], ["OCR_SERVICE_MISSING:ocr_service/ocr_cli.py"]
    try:
        completed = subprocess.run(
            [sys.executable, str(cli), "--image", str(image_path)],
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], [f"VISUAL_CELL_{item.get('cell_index', '')}:ocr.service:TIMEOUT"]
    except Exception as exc:
        return [], [f"VISUAL_CELL_{item.get('cell_index', '')}:ocr.service:ERROR:{type(exc).__name__}:{exc}"]

    events: list[str] = []
    if completed.stderr.strip():
        events.append(
            f"VISUAL_CELL_{item.get('cell_index', '')}:ocr.service:STDERR:{completed.stderr.strip()[:240]}"
        )
    if completed.returncode != 0:
        events.append(
            f"VISUAL_CELL_{item.get('cell_index', '')}:ocr.service:EXIT:{completed.returncode}"
        )
        return [], events
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return [], [f"VISUAL_CELL_{item.get('cell_index', '')}:ocr.service:BAD_JSON:{exc}"]

    page_index = int(item.get("page") or 1) - 1
    spans = [_span_from_payload(span, page_index) for span in payload.get("spans", [])]
    spans = [span for span in spans if span is not None]
    events.extend(f"VISUAL_CELL_{item.get('cell_index', '')}:{event}" for event in payload.get("events", []))
    return spans, events


def _span_from_payload(payload: dict[str, Any], page_index: int) -> TextSpan | None:
    text = str(payload.get("text", "")).strip()
    bbox_raw = payload.get("bbox", [])
    if not text or not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        return None
    try:
        bbox = tuple(float(x) for x in bbox_raw)
    except (TypeError, ValueError):
        return None
    confidence = payload.get("confidence", "")
    source = payload.get("source", "ocr_service")
    return TextSpan(
        text=text,
        bbox=bbox,  # type: ignore[arg-type]
        page=page_index,
        size=max(1.0, bbox[3] - bbox[1]),
        source=f"{source}:{confidence}",
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mark_visual_ocr_sources(evidence: CellEvidence) -> None:
    for key in [
        "item_number",
        "barcode_text",
        "packing_qty",
        "name_cn",
        "name_it",
        "name_es",
        "name_en",
        "name_pt",
        "name_latin",
        "name_unknown",
    ]:
        if key in evidence.evidence_sources:
            evidence.evidence_sources[key] = "visual_candidate_ocr"
