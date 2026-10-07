from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .barcode_engines import decode_barcodes
from .circuit_breaker import CircuitBreaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EngineProbeBreaker:
    max_total_seconds: float = 90.0
    max_engine_errors: int = 2
    max_engine_timeouts: int = 2
    max_consecutive_empty: int = 2
    started: float = field(default_factory=time.monotonic)
    errors: Counter[str] = field(default_factory=Counter)
    timeouts: Counter[str] = field(default_factory=Counter)
    empty: Counter[str] = field(default_factory=Counter)
    disabled: set[str] = field(default_factory=set)
    events: list[str] = field(default_factory=list)

    def allow(self, engine: str) -> bool:
        if engine in self.disabled:
            self.events.append(f"{engine}:SKIPPED_BY_BREAKER")
            return False
        if self.remaining_seconds() <= 0:
            self.disable(engine, "total_time_budget_exhausted")
            return False
        return True

    def remaining_seconds(self) -> float:
        return self.max_total_seconds - (time.monotonic() - self.started)

    def record_success(self, engine: str, count: int, elapsed: float) -> None:
        if count > 0:
            self.empty[engine] = 0
            self.events.append(f"{engine}:OK:{count}:{elapsed:.2f}s")
            return
        self.empty[engine] += 1
        self.events.append(f"{engine}:EMPTY:{self.empty[engine]}:{elapsed:.2f}s")
        if self.empty[engine] >= self.max_consecutive_empty:
            self.disable(engine, f"{self.empty[engine]} consecutive empty results")

    def record_error(self, engine: str, message: str) -> None:
        self.errors[engine] += 1
        self.events.append(f"{engine}:ERROR:{message}")
        if self.errors[engine] >= self.max_engine_errors:
            self.disable(engine, f"{self.errors[engine]} errors")

    def record_timeout(self, engine: str, seconds: float) -> None:
        self.timeouts[engine] += 1
        self.events.append(f"{engine}:TIMEOUT:{seconds:.1f}s")
        if self.timeouts[engine] >= self.max_engine_timeouts:
            self.disable(engine, f"{self.timeouts[engine]} timeouts")

    def disable(self, engine: str, reason: str) -> None:
        if engine not in self.disabled:
            self.disabled.add(engine)
            self.events.append(f"{engine}:DISABLED:{reason}")


def run_probe(
    image_paths: list[str | Path],
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    ocr_engines: list[str] | None = None,
    max_images: int = 4,
    max_total_seconds: float = 90.0,
    ocr_timeout_seconds: float = 18.0,
) -> dict[str, Any]:
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_engines = ocr_engines or ["rapidocr", "tesseract"]
    crop_by_image = _crop_by_image(manifest_path)

    images = [_resolve_path(path) for path in image_paths]
    images = [path for path in images if path.exists() and path.is_file()][:max_images]
    breaker = EngineProbeBreaker(max_total_seconds=max_total_seconds)
    barcode_breaker = CircuitBreaker(max_seconds=8.0, max_errors=2)
    items: list[dict[str, Any]] = []

    for image_path in images:
        if breaker.remaining_seconds() <= 0:
            breaker.events.append("PROBE_STOPPED:total_time_budget_exhausted")
            break
        item: dict[str, Any] = {
            "image": str(image_path),
            "crop": crop_by_image.get(str(image_path.resolve()), {}),
            "barcode": {},
            "ocr": {},
        }
        item["barcode"] = _probe_barcode(image_path, breaker, barcode_breaker)
        for engine in ocr_engines:
            item["ocr"][engine] = _probe_ocr(
                image_path=image_path,
                engine=engine,
                breaker=breaker,
                timeout_seconds=min(ocr_timeout_seconds, max(1.0, breaker.remaining_seconds())),
            )
        items.append(item)

    summary = _summarize(items, breaker, barcode_breaker)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": _probe_status(summary),
        "image_count": len(images),
        "output_dir": str(output_dir),
        "manifest": str(_resolve_path(manifest_path)) if manifest_path else "",
        "summary": summary,
        "items": items,
        "breaker_events": breaker.events + barcode_breaker.events,
    }
    (output_dir / "capability_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "capability_probe_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def image_paths_from_manifest(manifest_path: str | Path, max_images: int = 4) -> list[Path]:
    manifest_path = _resolve_path(manifest_path)
    data = _read_json(manifest_path)
    paths: list[Path] = []
    for item in data.get("items", []):
        if item.get("status") != "OK":
            continue
        image = item.get("image")
        if image:
            paths.append(_resolve_path(image))
        if len(paths) >= max_images:
            break
    return paths


def _probe_barcode(
    image_path: Path,
    breaker: EngineProbeBreaker,
    barcode_breaker: CircuitBreaker,
) -> dict[str, Any]:
    engine = "barcode"
    if not breaker.allow(engine):
        return {"status": "SKIPPED", "values": [], "elapsed_seconds": 0.0}
    started = time.monotonic()
    try:
        image = Image.open(image_path).convert("RGB")
        result = decode_barcodes(image, 0, float(image.width), float(image.height), barcode_breaker)
        values = sorted({item.value for item in result.evidences if item.value})
    except Exception as exc:
        breaker.record_error(engine, f"{type(exc).__name__}:{exc}")
        return {"status": "ERROR", "values": [], "elapsed_seconds": round(time.monotonic() - started, 3)}
    elapsed = time.monotonic() - started
    breaker.record_success(engine, len(values), elapsed)
    return {
        "status": "OK" if values else "EMPTY",
        "values": values,
        "elapsed_seconds": round(elapsed, 3),
    }


def _probe_ocr(
    image_path: Path,
    engine: str,
    breaker: EngineProbeBreaker,
    timeout_seconds: float,
) -> dict[str, Any]:
    key = f"ocr.{engine}"
    if not breaker.allow(key):
        return {"status": "SKIPPED", "span_count": 0, "elapsed_seconds": 0.0, "events": []}
    cli = PROJECT_ROOT / "ocr_service" / "ocr_cli.py"
    started = time.monotonic()
    cmd = [
        sys.executable,
        str(cli),
        "--image",
        str(image_path),
        "--engine",
        engine,
        "--min-confidence",
        "0.35",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        breaker.record_timeout(key, timeout_seconds)
        return {
            "status": "TIMEOUT",
            "span_count": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "events": [],
        }
    except Exception as exc:
        breaker.record_error(key, f"{type(exc).__name__}:{exc}")
        return {
            "status": "ERROR",
            "span_count": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "events": [],
        }

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()[:240]
        breaker.record_error(key, f"exit_{completed.returncode}:{message}")
        return {"status": "ERROR", "span_count": 0, "elapsed_seconds": round(elapsed, 3), "events": [message]}

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        breaker.record_error(key, f"bad_json:{exc}")
        return {"status": "ERROR", "span_count": 0, "elapsed_seconds": round(elapsed, 3), "events": []}
    spans = payload.get("spans", [])
    events = payload.get("events", [])
    breaker.record_success(key, len(spans), elapsed)
    return {
        "status": "OK" if spans else "EMPTY",
        "span_count": len(spans),
        "sample_text": " ".join(str(span.get("text", "")) for span in spans[:8]).strip(),
        "elapsed_seconds": round(elapsed, 3),
        "events": events,
    }


def _summarize(
    items: list[dict[str, Any]],
    breaker: EngineProbeBreaker,
    barcode_breaker: CircuitBreaker,
) -> dict[str, Any]:
    ocr_counts: Counter[str] = Counter()
    ocr_ok: Counter[str] = Counter()
    ocr_elapsed: Counter[str] = Counter()
    barcode_ok = 0
    barcode_elapsed = 0.0
    crop_methods: Counter[str] = Counter()
    crop_risks: Counter[str] = Counter()

    for item in items:
        barcode = item.get("barcode", {})
        if barcode.get("values"):
            barcode_ok += 1
        barcode_elapsed += float(barcode.get("elapsed_seconds") or 0.0)
        crop = item.get("crop", {})
        if crop.get("crop_method"):
            crop_methods[str(crop.get("crop_method"))] += 1
        quality = crop.get("crop_quality", {}) or {}
        if quality.get("association_unverified"):
            crop_risks["association_unverified"] += 1
        if quality.get("possible_other_items") or quality.get("foreign_item_anchors"):
            crop_risks["possible_other_items"] += 1
        if quality.get("touches_page_edge"):
            crop_risks["touches_page_edge"] += 1
        for engine, result in item.get("ocr", {}).items():
            ocr_counts[engine] += 1
            ocr_elapsed[engine] += float(result.get("elapsed_seconds") or 0.0)
            if result.get("span_count", 0) > 0:
                ocr_ok[engine] += 1

    image_count = len(items)
    return {
        "images_tested": image_count,
        "barcode_images_with_values": barcode_ok,
        "barcode_hit_rate": round(barcode_ok / max(image_count, 1), 3),
        "barcode_avg_seconds": round(barcode_elapsed / max(image_count, 1), 3),
        "ocr_images_with_text": dict(ocr_ok),
        "ocr_hit_rate": {
            engine: round(ocr_ok[engine] / max(ocr_counts[engine], 1), 3)
            for engine in sorted(ocr_counts)
        },
        "ocr_avg_seconds": {
            engine: round(ocr_elapsed[engine] / max(ocr_counts[engine], 1), 3)
            for engine in sorted(ocr_counts)
        },
        "crop_methods": dict(crop_methods),
        "crop_risks": dict(crop_risks),
        "disabled_engines": sorted(breaker.disabled),
        "engine_errors": dict(breaker.errors),
        "engine_timeouts": dict(breaker.timeouts),
        "barcode_breaker_disabled": sorted(barcode_breaker.disabled),
    }


def _probe_status(summary: dict[str, Any]) -> str:
    if summary.get("images_tested", 0) <= 0:
        return "NO_IMAGES"
    if summary.get("barcode_hit_rate", 0.0) <= 0 and not summary.get("ocr_images_with_text"):
        return "NO_SIGNAL"
    if summary.get("disabled_engines"):
        return "WEAK_WITH_BREAKER"
    return "PASS"


def _crop_by_image(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not manifest_path:
        return {}
    data = _read_json(_resolve_path(manifest_path))
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("items", []):
        image = item.get("image")
        if not image:
            continue
        try:
            out[str(_resolve_path(image).resolve())] = item
        except OSError:
            continue
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path | None) -> Path:
    if value is None:
        return PROJECT_ROOT
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# V5 Capability Probe",
        "",
        f"- status: {report.get('status')}",
        f"- generated_at: {report.get('generated_at')}",
        f"- images_tested: {summary.get('images_tested', 0)}",
        f"- barcode_hit_rate: {summary.get('barcode_hit_rate', 0)}",
        f"- disabled_engines: {', '.join(summary.get('disabled_engines', [])) or 'none'}",
        "",
        "## OCR",
        "",
    ]
    for engine, rate in sorted((summary.get("ocr_hit_rate") or {}).items()):
        avg = (summary.get("ocr_avg_seconds") or {}).get(engine, 0)
        lines.append(f"- {engine}: hit_rate={rate}, avg_seconds={avg}")
    lines.extend([
        "",
        "## Crop",
        "",
        f"- methods: {summary.get('crop_methods', {})}",
        f"- risks: {summary.get('crop_risks', {})}",
        "",
        "## Breaker Events",
        "",
    ])
    for event in report.get("breaker_events", []):
        lines.append(f"- {event}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Probe V5 OCR/barcode/crop capabilities with circuit breakers")
    parser.add_argument("--manifest", default="", help="mark_images_manifest.json to sample images from")
    parser.add_argument("--image", action="append", default=[], help="image path; repeatable")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-total-seconds", type=float, default=90.0)
    parser.add_argument("--ocr-timeout-seconds", type=float, default=18.0)
    parser.add_argument("--ocr-engine", action="append", default=[], help="OCR engine; repeatable")
    args = parser.parse_args()

    images = [_resolve_path(path) for path in args.image]
    if args.manifest:
        images.extend(image_paths_from_manifest(args.manifest, args.max_images))
    output_dir = args.output_dir or PROJECT_ROOT / "runs" / f"capability_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = run_probe(
        image_paths=images,
        output_dir=output_dir,
        manifest_path=args.manifest or None,
        ocr_engines=args.ocr_engine or None,
        max_images=args.max_images,
        max_total_seconds=args.max_total_seconds,
        ocr_timeout_seconds=args.ocr_timeout_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"PASS", "WEAK_WITH_BREAKER"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
