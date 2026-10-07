from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_RAPID_OCR: Any = None


def main() -> int:
    parser = argparse.ArgumentParser(description="V5 isolated OCR CLI")
    parser.add_argument("--image", required=True, help="image file to OCR")
    parser.add_argument("--engine", default="auto", choices=["auto", "rapidocr", "tesseract", "paddle"])
    parser.add_argument("--min-confidence", type=float, default=0.35)
    args = parser.parse_args()

    payload = run_ocr(Path(args.image), args.engine, args.min_confidence)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


def run_ocr(image_path: Path, engine: str = "auto", min_confidence: float = 0.35) -> dict[str, Any]:
    events: list[str] = []
    if not image_path.exists():
        return {"spans": [], "events": [f"OCR_IMAGE_MISSING:{image_path}"]}

    engines = [engine] if engine != "auto" else ["rapidocr", "tesseract"]
    if engine == "auto" and os.environ.get("V5_ENABLE_PADDLE_OCR") == "1":
        engines.append("paddle")
    for name in engines:
        try:
            if name == "rapidocr":
                spans = _run_rapidocr(image_path, min_confidence)
            elif name == "tesseract":
                spans = _run_tesseract(image_path, min_confidence)
            elif name == "paddle":
                spans = _run_paddle(image_path, min_confidence)
            else:
                events.append(f"OCR_ENGINE_UNKNOWN:{name}")
                continue
        except Exception as exc:
            events.append(f"ocr.{name}:ERROR:{type(exc).__name__}:{exc}")
            continue
        if spans:
            return {"spans": spans, "events": events + [f"ocr.{name}:OK:{len(spans)}"]}
        events.append(f"ocr.{name}:NO_TEXT")

    return {"spans": [], "events": events}


def _run_rapidocr(image_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    global _RAPID_OCR
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        try:
            from rapidocr import RapidOCR  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"rapidocr unavailable: {exc}") from exc

    if _RAPID_OCR is None:
        _RAPID_OCR = RapidOCR()
    result, _elapsed = _RAPID_OCR(str(image_path))
    spans: list[dict[str, Any]] = []
    for item in result or []:
        if len(item) < 3:
            continue
        points, text, confidence = item[0], str(item[1]), float(item[2] or 0.0)
        if confidence < min_confidence or not text.strip():
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        spans.append(_span(text, (min(xs), min(ys), max(xs), max(ys)), confidence, "ocr_service_rapidocr"))
    return spans


def _run_tesseract(image_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    try:
        data = pytesseract.image_to_data(image, lang="chi_sim+eng", output_type=pytesseract.Output.DICT)
    except Exception:
        data = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT)
    spans: list[dict[str, Any]] = []
    for idx, raw in enumerate(data.get("text", [])):
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            confidence = float(data.get("conf", ["-1"])[idx]) / 100.0
        except Exception:
            confidence = 0.0
        if confidence >= 0 and confidence < min_confidence:
            continue
        x = float(data["left"][idx])
        y = float(data["top"][idx])
        w = float(data["width"][idx])
        h = float(data["height"][idx])
        spans.append(_span(text, (x, y, x + w, y + h), confidence, "ocr_service_tesseract"))
    return spans


def _run_paddle(image_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import PaddleOCR

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, enable_mkldnn=False)
    except Exception:
        ocr = PaddleOCR(lang="ch")
    data = ocr.ocr(str(image_path), cls=True)
    spans: list[dict[str, Any]] = []
    for block in data or []:
        for line in block or []:
            if not line or len(line) < 2:
                continue
            points = line[0]
            text = str(line[1][0] if line[1] else "").strip()
            confidence = float(line[1][1] if line[1] and len(line[1]) > 1 else 0.0)
            if confidence < min_confidence or not text:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            spans.append(_span(text, (min(xs), min(ys), max(xs), max(ys)), confidence, "ocr_service_paddle"))
    return spans


def _span(text: str, bbox: tuple[float, float, float, float], confidence: float, source: str) -> dict[str, Any]:
    return {
        "text": text,
        "bbox": list(bbox),
        "confidence": round(float(confidence), 4),
        "source": source,
    }


if __name__ == "__main__":
    raise SystemExit(main())
