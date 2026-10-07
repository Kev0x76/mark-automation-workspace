from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .circuit_breaker import CircuitBreaker
from .models import TextSpan
from .normalize import normalize_text

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


@dataclass
class OCRResult:
    spans: list[TextSpan] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


_PADDLE_OCR: Any = None
_PADDLE_IMPORT_ERROR = ""


def should_run_ocr(has_text_layer: bool, spans: list[TextSpan], force: bool = False) -> bool:
    if force:
        return True
    if not has_text_layer:
        return True
    alpha_digit = sum(1 for span in spans for ch in span.text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return alpha_digit < 20


def run_ocr(
    image: Any,
    page_index: int,
    page_width: float,
    page_height: float,
    breaker: CircuitBreaker | None = None,
) -> OCRResult:
    breaker = breaker or CircuitBreaker(max_seconds=60, max_errors=3)
    result = OCRResult()
    if image is None:
        result.events.append("ocr:NO_IMAGE")
        return result

    if _paddle_available():
        spans = breaker.run(
            "ocr.paddle",
            lambda: _run_paddle(image, page_index, page_width, page_height),
            [],
        )
        if spans:
            result.spans = spans
            result.events = list(breaker.events)
            return result
    else:
        breaker.events.append(f"ocr.paddle:UNAVAILABLE:{_PADDLE_IMPORT_ERROR or 'not installed'}")

    if pytesseract is not None:
        spans = breaker.run(
            "ocr.tesseract",
            lambda: _run_tesseract(image, page_index, page_width, page_height),
            [],
        )
        result.spans = spans
    else:
        breaker.events.append("ocr.tesseract:UNAVAILABLE")
    result.events = list(breaker.events)
    return result


def _get_paddle() -> Any:
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        os.environ.setdefault("FLAGS_use_onednn", "0")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import PaddleOCR

        try:
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, enable_mkldnn=False)
        except Exception:
            try:
                _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ch", enable_mkldnn=False)
            except Exception:
                _PADDLE_OCR = PaddleOCR(lang="ch")
    return _PADDLE_OCR


def _paddle_available() -> bool:
    global _PADDLE_IMPORT_ERROR
    if _PADDLE_IMPORT_ERROR:
        return False
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    try:
        import paddleocr  # noqa: F401
    except Exception as exc:
        _PADDLE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    return True


def _run_paddle(image: Any, page_index: int, page_width: float, page_height: float) -> list[TextSpan]:
    ocr = _get_paddle()
    data = ocr.ocr(_image_to_numpy(image), cls=True)
    spans: list[TextSpan] = []
    sx = page_width / image.width
    sy = page_height / image.height
    for block in data or []:
        for line in block or []:
            if not line or len(line) < 2:
                continue
            points = line[0]
            text = normalize_text(line[1][0] if line[1] else "")
            confidence = float(line[1][1] if line[1] and len(line[1]) > 1 else 0.0)
            if not text or confidence < 0.35:
                continue
            xs = [float(p[0]) * sx for p in points]
            ys = [float(p[1]) * sy for p in points]
            spans.append(
                TextSpan(
                    text=text,
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    page=page_index,
                    size=max(1.0, (max(ys) - min(ys))),
                    source=f"ocr_paddle:{confidence:.2f}",
                )
            )
    return spans


def _image_to_numpy(image: Any) -> Any:
    try:
        import numpy as np

        return np.array(image.convert("RGB"))
    except Exception:
        return image


def _run_tesseract(image: Any, page_index: int, page_width: float, page_height: float) -> list[TextSpan]:
    if pytesseract is None:
        return []
    data = pytesseract.image_to_data(image, lang="chi_sim+eng", output_type=pytesseract.Output.DICT)
    sx = page_width / image.width
    sy = page_height / image.height
    spans: list[TextSpan] = []
    for idx, text_raw in enumerate(data.get("text", [])):
        text = normalize_text(text_raw)
        if not text:
            continue
        try:
            confidence = float(data.get("conf", ["-1"])[idx])
        except Exception:
            confidence = -1.0
        if confidence >= 0 and confidence < 35:
            continue
        x = float(data["left"][idx]) * sx
        y = float(data["top"][idx]) * sy
        w = float(data["width"][idx]) * sx
        h = float(data["height"][idx]) * sy
        spans.append(
            TextSpan(
                text=text,
                bbox=(x, y, x + w, y + h),
                page=page_index,
                size=max(1.0, h),
                source=f"ocr_tesseract:{confidence:.0f}",
            )
        )
    return spans
