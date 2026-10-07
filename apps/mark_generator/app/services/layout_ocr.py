"""Optional, CPU-friendly text detection for sales-image layout analysis.

The OCR dependency is intentionally isolated behind this adapter.  Import or
runtime failures are returned as explicit states so callers can fail closed
instead of silently treating an unavailable detector as "no text".
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ENGINE_NAME = "rapidocr-onnxruntime"
_ENGINE: Any | None = None
_ENGINE_INFO: dict[str, Any] | None = None
_ENGINE_ERROR: dict[str, Any] | None = None
_ENGINE_LOCK = threading.Lock()


def detect_text_regions(image_rgb: np.ndarray) -> dict[str, Any]:
    """Return source-coordinate text polygons and auditable detector metadata."""
    initialized = _get_engine()
    if initialized[0] is None:
        return dict(initialized[1])

    engine, engine_info = initialized
    started = time.perf_counter()
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        output, engine_elapsed = engine(
            image_bgr,
            use_det=True,
            use_cls=False,
            use_rec=False,
        )
        regions = _normalize_regions(output)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "status": "success",
            **engine_info,
            "elapsed_ms": elapsed_ms,
            "engine_elapsed_ms": [round(float(value) * 1000.0, 2) for value in (engine_elapsed or [])],
            "region_count": len(regions),
            "regions": regions,
        }
    except Exception as exc:  # optional dependency must never corrupt the pipeline
        return {
            "status": "failed",
            **engine_info,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "region_count": 0,
            "regions": [],
            "error_type": type(exc).__name__,
        }


def reset_ocr_engine_for_tests() -> None:
    """Clear lazy state.  Intended for deterministic dependency-state tests."""
    global _ENGINE, _ENGINE_INFO, _ENGINE_ERROR
    with _ENGINE_LOCK:
        _ENGINE = None
        _ENGINE_INFO = None
        _ENGINE_ERROR = None


def _get_engine() -> tuple[Any | None, dict[str, Any]]:
    global _ENGINE, _ENGINE_INFO, _ENGINE_ERROR
    if _ENGINE is not None:
        return _ENGINE, dict(_ENGINE_INFO or {})
    if _ENGINE_ERROR is not None:
        return None, dict(_ENGINE_ERROR)

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE, dict(_ENGINE_INFO or {})
        if _ENGINE_ERROR is not None:
            return None, dict(_ENGINE_ERROR)
        started = time.perf_counter()
        try:
            from rapidocr_onnxruntime import RapidOCR

            # Disabling classifier and recognizer here avoids loading models the
            # layout stage does not use.  Only text geometry is required.
            engine = RapidOCR(use_cls=False, use_rec=False)
            model_path = _detector_model_path(engine)
            version = importlib.metadata.version(ENGINE_NAME)
            _ENGINE = engine
            _ENGINE_INFO = {
                "engine": ENGINE_NAME,
                "version": version,
                "model_sha256": _sha256_file(model_path) if model_path else None,
                "initialization_ms": round((time.perf_counter() - started) * 1000.0, 2),
            }
            return _ENGINE, dict(_ENGINE_INFO)
        except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
            _ENGINE_ERROR = {
                "status": "unavailable",
                "engine": ENGINE_NAME,
                "version": None,
                "model_sha256": None,
                "initialization_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "elapsed_ms": 0.0,
                "region_count": 0,
                "regions": [],
                "error_type": type(exc).__name__,
            }
            return None, dict(_ENGINE_ERROR)
        except Exception as exc:
            _ENGINE_ERROR = {
                "status": "failed",
                "engine": ENGINE_NAME,
                "version": None,
                "model_sha256": None,
                "initialization_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "elapsed_ms": 0.0,
                "region_count": 0,
                "regions": [],
                "error_type": type(exc).__name__,
            }
            return None, dict(_ENGINE_ERROR)


def _normalize_regions(output: Any) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for value in output or []:
        points_value = value
        array = np.asarray(points_value)
        if array.shape != (4, 2) and isinstance(value, (list, tuple)) and value:
            points_value = value[0]
            array = np.asarray(points_value)
        if array.shape != (4, 2):
            continue
        points = [[int(round(float(x))), int(round(float(y)))] for x, y in array]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        regions.append(
            {
                "polygon": points,
                "bbox": [x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)],
            }
        )
    return regions


def _detector_model_path(engine: Any) -> Path | None:
    session = getattr(getattr(getattr(engine, "text_det", None), "infer", None), "session", None)
    value = getattr(session, "_model_path", None)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
