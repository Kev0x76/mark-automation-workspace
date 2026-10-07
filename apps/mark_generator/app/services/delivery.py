from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


MAX_DELIVERY_STEM_LENGTH = 160
INVALID_FILENAME_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]+')


def normalize_original_contract_filename(
    filename: str | None,
    *,
    extension: str,
) -> str:
    """Return a safe leaf filename while preserving human-readable Unicode."""

    text = unicodedata.normalize("NFC", str(filename or ""))
    leaf = text.replace("\\", "/").rsplit("/", 1)[-1]
    leaf = INVALID_FILENAME_CHARS_RE.sub("_", leaf)
    leaf = re.sub(r"\s+", " ", leaf).strip(" .")
    suffix = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    if not leaf:
        return f"contract{suffix}"
    if not Path(leaf).suffix:
        leaf = f"{leaf}{suffix}"
    stem = Path(leaf).stem.strip(" .") or "contract"
    file_suffix = Path(leaf).suffix.lower() or suffix
    max_stem = max(1, 220 - len(file_suffix))
    return f"{stem[:max_stem].rstrip(' .') or 'contract'}{file_suffix}"


def delivery_pdf_filename(project_dir: Path, project_id: str) -> str:
    """Build the approved external PDF name, falling back for legacy projects."""

    original_filename = _snapshot_original_filename(project_dir)
    source = original_filename or project_id
    stem = Path(source).stem if original_filename else source
    stem = _safe_delivery_stem(stem) or _safe_delivery_stem(project_id) or "project"
    return f"{stem}-唛头.pdf"


def _snapshot_original_filename(project_dir: Path) -> str | None:
    path = project_dir / "data" / "contract.snapshot.json"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    source_contract = payload.get("source_contract")
    if not isinstance(source_contract, dict):
        return None
    value = source_contract.get("original_filename")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_delivery_stem(value: str) -> str:
    stem = unicodedata.normalize("NFC", str(value or ""))
    stem = INVALID_FILENAME_CHARS_RE.sub("_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem[:MAX_DELIVERY_STEM_LENGTH].rstrip(" .")
