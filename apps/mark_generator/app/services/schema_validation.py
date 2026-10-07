from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_DIR = Path(__file__).resolve().parents[4] / "shared" / "schemas"
_STANDARD_WRITE_LOCK = threading.Lock()


class ContractValidationError(ValueError):
    """Raised before an invalid standard artifact can replace a valid one."""


def validate_standard_payload(payload: dict[str, Any], schema_name: str) -> None:
    schema_path = SCHEMA_DIR / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    details = []
    for error in errors[:8]:
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        details.append(f"{location}: {error.message}")
    if len(errors) > 8:
        details.append(f"... and {len(errors) - 8} more validation errors")
    raise ContractValidationError(
        f"Payload does not satisfy {schema_name}: " + "; ".join(details)
    )


def write_standard_json(path: Path, payload: dict[str, Any], schema_name: str) -> None:
    """Validate, then atomically replace a standard JSON artifact."""

    validate_standard_payload(payload, schema_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows can reject two simultaneous replacements of the same target
        # even when each writer uses a unique temporary file. The local app is
        # single-process but serves requests on multiple worker threads, so
        # serialize only the final atomic replacement.
        with _STANDARD_WRITE_LOCK:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
