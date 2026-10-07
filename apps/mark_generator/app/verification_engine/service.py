from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

from .payload import build_run_payload
from .pipeline import run_pipeline

ALLOWED_CONTRACT_EXTENSIONS = {".xlsx", ".xls"}
ALLOWED_CONTRACT_SNAPSHOT_EXTENSIONS = {".json"}
ALLOWED_MARK_EXTENSIONS = {".pdf", ".cdr", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_EVIDENCE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class VerificationEngineError(ValueError):
    pass


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def run_verification(
    contract_path: Path | str,
    mark_paths: Iterable[Path | str],
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    lineart_evidence_dir: Path | str | None = None,
    contract_snapshot_path: Path | str | None = None,
    snapshot_inputs: bool = True,
) -> dict[str, Any]:
    contract = Path(contract_path).resolve()
    contract_snapshot = (
        Path(contract_snapshot_path).resolve() if contract_snapshot_path is not None else None
    )
    marks = [Path(path).resolve() for path in mark_paths]
    target = Path(run_dir).resolve()
    run_id = run_id or new_run_id()
    _validate_run_id(run_id)
    _validate_input(contract, ALLOWED_CONTRACT_EXTENSIONS, "contract")
    if contract_snapshot is not None:
        _validate_input(
            contract_snapshot,
            ALLOWED_CONTRACT_SNAPSHOT_EXTENSIONS,
            "contract snapshot",
        )
    if not marks:
        raise VerificationEngineError("At least one mark file is required.")
    for mark in marks:
        _validate_input(mark, ALLOWED_MARK_EXTENSIONS, "mark")

    target.mkdir(parents=True, exist_ok=False)
    try:
        if snapshot_inputs:
            contract, marks, evidence, contract_snapshot = _snapshot_inputs(
                contract,
                marks,
                Path(lineart_evidence_dir).resolve() if lineart_evidence_dir else None,
                contract_snapshot,
                target,
            )
        else:
            evidence = Path(lineart_evidence_dir).resolve() if lineart_evidence_dir else None

        output_dir = target / "artifacts"
        summary = run_pipeline(
            contract,
            marks,
            output_dir,
            evidence,
            contract_snapshot_path=contract_snapshot,
        )
        payload = build_run_payload(run_id, output_dir, summary)
        manifest = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "engine": "embedded_v5",
            "rule_version": "v5",
            "inputs": {
                "contract": _file_record(contract, target),
                "contract_snapshot": (
                    _file_record(contract_snapshot, target) if contract_snapshot is not None else None
                ),
                "marks": [_file_record(mark, target) for mark in marks],
            },
            "artifacts_dir": "artifacts",
            "status_counts": payload["status_counts"],
            "evidence_status_counts": payload["evidence_status_counts"],
        }
        (target / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (target / "payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload["run_manifest"] = manifest
        return payload
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def load_run(run_dir: Path | str) -> dict[str, Any]:
    target = Path(run_dir)
    payload_path = target / "payload.json"
    summary_path = target / "artifacts" / "v5_summary.json"
    if not payload_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Verification run not found.")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["run_manifest"] = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
    return payload


def _snapshot_inputs(
    contract: Path,
    marks: list[Path],
    evidence_dir: Path | None,
    contract_snapshot: Path | None,
    target: Path,
) -> tuple[Path, list[Path], Path | None, Path | None]:
    inputs = target / "inputs"
    inputs.mkdir()
    contract_target = inputs / f"contract{contract.suffix.lower()}"
    _link_or_copy(contract, contract_target)
    contract_snapshot_target = None
    if contract_snapshot is not None:
        contract_snapshot_target = inputs / "contract.snapshot.json"
        _link_or_copy(contract_snapshot, contract_snapshot_target)
    mark_dir = inputs / "marks"
    mark_dir.mkdir()
    mark_targets = []
    for index, mark in enumerate(marks, start=1):
        mark_target = mark_dir / f"{index:02d}-{_safe_filename(mark.name)}"
        _link_or_copy(mark, mark_target)
        mark_targets.append(mark_target)

    evidence_target = None
    if evidence_dir and evidence_dir.is_dir():
        evidence_target = inputs / "lineart_evidence"
        evidence_target.mkdir()
        for path in sorted(evidence_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_EVIDENCE_EXTENSIONS:
                _link_or_copy(path, evidence_target / _safe_filename(path.name))
    return contract_target, mark_targets, evidence_target, contract_snapshot_target


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _validate_input(path: Path, extensions: set[str], label: str) -> None:
    if not path.is_file():
        raise VerificationEngineError(f"{label} file not found: {path.name}")
    if path.suffix.lower() not in extensions:
        raise VerificationEngineError(f"Unsupported {label} file type: {path.suffix}")
    if path.stat().st_size <= 0:
        raise VerificationEngineError(f"{label} file is empty: {path.name}")


def _validate_run_id(run_id: str) -> None:
    if not run_id or any(character not in "0123456789abcdef" for character in run_id):
        raise VerificationEngineError("Invalid verification run ID.")


def _safe_filename(name: str) -> str:
    safe = "".join(character for character in Path(name).name if character.isalnum() or character in "._-")
    return safe or "upload.bin"


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
