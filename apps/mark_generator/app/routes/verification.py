from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..settings import get_workspace_root
from ..verification_engine.payload import build_run_payload
from ..verification_engine.service import (
    ALLOWED_CONTRACT_EXTENSIONS,
    ALLOWED_MARK_EXTENSIONS,
    VerificationEngineError,
    load_run,
    new_run_id,
    run_verification,
)

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_CONTRACT_BYTES = 100 * 1024 * 1024
MAX_MARK_BYTES = 256 * 1024 * 1024
MAX_MARK_FILES = 20
DOWNLOADS = {
    "summary": ("v5_item_summary.csv", "text/csv"),
    "details": ("v5_result.csv", "text/csv"),
    "json": ("v5_result.json", "application/json"),
    "task": ("v5_summary.json", "application/json"),
    "visual": ("visual_report.html", "text/html"),
}


@router.post("/api/verification-runs")
async def create_verification_run(
    contract: Annotated[UploadFile, File()],
    marks: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    if len(marks) > MAX_MARK_FILES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_MARK_FILES} mark files may be uploaded.")
    run_id = new_run_id()
    runs_root = get_workspace_root() / "verification_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"verification-{run_id}-", dir=runs_root))
    target = runs_root / run_id
    try:
        contract_path = await _save_upload(
            contract, stage / "contract", ALLOWED_CONTRACT_EXTENSIONS, MAX_CONTRACT_BYTES
        )
        mark_dir = stage / "marks"
        mark_dir.mkdir()
        mark_paths = []
        for index, mark in enumerate(marks, start=1):
            mark_paths.append(
                await _save_upload(
                    mark,
                    mark_dir / f"{index:02d}",
                    ALLOWED_MARK_EXTENSIONS,
                    MAX_MARK_BYTES,
                )
            )
        payload = run_verification(contract_path, mark_paths, target, run_id=run_id)
        return payload
    except VerificationEngineError as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        logger.exception("Embedded verification run %s failed", run_id)
        raise HTTPException(
            status_code=500,
            detail="Verification failed; details were written to the local service log.",
        ) from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@router.get("/api/verification-runs/{run_id}")
def get_verification_run(run_id: str) -> dict[str, Any]:
    return load_run(_run_dir(run_id))


@router.get("/api/verification-runs/{run_id}/downloads/{name}")
def download_verification_artifact(run_id: str, name: str) -> FileResponse:
    item = DOWNLOADS.get(name)
    if not item:
        raise HTTPException(status_code=404, detail="Verification artifact not found.")
    filename, media_type = item
    path = _run_dir(run_id) / "artifacts" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Verification artifact not found.")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/api/verification-runs/{run_id}/images/{item_number}")
def verification_mark_image(run_id: str, item_number: str) -> FileResponse:
    run_dir = _run_dir(run_id)
    manifest = _read_json(run_dir / "artifacts" / "mark_images_manifest.json")
    for item in manifest.get("items", []):
        if item.get("item_number") != item_number or item.get("status") not in {"OK", "PAGE_FALLBACK"}:
            continue
        path = _resolved_artifact_path(item.get("image", ""), run_dir)
        if path:
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Verification image not found.")


@router.get("/api/verification-runs/{run_id}/visual-cells")
def verification_visual_cells(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    manifest = _read_json(run_dir / "artifacts" / "visual_cells_manifest.json")
    items = []
    for index, item in enumerate(manifest.get("items", [])):
        if item.get("status") != "VISUAL_CELL_CANDIDATE":
            continue
        items.append(
            {
                "index": index,
                "page": item.get("page", ""),
                "cell_index": item.get("cell_index", ""),
                "status": "VISUAL_CELL_CANDIDATE",
                "association": "NEEDS_MANUAL_REVIEW",
                "image_url": f"/api/verification-runs/{run_id}/visual-cells/{index}/image",
                "bbox_px": item.get("bbox_px", []),
                "message": item.get("message", "") or "视觉候选切块，未做核对，请人工复核",
            }
        )
    return {
        "run_id": run_id,
        "candidate_count": len(items),
        "items": items,
        "events": manifest.get("events", []),
    }


@router.get("/api/verification-runs/{run_id}/visual-cells/{cell_index}/image")
def verification_visual_cell_image(run_id: str, cell_index: int) -> FileResponse:
    run_dir = _run_dir(run_id)
    manifest = _read_json(run_dir / "artifacts" / "visual_cells_manifest.json")
    items = manifest.get("items", [])
    if cell_index < 0 or cell_index >= len(items):
        raise HTTPException(status_code=404, detail="Visual candidate not found.")
    item = items[cell_index]
    if item.get("status") != "VISUAL_CELL_CANDIDATE":
        raise HTTPException(status_code=404, detail="Visual candidate not found.")
    path = _resolved_artifact_path(item.get("image", ""), run_dir)
    if not path:
        raise HTTPException(status_code=404, detail="Visual candidate not found.")
    return FileResponse(path)


@router.post("/api/verify", include_in_schema=False)
async def legacy_verify(
    contract: Annotated[UploadFile, File()],
    marks: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    return await create_verification_run(contract, marks)


@router.get("/api/runs/{run_id}", include_in_schema=False)
def legacy_get_run(run_id: str) -> dict[str, Any]:
    return get_verification_run(run_id)


async def _save_upload(
    upload: UploadFile,
    stem: Path,
    allowed: set[str],
    limit: int,
) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename or suffix}")
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"{upload.filename or 'upload'} exceeds the size limit.")
    if not data:
        raise HTTPException(status_code=400, detail=f"Upload is empty: {upload.filename or 'file'}")
    path = stem.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _run_dir(run_id: str) -> Path:
    if not run_id or any(character not in "0123456789abcdef" for character in run_id):
        raise HTTPException(status_code=404, detail="Verification run not found.")
    path = get_workspace_root() / "verification_runs" / run_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Verification run not found.")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolved_artifact_path(raw_path: str, run_dir: Path) -> Path | None:
    path = Path(str(raw_path))
    candidates = [path] if path.is_absolute() else [run_dir / "artifacts" / path, run_dir / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(run_dir.resolve())
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None
