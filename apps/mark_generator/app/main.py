from __future__ import annotations

import hashlib
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .routes.verification import router as verification_router

from .services.excel_contract import SUPPORTED_CONTRACT_EXTENSIONS, extract_contract_project
from .services.contract_item_region import ContractExtractionError
from .services.delivery import delivery_pdf_filename
from .services.lineart_jobs import get_lineart_job, start_lineart_job
from .services.lineart import (
    LineartError,
    approve_eligible_lineart_candidates,
    approve_lineart_candidate,
    generate_lineart_candidates,
    generate_lineart_candidate,
    refine_lineart_from_existing,
    get_project_lineart_prompt,
    get_lineart_state,
    reject_lineart,
    resolve_project_dir,
    resolve_project_file,
    save_lineart_candidate,
    save_project_lineart_prompt,
    upload_lineart,
)
from .services.mark_preview import (
    MarkPreviewError,
    render_standard_mark_pair_png,
)
from .services.image_variants import (
    ImageVariantError,
    project_image_variant,
    standard_mark_preview_etag,
    standard_mark_preview_variant,
)
from .services.lineart_sharing import (
    create_manual_lineart_group,
    get_lineart_group_state,
    share_lineart_with_group,
)
from .services.manual_lineart import upload_manual_lineart_without_product_region
from .services.pdf_generator import GenerationError, generate_project_pdf, get_template_mode
from .services.product_regions import (
    ProductRegionError,
    create_manual_product_region,
    generate_product_regions,
    get_product_region_state,
    reject_product_region,
    select_product_region,
)
from .services.product_region_jobs import (
    get_product_region_job,
    start_product_region_job,
)
from .services.production_readiness import get_production_readiness
from .services.verification import VerificationError, verify_project
from .services.verification_jobs import get_verification_job, start_verification_job
from .services.verification_manual_release import (
    ManualReleaseConflict,
    ManualReleaseError,
    create_manual_release,
)
from .settings import (
    get_active_lineart_provider,
    get_lineart_provider_status,
    get_workspace_root,
    save_grok_api_key,
    save_openai_api_key,
    save_lineart_provider_selection,
)


app = FastAPI(title="Mark Automation Workbench", version="1.0.0")
app.include_router(verification_router)

FRONTEND_ROOT = Path(__file__).resolve().parent / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT), name="frontend-assets")

# Browsers must revalidate anything that can change in place; only versioned
# front-end assets (``?v=...``) are immutable. Without an explicit policy,
# FileResponse's Last-Modified lets browsers heuristically reuse stale
# project JSON, and the runtime pack's epoch mtimes would pin index.html.
REVALIDATE = "no-cache"
IMMUTABLE = "public, max-age=31536000, immutable"


@app.middleware("http")
async def cache_policy(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/") and request.query_params.get("v"):
        response.headers["Cache-Control"] = IMMUTABLE
    elif "cache-control" not in response.headers and (
        path == "/" or path.startswith("/assets/") or path.startswith("/api/")
    ):
        response.headers["Cache-Control"] = REVALIDATE
    return response


def _etag_matches(request: Request, etag: str) -> bool:
    header = request.headers.get("if-none-match", "")
    if not header:
        return False
    if header.strip() == "*":
        return True
    # nginx may weaken ETags (W/) when it compresses; compare opaque tags.
    candidates = {value.strip().removeprefix("W/") for value in header.split(",")}
    return etag in candidates


def _not_modified(etag: str) -> Response:
    return Response(status_code=304, headers={"ETag": etag, "Cache-Control": REVALIDATE})

MAX_CONTRACT_BYTES = 100 * 1024 * 1024
MAX_LINEART_BYTES = 25 * 1024 * 1024


async def read_upload_limited(upload: UploadFile, limit: int, label: str) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"{label} exceeds the {limit // (1024 * 1024)} MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    return data


_ASSET_REF = re.compile(r'(/assets/)([A-Za-z0-9_.-]+)\?v=[A-Za-z0-9_.-]+')
_index_cache: dict[str, object] = {}


def _asset_version(name: str) -> str:
    path = FRONTEND_ROOT / name
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return "missing"


def _render_index() -> str:
    """index.html with every ``/assets/x?v=...`` pinned to x's content hash.

    Versioned assets are served immutable, so the version must change whenever
    the file does; deriving it from content makes a forgotten manual bump
    impossible. Re-rendered only when a front-end file changes.
    """

    files = sorted(p for p in FRONTEND_ROOT.iterdir() if p.is_file())
    stamp = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    if _index_cache.get("stamp") != stamp:
        html = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")
        html = _ASSET_REF.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={_asset_version(m.group(2))}", html)
        _index_cache.update(stamp=stamp, html=html)
    return str(_index_cache["html"])


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())



@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "app": "mark_automation",
        "verification_engine": {"status": "ready", "engine": "embedded_v5"},
    }


@app.get("/api/lineart-provider")
def lineart_provider_status() -> dict[str, object]:
    """Return non-secret provider/model configuration for setup diagnostics."""
    return get_lineart_provider_status()


@app.post("/api/lineart-provider/select")
def select_lineart_provider(
    provider: Annotated[str, Form()],
    model: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    """Persist the provider choice for subsequent lineart jobs."""

    try:
        return save_lineart_provider_selection(provider, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _key_save_forbidden(request: Request) -> bool:
    """Block browser key saves in production or when the caller is not local.

    Behind nginx the peer is always 127.0.0.1, so local-only host checks are not
    enough on a public reverse proxy. Set MARK_DISABLE_KEY_SAVE=1 on servers.
    """
    flag = os.environ.get("MARK_DISABLE_KEY_SAVE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    client_host = request.client.host if request.client else ""
    return client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}


@app.post("/api/lineart-provider/grok-key")
def save_grok_key(
    request: Request,
    api_key: Annotated[str, Form()],
    append: Annotated[bool, Form()] = True,
) -> dict[str, object]:
    """Save a Grok key only when the workbench is being used locally."""

    if _key_save_forbidden(request):
        raise HTTPException(status_code=403, detail="Grok key saving is disabled on this deployment.")
    try:
        saved = save_grok_api_key(api_key, append=append)
        return {**get_lineart_provider_status(), **saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/lineart-provider/openai-key")
def save_openai_key(
    request: Request,
    api_key: Annotated[str, Form()],
    append: Annotated[bool, Form()] = True,
) -> dict[str, object]:
    """Save an OpenAI-compatible key only on the local workbench."""

    if _key_save_forbidden(request):
        raise HTTPException(status_code=403, detail="OpenAI key saving is disabled on this deployment.")
    try:
        saved = save_openai_api_key(api_key, append=append)
        return {**get_lineart_provider_status(), **saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/workspace")
def workspace_info() -> dict[str, object]:
    workspace_root = get_workspace_root()
    return {
        "workspace_root": str(workspace_root),
        "generator_root": str(Path(__file__).resolve().parents[1]),
        "projects_root": str(workspace_root / "projects"),
        "verification_runs_root": str(workspace_root / "verification_runs"),
        "verification_engine": {"status": "ready", "engine": "embedded_v5"},
    }


@app.get("/api/projects/{project_id}")
def project_detail(project_id: str) -> dict[str, object]:
    try:
        project_dir = resolve_project_dir(get_workspace_root(), project_id)
        return {
            "project_id": project_id,
            "project_dir": str(project_dir),
            "linearts": get_lineart_state(get_workspace_root(), project_id),
            "readiness": get_production_readiness(get_workspace_root(), project_id),
            "paths": {
                "items_normalized": "data/items.normalized.json",
                "contract_snapshot": "data/contract.snapshot.json",
                "lineart_groups": "reports/lineart_groups.json",
                "lineart_prompt": "data/lineart_prompt.json",
                "preview_pdf": "generated/preview.pdf",
                "standard_mark_pdf": "generated/standard_mark.pdf",
                "manifest": "manifest.json",
                "generation_report": "reports/generation_report.json",
                "verification_manual_release": "reports/verification_manual_release.json",
            },
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/lineart-groups")
def project_lineart_groups(project_id: str) -> dict[str, object]:
    try:
        return get_lineart_group_state(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/lineart-groups")
def create_project_lineart_group(
    project_id: str,
    item_nos: Annotated[str, Form()],
    group_name: Annotated[str, Form()],
    reviewer: Annotated[str, Form()],
    reason: Annotated[str, Form()],
) -> dict[str, object]:
    try:
        return create_manual_lineart_group(
            get_workspace_root(),
            project_id,
            item_nos=item_nos.split(","),
            group_name=group_name,
            reviewer=reviewer,
            reason=reason,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/lineart-groups/{group_id}/share")
def share_project_lineart_group(
    project_id: str,
    group_id: str,
    representative_item_no: Annotated[str, Form()],
    reviewer: Annotated[str, Form()],
    reason: Annotated[str, Form()],
    allowed_variations: Annotated[str, Form()] = "size,color,pattern",
    confirm_same_shape: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    try:
        return share_lineart_with_group(
            get_workspace_root(),
            project_id,
            group_id,
            representative_item_no=representative_item_no,
            reviewer=reviewer,
            reason=reason,
            allowed_variations=allowed_variations.split(","),
            confirm_same_shape=confirm_same_shape,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/extract")
async def extract_project(
    contract: Annotated[UploadFile, File()],
    project_id: Annotated[str | None, Form()] = None,
) -> dict:
    suffix = Path(contract.filename or "contract.xlsx").suffix or ".xlsx"
    suffix = suffix.lower()
    if suffix not in SUPPORTED_CONTRACT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_CONTRACT_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Contract must be one of: {allowed}.")
    contract_bytes = await read_upload_limited(contract, MAX_CONTRACT_BYTES, "Contract")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(contract_bytes)
        temp_path = Path(temp_file.name)
    try:
        try:
            result = extract_contract_project(
                temp_path,
                workspace_root=get_workspace_root(),
                project_id=project_id.strip() if project_id else None,
                original_filename=contract.filename,
            )
        except ContractExtractionError as exc:
            raise HTTPException(status_code=422, detail=exc.as_detail()) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        readiness = get_production_readiness(get_workspace_root(), result["normalized"]["project_id"])
        return {
            "project_id": result["normalized"]["project_id"],
            "total_items": result["report"]["total_items"],
            "matched_images": result["report"]["matched_images"],
            "warnings": result["report"]["warnings"],
            "contract_item_region": result["report"].get("contract_item_region"),
            "paths": result["paths"],
            "readiness": readiness,
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/projects/{project_id}/linearts")
def project_linearts(project_id: str) -> dict[str, object]:
    try:
        return get_lineart_state(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/product-regions")
def project_product_regions(project_id: str) -> dict[str, object]:
    try:
        return get_product_region_state(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProductRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/product-regions/generate", status_code=202)
def generate_project_product_regions(
    project_id: str,
    force: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    try:
        return start_product_region_job(get_workspace_root(), project_id, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ProductRegionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/product-regions/progress")
def project_product_region_progress(project_id: str) -> dict[str, object]:
    try:
        return get_product_region_job(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ProductRegionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/product-regions/{item_no}/generate")
def generate_item_product_regions(
    project_id: str,
    item_no: str,
    force: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    try:
        return generate_product_regions(get_workspace_root(), project_id, item_no=item_no, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProductRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/product-regions/{item_no}/select")
def select_project_product_region(
    project_id: str,
    item_no: str,
    candidate_id: Annotated[str, Form()],
) -> dict[str, object]:
    try:
        return select_product_region(get_workspace_root(), project_id, item_no, candidate_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProductRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/product-regions/{item_no}/manual-crop")
def create_project_manual_product_region(
    project_id: str,
    item_no: str,
    x: Annotated[float, Form()],
    y: Annotated[float, Form()],
    width: Annotated[float, Form()],
    height: Annotated[float, Form()],
    reviewer: Annotated[str, Form()],
    reason: Annotated[str, Form()],
) -> dict[str, object]:
    try:
        return create_manual_product_region(
            get_workspace_root(),
            project_id,
            item_no,
            x=x,
            y=y,
            width=width,
            height=height,
            reviewer=reviewer,
            reason=reason,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProductRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/product-regions/{item_no}/reject")
def reject_project_product_region(
    project_id: str,
    item_no: str,
    candidate_id: Annotated[str, Form()],
) -> dict[str, object]:
    try:
        return reject_product_region(get_workspace_root(), project_id, item_no, candidate_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProductRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/candidates")
async def upload_lineart_candidate(
    project_id: str,
    item_no: str,
    file: Annotated[UploadFile, File()],
) -> dict[str, object]:
    content = await read_upload_limited(file, MAX_LINEART_BYTES, "Lineart")
    try:
        return save_lineart_candidate(
            get_workspace_root(),
            project_id,
            item_no,
            file.filename or "lineart.png",
            BytesIO(content),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/generate-candidates")
def generate_linearts(
    project_id: str,
    provider: Annotated[str, Form()] = "",
) -> dict[str, object]:
    """Synchronous batch path kept for tests and scripts."""

    try:
        provider_name = provider.strip() or get_active_lineart_provider()
        return generate_lineart_candidates(get_workspace_root(), project_id, provider_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/generate-job", status_code=202)
def start_project_lineart_job(
    project_id: str,
    provider: Annotated[str, Form()] = "",
    concurrency: Annotated[str, Form()] = "",
    strategy: Annotated[str, Form()] = "group_representatives",
) -> dict[str, object]:
    """Start background generation of missing approved linearts and return immediately."""

    try:
        provider_name = provider.strip() or get_active_lineart_provider()
        workers = None
        if concurrency.strip():
            workers = int(concurrency.strip())
        return start_lineart_job(
            get_workspace_root(),
            project_id,
            provider_name=provider_name,
            concurrency=workers,
            strategy=strategy.strip() or "group_representatives",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/linearts/generate-progress")
def project_lineart_job_progress(project_id: str) -> dict[str, object]:
    try:
        return get_lineart_job(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/lineart-prompt")
def project_lineart_prompt(project_id: str) -> dict[str, object]:
    try:
        return get_project_lineart_prompt(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/lineart-prompt")
def update_project_lineart_prompt(
    project_id: str,
    prompt_instruction: Annotated[str, Form()] = "",
) -> dict[str, object]:
    try:
        return save_project_lineart_prompt(
            get_workspace_root(),
            project_id,
            prompt_instruction,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/approve-eligible")
def approve_eligible_linearts(project_id: str) -> dict[str, object]:
    try:
        return approve_eligible_lineart_candidates(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/generate")
def generate_lineart(
    project_id: str,
    item_no: str,
    provider: Annotated[str, Form()] = "",
    prompt_instruction: Annotated[str, Form()] = "",
) -> dict[str, object]:
    try:
        provider_name = provider.strip() or get_active_lineart_provider()
        return generate_lineart_candidate(
            get_workspace_root(),
            project_id,
            item_no,
            provider_name,
            prompt_instruction=prompt_instruction,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/refine")
def refine_lineart(
    project_id: str,
    item_no: str,
    prompt_instruction: Annotated[str, Form()],
    provider: Annotated[str, Form()] = "",
) -> dict[str, object]:
    """Edit the current lineart with a one-shot prompt and auto-adopt the result."""

    try:
        provider_name = provider.strip() or get_active_lineart_provider()
        return refine_lineart_from_existing(
            get_workspace_root(),
            project_id,
            item_no,
            prompt_instruction=prompt_instruction,
            provider_name=provider_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/upload")
async def upload_manual_lineart(
    project_id: str,
    item_no: str,
    file: Annotated[UploadFile, File()],
) -> dict[str, object]:
    content = await read_upload_limited(file, MAX_LINEART_BYTES, "Lineart")
    try:
        return upload_lineart(
            get_workspace_root(),
            project_id,
            item_no,
            file.filename or "uploaded.png",
            BytesIO(content),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/upload-without-product-image")
async def upload_manual_lineart_fallback(
    project_id: str,
    item_no: str,
    file: Annotated[UploadFile, File()],
    reviewer: Annotated[str, Form()],
    reason: Annotated[str, Form()],
    confirm_item_identity: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    content = await read_upload_limited(file, MAX_LINEART_BYTES, "Lineart")
    try:
        return upload_manual_lineart_without_product_region(
            get_workspace_root(),
            project_id,
            item_no,
            file.filename or "uploaded.png",
            content,
            reviewer=reviewer,
            reason=reason,
            confirm_item_identity=confirm_item_identity,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/approve")
def approve_lineart(
    project_id: str,
    item_no: str,
    candidate_path: Annotated[str, Form()],
    approval_mode: Annotated[str, Form()] = "automatic",
    reviewer: Annotated[str | None, Form()] = None,
    reason: Annotated[str | None, Form()] = None,
    confirm_target_size_legible: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    try:
        return approve_lineart_candidate(
            get_workspace_root(),
            project_id,
            item_no,
            candidate_path,
            approval_mode=approval_mode,
            reviewer=reviewer,
            reason=reason,
            confirm_target_size_legible=confirm_target_size_legible,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/linearts/{item_no}/reject")
def reject_lineart_candidate(
    project_id: str,
    item_no: str,
    candidate_path: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    try:
        return reject_lineart(get_workspace_root(), project_id, item_no, candidate_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/generate-preview")
def generate_preview(project_id: str) -> dict[str, object]:
    try:
        return generate_project_pdf(get_workspace_root(), project_id, mode="preview")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, GenerationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/generate-standard")
def generate_standard(project_id: str, verify: bool = False) -> dict[str, object]:
    try:
        result = generate_project_pdf(get_workspace_root(), project_id, mode="standard")
        if verify and result.get("status") in {"success", "success_with_warnings"}:
            # Keep PDF generation synchronous; queue independent verification so the UI can poll.
            generation_status = str(result["status"])
            job = start_verification_job(
                get_workspace_root(),
                project_id,
                template_mode=get_template_mode(),
            )
            result["generation_status"] = generation_status
            result["verification_job"] = job
            result["verification_status"] = job.get("status")
            # Do not overwrite generation status with verification outcome; poll /verify/progress.
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, GenerationError, VerificationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/api/projects/{project_id}/verify", status_code=202)
def verify_generated_project(project_id: str) -> dict[str, object]:
    try:
        return start_verification_job(
            get_workspace_root(),
            project_id,
            template_mode=get_template_mode(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, VerificationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/verify/progress")
def verify_project_progress(project_id: str) -> dict[str, object]:
    try:
        return get_verification_job(get_workspace_root(), project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/verification/manual-release")
def manual_release_failed_verification(
    project_id: str,
    reviewer: Annotated[str, Form()],
    reason: Annotated[str, Form()],
    confirm_all_issues: Annotated[bool, Form()] = False,
    confirmed_issue_ids: Annotated[str, Form()] = "[]",
) -> dict[str, object]:
    """Record a human release without rewriting the automated V5 result."""

    try:
        return create_manual_release(
            get_workspace_root(),
            project_id,
            reviewer=reviewer,
            reason=reason,
            confirm_all_issues=confirm_all_issues,
            confirmed_issue_ids_json=confirmed_issue_ids,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManualReleaseConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ManualReleaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/standard-marks/{item_no}/preview.png")
def standard_mark_preview(project_id: str, item_no: str, request: Request) -> Response:
    try:
        etag = standard_mark_preview_etag(get_workspace_root(), project_id, item_no)
        if _etag_matches(request, etag):
            return _not_modified(etag)
        png = render_standard_mark_pair_png(get_workspace_root(), project_id, item_no)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, MarkPreviewError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": REVALIDATE, "ETag": etag},
    )


@app.get("/api/projects/{project_id}/standard-marks/{item_no}/preview.webp")
def standard_mark_preview_webp(
    project_id: str, item_no: str, request: Request, w: int | None = None
) -> Response:
    """Display variant: lossy thumbnail with ``w``, lossless full render without."""

    try:
        variant = standard_mark_preview_variant(get_workspace_root(), project_id, item_no, w)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, MarkPreviewError, ImageVariantError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _etag_matches(request, variant.etag):
        return _not_modified(variant.etag)
    return Response(
        content=variant.body,
        media_type=variant.media_type,
        headers={"Cache-Control": REVALIDATE, "ETag": variant.etag},
    )


@app.get("/api/projects/{project_id}/thumbs/{file_path:path}")
def project_image_thumbnail(project_id: str, file_path: str, request: Request, w: int = 320) -> Response:
    """Downscaled WebP of a project image for display; originals stay under /files/."""

    try:
        project_dir = resolve_project_dir(get_workspace_root(), project_id)
        path = resolve_project_file(project_dir, file_path)
        variant = project_image_variant(get_workspace_root(), path, w)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LineartError, ImageVariantError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _etag_matches(request, variant.etag):
        return _not_modified(variant.etag)
    return Response(
        content=variant.body,
        media_type=variant.media_type,
        headers={"Cache-Control": REVALIDATE, "ETag": variant.etag},
    )


def _file_etag(path: Path) -> str:
    # Same derivation as Starlette's FileResponse so both paths agree.
    stat = path.stat()
    base = f"{stat.st_mtime}-{stat.st_size}"
    return f'"{hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()}"'


@app.get("/api/projects/{project_id}/files/{file_path:path}")
def project_file(project_id: str, file_path: str, request: Request) -> Response:
    try:
        project_dir = resolve_project_dir(get_workspace_root(), project_id)
        path = resolve_project_file(project_dir, file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LineartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Project file not found.")
    delivery_pdf_path = (project_dir / "generated" / "standard_mark.pdf").resolve()
    if path.resolve() == delivery_pdf_path:
        readiness = get_production_readiness(get_workspace_root(), project_id)
        if not readiness.get("production_ready"):
            raise HTTPException(
                status_code=409,
                detail="Standard PDF is not verified against its current inputs and cannot be released.",
            )
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=delivery_pdf_filename(project_dir, project_id),
        )
    etag = _file_etag(path)
    if _etag_matches(request, etag):
        return _not_modified(etag)
    return FileResponse(path)
