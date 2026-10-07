from __future__ import annotations

import hashlib
import base64
import json
import time
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image

from .image_postprocess import (
    GENERATIVE_CANVAS_MARGIN_RATIO,
    pad_generative_canvas,
    postprocess_generative_lineart_bytes,
    postprocess_lineart_file,
)
from .api_key_pool import get_api_key_pool
from .lineart_engine import generate_lineart_bytes, generate_lineart_file, generate_lineart_file_v3
from .lineart_target_qc import evaluate_golden_style_alignment, evaluate_target_size
from ..settings import (
    get_lineart_provider_model,
    get_openai_base_url,
    get_openai_image_quality,
    get_xai_api_key,
    get_xai_api_keys,
    get_openai_api_keys,
    get_xai_base_url,
    get_xai_image_model,
)


class LineartProviderError(ValueError):
    pass


def merge_target_size_qc(
    quality: dict[str, Any], target_size_qc: dict[str, Any]
) -> dict[str, Any]:
    """Make target-canvas failures part of the automatic approval decision."""

    merged = dict(quality)
    merged["target_size_qc"] = target_size_qc
    target_failures = list(target_size_qc.get("failure_reasons") or [])
    if target_failures:
        merged["failure_reasons"] = list(
            dict.fromkeys(list(merged.get("failure_reasons") or []) + target_failures)
        )
        merged["auto_qc_passed"] = False
        merged["technical_qc_passed"] = False
    return merged


@dataclass(frozen=True)
class LineartCandidate:
    filename: str
    data: bytes
    media_type: str
    provider: str
    model: str | None
    prompt: str | None
    warnings: list[dict[str, Any]]
    algorithm_version: str | None = None
    mode: str | None = None
    quality: dict[str, Any] | None = None
    diagnostics: dict[str, bytes] | None = None


class LineartProvider(Protocol):
    name: str

    def generate(
        self,
        *,
        project_dir: Path,
        item: dict,
        region_path: Path,
        mask_path: Path | None = None,
    ) -> LineartCandidate:
        ...


class MockLineartProvider:
    """Legacy workflow stub retained for explicit compatibility tests only."""

    name = "mock"

    def generate(
        self,
        *,
        project_dir: Path,
        item: dict,
        region_path: Path,
        mask_path: Path | None = None,
    ) -> LineartCandidate:
        if not region_path.exists() or not region_path.is_file():
            raise LineartProviderError("Selected product-region image was not found.")

        output, process_warnings = postprocess_lineart_file(region_path)
        return LineartCandidate(
            filename=f"{item.get('item_no', 'item')}-mock-lineart.png",
            data=output,
            media_type="image/png",
            provider=self.name,
            model=None,
            prompt=None,
            warnings=[
                {
                    "severity": "needs_review",
                    "code": "mock_lineart_candidate",
                    "message": "MockLineartProvider generated a workflow candidate only; it is not production line art.",
                }
            ]
            + process_warnings,
        )


class OpenCVV2LineartProvider:
    """Deterministic provider that only preserves source-supported structure."""

    name = "opencv_v2"

    def generate(
        self,
        *,
        project_dir: Path,
        item: dict,
        region_path: Path,
        mask_path: Path | None = None,
    ) -> LineartCandidate:
        if not region_path.exists() or not region_path.is_file():
            raise LineartProviderError("Selected product-region image was not found.")

        try:
            result = generate_lineart_file(region_path, mask_path)
        except Exception as exc:
            raise LineartProviderError(f"OpenCV lineart generation failed: {exc}") from exc
        return LineartCandidate(
            filename=f"{item.get('item_no', 'item')}-opencv-v2-lineart.png",
            data=result.data,
            media_type="image/png",
            provider=self.name,
            model=None,
            prompt=None,
            warnings=result.warnings,
            algorithm_version=result.algorithm_version,
            mode=result.mode,
            quality=result.quality,
        )


class OpenCVV3LineartProvider:
    """OCR-aware deterministic provider for selected product regions only."""

    name = "opencv_v3"

    def generate(
        self,
        *,
        project_dir: Path,
        item: dict,
        region_path: Path,
        mask_path: Path | None = None,
    ) -> LineartCandidate:
        if not region_path.exists() or not region_path.is_file():
            raise LineartProviderError("Selected product-region image was not found.")

        region = item.get("selected_product_region") or {}
        if not isinstance(region, dict):
            raise LineartProviderError("Selected product-region metadata is invalid.")
        (
            text_mask_path,
            text_regions,
            text_mask_status,
            ocr_evidence,
            evidence_warnings,
        ) = resolve_v3_ocr_evidence(project_dir, region)
        try:
            result = generate_lineart_file_v3(
                region_path,
                mask_path,
                text_mask_path=text_mask_path,
                text_regions=text_regions,
                text_mask_status=text_mask_status,
                mask_semantics=str(region.get("mask_semantics") or "foreground_evidence"),
            )
        except LineartProviderError:
            raise
        except Exception as exc:
            raise LineartProviderError(f"OpenCV v3 lineart generation failed: {exc}") from exc

        quality = dict(result.quality)
        quality["ocr_evidence"] = ocr_evidence
        return LineartCandidate(
            filename=f"{item.get('item_no', 'item')}-opencv-v3-lineart.png",
            data=result.data,
            media_type="image/png",
            provider=self.name,
            model=None,
            prompt=None,
            warnings=result.warnings + evidence_warnings,
            algorithm_version=result.algorithm_version,
            mode=result.mode,
            quality=quality,
            diagnostics=result.diagnostics,
        )


class OpenAIImageLineartProvider:
    """Generate a finished line-art asset from the selected product region.

    The provider deliberately receives only the immutable selected cutout.  It
    never sends the complete contract sales page to the image model.
    """

    name = "openai_image"
    algorithm_version = "openai-image-edit-v4-1k-distilled-style"

    def generate(
        self,
        *,
        project_dir: Path,
        item: dict,
        region_path: Path,
        mask_path: Path | None = None,
    ) -> LineartCandidate:
        if not region_path.exists() or not region_path.is_file():
            raise LineartProviderError("Selected product-region image was not found.")
        keys = get_openai_api_keys()
        if not keys:
            raise LineartProviderError(
                "OPENAI_API_KEY is not configured; generative lineart cannot run."
            )

        model = get_lineart_provider_model(self.name) or get_openai_image_model()
        quality = get_openai_image_quality()
        base_url = get_openai_base_url()
        prompt = build_generative_lineart_prompt(item)
        # Keep the longest edge at the gateway's 1K tier while matching the
        # selected product's broad aspect ratio.  A square canvas is not a
        # visual requirement and wastes space for tall or wide products.
        size = preferred_output_size(region_path)
        pool = get_api_key_pool(self.name, keys)
        lease = pool.acquire()
        rate_limited = False
        hard_fail = False
        try:
            from openai import OpenAI

            client = OpenAI(api_key=lease.key, base_url=base_url) if base_url else OpenAI(api_key=lease.key)
            with region_path.open("rb") as image_file:
                response = client.images.edit(
                    model=model,
                    image=image_file,
                    prompt=prompt,
                    quality=quality,
                    size=size,
                    output_format="png",
                    background="opaque",
                )
            encoded = response.data[0].b64_json if response.data else None
            if not encoded:
                raise LineartProviderError("OpenAI image edit returned no image data.")
            raw = base64.b64decode(encoded, validate=True)
            validate_generated_image(raw)
            raw = pad_generative_canvas(raw)
            # Technical QC uses a deterministic binary interpretation, while
            # the saved production candidate retains clean grayscale
            # antialiasing and line-weight hierarchy.
            result = generate_lineart_bytes(raw)
            processed, process_warnings = postprocess_generative_lineart_bytes(raw)
            target_size_qc, _ = evaluate_target_size(processed)
        except LineartProviderError:
            hard_fail = True
            raise
        except Exception as exc:
            message = str(exc).lower()
            rate_limited = "429" in message or "rate" in message
            hard_fail = "401" in message or "403" in message or "auth" in message
            raise LineartProviderError(f"OpenAI image lineart generation failed: {exc}") from exc
        finally:
            pool.release(lease, rate_limited=rate_limited, hard_fail=hard_fail)

        result_quality = merge_target_size_qc(result.quality, target_size_qc)
        result_quality.update(
            {
                "generative_ai": True,
                "model": model,
                # The production policy trusts the configured AI lineart
                # provider after deterministic artifact QC.  The former V5
                # visual/OC comparison was slow and is intentionally disabled.
                "visual_qc_required": False,
                "visual_review_policy": "trusted_ai_generation_no_visual_oc",
                "golden_style_profile": "temporary_golden_lineart_v1_distilled_prompt_only",
                "requested_output_size": size,
                "api_key_slot": lease.slot,
                "api_key_pool_size": pool.size,
                "canvas_safety_margin_ratio": GENERATIVE_CANVAS_MARGIN_RATIO,
                "golden_style_alignment": evaluate_golden_style_alignment(target_size_qc),
            }
        )
        return LineartCandidate(
            filename=f"{item.get('item_no', 'item')}-openai-lineart.png",
            data=processed,
            media_type="image/png",
            provider=self.name,
            model=model,
            prompt=prompt,
            warnings=result.warnings + process_warnings,
            algorithm_version=f"{self.algorithm_version}+{result.algorithm_version}",
            mode=result.mode,
            quality=result_quality,
        )


class GrokImageLineartProvider:
    """Generate line art through xAI's JSON image-edit endpoint.

    xAI intentionally does not accept the OpenAI SDK's multipart
    ``images.edit`` request.  The selected product cutout is therefore sent
    as a base64 data URI in a JSON request, and the returned temporary image
    URL is downloaded immediately into the immutable candidate directory.
    """

    name = "grok_image"
    algorithm_version = "grok-imagine-image-edit-v1-1k-distilled-style"
    timeout_seconds = 180.0
    # Transient TLS/proxy drops (e.g. UNEXPECTED_EOF_WHILE_READING) are common on
    # unstable routes to api.x.ai; retry with a fresh key lease before failing.
    max_attempts = 3
    retry_backoff_seconds = (0.8, 1.6, 3.0)

    def generate(
        self,
        *,
        project_dir: Path,
        item: dict,
        region_path: Path,
        mask_path: Path | None = None,
    ) -> LineartCandidate:
        if not region_path.exists() or not region_path.is_file():
            raise LineartProviderError("Selected product-region image was not found.")
        keys = get_xai_api_keys()
        if not keys:
            raise LineartProviderError(
                "XAI_API_KEY is not configured; select Grok and save a local key first."
            )

        model = get_lineart_provider_model(self.name) or get_xai_image_model()
        prompt = build_generative_lineart_prompt(item)
        pool = get_api_key_pool(self.name, keys)
        encoded = base64.b64encode(region_path.read_bytes()).decode("ascii")
        source_media_type = media_type_for_extension(region_path.suffix.lower().lstrip("."))
        if source_media_type not in {"image/png", "image/jpeg", "image/webp"}:
            source_media_type = "image/png"
        payload = {
            "model": model,
            "prompt": prompt,
            "image": {
                "url": f"data:{source_media_type};base64,{encoded}",
                "type": "image_url",
            },
            # Prefer inline bytes. Some VPS IPs get 403 on imgen.x.ai temporary
            # URLs ("abusive traffic patterns"), which breaks URL-only results.
            "response_format": "b64_json",
        }

        last_error: Exception | None = None
        used_slot: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            lease = pool.acquire()
            used_slot = lease.slot
            rate_limited = False
            hard_fail = False
            try:
                response = httpx.post(
                    f"{get_xai_base_url()}/images/edits",
                    headers={
                        "Authorization": f"Bearer {lease.key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                data = body.get("data") if isinstance(body, dict) else None
                first = data[0] if isinstance(data, list) and data else None
                if not isinstance(first, dict):
                    raise LineartProviderError("Grok image edit returned no image data.")
                raw = self._decode_result(first)
                validate_generated_image(raw)
                raw = pad_generative_canvas(raw)
                result = generate_lineart_bytes(raw)
                processed, process_warnings = postprocess_generative_lineart_bytes(raw)
                target_size_qc, _ = evaluate_target_size(processed)
                pool.release(lease, rate_limited=False, hard_fail=False)
                break
            except LineartProviderError:
                hard_fail = True
                pool.release(lease, rate_limited=rate_limited, hard_fail=hard_fail)
                raise
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                rate_limited = code == 429
                hard_fail = code in {401, 403}
                detail = ""
                try:
                    detail = (exc.response.text or "")[:180].replace("\n", " ")
                except Exception:
                    detail = ""
                pool.release(lease, rate_limited=rate_limited, hard_fail=hard_fail)
                # Retry only rate limits; auth errors fail closed immediately.
                if code == 429 and attempt < self.max_attempts:
                    last_error = exc
                    time.sleep(self.retry_backoff_seconds[min(attempt - 1, len(self.retry_backoff_seconds) - 1)])
                    continue
                suffix = f" body={detail}" if detail else ""
                raise LineartProviderError(
                    f"Grok image lineart generation failed (HTTP {code}).{suffix}"
                ) from exc
            except httpx.HTTPError as exc:
                # ConnectError/ReadError/SSL EOF etc. — retry with another key if possible.
                pool.release(lease, rate_limited=False, hard_fail=False)
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds[min(attempt - 1, len(self.retry_backoff_seconds) - 1)])
                    continue
                raise LineartProviderError(
                    f"Grok image lineart request failed after {self.max_attempts} attempts "
                    f"({type(exc).__name__}: {exc}). Check proxy/TLS route to api.x.ai."
                ) from exc
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                pool.release(lease, rate_limited=False, hard_fail=True)
                raise LineartProviderError(
                    f"Grok image lineart returned an unreadable response: {exc}"
                ) from exc
            except Exception as exc:
                pool.release(lease, rate_limited=False, hard_fail=True)
                raise LineartProviderError(f"Grok image lineart generation failed: {exc}") from exc
        else:
            raise LineartProviderError(
                f"Grok image lineart request failed after {self.max_attempts} attempts "
                f"({type(last_error).__name__ if last_error else 'unknown'})."
            )

        result_quality = merge_target_size_qc(result.quality, target_size_qc)
        result_quality.update(
            {
                "generative_ai": True,
                "model": model,
                "visual_qc_required": False,
                "visual_review_policy": "trusted_ai_generation_no_visual_oc",
                "golden_style_profile": "temporary_golden_lineart_v1_distilled_prompt_only",
                "requested_output_size": "provider_default_1k",
                "api_key_slot": used_slot,
                "api_key_pool_size": pool.size,
                "canvas_safety_margin_ratio": GENERATIVE_CANVAS_MARGIN_RATIO,
                "golden_style_alignment": evaluate_golden_style_alignment(target_size_qc),
            }
        )
        return LineartCandidate(
            filename=f"{item.get('item_no', 'item')}-grok-lineart.png",
            data=processed,
            media_type="image/png",
            provider=self.name,
            model=model,
            prompt=prompt,
            warnings=result.warnings + process_warnings,
            algorithm_version=f"{self.algorithm_version}+{result.algorithm_version}",
            mode=result.mode,
            quality=result_quality,
        )

    def _decode_result(self, image: dict[str, Any]) -> bytes:
        encoded = image.get("b64_json") or image.get("base64")
        if encoded:
            try:
                return base64.b64decode(str(encoded), validate=True)
            except (ValueError, TypeError) as exc:
                raise LineartProviderError("Grok image edit returned invalid base64 image data.") from exc
        url = image.get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise LineartProviderError("Grok image edit returned neither image data nor a safe image URL.")
        try:
            image_response = httpx.get(url, timeout=self.timeout_seconds, follow_redirects=True)
            image_response.raise_for_status()
            return image_response.content
        except httpx.HTTPError as exc:
            raise LineartProviderError("Grok image result could not be downloaded.") from exc


def get_lineart_provider(name: str) -> LineartProvider:
    normalized = (name or "opencv_v2").strip().lower()
    if normalized in {"opencv_v2", "opencv-v2"}:
        return OpenCVV2LineartProvider()
    if normalized in {"opencv_v3", "opencv-v3"}:
        return OpenCVV3LineartProvider()
    if normalized == "mock":
        return MockLineartProvider()
    if normalized in {"gpt", "openai", "openai_image", "openai-image"}:
        return OpenAIImageLineartProvider()
    if normalized in {"grok", "xai", "grok_image", "grok-image"}:
        return GrokImageLineartProvider()
    raise LineartProviderError(f"Unsupported lineart provider: {name}")


MAX_GENERATIVE_LINEART_PROMPT_CHARS = 1000
MAX_LINEART_REFINEMENT_CHARS = 180


def build_generative_lineart_prompt(item: dict[str, Any]) -> str:
    prompt_override = " ".join(str(item.get("lineart_prompt_override") or "").split())
    if prompt_override:
        return prompt_override[:MAX_GENERATIVE_LINEART_PROMPT_CHARS]
    names = [
        " ".join(str(item.get(key) or "").split())
        for key in ("cn_name", "en_name", "it_name", "es_name")
        if str(item.get(key) or "").strip()
    ]
    product_hint = (" / ".join(names[:4]) or "the product in the selected cutout")[:60].rstrip(" /,;")
    prompt = (
        f"Convert {product_hint} from the selected image to B&W line art. Preserve orientation, layout, silhouette, "
        "proportions, all pieces/counts and key structures; keep complete; never add, remove, merge, restage or redesign "
        "parts. Golden style: bold silhouette, thin inner lines, white space; crisp at 160 x 184 pt. Preserve only markings "
        "physically printed, engraved, embossed or molded on the product. If unclear, copy visible shapes; never invent, "
        "translate or correct. Remove external page text/graphics, prices, dimensions, watermarks and background. Pure white "
        "black strokes; no fill, shading, photo texture, noise or double edges; longest edge 1024 px."
    )
    refinement = " ".join(str(item.get("lineart_prompt_addition") or "").split())
    if refinement:
        refinement = refinement[:MAX_LINEART_REFINEMENT_CHARS].rstrip(" .,;")
        prompt += (
            " Operator refinement (subject to every preservation and safety rule above): "
            f"{refinement}."
        )
    prompt += " Return one image."
    if len(prompt) <= MAX_GENERATIVE_LINEART_PROMPT_CHARS:
        return prompt
    # Keep the contract-bound identity and style instructions intact if a
    # future caller supplies unusually long product names.  The hard cap is
    # enforced before the request leaves the workbench.
    clipped = prompt[:MAX_GENERATIVE_LINEART_PROMPT_CHARS]
    cut = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(" "))
    return clipped[: max(cut, 1)].rstrip(" .,;") + "."


def preferred_output_size(path: Path) -> str:
    with Image.open(path) as image:
        width, height = image.size
    if height > width * 1.15:
        return "768x1024"
    if width > height * 1.15:
        return "1024x768"
    return "1024x1024"


def validate_generated_image(data: bytes) -> None:
    if not data or len(data) > 25 * 1024 * 1024:
        raise LineartProviderError("Generated image is empty or exceeds the 25 MB limit.")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
            width, height = image.size
    except Exception as exc:
        raise LineartProviderError("Generated response is not a valid image.") from exc
    if width < 256 or height < 256 or width * height > 16_000_000:
        raise LineartProviderError("Generated image dimensions are outside the accepted range.")


def resolve_v3_ocr_evidence(
    project_dir: Path, region: dict[str, Any]
) -> tuple[
    Path | None,
    list[dict[str, Any]],
    str,
    dict[str, Any],
    list[dict[str, Any]],
]:
    nested_ocr = region.get("ocr") if isinstance(region.get("ocr"), dict) else {}
    relative_path = region.get("text_mask_path") or nested_ocr.get("text_mask_path")
    expected_hash = region.get("text_mask_sha256") or nested_ocr.get("text_mask_sha256")
    raw_regions = region.get("text_regions", nested_ocr.get("text_regions", []))
    text_regions = [entry for entry in raw_regions if isinstance(entry, dict)] if isinstance(raw_regions, list) else []
    ocr_status = str(region.get("ocr_status") or nested_ocr.get("status") or "missing").strip().lower()
    warnings: list[dict[str, Any]] = []

    evidence = {
        "status": "missing",
        "ocr_status": ocr_status,
        "text_mask_path": str(relative_path) if relative_path else None,
        "text_mask_sha256": str(expected_hash) if expected_hash else None,
        "text_mask_sha256_verified": False,
        "text_regions_count": len(text_regions),
    }
    if relative_path:
        project_root = project_dir.resolve()
        path = (project_root / str(relative_path)).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise LineartProviderError("OCR text mask path escapes the project directory.") from exc
        if path.exists() and path.is_file():
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected_hash and actual_hash != str(expected_hash).lower():
                raise LineartProviderError("OCR text mask hash mismatch; regenerate product-region candidates.")
            status = "provided" if expected_hash else "provided_unverified"
            evidence.update(
                {
                    "status": status,
                    "text_mask_sha256": actual_hash,
                    "text_mask_sha256_verified": bool(expected_hash),
                }
            )
            if not expected_hash:
                warnings.append(
                    {
                        "severity": "needs_review",
                        "code": "lineart_text_mask_hash_missing",
                        "message": "OCR text mask has no immutable hash and cannot support automatic QC.",
                    }
                )
            return path, text_regions, status, evidence, warnings
        warnings.append(
            {
                "severity": "needs_review",
                "code": "lineart_text_mask_missing",
                "message": "Product-region metadata references an OCR text mask that is missing.",
            }
        )
        if text_regions:
            evidence["status"] = "text_regions"
            return None, text_regions, "text_regions", evidence, warnings
        evidence["status"] = "missing_file"
        return None, [], "missing_file", evidence, warnings

    if text_regions:
        evidence["status"] = "text_regions"
        return None, text_regions, "text_regions", evidence, warnings
    if ocr_status in {"success", "succeeded", "ok", "completed"}:
        evidence["status"] = "ocr_success_no_text"
        return None, [], "ocr_success_no_text", evidence, warnings
    return None, [], "missing", evidence, warnings


def media_type_for_extension(extension: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }.get(extension, "application/octet-stream")
