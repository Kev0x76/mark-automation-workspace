"""Orchestration: carton facts + contract items -> chosen plans -> PDF + report.

Failure is closed. A SKU whose every candidate layout breaks a hard constraint
is reported as ``manual_layout_required`` and is not drawn, because silently
shrinking content until it fits is exactly how an unscannable barcode or an
illegible name reaches a factory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import tempfile
from uuid import uuid4
from typing import Any, Callable

from ..carton_specs import CartonFacts
from ..schema_validation import SCHEMA_DIR
from ..template_renderer import TemplateAssetError
from .atlas import Atlas, load_atlas
from .constraints import select_face
from .families import build_candidates
from .model import Box, FacePlan, LayoutPlan, LayoutRevision, mm
from .renderer import DEFAULT_GUTTER_MM, AdaptiveRenderer, page_geometry

#: The template renderer's refusal to draw content it cannot carry without
#: clipping or distorting. Treated as a layout failure, not a crash.
_RenderRefused = TemplateAssetError


def plan_item(
    atlas: Atlas,
    item: dict[str, Any],
    carton: CartonFacts,
    *,
    gutter_mm: float = DEFAULT_GUTTER_MM,
) -> tuple[LayoutPlan | None, list[dict[str, Any]]]:
    """Choose the best valid front/side pair for one SKU."""
    page_size, front_box, side_box, gutter = page_geometry(
        carton.front_face_mm, carton.side_face_mm, gutter_mm=gutter_mm
    )

    issues: list[dict[str, Any]] = []
    faces: dict[str, FacePlan] = {}
    scores: dict[str, float] = {}
    runners: dict[str, float | None] = {}

    for kind, box in (("front", front_box), ("side", side_box)):
        local = Box(0.0, 0.0, box.width, box.height)
        candidates = build_candidates(
            atlas, kind, local, item,  # type: ignore[arg-type]
            front_aspect=front_box.aspect, side_aspect=side_box.aspect,
        )
        winner, face_issues, score, runner_up = select_face(atlas, candidates, item)
        issues.extend(face_issues)
        if winner is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "manual_layout_required",
                    "message": (
                        f"No valid {kind} layout exists for a "
                        f"{mm(box.width):.0f} x {mm(box.height):.0f} mm face; manual layout is required."
                    ),
                    "item_no": item.get("item_no"),
                    "face": kind,
                }
            )
            return None, issues
        # re-place the chosen face at its absolute position on the page
        winner.panel = box
        winner.content = _shift(winner.content, box.x0, box.y0)
        winner.statics = [_shift_static(s, box.x0, box.y0) for s in winner.statics]
        winner.fields = _shift_fields(winner.fields, box.x0, box.y0)
        if winner.barcode is not None:
            winner.barcode.bars = _shift(winner.barcode.bars, box.x0, box.y0)
        faces[kind] = winner
        scores[kind] = score
        runners[kind] = runner_up

    combined = (scores["front"] + scores["side"]) / 2.0
    runner_values = [v for v in runners.values() if v is not None]
    runner_combined = sum(runner_values) / len(runner_values) if runner_values else None

    margin = float(atlas.scoring.get("auto_select_margin", 0.08))
    selection = "auto_selected"
    if runner_combined is not None and (combined - runner_combined) < margin:
        selection = "needs_review"
        issues.append(
            {
                "severity": "warning",
                "code": "layout_choice_ambiguous",
                "message": (
                    f"Best layout scored {combined:.3f} against a runner-up of {runner_combined:.3f}, "
                    f"inside the {margin} decision margin; a human should confirm the arrangement."
                ),
                "item_no": item.get("item_no"),
            }
        )

    plan = LayoutPlan(
        item_no=str(item.get("item_no") or ""),
        carton_mm=(carton.length_mm, carton.width_mm, carton.height_mm),
        front=faces["front"],
        side=faces["side"],
        page_size_pt=page_size,
        gutter_pt=gutter,
        score=combined,
        runner_up_score=runner_combined,
        selection=selection,
        issues=issues,
    )
    return plan, issues


@lru_cache(maxsize=1)
def _issue_schema_properties() -> dict[str, Any]:
    schema = json.loads((SCHEMA_DIR / "generation_report.schema.json").read_text(encoding="utf-8"))
    return schema["$defs"]["issue"]["properties"]


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "array": (list,),
}


def as_report_issue(finding: dict[str, Any]) -> dict[str, Any]:
    """Project an engine or audit finding onto the generation report's issue shape.

    The report schema is closed, while audit findings carry measurements
    (band, occluded_px, spread, ...). The complete finding stays in
    ``reports/layout_audit.json``; the report keeps only fields its contract
    declares, each with the declared type, so a new measurement never makes a
    valid PDF unreportable.
    """
    properties = _issue_schema_properties()
    issue: dict[str, Any] = {}
    for key, spec in properties.items():
        value = finding.get(key)
        if value is None:
            continue
        expected = _JSON_TYPES.get(str(spec.get("type")))
        if expected and (not isinstance(value, expected) or isinstance(value, bool)):
            continue
        if "enum" in spec and value not in spec["enum"]:
            continue
        if expected == (list,):
            item_type = _JSON_TYPES.get(str(spec.get("items", {}).get("type")))
            if item_type and not all(isinstance(v, item_type) and not isinstance(v, bool) for v in value):
                continue
        issue[key] = value
    issue["code"] = str(finding.get("code") or "unspecified_layout_issue")
    severity = finding.get("severity")
    issue["severity"] = severity if severity in properties["severity"]["enum"] else "warning"
    issue["message"] = str(finding.get("message") or issue["code"])
    return issue


def _shift(box: Box, dx: float, dy: float) -> Box:
    return Box(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy)


def _shift_static(placement, dx: float, dy: float):
    # replace() rather than re-listing fields (as renderer._shift_static does): a
    # re-listed copy silently dropped slice_y / glyph_scale, so a taller table
    # row was drawn with its words scaled to the row height.
    from dataclasses import replace

    return replace(placement, target=_shift(placement.target, dx, dy))


def _shift_fields(fields: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    from .renderer import _shift_fields as shift

    return shift(fields, dx, dy)


def build_revision(
    items: list[dict[str, Any]],
    cartons: dict[str, CartonFacts],
    *,
    atlas: Atlas | None = None,
    gutter_mm: float = DEFAULT_GUTTER_MM,
) -> LayoutRevision:
    atlas = atlas or load_atlas()
    revision = LayoutRevision(
        policy_id=str(atlas.policy["policy_id"]),
        schema_version=str(atlas.policy["schema_version"]),
        template_id=atlas.template_id,
    )
    for item in items:
        item_no = str(item.get("item_no") or "")
        carton = cartons.get(item_no)
        if carton is None:
            revision.issues.append(
                {
                    "severity": "error",
                    "code": "carton_dimensions_missing",
                    "message": f"{item_no or '(unnamed item)'} has no carton dimensions; skipped.",
                    "item_no": item_no or None,
                }
            )
            continue
        plan, issues = plan_item(atlas, item, carton, gutter_mm=gutter_mm)
        if plan is None:
            revision.issues.extend(issues)
            continue
        revision.plans.append(plan)
    return revision


def render_adaptive_marks(
    project_dir: Path,
    items: list[dict[str, Any]],
    cartons: dict[str, CartonFacts],
    output_path: Path,
    *,
    mode: str,
    resolve_image: Callable[[dict[str, Any]], Path | None],
    warnings: list[dict[str, Any]],
    gutter_mm: float = DEFAULT_GUTTER_MM,
    atlas: Atlas | None = None,
    audit_layers: bool = False,
) -> dict[str, Any]:
    """Render, independently audit the exact bytes, then publish atomically.

    ``audit_layers`` is retained for caller compatibility. Element layers are
    now mandatory: without them the final-artifact checks cannot be performed.
    A failed run is archived for diagnosis and never occupies ``output_path``.
    """
    atlas = atlas or load_atlas()
    revision = build_revision(items, cartons, atlas=atlas, gutter_mm=gutter_mm)
    warnings.extend(as_report_issue(i) for i in revision.issues)
    # Per-plan findings must reach the caller too. A barcode under the GS1
    # distribution magnification, or an ambiguous family choice, is exactly the
    # kind of thing that must not stay buried inside the layout report.
    for plan in revision.plans:
        warnings.extend(as_report_issue(i) for i in plan.issues)

    if not revision.plans or any(i.get("severity") == "error" for i in revision.issues):
        output_path.unlink(missing_ok=True)
        return {
            "pages": 0,
            "layout": "carton_adaptive",
            "layout_revision": revision.to_dict(),
            "status": "no_layout",
        }

    items_by_no = {str(item.get("item_no") or ""): item for item in items}
    from ..layout_audit.audit import audit_revision

    renderer = AdaptiveRenderer(atlas=atlas, audit_layers=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = "preview_layout_audit.json" if mode == "preview" else "layout_audit.json"
    with tempfile.TemporaryDirectory(prefix=".adaptive-", dir=output_path.parent) as temporary:
        staged = Path(temporary) / output_path.name
        try:
            result = renderer.render(
                project_dir, revision.plans, items_by_no, staged,
                mode=mode, resolve_image=resolve_image, warnings=warnings,
                draw_cut_marks=False,
            )
        except _RenderRefused as exc:
            # The renderer refuses to draw a face it cannot carry without
            # clipping or horizontally distorting type - a long Spanish name in
            # the side NAME cell is the live case (2026-09-15). Refusing is the
            # right call, but letting it escape would lose the run's report and
            # surface as a crash instead of a blocking result, so it becomes the
            # same fail-closed outcome a failed audit produces.
            output_path.unlink(missing_ok=True)
            return {
                "pages": 0,
                "layout": "carton_adaptive",
                "layout_revision": revision.to_dict(),
                "status": "render_failed",
                "errors": [
                    as_report_issue(
                        {
                            "severity": "error",
                            "code": "layout_render_failed",
                            "message": f"The chosen layout could not be drawn: {exc}",
                        }
                    )
                ],
            }
        try:
            audit = audit_revision(staged, revision, items_by_no, atlas=atlas)
        except Exception as exc:
            # A crashed/unavailable checker is not a passing check.
            audit = {
                "schema_version": "1.0", "faces": [],
                "summary": {"faces": 0, "errors": 1, "warnings": 0, "info": 0},
                "findings": [{"severity": "error", "code": "layout_audit_unavailable",
                              "message": f"Final PDF audit failed to run: {type(exc).__name__}: {exc}"}],
            }
        audit["source_pdf"] = output_path.name
        audit["pdf_sha256"] = hashlib.sha256(staged.read_bytes()).hexdigest()
        audit["mode"] = mode
        audit["plan_fingerprints"] = revision.to_dict()["plan_fingerprints"]
        failures = [f for f in audit["findings"] if f.get("severity") == "error"]
        audit["status"] = "failed" if failures else "passed"
        warnings.extend(as_report_issue(f) for f in audit["findings"] if f.get("severity") != "error")
        if failures:
            archive = reports_dir / "layout_audit_runs" / f"{mode}-{uuid4().hex}"
            archive.mkdir(parents=True)
            staged.replace(archive / "rejected.pdf")
            audit["rejected_pdf"] = str((archive / "rejected.pdf").relative_to(project_dir)).replace("\\", "/")
            (archive / "layout_revision.json").write_text(
                json.dumps(revision.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            output_path.unlink(missing_ok=True)
            result["status"] = "layout_audit_failed"
            result["pages"] = 0
            result["errors"] = [as_report_issue(f) for f in failures]
        else:
            # The audited PDF is the delivered PDF; there is no second render.
            staged.replace(output_path)
            result["status"] = "rendered"
        encoded = json.dumps(audit, ensure_ascii=False, indent=2)
        (reports_dir / report_name).write_text(encoded, encoding="utf-8")
        if failures:
            (archive / "layout_audit.json").write_text(encoded, encoding="utf-8")
        result["layout_audit"] = audit
    result["layout_revision"] = revision.to_dict()
    return result


__all__ = ["build_revision", "plan_item", "render_adaptive_marks"]
