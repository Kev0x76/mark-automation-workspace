from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import VerificationResult


@dataclass(frozen=True)
class VisualReportConfig:
    max_items: int = 300
    max_consecutive_errors: int = 2


FIELDS = [
    ("item_number", "货号"),
    ("barcode", "合同条码"),
    ("barcode_scan", "真实扫码"),
    ("barcode_text", "条码数字"),
    ("name_cn", "中文品名"),
    ("name_it", "意文品名"),
    ("name_es", "西文品名"),
    ("name_en", "英文品名"),
    ("packing_qty", "装箱数"),
]


def write_visual_report(
    output_dir: str | Path,
    results: list[VerificationResult],
    crop_manifest: dict[str, Any],
    config: VisualReportConfig = VisualReportConfig(),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report_path = output_dir / "visual_report.html"
    image_by_item = {
        item.get("item_number", ""): item
        for item in crop_manifest.get("items", [])
    }
    events: list[str] = []
    cards: list[str] = []
    consecutive_errors = 0

    for index, result in enumerate(results):
        if index >= config.max_items:
            events.append(f"VISUAL_REPORT_STOPPED:max_items:{config.max_items}")
            break
        try:
            cards.append(_render_item_card(output_dir, result, image_by_item.get(result.item_number)))
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            events.append(f"{result.item_number}:VISUAL_CARD_FAILED:{type(exc).__name__}:{exc}")
            if consecutive_errors >= config.max_consecutive_errors:
                events.append("VISUAL_REPORT_STOPPED:too_many_consecutive_errors")
                break

    html_text = _page_html(results, cards, events)
    report_path.write_text(html_text, encoding="utf-8")
    manifest = {
        "html": str(report_path),
        "cards": len(cards),
        "events": events,
    }
    (output_dir / "visual_report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _render_item_card(output_dir: Path, result: VerificationResult, image_item: dict[str, Any] | None) -> str:
    review = result.review or {}
    evidence_grade = review.get("evidence_grade", {})
    status = _overall_status(result)
    image_html = _image_html(output_dir, image_item)
    rows = "\n".join(_field_row(field, label, result) for field, label in FIELDS)
    return f"""
    <section class="card status-{html.escape(status.lower())}">
      <div class="card-head">
        <div>
          <h2>{html.escape(result.item_number)}</h2>
          <p class="evidence-grade">{html.escape(evidence_grade.get("label", ""))} {html.escape(evidence_grade.get("reason", ""))}</p>
          <p>状态：<strong>{html.escape(status)}</strong> ｜ 关联置信度：{result.association_confidence:.3f}</p>
        </div>
        <div class="cell-id">{html.escape(result.cell_id)}</div>
      </div>
      <div class="card-body">
        <div class="image-panel">{image_html}</div>
        <div class="compare-panel">
          <table>
            <thead>
              <tr>
                <th>字段</th>
                <th>合同</th>
                <th>唛头识别</th>
                <th>状态</th>
                <th>备注</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
          <div class="advice">{html.escape(review.get("suggested_action", ""))}</div>
        </div>
      </div>
    </section>
    """


def _field_row(field: str, label: str, result: VerificationResult) -> str:
    found = next((item for item in result.fields if item.field == field), None)
    if not found:
        return f"<tr><td>{html.escape(label)}</td><td></td><td></td><td></td><td></td></tr>"
    return f"""
      <tr class="field-{html.escape(found.status.lower())}">
        <td>{html.escape(label)}</td>
        <td>{html.escape(found.contract_value)}</td>
        <td>{html.escape(found.mark_value)}</td>
        <td>{html.escape(found.status)}</td>
        <td>{html.escape(found.note)}</td>
      </tr>
    """


def _image_html(output_dir: Path, image_item: dict[str, Any] | None) -> str:
    if not image_item or image_item.get("status") not in {"OK", "PAGE_FALLBACK"} or not image_item.get("image"):
        status = image_item.get("status", "NO_IMAGE") if image_item else "NO_IMAGE"
        return f'<div class="no-image">唛头原图未生成：{html.escape(status)}</div>'
    path = Path(image_item["image"])
    try:
        rel = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    if image_item.get("status") == "PAGE_FALLBACK":
        message = html.escape(image_item.get("message", "未定位到该货号的单个唛头，当前显示整页预览供人工查看。"))
        return f'<div class="fallback-note">{message}</div><img src="{html.escape(rel)}" alt="mark page preview">'
    return f'<img src="{html.escape(rel)}" alt="mark image">'


def _page_html(results: list[VerificationResult], cards: list[str], events: list[str]) -> str:
    counts: dict[str, int] = {}
    for result in results:
        status = _overall_status(result)
        counts[status] = counts.get(status, 0) + 1
    count_text = " ｜ ".join(f"{html.escape(k)}: {v}" for k, v in counts.items())
    events_html = "".join(f"<li>{html.escape(event)}</li>" for event in events)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>唛头视觉核对报告</title>
  <style>
    body {{ margin: 0; background: #f4f6f7; color: #1f2933; font-family: "Microsoft YaHei", Arial, sans-serif; }}
    main {{ width: min(1400px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1, h2, p {{ margin: 0; }}
    header {{ margin-bottom: 18px; }}
    header p {{ margin-top: 8px; color: #5e6b73; }}
    .card {{ background: #fff; border: 1px solid #d8e0e4; margin-bottom: 16px; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 16px; padding: 14px 16px; border-bottom: 1px solid #d8e0e4; }}
    .cell-id {{ color: #66757f; font-size: 13px; text-align: right; overflow-wrap: anywhere; }}
    .evidence-grade {{ margin-top: 6px; color: #365f56; font-size: 13px; font-weight: 700; }}
    .card-body {{ display: grid; grid-template-columns: minmax(360px, 0.95fr) minmax(520px, 1.05fr); gap: 14px; padding: 14px 16px 16px; }}
    .image-panel {{ border: 1px solid #d8e0e4; background: #fafafa; display: grid; place-items: center; min-height: 260px; overflow: auto; }}
    .image-panel img {{ max-width: 100%; height: auto; display: block; }}
    .no-image {{ color: #8a4b08; padding: 24px; text-align: center; }}
    .fallback-note {{ width: 100%; padding: 8px 10px; background: #fff8e8; color: #5f3b05; border-bottom: 1px solid #d8e0e4; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border: 1px solid #d8e0e4; padding: 8px; vertical-align: top; font-size: 13px; overflow-wrap: anywhere; }}
    th {{ background: #eef3f3; }}
    .field-pass td:nth-child(4) {{ color: #18703a; font-weight: 700; }}
    .field-needs_review td:nth-child(4), .field-missing td:nth-child(4) {{ color: #a15c07; font-weight: 700; }}
    .field-mismatch td:nth-child(4) {{ color: #b42318; font-weight: 700; }}
    .advice {{ margin-top: 10px; padding: 10px; background: #f7faf9; border: 1px solid #d8e0e4; color: #314047; }}
    .events {{ margin-top: 20px; background: #fff; border: 1px solid #d8e0e4; padding: 12px 16px; }}
    @media (max-width: 980px) {{ .card-body {{ grid-template-columns: 1fr; }} .card-head {{ display: grid; }} .cell-id {{ text-align: left; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>唛头视觉核对报告</h1>
      <p>{count_text}</p>
    </header>
    {''.join(cards)}
    <section class="events">
      <h2>报告事件</h2>
      <ul>{events_html or '<li>无</li>'}</ul>
    </section>
  </main>
</body>
</html>"""


def _overall_status(result: VerificationResult) -> str:
    if result.error:
        return "NEEDS_MANUAL_REVIEW"
    statuses = {field.status for field in result.fields}
    if "MISMATCH" in statuses:
        return "BLOCKING_ISSUE"
    if "MISSING" in statuses or "NEEDS_REVIEW" in statuses:
        return "NEEDS_MANUAL_REVIEW"
    return "OK"
