from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CORE_FIELDS = {
    "item_number",
    "barcode",
    "barcode_scan",
    "barcode_text",
    "name_cn",
    "name_it",
    "name_es",
    "name_en",
    "packing_qty",
    "line_art",
}
MAX_ITEMS_IN_RESPONSE = 500


def build_run_payload(run_id: str, run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    item_rows = _read_csv(run_dir / "v5_item_summary.csv")
    sorted_items = sorted(item_rows, key=_item_sort_key)
    details = _details_by_item(run_dir / "v5_result.json")
    image_manifest = _read_json(run_dir / "mark_images_manifest.json")
    images = _images_by_item(image_manifest, run_id)

    events = _page_events(summary)
    if len(sorted_items) > MAX_ITEMS_IN_RESPONSE:
        events.append(f"RESPONSE_TRUNCATED:{len(sorted_items)}>{MAX_ITEMS_IN_RESPONSE}")
    if summary.get("contract_items", 0) and not summary.get("mark_images_count", 0):
        events.append(
            "NO_MARK_IMAGES:未能定位到可裁剪的单个唛头；通常是图片型 PDF、未扫到条码或 OCR 不可用"
        )

    payload_items = []
    for row in sorted_items[:MAX_ITEMS_IN_RESPONSE]:
        item_number = row.get("item_number", "")
        image_info = images.get(item_number, {})
        payload_items.append(
            {
                **row,
                "fields": details.get(item_number, []),
                "image_url": image_info.get("url", ""),
                "image_status": image_info.get("status", ""),
                "image_message": image_info.get("message", ""),
            }
        )

    return {
        "run_id": run_id,
        "summary": summary,
        "status_counts": _count_status(item_rows),
        "evidence_status_counts": _count_evidence_status(image_manifest),
        "items": payload_items,
        "downloads": download_links(run_id),
        "events": events,
    }


def download_links(run_id: str) -> dict[str, str]:
    base = f"/api/verification-runs/{run_id}"
    return {
        "summary": f"{base}/downloads/summary",
        "details": f"{base}/downloads/details",
        "json": f"{base}/downloads/json",
        "task": f"{base}/downloads/task",
        "visual": f"{base}/downloads/visual",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _details_by_item(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for result in _read_json(path).get("results", []):
        item_number = result.get("item_number", "")
        fields = []
        for field in result.get("fields", []):
            if field.get("field") not in CORE_FIELDS:
                continue
            fields.append(
                {
                    "field": field.get("field", ""),
                    "contract_value": field.get("contract_value", ""),
                    "mark_value": field.get("mark_value", ""),
                    "status": field.get("status", ""),
                    "note": field.get("note", ""),
                    "source": field.get("source", ""),
                }
            )
        out[item_number] = fields
    return out


def _images_by_item(data: dict[str, Any], run_id: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in data.get("items", []):
        item_number = item.get("item_number", "")
        if item_number and item.get("status") in {"OK", "PAGE_FALLBACK"}:
            out[item_number] = {
                "url": f"/api/verification-runs/{run_id}/images/{item_number}",
                "status": item.get("status", ""),
                "message": item.get("message", ""),
            }
    return out


def _count_evidence_status(data: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in data.get("items", []):
        status = str(item.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _count_status(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("overall_status", "") or "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _item_sort_key(row: dict[str, str]) -> tuple[int, str]:
    order = {"BLOCKING_ISSUE": 0, "NEEDS_MANUAL_REVIEW": 1, "OK": 2}
    return (order.get(row.get("overall_status", ""), 9), row.get("item_number", ""))


def _page_events(summary: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for page in summary.get("pages", []):
        if page.get("has_text_layer") is False:
            events.append(f"PAGE_NO_TEXT_LAYER:p{page.get('page')}")
        if not page.get("barcodes"):
            events.append(f"PAGE_NO_BARCODES:p{page.get('page')}")
        events.extend(str(event) for event in page.get("events", []))
    return events
