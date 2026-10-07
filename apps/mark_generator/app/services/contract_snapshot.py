"""Canonical contract facts shared by generation and verification.

The mutable ``items.normalized.json`` file gains product-region and lineart
state as a project moves through production.  It therefore must not be the
verification truth.  This module creates and validates a content-addressed
snapshot of the contract fields produced by the one authorised Excel parse.
The digest detects ordinary drift; it is not a cryptographic signature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = "1.0.0"
PARSER_NAME = "mark_generator.excel_contract"
PARSER_VERSION = "2.2.0"


class ContractSnapshotError(ValueError):
    pass


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_item(
    item: Mapping[str, Any],
    *,
    source_row: int,
    columns: Mapping[str, str],
) -> dict[str, Any]:
    """Copy only contract-owned fields and their source-cell coordinates."""

    fields = ("item_no", "barcode", "cn_name", "it_name", "es_name", "en_name", "qty")
    source_cells = {
        field: f"{columns[field]}{source_row}"
        for field in fields
        if isinstance(columns.get(field), str) and columns[field]
    }
    return {
        "item_number": str(item.get("item_no") or ""),
        "barcode": str(item.get("barcode") or ""),
        "names": {
            "cn": str(item.get("cn_name") or ""),
            "it": str(item.get("it_name") or ""),
            "es": str(item.get("es_name") or ""),
            "en": str(item.get("en_name") or ""),
            "pt": "",
        },
        "packing_qty": item.get("qty"),
        "source": {
            "row": int(source_row),
            "cells": source_cells,
        },
    }


def build_contract_snapshot(
    *,
    project_id: str,
    contract_path: Path | str,
    extraction_report: Mapping[str, Any],
    items: list[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = Path(contract_path)
    table: dict[str, Any] = {
        "sheet": str(extraction_report.get("selected_sheet") or ""),
        "header_start_row": int(
            extraction_report.get("header_start_row")
            or extraction_report.get("header_row")
            or 0
        ),
        "header_row": int(extraction_report.get("header_row") or 0),
        "columns": dict(extraction_report.get("columns") or {}),
    }
    contract_item_region = extraction_report.get("contract_item_region")
    if isinstance(contract_item_region, Mapping):
        table["contract_item_region"] = {
            "method": str(contract_item_region.get("method") or ""),
            "first_product_row": int(contract_item_region.get("first_product_row") or 0),
            "last_product_row": int(contract_item_region.get("last_product_row") or 0),
            "item_rows": [int(row) for row in contract_item_region.get("item_rows") or []],
            "spans": [
                {
                    "start_row": int(span.get("start_row") or 0),
                    "end_row": int(span.get("end_row") or 0),
                    "item_count": int(span.get("item_count") or 0),
                }
                for span in contract_item_region.get("spans") or []
                if isinstance(span, Mapping)
            ],
            "ignored_non_product_rows_count": int(
                contract_item_region.get("ignored_non_product_rows_count") or 0
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "project_id": project_id,
        "source_contract": {
            "path": str(extraction_report.get("source_contract") or ""),
            "format": str(extraction_report.get("contract_format") or contract.suffix.lstrip(".")),
            "original_filename": str(extraction_report.get("original_filename") or ""),
            "sha256": file_sha256(contract),
        },
        "parser": {
            "name": PARSER_NAME,
            "version": PARSER_VERSION,
        },
        "table": table,
        "items": [dict(item) for item in items],
    }
    payload["integrity"] = {
        "payload_sha256": snapshot_payload_sha256(payload),
    }
    return payload


def snapshot_payload_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "integrity"}
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_contract_snapshot(
    payload: Mapping[str, Any],
    *,
    contract_path: Path | str | None = None,
) -> None:
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ContractSnapshotError("Unsupported contract snapshot schema version.")
    if not isinstance(payload.get("items"), list):
        raise ContractSnapshotError("Contract snapshot items must be a list.")
    expected_payload_hash = (payload.get("integrity") or {}).get("payload_sha256")
    actual_payload_hash = snapshot_payload_sha256(payload)
    if not expected_payload_hash or expected_payload_hash != actual_payload_hash:
        raise ContractSnapshotError("Contract snapshot content hash does not match.")
    if contract_path is not None:
        expected_contract_hash = (payload.get("source_contract") or {}).get("sha256")
        actual_contract_hash = file_sha256(contract_path)
        if not expected_contract_hash or expected_contract_hash != actual_contract_hash:
            raise ContractSnapshotError("Contract snapshot does not match the source contract file.")


def read_contract_snapshot(
    path: Path | str,
    *,
    contract_path: Path | str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractSnapshotError("Contract snapshot is missing or unreadable.") from exc
    if not isinstance(payload, dict):
        raise ContractSnapshotError("Contract snapshot root must be an object.")
    validate_contract_snapshot(payload, contract_path=contract_path)
    return payload
