"""Deterministic category evidence for future shared-lineart workflows.

This module deliberately stops at category grouping.  An equal normalized
Chinese name is evidence that items belong to the same business category; it
is not, by itself, proof that every visible structural difference is limited
to size, colour, or pattern.  The current item contract also has no variant
evidence or shared-lineart provenance model, so this service never copies an
asset and never marks an item approved.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


GROUPING_METHOD = "exact_normalized_cn_name_v1"
GROUPING_SCOPE = "category_evidence_only"
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_INTERNAL_SPACE = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])\s+"
    r"(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])"
)
_ZERO_WIDTH = str.maketrans("", "", "\u200b\u200c\u200d\ufeff")


def normalize_cn_name(value: object) -> str:
    """Return a conservative core name without deleting semantic tokens.

    Only Unicode compatibility and whitespace differences are normalized.
    Size, colour, pattern, gender, shape, and style words are intentionally
    preserved; silently stripping them would create unsafe fuzzy groups.
    """

    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).translate(_ZERO_WIDTH)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Spreadsheet authors often insert spaces between Chinese characters for
    # presentation.  Removing only CJK-to-CJK spaces does not drop any word.
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = _CJK_INTERNAL_SPACE.sub("", normalized)
    return normalized


def stable_group_id(normalized_cn_name: str) -> str:
    """Build an order-independent, content-addressed category identifier."""

    digest = hashlib.sha256(normalized_cn_name.encode("utf-8")).hexdigest()[:16]
    return f"cn-{digest}"


def build_lineart_groups_report(normalized_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable report from ``items.normalized.json`` data.

    The caller may persist the returned value as
    ``reports/lineart_groups.json``.  This function performs no filesystem
    writes so it can be safely wired into the project workflow later.
    """

    items = normalized_payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("normalized payload items must be a list")
    project_id = str(normalized_payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("normalized payload project_id must be a non-empty string")
    return analyze_lineart_groups(items, project_id=project_id)


def analyze_lineart_groups(
    items: Iterable[Mapping[str, Any]],
    *,
    project_id: object = None,
) -> dict[str, Any]:
    """Group valid items by exact normalized Chinese name.

    Multi-item groups are category candidates only.  Their variant evidence is
    explicitly reported as missing because the current normalized item schema
    does not retain size/colour/pattern versus structural-difference evidence.
    """

    materialized = list(items)
    item_numbers = [_item_number(item) for item in materialized]
    duplicate_numbers = {
        item_no for item_no, count in Counter(item_numbers).items() if item_no and count > 1
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review_items: list[dict[str, str]] = []
    for item in materialized:
        item_no = _item_number(item)
        raw_name = item.get("cn_name")
        normalized_cn_name = normalize_cn_name(raw_name)
        reason = _review_reason(
            item_no=item_no,
            normalized_cn_name=normalized_cn_name,
            duplicate_numbers=duplicate_numbers,
        )
        if reason:
            review_items.append(
                {
                    "item_no": item_no,
                    "cn_name": str(raw_name or "").strip(),
                    "reason": reason,
                }
            )
            continue
        grouped[normalized_cn_name].append(
            {
                "item_no": item_no,
                "cn_name": str(raw_name).strip(),
                "normalization_changed": str(raw_name).strip() != normalized_cn_name,
            }
        )

    groups = []
    for normalized_cn_name in sorted(grouped):
        members = sorted(grouped[normalized_cn_name], key=lambda member: member["item_no"])
        is_singleton = len(members) == 1
        groups.append(
            {
                "group_id": stable_group_id(normalized_cn_name),
                "normalized_cn_name": normalized_cn_name,
                "members": members,
                "evidence": {
                    "method": GROUPING_METHOD,
                    "member_count": len(members),
                    "original_cn_names": sorted({member["cn_name"] for member in members}),
                    "normalization": [
                        "unicode_nfkc",
                        "remove_zero_width_characters",
                        "collapse_whitespace",
                        "remove_cjk_internal_whitespace",
                    ],
                },
                "variant_evidence_status": (
                    "not_applicable_singleton" if is_singleton else "missing_from_item_contract"
                ),
                "reuse_decision": (
                    "not_applicable_singleton" if is_singleton else "needs_variant_evidence"
                ),
                "auto_lineart_reuse": False,
            }
        )

    review_items.sort(key=lambda item: (item["item_no"], item["reason"]))
    multi_item_groups = sum(1 for group in groups if len(group["members"]) > 1)
    return {
        "schema_version": "1.0.0",
        "project_id": str(project_id).strip() if project_id is not None else None,
        "method": GROUPING_METHOD,
        "scope": GROUPING_SCOPE,
        "policy": {
            "auto_category_grouping": True,
            "auto_lineart_reuse": False,
            "reuse_requires": [
                "same deterministic category group",
                "evidence that differences are limited to size, color, or pattern",
                "shared-lineart provenance that does not misrepresent a target product region",
            ],
            "fuzzy_name_matching": False,
        },
        "summary": {
            "input_items": len(materialized),
            "grouped_items": sum(len(group["members"]) for group in groups),
            "category_groups": len(groups),
            "multi_item_groups": multi_item_groups,
            "review_items": len(review_items),
            "auto_reuse_groups": 0,
        },
        "groups": groups,
        "review_items": review_items,
    }


def _item_number(item: Mapping[str, Any]) -> str:
    value = item.get("item_no")
    return str(value).strip() if value is not None else ""


def _review_reason(
    *,
    item_no: str,
    normalized_cn_name: str,
    duplicate_numbers: set[str],
) -> str | None:
    if not item_no:
        return "missing_item_no"
    if item_no in duplicate_numbers:
        return "duplicate_item_no"
    if not normalized_cn_name:
        return "missing_cn_name"
    if len(_CJK_PATTERN.findall(normalized_cn_name)) < 2:
        return "insufficient_chinese_category_evidence"
    return None
