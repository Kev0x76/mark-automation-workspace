from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from .models import CellEvidence


def ai_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def enrich_cells_with_ai(evidences: list[CellEvidence]) -> list[CellEvidence]:
    """AI fallback 入口。

    第一阶段保持严格 gated：没有 API key 时不调用；有 key 时也只处理 needs_ai 的 cell。
    为了避免在未确认成本前产生调用，这里先返回原证据并标记需要 AI。
    后续可在这里接 Claude Vision 的 per-cell JSON 抽取。
    """
    if not ai_available():
        return evidences
    # Phase 1: no automatic paid calls until prompt and cost policy are confirmed.
    return evidences


def ai_stub_payload(evidence: CellEvidence) -> dict[str, Any]:
    data = asdict(evidence)
    return json.loads(json.dumps(data, default=str, ensure_ascii=False))
