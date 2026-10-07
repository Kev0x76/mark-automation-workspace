"""L2 style conformance: bands measured from real factory marks.

Four quantities, kept together on purpose. Any one of them alone can be improved
by making the layout worse, which the project has already measured once: spacing
elements out cut the largest empty rectangle from 18.5% to 12.2% and looked
visibly worse. The four pull against each other -

=========================  ==========================
layout goes wrong          the quantity that reacts
=========================  ==========================
too empty                  ink coverage falls
one big hole               largest empty rectangle rises
everything in one corner   spread falls, balance rises
elements scattered         alignment falls
=========================  ==========================

- so an optimiser cannot satisfy all four by cheating on the arrangement.

They are acceptance gates, not objectives. Being inside a band does not make a
layout good; being outside one means a person should look. Out-of-band is
therefore a review trigger and never a silent failure, and whether it should
ever block a release is an owner decision this module does not take.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .raster import PanelMetrics

METRICS = ("ink_coverage", "largest_empty_rect", "spread", "balance")


@dataclass(frozen=True)
class Band:
    """An accepted range for one quantity on one kind of face.

    Two ranges, not one, because a small sample with an outlier needs both.
    Front-face ink coverage on the factory corpus runs 29.0-35.3% across
    eighteen panels and then jumps to 41.7% on the nineteenth. A min-to-max band
    accepts anything up to 41.7% and would have said nothing about the 40.6% the
    2026-09-02 samples measured - which sits above eighteen of the nineteen.
    ``p10``/``p90`` carry that information; ``low``/``high`` stay the gate.
    """

    metric: str
    low: float
    high: float
    n: int
    median: float
    p10: float = 0.0
    p90: float = 0.0

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def is_typical(self, value: float) -> bool:
        if self.p90 <= self.p10:
            return True
        return self.p10 <= value <= self.p90

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": round(self.low, 5),
            "high": round(self.high, 5),
            "p10": round(self.p10, 5),
            "median": round(self.median, 5),
            "p90": round(self.p90, 5),
            "n": self.n,
        }


def calibrate(samples: Sequence[PanelMetrics], *, margin: float = 0.0) -> dict[str, Band]:
    """Full observed range per metric.

    Min-to-max rather than a percentile band: the sample is small enough that
    trimming it would throw away real factory practice, and a band that rejects
    marks the factory actually ships is worse than a wide one. ``margin`` widens
    the range by a fraction of its own width when a caller wants slack.
    """
    out: dict[str, Band] = {}
    for metric in METRICS:
        values = np.array([getattr(sample, metric) for sample in samples], dtype=float)
        if values.size == 0:
            continue
        low, high = float(values.min()), float(values.max())
        pad = (high - low) * margin
        out[metric] = Band(
            metric=metric,
            low=low - pad,
            high=high + pad,
            n=int(values.size),
            median=float(np.median(values)),
            p10=float(np.percentile(values, 10)),
            p90=float(np.percentile(values, 90)),
        )
    return out


def load_bands(path: str | Path) -> dict[str, dict[str, Band]]:
    """``{face: {metric: Band}}`` from a calibration file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Band]] = {}
    for face, metrics in payload.get("bands", {}).items():
        out[face] = {
            metric: Band(
                metric=metric,
                low=float(entry["low"]),
                high=float(entry["high"]),
                n=int(entry.get("n", 0)),
                median=float(entry.get("median", 0.0)),
                p10=float(entry.get("p10", 0.0)),
                p90=float(entry.get("p90", 0.0)),
            )
            for metric, entry in metrics.items()
        }
    return out


def dump_bands(
    bands: dict[str, dict[str, Band]], *, provenance: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "policy_id": "kmt_style_v1_layout_bands_v1",
        "provenance": provenance,
        "bands": {
            face: {metric: band.to_dict() for metric, band in sorted(metrics.items())}
            for face, metrics in sorted(bands.items())
        },
    }


def check_metrics(
    metrics: PanelMetrics,
    bands: dict[str, Band],
    *,
    face: str,
    item_no: str | None = None,
) -> list[dict[str, Any]]:
    """Compare one face against its band. Out-of-band is a warning, by design."""
    findings: list[dict[str, Any]] = []
    for name, band in sorted(bands.items()):
        value = float(getattr(metrics, name))
        if not band.contains(value):
            direction = "below" if value < band.low else "above"
            findings.append(
                {
                    "severity": "warning",
                    "code": "style_metric_outside_factory_band",
                    "message": (
                        f"{name} on the {face} face is {value:.3f}, {direction} the "
                        f"{band.low:.3f}-{band.high:.3f} range measured on {band.n} factory "
                        "panels. Not a defect on its own - it means a person should look."
                    ),
                    "face": face,
                    "metric": name,
                    "value": round(value, 5),
                    "band": band.to_dict(),
                    "item_no": item_no,
                }
            )
        elif not band.is_typical(value):
            side = "below" if value < band.p10 else "above"
            findings.append(
                {
                    "severity": "info",
                    "code": "style_metric_unusual_for_factory",
                    "message": (
                        f"{name} on the {face} face is {value:.3f} - inside the factory range "
                        f"but {side} the {band.p10:.3f}-{band.p90:.3f} middle where {band.n} "
                        "factory panels sit. Worth a look when several of these land together."
                    ),
                    "face": face,
                    "metric": name,
                    "value": round(value, 5),
                    "band": band.to_dict(),
                    "item_no": item_no,
                }
            )
    return findings


def summarise_samples(samples: Iterable[PanelMetrics]) -> dict[str, dict[str, float]]:
    """Median / min / max per metric, for a calibration report."""
    collected = list(samples)
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = np.array([getattr(sample, metric) for sample in collected], dtype=float)
        if values.size == 0:
            continue
        out[metric] = {
            "n": int(values.size),
            "min": round(float(values.min()), 5),
            "median": round(float(np.median(values)), 5),
            "max": round(float(values.max()), 5),
        }
    return out


__all__ = [
    "METRICS",
    "Band",
    "calibrate",
    "check_metrics",
    "dump_bands",
    "load_bands",
    "summarise_samples",
]
