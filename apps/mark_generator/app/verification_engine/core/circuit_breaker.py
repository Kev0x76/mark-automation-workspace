from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class CircuitBreaker:
    max_seconds: float = 45.0
    max_errors: int = 3
    disabled: set[str] = field(default_factory=set)
    errors: dict[str, int] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def run(self, key: str, fn: Callable[[], T], fallback: T) -> T:
        if key in self.disabled:
            self.events.append(f"{key}:SKIPPED_BY_CIRCUIT_BREAKER")
            return fallback
        start = time.monotonic()
        try:
            result = fn()
        except Exception as exc:
            self._record_error(key, f"{type(exc).__name__}: {exc}")
            return fallback
        elapsed = time.monotonic() - start
        if elapsed > self.max_seconds:
            self.disable(key, f"timeout {elapsed:.1f}s")
        return result

    def _record_error(self, key: str, message: str) -> None:
        count = self.errors.get(key, 0) + 1
        self.errors[key] = count
        self.events.append(f"{key}:ERROR:{message}")
        if count >= self.max_errors:
            self.disable(key, f"{count} consecutive errors")

    def disable(self, key: str, reason: str) -> None:
        self.disabled.add(key)
        self.events.append(f"{key}:DISABLED:{reason}")
