"""Thread-safe API key pools for generative lineart providers.

First principles:
- Concurrency only helps when each call can use an independent quota unit (key).
- Keys never appear in API responses or logs.
- 429 / auth failures cool a key briefly instead of burning the whole pool.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _KeyState:
    key: str
    in_use: int = 0
    cool_until: float = 0.0
    uses: int = 0
    failures: int = 0


@dataclass
class ApiKeyLease:
    provider: str
    key: str
    slot: int


class ApiKeyPool:
    def __init__(self, provider: str, keys: list[str], *, cooldown_seconds: float = 20.0) -> None:
        self.provider = provider
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._keys = [_KeyState(key=key) for key in keys if key]

    @property
    def size(self) -> int:
        return len(self._keys)

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            return {
                "provider": self.provider,
                "key_count": len(self._keys),
                "available": sum(1 for state in self._keys if state.in_use == 0 and state.cool_until <= now),
                "in_use": sum(state.in_use for state in self._keys),
                "cooling": sum(1 for state in self._keys if state.cool_until > now),
            }

    def acquire(self, *, timeout_seconds: float = 120.0) -> ApiKeyLease:
        if not self._keys:
            raise RuntimeError(f"No API keys configured for provider {self.provider}")
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while True:
            with self._lock:
                now = time.monotonic()
                candidates = [
                    (index, state)
                    for index, state in enumerate(self._keys)
                    if state.cool_until <= now
                ]
                if candidates:
                    # Prefer free keys; among free, least used; then least concurrent.
                    candidates.sort(key=lambda item: (item[1].in_use, item[1].uses, item[0]))
                    index, state = candidates[0]
                    state.in_use += 1
                    state.uses += 1
                    return ApiKeyLease(provider=self.provider, key=state.key, slot=index)
                wait_for = min((state.cool_until for state in self._keys), default=now + 0.5) - now
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for an API key for {self.provider}")
            time.sleep(max(0.05, min(0.5, wait_for)))

    def release(self, lease: ApiKeyLease | None, *, rate_limited: bool = False, hard_fail: bool = False) -> None:
        if lease is None:
            return
        with self._lock:
            if lease.slot < 0 or lease.slot >= len(self._keys):
                return
            state = self._keys[lease.slot]
            state.in_use = max(0, state.in_use - 1)
            if rate_limited or hard_fail:
                state.failures += 1
                cool = self.cooldown_seconds * (3 if hard_fail else 1)
                state.cool_until = max(state.cool_until, time.monotonic() + cool)


_POOLS: dict[str, ApiKeyPool] = {}
_POOLS_LOCK = threading.Lock()
_POOL_SIGNATURES: dict[str, tuple[str, ...]] = {}


def _split_keys(*values: str | None) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for part in value.replace("\n", ",").replace(";", ",").split(","):
            key = part.strip()
            if not key or key in seen:
                continue
            if any(char.isspace() for char in key) or "=" in key:
                continue
            seen.add(key)
            found.append(key)
    return found


def reset_key_pools() -> None:
    with _POOLS_LOCK:
        _POOLS.clear()
        _POOL_SIGNATURES.clear()


def get_api_key_pool(provider: str, keys: list[str], *, cooldown_seconds: float = 20.0) -> ApiKeyPool:
    signature = tuple(keys)
    with _POOLS_LOCK:
        existing = _POOLS.get(provider)
        if existing is not None and _POOL_SIGNATURES.get(provider) == signature:
            return existing
        pool = ApiKeyPool(provider, keys, cooldown_seconds=cooldown_seconds)
        _POOLS[provider] = pool
        _POOL_SIGNATURES[provider] = signature
        return pool
