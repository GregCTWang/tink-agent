"""Frontmost-app identity, ignoring tink-agent / Python itself."""
from __future__ import annotations

_SELF_MARKERS = (
    "org.python.python",
    "python.org",
    "tink-agent",
    "tink agent",
    "tink_agent",
)


def is_self_app(front: str) -> bool:
    f = (front or "").lower()
    return any(m in f for m in _SELF_MARKERS)


class FrontmostTracker:
    """Return last non-self frontmost app when Python/tink-agent steals focus."""

    def __init__(self, raw_fn):
        self._raw = raw_fn
        self._last_real = ""

    def __call__(self) -> str:
        try:
            cur = self._raw() or ""
        except Exception:  # noqa: BLE001
            cur = ""
        if cur.strip() and not is_self_app(cur):
            self._last_real = cur
            return cur
        return self._last_real
