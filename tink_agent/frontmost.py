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


def default_frontmost_pid() -> int:
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return 0
        return int(app.processIdentifier())
    except Exception:  # noqa: BLE001
        return 0


class FrontmostTracker:
    """Return last non-self frontmost app when Python/tink-agent steals focus."""

    def __init__(self, raw_fn, raw_pid_fn=None, on_frontmost_pid=None):
        self._raw = raw_fn
        self._raw_pid = raw_pid_fn or default_frontmost_pid
        self._on_frontmost_pid = on_frontmost_pid or (lambda _pid: None)
        self._last_real = ""
        self.last_pid = 0

    def __call__(self) -> str:
        try:
            cur = self._raw() or ""
        except Exception:  # noqa: BLE001
            cur = ""
        pid = 0
        try:
            pid = int(self._raw_pid() or 0)
        except Exception:  # noqa: BLE001
            pid = 0
        if cur.strip() and not is_self_app(cur):
            self._last_real = cur
            if pid and pid != self.last_pid:
                self.last_pid = pid
                self._on_frontmost_pid(pid)
            return cur
        return self._last_real
