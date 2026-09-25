"""Optional activity log: voice transcripts, the app they were typed into, and
button actions. Off by default.

Optimized for the low event rate of a person talking / pressing buttons:
- Does nothing (no file opened, no I/O) while disabled.
- Keeps a single line-buffered append handle open instead of reopening per write.
- One compact tab-separated line per event: <iso-time>\t<kind>\t<app>\t<detail>.
Thread-safe: actions are logged from the audio thread, transcripts from the
transcription worker thread.
"""
from __future__ import annotations
import threading
import time
from pathlib import Path

DEFAULT_LOG = Path.home() / "Library" / "Logs" / "TinkAgent-activity.log"


def _clean(s) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", " ")


class ActivityLogger:
    def __init__(self, enabled: bool = False, path=None, now_fn=None):
        self.enabled = bool(enabled)
        self.path = Path(path) if path else DEFAULT_LOG
        self._lock = threading.Lock()
        self._fh = None
        self._now = now_fn or (lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self.enabled = bool(value)
            if not self.enabled and self._fh is not None:
                self._fh.close()
                self._fh = None

    def _write(self, kind: str, app, detail) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self.enabled:
                return
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
            self._fh.write(
                f"{self._now()}\t{kind}\t{_clean(app or '-')}\t{_clean(detail)}\n")

    def transcript(self, text: str, app=None) -> None:
        self._write("voice", app, text)

    def action(self, slot: int, action: str, app=None) -> None:
        self._write("action", app, f"slot{slot}:{action}")

    def blocked(self, what, app=None) -> None:
        self._write("blocked", app, what)

    def dictation(self, app, detail: str) -> None:
        self._write("dictation", app, detail)

    def tone(self, slot: int, app=None, detail: str = "") -> None:
        self._write("tone", app, f"slot{slot}{(' ' + detail) if detail else ''}")

    def knob(self, app, detail: str) -> None:
        self._write("knob", app, detail)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
