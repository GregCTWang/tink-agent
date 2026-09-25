"""Push-to-talk dictation: voice onset starts app shortcuts; grey button ends."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_PROFILES: list[dict] = [
    {
        "match": "Grok Bot",
        "mode": "toggle",
        "keys": ["cmd", "d"],
        "auto_send": True,
        "send_delay_ms": 200,
    },
    {
        "match": "Claude",
        "mode": "toggle",
        "keys": ["cmd", "d"],
        "auto_send": True,
        "send_delay_ms": 200,
    },
    {
        "match": "Cursor",
        "mode": "hold",
        "keys": ["ctrl", "m"],
        "auto_send": True,
        "send_delay_ms": 200,
        "cancel_escape": True,
    },
]

DEFAULT_DEBUG_LOG = Path.home() / "Library" / "Logs" / "TinkAgent-dictation-debug.log"


def block_rms(block: np.ndarray, start: int = 0, length: int | None = None) -> float:
    block = np.asarray(block, dtype=np.float64).reshape(-1)
    if length is not None:
        block = block[start : start + length]
    elif start:
        block = block[start:]
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block * block)))


def match_profile(front: str, profiles: list[dict]) -> dict | None:
    f = (front or "").lower()
    if not f:
        return None
    for p in profiles:
        m = str(p.get("match") or "").strip()
        if m and m.lower() in f:
            return p
    return None


def end_action_for_slot(config, slot: int, router) -> str:
    custom = getattr(config, "dictation_end_actions", None) or {}
    if str(slot) in custom:
        return str(custom[str(slot)])
    if slot in custom:
        return str(custom[slot])
    return str(router.slot_actions.get(slot, "noop"))


@dataclass
class DictationOnsetGate:
    """Voice onset only; no silence-based session end."""

    onset_rms: float
    onset_min_ms: int
    sample_rate: int
    block_size: int
    onset_window_samples: int
    max_session_ms: int

    _state: str = field(default="idle", init=False)
    _onset_blocks: int = field(default=0, init=False)
    _session_blocks: int = field(default=0, init=False)
    last_stop_reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        block_ms = 1000.0 * self.block_size / self.sample_rate
        self._onset_min_blocks = max(1, int(round(self.onset_min_ms / block_ms)))
        self._max_session_blocks = max(1, int(round(self.max_session_ms / block_ms)))

    @property
    def active(self) -> bool:
        return self._state == "active"

    def reset(self) -> None:
        self._state = "idle"
        self._onset_blocks = 0
        self._session_blocks = 0
        self.last_stop_reason = ""

    def process(self, block: np.ndarray, button_active: bool) -> str | None:
        onset_rms = block_rms(block, 0, min(self.onset_window_samples, block.size))

        if self._state == "idle":
            if button_active:
                self._onset_blocks = 0
                return None
            if onset_rms >= self.onset_rms:
                self._onset_blocks += 1
                if self._onset_blocks >= self._onset_min_blocks:
                    self._state = "active"
                    self._onset_blocks = 0
                    self._session_blocks = 0
                    return "start"
            else:
                self._onset_blocks = 0
            return None

        self._session_blocks += 1
        if self._session_blocks >= self._max_session_blocks:
            self._state = "idle"
            self.last_stop_reason = "max"
            return "max"
        return None


class DictationDebugLogger:
    def __init__(self, enabled: bool = False, path=None):
        self.enabled = bool(enabled)
        self.path = Path(path) if path else DEFAULT_DEBUG_LOG
        self._lock = threading.Lock()
        self._fh = None
        self.rms_min = 500.0

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self.enabled = bool(value)
            if not self.enabled and self._fh is not None:
                self._fh.close()
                self._fh = None

    def _write(self, line: str) -> None:
        with self._lock:
            if not self.enabled:
                return
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
            self._fh.write(line + "\n")

    def log_block(
        self,
        *,
        session_active: bool,
        rms: float,
        button_active: bool,
        slot: int | None,
        metrics: dict | None,
    ) -> None:
        if not self.enabled:
            return
        loud = rms >= self.rms_min
        if not session_active and not loud and not button_active and slot is None:
            return
        m = metrics or {}
        self._write(
            f"block rms={rms:.1f} btn={int(button_active)} slot={slot or '-'} "
            f"sim={m.get('similarity', '-')} sq={m.get('square_score', '-')} "
            f"session={int(session_active)}"
        )

    def log_detection(self, detail: str) -> None:
        self._write(f"detect {detail}")

    def log_session(self, detail: str) -> None:
        self._write(f"session {detail}")


class DictationController:
    def __init__(
        self,
        config,
        router,
        frontmost_fn: Callable[[], str],
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        delay_fn: Callable[[int, Callable[[], None]], None] | None = None,
        on_event: Callable[[str, object], None] | None = None,
        logger=None,
        debug_logger: DictationDebugLogger | None = None,
    ):
        self.config = config
        self.router = router
        self._frontmost_fn = frontmost_fn
        self._dispatch = dispatch or (lambda fn: fn())
        self._delay = delay_fn or (lambda _ms, fn: fn())
        self._on_event = on_event or (lambda _k, _p: None)
        self.logger = logger
        self.debug_logger = debug_logger or DictationDebugLogger(False)
        self._gate = self._make_gate()
        self._profile: dict | None = None

    def _make_gate(self) -> DictationOnsetGate:
        c = self.config
        window_samples = max(
            1,
            int(c.sample_rate * getattr(c, "dictation_onset_window_ms", 25) / 1000.0),
        )
        return DictationOnsetGate(
            onset_rms=c.dictation_onset_rms,
            onset_min_ms=c.dictation_onset_min_ms,
            sample_rate=c.sample_rate,
            block_size=c.block_size,
            onset_window_samples=window_samples,
            max_session_ms=c.dictation_max_session_ms,
        )

    def reload_config(self) -> None:
        if self._gate.active:
            self.force_release("config_reload")
        self._gate = self._make_gate()
        self.debug_logger.set_enabled(getattr(self.config, "dictation_debug_log", False))
        if getattr(self.config, "dictation_debug_path", ""):
            self.debug_logger.path = Path(self.config.dictation_debug_path)

    @property
    def is_active(self) -> bool:
        return self._gate.active

    def observe_block(
        self,
        block: np.ndarray,
        button_active: bool,
        *,
        slot: int | None = None,
        tone_metrics: dict | None = None,
    ) -> None:
        if not getattr(self.config, "dictation_enabled", True):
            return
        full_rms = block_rms(block)
        self.debug_logger.rms_min = getattr(self.config, "dictation_debug_rms_min", 500.0)
        self.debug_logger.log_block(
            session_active=self._gate.active,
            rms=full_rms,
            button_active=button_active,
            slot=slot,
            metrics=tone_metrics,
        )
        transition = self._gate.process(block, button_active)
        if transition == "start":
            self._begin_session()
        elif transition == "max":
            self._end_session(reason="max", post_action=None)

    def handle_button_end(self, slot: int, detection: dict | None = None) -> bool:
        if not self._gate.active or slot is None or slot > 4:
            return False
        det = detection or {}
        self.debug_logger.log_detection(
            f"slot={slot} sim={det.get('similarity', '-')} rms={det.get('rms', '-')} "
            f"dur_ms={det.get('duration_ms', '-')} ended_session=1"
        )
        action = end_action_for_slot(self.config, slot, self.router)
        self._gate.reset()
        self._end_session(reason=f"button_slot_{slot}", post_action=action)
        return True

    def force_release(self, reason: str = "cleanup") -> None:
        if not self.router.has_held_keys() and not self._gate.active:
            return
        self._gate.reset()
        self._end_session(reason=reason, post_action=None)

    def _front(self) -> str:
        try:
            return self._frontmost_fn() or ""
        except Exception:  # noqa: BLE001
            return ""

    def _log_dictation(self, detail: str) -> None:
        if self.logger is not None:
            self.logger.dictation(self._front(), detail)
        self.debug_logger.log_session(detail)

    def _begin_session(self) -> None:
        front = self._front()
        profile = match_profile(front, self.config.dictation_profiles or [])
        if profile is None:
            self._gate.reset()
            self._on_event("dictation", "no_profile")
            self._log_dictation(f"start skipped reason=no_profile front={front[:40]}")
            return
        self._profile = profile
        mode = profile.get("mode", "toggle")
        match_name = profile.get("match", "")
        self._on_event("dictation", ("start", match_name, mode))
        self._log_dictation(f"start profile={match_name} mode={mode}")
        if mode == "hold":
            self._dispatch(lambda: self.router.dictation_press(list(profile.get("keys") or [])))
        else:
            self._dispatch(lambda: self.router.dictation_tap(list(profile.get("keys") or [])))

    def _end_session(self, *, reason: str, post_action: str | None) -> None:
        profile = self._profile
        self._profile = None
        if profile is None:
            self.router.dictation_release_all()
            return

        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        delay_ms = int(profile.get("send_delay_ms", self.config.dictation_send_delay_ms))
        match_name = profile.get("match", "")

        def stop_keys() -> None:
            if mode == "hold":
                self.router.dictation_release_all()
            else:
                self.router.dictation_tap(keys)

        self._dispatch(stop_keys)
        self._on_event("dictation", ("stop", reason))
        self._log_dictation(f"stop reason={reason} profile={match_name}")

        if post_action and post_action not in ("noop", "unmapped"):
            def post() -> None:
                self.router.fire_named_action(post_action)

            self._delay(delay_ms, lambda: self._dispatch(post))
