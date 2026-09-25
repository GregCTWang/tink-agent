"""Push-to-talk dictation: voice-onset sessions that drive per-app shortcuts.

Independent of VoiceGate / STT — uses its own RMS gates and hangover. Beeps
(tone_active) are ignored for onset/offset so loud button tones do not start
or end sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class DictationOnsetGate:
    """Energy gate for dictation only (not STT). Fast onset via a short RMS window."""

    onset_rms: float
    release_rms: float
    onset_min_ms: int
    hangover_ms: int
    sample_rate: int
    block_size: int
    onset_window_samples: int
    max_session_ms: int

    _state: str = field(default="idle", init=False)
    _onset_blocks: int = field(default=0, init=False)
    _hangover_blocks: int = field(default=0, init=False)
    _session_blocks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        block_ms = 1000.0 * self.block_size / self.sample_rate
        self._onset_min_blocks = max(1, int(round(self.onset_min_ms / block_ms)))
        self._hangover_blocks_max = max(1, int(round(self.hangover_ms / block_ms)))
        self._max_session_blocks = max(1, int(round(self.max_session_ms / block_ms)))

    @property
    def active(self) -> bool:
        return self._state == "active"

    def reset(self) -> None:
        self._state = "idle"
        self._onset_blocks = 0
        self._hangover_blocks = 0
        self._session_blocks = 0

    def process(self, block: np.ndarray, tone_active: bool) -> str | None:
        """Return 'start', 'stop', or 'max' (session cap); None if unchanged."""
        if tone_active:
            # Loud beeps must not open/close dictation sessions.
            if self._state == "idle":
                self._onset_blocks = 0
            elif self._state == "active":
                self._hangover_blocks = 0
            return None

        onset_rms = block_rms(block, 0, min(self.onset_window_samples, block.size))
        full_rms = block_rms(block)

        if self._state == "idle":
            if onset_rms >= self.onset_rms:
                self._onset_blocks += 1
                if self._onset_blocks >= self._onset_min_blocks:
                    self._state = "active"
                    self._onset_blocks = 0
                    self._hangover_blocks = 0
                    self._session_blocks = 0
                    return "start"
            else:
                self._onset_blocks = 0
            return None

        # active
        self._session_blocks += 1
        if self._session_blocks >= self._max_session_blocks:
            self._state = "idle"
            return "max"

        if full_rms < self.release_rms:
            self._hangover_blocks += 1
            if self._hangover_blocks >= self._hangover_blocks_max:
                self._state = "idle"
                self._hangover_blocks = 0
                self._session_blocks = 0
                return "stop"
        else:
            self._hangover_blocks = 0
        return None


class DictationController:
    """Session state + keyboard side-effects via ActionRouter dictation helpers."""

    def __init__(
        self,
        config,
        router,
        frontmost_fn: Callable[[], str],
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        delay_fn: Callable[[int, Callable[[], None]], None] | None = None,
        on_event: Callable[[str, object], None] | None = None,
    ):
        self.config = config
        self.router = router
        self._frontmost_fn = frontmost_fn
        self._dispatch = dispatch or (lambda fn: fn())
        self._delay = delay_fn or (lambda _ms, fn: fn())
        self._on_event = on_event or (lambda _k, _p: None)
        self._gate = self._make_gate()
        self._profile: dict | None = None
        self._cancel_requested = False
        self._send_pending = False

    def _make_gate(self) -> DictationOnsetGate:
        c = self.config
        window_samples = max(
            1,
            int(c.sample_rate * getattr(c, "dictation_onset_window_ms", 25) / 1000.0),
        )
        return DictationOnsetGate(
            onset_rms=c.dictation_onset_rms,
            release_rms=c.dictation_release_rms,
            onset_min_ms=c.dictation_onset_min_ms,
            hangover_ms=c.dictation_hangover_ms,
            sample_rate=c.sample_rate,
            block_size=c.block_size,
            onset_window_samples=window_samples,
            max_session_ms=c.dictation_max_session_ms,
        )

    def reload_config(self) -> None:
        self._gate = self._make_gate()

    @property
    def is_active(self) -> bool:
        return self._gate.active

    def process_block(self, block: np.ndarray, tone_active: bool) -> None:
        if not getattr(self.config, "dictation_enabled", True):
            return
        transition = self._gate.process(block, tone_active)
        if transition == "start":
            self._begin_session()
        elif transition in ("stop", "max"):
            self._end_session(send=not self._cancel_requested)
            self._cancel_requested = False

    def handle_tone(self, slot: int) -> bool:
        """Return True if the tone was consumed (dictation_cancel during session)."""
        action = self.router.slot_actions.get(slot)
        if action != "dictation_cancel" or not self._gate.active:
            return False
        self._cancel_requested = True
        self._gate.reset()
        self._end_session(send=False, cancelled=True)
        self._on_event("dictation", "cancel")
        return True

    def force_release(self, reason: str = "cleanup") -> None:
        if not self.router.has_held_keys() and not self._gate.active:
            return
        self._cancel_requested = True
        self._gate.reset()
        self._end_session(send=False, cancelled=True, reason=reason)

    def _front(self) -> str:
        try:
            return self._frontmost_fn() or ""
        except Exception:  # noqa: BLE001
            return ""

    def _begin_session(self) -> None:
        front = self._front()
        profile = match_profile(front, self.config.dictation_profiles or [])
        if profile is None:
            self._gate.reset()
            self._on_event("dictation", "no_profile")
            return
        self._profile = profile
        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        self._on_event("dictation", ("start", profile.get("match"), mode))
        if mode == "hold":
            self._dispatch(lambda: self.router.dictation_press(keys))
        else:
            self._dispatch(lambda: self.router.dictation_tap(keys))

    def _end_session(
        self,
        *,
        send: bool,
        cancelled: bool = False,
        reason: str = "",
    ) -> None:
        profile = self._profile
        self._profile = None
        if profile is None:
            self.router.dictation_release_all()
            return

        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        auto_send = bool(profile.get("auto_send", True))
        delay_ms = int(profile.get("send_delay_ms", self.config.dictation_send_delay_ms))

        def stop_keys() -> None:
            if cancelled and profile.get("cancel_escape"):
                self.router.dictation_tap(["esc"])
            if mode == "hold":
                self.router.dictation_release_all()
            else:
                self.router.dictation_tap(keys)

        self._dispatch(stop_keys)
        self._on_event("dictation", ("stop", cancelled, reason))

        if send and auto_send and not cancelled:
            self._send_pending = True

            def send_enter() -> None:
                self._send_pending = False
                self.router.dictation_tap(["enter"])

            self._delay(delay_ms, lambda: self._dispatch(send_enter))
