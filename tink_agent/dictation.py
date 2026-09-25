"""Push-to-talk dictation: voice-onset sessions that drive per-app shortcuts.

Independent of VoiceGate / STT — uses its own RMS gates and hangover. Beeps
(tone_active) are ignored for onset/offset so loud button tones do not start
or end sessions. Release uses an adaptive threshold between the knob-released
noise floor and the knob-held-silent floor when auto-calibration is enabled.
"""
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


@dataclass
class KnobFloorCalibrator:
    """Track released vs knob-held-silent RMS floors; release threshold sits between."""

    released_floor: float
    held_floor: float
    auto_floor: bool = True
    fixed_release_rms: float = 35.0
    ema_alpha: float = 0.08

    def release_threshold(self) -> float:
        if not self.auto_floor:
            return self.fixed_release_rms
        mid = (self.released_floor + self.held_floor) / 2.0
        lo = self.released_floor + 1.5
        hi = self.held_floor - 1.5
        if lo >= hi:
            return self.fixed_release_rms
        return max(lo, min(mid, hi))

    def observe_idle(self, rms: float, *, tone_active: bool, onset_rms: float) -> None:
        if not self.auto_floor or tone_active or rms >= onset_rms:
            return
        if rms < 55:
            a = self.ema_alpha
            self.released_floor = (1 - a) * self.released_floor + a * rms

    def observe_active_quiet(self, rms: float, *, tone_active: bool, onset_rms: float) -> None:
        if not self.auto_floor or tone_active:
            return
        if rms < onset_rms:
            a = self.ema_alpha
            self.held_floor = (1 - a) * self.held_floor + a * rms


@dataclass
class DictationOnsetGate:
    """Energy gate for dictation only (not STT). Fast onset via a short RMS window."""

    onset_rms: float
    onset_min_ms: int
    hangover_ms: int
    sample_rate: int
    block_size: int
    onset_window_samples: int
    max_session_ms: int
    floors: KnobFloorCalibrator

    _state: str = field(default="idle", init=False)
    _onset_blocks: int = field(default=0, init=False)
    _hangover_blocks: int = field(default=0, init=False)
    _session_blocks: int = field(default=0, init=False)
    last_stop_reason: str = field(default="", init=False)

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
        self.last_stop_reason = ""

    def process(self, block: np.ndarray, tone_active: bool) -> str | None:
        """Return 'start', 'stop', or 'max' (session cap); None if unchanged."""
        onset_rms = block_rms(block, 0, min(self.onset_window_samples, block.size))
        full_rms = block_rms(block)

        if self._state == "idle":
            self.floors.observe_idle(full_rms, tone_active=tone_active, onset_rms=self.onset_rms)
            if tone_active:
                self._onset_blocks = 0
                return None
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

        # active — beeps must not advance hangover toward release
        if tone_active:
            self._hangover_blocks = 0
        else:
            self.floors.observe_active_quiet(
                full_rms, tone_active=False, onset_rms=self.onset_rms)

        self._session_blocks += 1
        if self._session_blocks >= self._max_session_blocks:
            self._state = "idle"
            self.last_stop_reason = "max"
            return "max"

        release_cutoff = self.floors.release_threshold()
        if tone_active:
            return None

        if full_rms < release_cutoff:
            self._hangover_blocks += 1
            if self._hangover_blocks >= self._hangover_blocks_max:
                self._state = "idle"
                self._hangover_blocks = 0
                self._session_blocks = 0
                self.last_stop_reason = "hangover"
                return "stop"
        else:
            self._hangover_blocks = 0
        return None


class DictationDebugLogger:
    """Optional per-block RMS / tone metrics (separate file, default off)."""

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

    def maybe_log(
        self,
        *,
        session_active: bool,
        rms: float,
        tone_active: bool,
        slot: int | None,
        metrics: dict | None,
        release_threshold: float,
    ) -> None:
        if not self.enabled:
            return
        loud = rms >= self.rms_min
        if not session_active and not loud and not tone_active and slot is None:
            return
        m = metrics or {}
        freq = m.get("best_freq")
        dom = m.get("dominance")
        tonal = m.get("tonality")
        detail = (
            f"rms={rms:.1f} thr={release_threshold:.1f} "
            f"tone={int(bool(tone_active))} slot={slot or '-'} "
            f"freq={freq if freq is not None else '-'} "
            f"dom={dom if dom is not None else '-'} "
            f"tonal={tonal if tonal is not None else '-'}"
        )
        with self._lock:
            if not self.enabled:
                return
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
            self._fh.write(detail + "\n")


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
        self._cancel_requested = False

    def _make_gate(self) -> DictationOnsetGate:
        c = self.config
        window_samples = max(
            1,
            int(c.sample_rate * getattr(c, "dictation_onset_window_ms", 25) / 1000.0),
        )
        floors = KnobFloorCalibrator(
            released_floor=c.dictation_floor_released_rms,
            held_floor=c.dictation_floor_held_rms,
            auto_floor=c.dictation_auto_floor,
            fixed_release_rms=c.dictation_release_rms,
            ema_alpha=c.dictation_floor_ema_alpha,
        )
        return DictationOnsetGate(
            onset_rms=c.dictation_onset_rms,
            onset_min_ms=c.dictation_onset_min_ms,
            hangover_ms=c.dictation_hangover_ms,
            sample_rate=c.sample_rate,
            block_size=c.block_size,
            onset_window_samples=window_samples,
            max_session_ms=c.dictation_max_session_ms,
            floors=floors,
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
        tone_active: bool,
        *,
        slot: int | None = None,
        tone_metrics: dict | None = None,
    ) -> None:
        """Process one audio block; optional tone slot/metrics for debug logging."""
        if not getattr(self.config, "dictation_enabled", True):
            return
        full_rms = block_rms(block)
        thr = self._gate.floors.release_threshold()
        self.debug_logger.rms_min = getattr(
            self.config, "dictation_debug_rms_min", 500.0)
        self.debug_logger.maybe_log(
            session_active=self._gate.active,
            rms=full_rms,
            tone_active=tone_active,
            slot=slot,
            metrics=tone_metrics,
            release_threshold=thr,
        )
        transition = self._gate.process(block, tone_active)
        if transition == "start":
            self._begin_session()
        elif transition == "stop":
            reason = self._gate.last_stop_reason or "hangover"
            self._end_session(send=not self._cancel_requested, reason=reason)
            self._cancel_requested = False
        elif transition == "max":
            self._end_session(send=not self._cancel_requested, reason="max")
            self._cancel_requested = False

    def handle_tone(self, slot: int) -> bool:
        """Return True if the tone was consumed (cancel during session)."""
        if not self._gate.active:
            return False
        cancel_any = getattr(self.config, "dictation_cancel_on_any_tone", True)
        action = self.router.slot_actions.get(slot)
        if cancel_any or action == "dictation_cancel":
            self._cancel_session("cancel" if action == "dictation_cancel" else "tone")
            return True
        return False

    def _cancel_session(self, reason: str) -> None:
        self._cancel_requested = True
        self._gate.reset()
        self._end_session(send=False, cancelled=True, reason=reason)
        self._on_event("dictation", ("cancel", reason))

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

    def _log_dictation(self, detail: str) -> None:
        if self.logger is not None:
            self.logger.dictation(self._front(), detail)

    def _begin_session(self) -> None:
        front = self._front()
        profile = match_profile(front, self.config.dictation_profiles or [])
        if profile is None:
            self._gate.reset()
            self._on_event("dictation", "no_profile")
            self._log_dictation("start skipped reason=no_profile")
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

    def _end_session(
        self,
        *,
        send: bool,
        cancelled: bool = False,
        reason: str = "",
    ) -> None:
        profile = self._profile
        self._profile = None
        stop_reason = reason or ("cancel" if cancelled else "hangover")
        if profile is None:
            self.router.dictation_release_all()
            if stop_reason not in ("cleanup",):
                self._log_dictation(f"stop reason={stop_reason} profile=-")
            return

        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        auto_send = bool(profile.get("auto_send", True))
        delay_ms = int(profile.get("send_delay_ms", self.config.dictation_send_delay_ms))
        match_name = profile.get("match", "")

        def stop_keys() -> None:
            if cancelled and profile.get("cancel_escape"):
                self.router.dictation_tap(["esc"])
            if mode == "hold":
                self.router.dictation_release_all()
            else:
                self.router.dictation_tap(keys)

        self._dispatch(stop_keys)
        self._on_event("dictation", ("stop", cancelled, stop_reason))
        self._log_dictation(
            f"stop reason={stop_reason} profile={match_name} cancelled={int(cancelled)}")

        if send and auto_send and not cancelled:

            def send_enter() -> None:
                self.router.dictation_tap(["enter"])

            self._delay(delay_ms, lambda: self._dispatch(send_enter))
