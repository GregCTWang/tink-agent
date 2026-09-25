"""Push-to-talk dictation: serial knob (preferred) or audio fallback."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .fx_button import tracks_audio_grey_buzz

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
    """Voice onset; optional knob-held gate when serial is linked."""

    onset_rms: float
    onset_min_ms: int
    sample_rate: int
    block_size: int
    onset_window_samples: int

    _state: str = field(default="idle", init=False)
    _onset_blocks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        block_ms = 1000.0 * self.block_size / self.sample_rate
        self._onset_min_blocks = max(1, int(round(self.onset_min_ms / block_ms)))
        self._block_ms = block_ms

    @property
    def active(self) -> bool:
        return self._state == "active"

    def reset(self) -> None:
        self._state = "idle"
        self._onset_blocks = 0

    def activate(self) -> None:
        self._state = "active"
        self._onset_blocks = 0

    def process(
        self,
        block: np.ndarray,
        *,
        button_active: bool,
        knob_held: bool,
        require_knob: bool,
    ) -> str | None:
        onset_rms = block_rms(block, 0, min(self.onset_window_samples, block.size))

        if self._state == "idle":
            if button_active:
                self._onset_blocks = 0
                return None
            if require_knob and not knob_held:
                self._onset_blocks = 0
                return None
            if onset_rms >= self.onset_rms:
                self._onset_blocks += 1
                if self._onset_blocks >= self._onset_min_blocks:
                    self._state = "active"
                    self._onset_blocks = 0
                    return "start"
            else:
                self._onset_blocks = 0
            return None
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
        knob_held: bool,
        knob_source: str,
    ) -> None:
        if not self.enabled:
            return
        loud = rms >= self.rms_min
        if not session_active and not loud and not button_active and slot is None:
            return
        m = metrics or {}
        sq = m.get("square_score", "-")
        self._write(
            f"block rms={rms:.1f} knob={int(knob_held)} src={knob_source} "
            f"btn={int(button_active)} slot={slot or '-'} "
            f"sim={m.get('similarity', '-')} sq={sq} session={int(session_active)}"
        )

    def log_serial(self, line: str) -> None:
        self._write(f"serial {line}")

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
        monotonic_fn: Callable[[], float] | None = None,
    ):
        self.config = config
        self.router = router
        self._frontmost_fn = frontmost_fn
        self._dispatch = dispatch or (lambda fn: fn())
        self._delay = delay_fn or (lambda _ms, fn: fn())
        self._on_event = on_event or (lambda _k, _p: None)
        self.logger = logger
        self.debug_logger = debug_logger or DictationDebugLogger(False)
        self._mono = monotonic_fn or time.monotonic
        self._gate = self._make_gate()
        self._profile: dict | None = None
        self._serial_linked = False
        self.knob_held = False
        self._skip_release_send = False
        self._idle_ms = 0.0
        self._last_activity_mono = self._mono()
        self._grey_armed_at: float | None = None
        self._grey_audio_window_s = 0.3
        self._speech_detected = False
        self._knob_start_generation = 0
        self._session_toggle_tap_at: float | None = None
        self._audio_button_dead_until = 0.0
        self._audio_session_cooldown_until = 0.0
        self._grey_buzz_blocks = 0
        self._grey_hold_cancelled = False
        self._grey_burst_consumed = False

    @property
    def knob_source(self) -> str:
        if self._serial_linked and getattr(self.config, "knob_serial_enabled", True):
            return "serial"
        return "audio_fallback"

    def uses_audio_button_end(self) -> bool:
        return self.knob_source == "audio_fallback"

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
        )

    def reload_config(self) -> None:
        if self._gate.active:
            self.force_release("config_reload")
        self._gate = self._make_gate()
        self.debug_logger.set_enabled(getattr(self.config, "dictation_debug_log", False))
        if getattr(self.config, "dictation_debug_path", ""):
            self.debug_logger.path = Path(self.config.dictation_debug_path)

    def set_serial_link(self, connected: bool, detail: str = "") -> None:
        self._serial_linked = connected
        src = self.knob_source
        msg = f"knob_source={src} link={int(connected)} {detail}".strip()
        self._log_knob(msg)
        if not connected:
            self.knob_held = False

    @property
    def is_active(self) -> bool:
        return self._gate.active

    def _cancel_knob_start_debounce(self) -> None:
        self._knob_start_generation += 1

    def _reset_grey_buzz(self) -> None:
        self._grey_buzz_blocks = 0
        self._grey_hold_cancelled = False
        self._grey_burst_consumed = False

    def _schedule_knob_start(self) -> None:
        if self.knob_source != "serial":
            return
        if self._mono() < self._audio_session_cooldown_until:
            return
        if self._mono() < self._audio_button_dead_until:
            return
        debounce_ms = int(getattr(self.config, "knob_start_debounce_ms", 150))
        gen = self._knob_start_generation + 1
        self._knob_start_generation = gen

        def fire() -> None:
            if gen != self._knob_start_generation:
                return
            if not self.knob_held or self._gate.active:
                return
            self._speech_detected = False
            self._skip_release_send = False
            self._begin_session()

        self._delay(debounce_ms, fire)

    def on_serial_line(self, token: str) -> None:
        if token == "HB":
            return
        if token == "K1":
            self.knob_held = True
            self._idle_ms = 0.0
            self._last_activity_mono = self._mono()
            self._log_knob("knob_down")
            if self.knob_source == "serial" and not self._gate.active:
                self._schedule_knob_start()
            return
        if token == "K0":
            self.knob_held = False
            self._log_knob("knob_up")
            self._cancel_knob_start_debounce()
            if self._skip_release_send:
                self._skip_release_send = False
                return
            if self._gate.active:
                self._gate.reset()
                if self._speech_detected:
                    action = getattr(self.config, "dictation_release_action", "enter")
                    self._end_session(reason="release_send", post_action=action)
                else:
                    self._end_session(reason="release_nospeech", post_action=None)
            return
        if token == "G0":
            self._log_knob("grey_press")
            if self._gate.active and self.knob_held:
                self._skip_release_send = True
                self._gate.reset()
                self._end_session(reason="grey_cancel", post_action=None, cancelled=True)
            else:
                self._grey_armed_at = self._mono()
            return

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
        src = self.knob_source
        if src == "audio_fallback":
            self.knob_held = False
        require_knob = src == "serial"
        self.debug_logger.rms_min = getattr(self.config, "dictation_debug_rms_min", 500.0)
        m = dict(tone_metrics or {})
        if "square_score" not in m and m.get("similarity") is not None:
            m.setdefault("square_score", m.get("square_score", "-"))
        self.debug_logger.log_block(
            session_active=self._gate.active,
            rms=full_rms,
            button_active=button_active,
            slot=slot,
            metrics=m,
            knob_held=self.knob_held,
            knob_source=src,
        )
        if full_rms >= self.config.dictation_onset_rms:
            self._last_activity_mono = self._mono()
            self._idle_ms = 0.0
            if self._gate.active:
                self._speech_detected = True

        if src == "audio_fallback":
            self._observe_audio_grey_buzz(
                button_active,
                tone_metrics=m or {},
                slot=slot,
            )
        elif src != "serial":
            transition = self._gate.process(
                block,
                button_active=button_active,
                knob_held=self.knob_held,
                require_knob=require_knob,
            )
            if transition == "start":
                self._skip_release_send = False
                self._speech_detected = False
                self._begin_session()

        if self._gate.active and src == "serial":
            block_ms = self._gate._block_ms
            if self.knob_held or full_rms >= self.config.dictation_onset_rms:
                self._idle_ms = 0.0
            else:
                self._idle_ms += block_ms
                cap = getattr(self.config, "dictation_idle_cap_ms", 180000)
                if self._idle_ms >= cap:
                    self._gate.reset()
                    self._end_session(reason="idle_cap", post_action=None)

        if self._gate.active and src == "audio_fallback":
            block_ms = self._gate._block_ms
            if full_rms >= self.config.dictation_onset_rms:
                self._idle_ms = 0.0
            else:
                self._idle_ms += block_ms
                cap = int(getattr(self.config, "dictation_audio_idle_cap_ms", 60000))
                if self._idle_ms >= cap:
                    self._gate.reset()
                    self._end_session(reason="idle_cap", post_action=None)

    def _mark_audio_button_cooldown(self) -> None:
        post_ms = int(getattr(self.config, "dictation_audio_post_button_ms", 900))
        self._audio_button_dead_until = self._mono() + post_ms / 1000.0

    def _observe_audio_grey_buzz(
        self,
        button_active: bool,
        *,
        tone_metrics: dict,
        slot: int | None,
    ) -> None:
        """Track live slot-1 buzz for hold-cancel; ignore knob/clicks/voice onset."""
        if self._mono() < self._audio_button_dead_until:
            if not button_active:
                self._reset_grey_buzz()
            return
        if slot is not None and slot != 1:
            return
        if tracks_audio_grey_buzz(tone_metrics, self.config):
            self._grey_buzz_blocks += 1
            block_ms = self._gate._block_ms
            dur_ms = self._grey_buzz_blocks * block_ms
            hold_min = int(getattr(self.config, "dictation_grey_hold_min_ms", 700))
            if (
                self._gate.active
                and dur_ms >= hold_min
                and not self._grey_hold_cancelled
            ):
                self._grey_hold_cancelled = True
                self._grey_burst_consumed = True
                self._skip_release_send = True
                self._gate.reset()
                self.debug_logger.log_detection(
                    f"slot=1 dur_ms={dur_ms:.0f} grey_hold_cancel=1"
                )
                self._end_session(
                    reason="grey_hold_cancel",
                    post_action=None,
                    cancelled=True,
                )
                self._mark_audio_button_cooldown()
        elif not button_active and not self._grey_hold_cancelled:
            self._grey_buzz_blocks = 0

    def handle_audio_grey(self, slot: int, detection: dict | None = None) -> bool:
        """Grey sample-light-1 only: tap start/send, hold cancel, consume tail."""
        if self.knob_source != "audio_fallback" or slot != 1:
            return False
        det = detection or {}
        dur_ms = float(det.get("duration_ms") or 0.0)
        tap_max = int(getattr(self.config, "dictation_grey_tap_max_ms", 350))
        hold_min = int(getattr(self.config, "dictation_grey_hold_min_ms", 700))
        self.debug_logger.log_detection(
            f"slot=1 dur_ms={dur_ms:.0f} sim={det.get('similarity', '-')} audio_grey=1"
        )
        if self._grey_burst_consumed or self._grey_hold_cancelled:
            self._mark_audio_button_cooldown()
            self._reset_grey_buzz()
            return True
        if dur_ms >= hold_min:
            self._log_dictation("grey_hold_idle ignored")
            self._mark_audio_button_cooldown()
            self._reset_grey_buzz()
            return True
        if dur_ms > tap_max:
            self._log_dictation(f"grey_ambiguous dur_ms={dur_ms:.0f}")
            self._mark_audio_button_cooldown()
            self._reset_grey_buzz()
            return True
        self._mark_audio_button_cooldown()
        if not self._gate.active:
            self._speech_detected = False
            self._skip_release_send = False
            self._begin_session()
            self._reset_grey_buzz()
            return True
        self._gate.reset()
        action = getattr(self.config, "dictation_release_action", "enter")
        self._end_session(reason="grey_send", post_action=action)
        self._reset_grey_buzz()
        return True

    def try_serial_grey_audio_action(self, slot: int) -> bool:
        """True if serial G0 recently armed and this audio slot may fire (outside session)."""
        if self.knob_source != "serial" or self._gate.active:
            return False
        if self._grey_armed_at is None:
            return False
        if self._mono() - self._grey_armed_at > self._grey_audio_window_s:
            self._grey_armed_at = None
            return False
        self._grey_armed_at = None
        return True

    def handle_button_end(self, slot: int, detection: dict | None = None) -> bool:
        if self.knob_source == "audio_fallback":
            return False
        if not self.uses_audio_button_end():
            return False
        if not self._gate.active or slot is None or slot > 4:
            return False
        det = detection or {}
        self.debug_logger.log_detection(
            f"slot={slot} sim={det.get('similarity', '-')} ended_session=1 fallback=1"
        )
        action = end_action_for_slot(self.config, slot, self.router)
        self._gate.reset()
        self._end_session(reason=f"fallback_button_slot_{slot}", post_action=action)
        return True

    def force_release(self, reason: str = "cleanup") -> None:
        if not self.router.has_held_keys() and not self._gate.active:
            return
        self._cancel_knob_start_debounce()
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

    def _log_knob(self, detail: str) -> None:
        if self.logger is not None:
            self.logger.knob(self._front(), detail)
        self.debug_logger.log_session(f"knob {detail}")

    @staticmethod
    def _keys_label(keys: list) -> str:
        return "+".join(str(k) for k in keys)

    def _log_keys(self, detail: str) -> None:
        front = self._front()
        if self.logger is not None and hasattr(self.logger, "keys"):
            self.logger.keys(front, detail)
        elif self.logger is not None:
            self.logger.dictation(front, f"keys {detail}")

    def _begin_session(self) -> None:
        if (
            self.knob_source == "audio_fallback"
            and self._mono() < self._audio_session_cooldown_until
        ):
            self._log_dictation("start skipped reason=audio_cooldown")
            return
        front = self._front()
        profile = match_profile(front, self.config.dictation_profiles or [])
        if profile is None:
            self._gate.reset()
            self._on_event("dictation", "no_profile")
            self._log_dictation(f"start skipped reason=no_profile front={front[:40]}")
            return
        self._gate.activate()
        self._profile = profile
        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        match_name = profile.get("match", "")
        self._on_event("dictation", ("start", match_name, mode))
        self._log_dictation(
            f"start profile={match_name} mode={mode} knob_source={self.knob_source}")
        label = self._keys_label(keys)

        def start_keys() -> None:
            if mode == "hold":
                self.router.dictation_press(keys)
                self._log_keys(f"{label} down (start)")
            else:
                self.router.dictation_tap(keys)
                self._session_toggle_tap_at = self._mono()
                self._log_keys(f"{label} (start)")

        self._dispatch(start_keys)

    def _end_session(
        self,
        *,
        reason: str,
        post_action: str | None,
        cancelled: bool = False,
    ) -> None:
        profile = self._profile
        self._profile = None
        self._idle_ms = 0.0
        if profile is None:
            self.router.dictation_release_all()
            return

        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        delay_ms = int(profile.get("send_delay_ms", self.config.dictation_send_delay_ms))
        match_name = profile.get("match", "")
        label = self._keys_label(keys)
        min_gap_ms = int(getattr(self.config, "dictation_min_toggle_gap_ms", 400))

        def stop_keys() -> None:
            if cancelled and (
                profile.get("cancel_escape")
                or (
                    self.knob_source == "audio_fallback"
                    and reason in ("grey_cancel", "grey_hold_cancel")
                )
            ):
                self.router.dictation_tap(["esc"])
                self._log_keys("escape")
            if mode == "hold":
                self.router.dictation_release_all()
                self._log_keys(f"{label} up (stop)")
            else:
                self.router.dictation_tap(keys)
                self._log_keys(f"{label} (stop)")
            self._session_toggle_tap_at = None

        def dispatch_stop() -> None:
            self._dispatch(stop_keys)

        self._on_event("dictation", ("stop", reason))
        self._log_dictation(
            f"stop reason={reason} profile={match_name} knob_source={self.knob_source}")

        if self.knob_source == "audio_fallback":
            cd_ms = int(getattr(self.config, "dictation_audio_session_cooldown_ms", 800))
            self._audio_session_cooldown_until = self._mono() + cd_ms / 1000.0

        if mode == "toggle" and self._session_toggle_tap_at is not None:
            elapsed_ms = (self._mono() - self._session_toggle_tap_at) * 1000.0
            wait_ms = int(max(0, min_gap_ms - elapsed_ms))
            if wait_ms > 0:
                self._delay(wait_ms, dispatch_stop)
            else:
                dispatch_stop()
        else:
            dispatch_stop()

        if post_action and post_action not in ("noop", "unmapped") and not cancelled:
            def post() -> None:
                self.router.fire_named_action(post_action)
                self._log_keys(post_action.replace("_", "+"))

            self._delay(delay_ms, lambda: self._dispatch(post))
