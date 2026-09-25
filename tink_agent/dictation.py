"""Push-to-talk dictation: serial knob (preferred) or audio fallback."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .ax_restore import (
    CancelRestoreEngine,
    FocusSnapshot,
    SnapshotAttempt,
    is_safe_snapshot_target,
)

DEFAULT_PROFILES: list[dict] = [
    {
        "match": "Grok Bot",
        "mode": "toggle",
        "keys": ["cmd", "d"],
        "auto_send": True,
        "send_delay_ms": 200,
        "restore_on_cancel": False,
        "cancel_method": "stop_then_undo",
        "undo_delay_ms": 1500,
        "cancel_sequence": "escape_then_release",
    },
    {
        "match": "Claude",
        "mode": "toggle",
        "keys": ["cmd", "d"],
        "auto_send": True,
        "send_delay_ms": 200,
        "restore_on_cancel": False,
        "cancel_method": "stop_then_undo",
        "undo_delay_ms": 1500,
        "cancel_sequence": "escape_then_release",
    },
    {
        "match": "Cursor",
        "mode": "hold",
        "keys": ["ctrl", "m"],
        "auto_send": True,
        "send_delay_ms": 200,
        "restore_on_cancel": False,
        "cancel_method": "stop_then_undo",
        "undo_delay_ms": 1500,
        "cancel_sequence": "escape_then_release",
    },
]


def resolve_cancel_method(profile: dict) -> str:
    raw = profile.get("cancel_method")
    if raw:
        return str(raw).strip().lower()
    if profile.get("restore_on_cancel"):
        return "restore"
    return "escape"

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
        self._write(
            f"block rms={rms:.1f} knob={int(knob_held)} src={knob_source} "
            f"btn={int(button_active)} slot={slot or '-'} "
            f"sim={m.get('similarity', '-')} session={int(session_active)}"
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
        ax_port_factory: Callable[[object, Callable], object | None] | None = None,
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
        self._cancel_snapshot: FocusSnapshot | None = None
        self._ax_port_factory = ax_port_factory
        self._ax_port = None
        self._session_gen = 0

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

    def _schedule_knob_start(self) -> None:
        if self.knob_source != "serial":
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
        require_knob = src == "serial"
        self.debug_logger.rms_min = getattr(self.config, "dictation_debug_rms_min", 500.0)
        self.debug_logger.log_block(
            session_active=self._gate.active,
            rms=full_rms,
            button_active=button_active,
            slot=slot,
            metrics=tone_metrics,
            knob_held=self.knob_held,
            knob_source=src,
        )
        if full_rms >= self.config.dictation_onset_rms:
            self._last_activity_mono = self._mono()
            self._idle_ms = 0.0
            if self._gate.active:
                self._speech_detected = True

        if src == "serial":
            pass
        else:
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

    def _log_restore(self, detail: str) -> None:
        front = self._front()
        if self.logger is not None and hasattr(self.logger, "restore"):
            self.logger.restore(front, detail)
        elif self.logger is not None:
            self.logger.dictation(front, f"restore {detail}")

    def _front_matches_session(self, session_front: str) -> bool:
        if not session_front:
            return True
        cur = (self._front() or "").lower()
        exp = session_front.lower()
        return exp in cur or cur in exp

    def _get_ax_port(self):
        if self._ax_port is not None:
            return self._ax_port
        if self._ax_port_factory is None:
            return None
        self._ax_port = self._ax_port_factory(self.router, self._dispatch)
        return self._ax_port

    def _snapshot_for_cancel(self, profile: dict) -> None:
        self._cancel_snapshot = None
        if resolve_cancel_method(profile) != "restore":
            return
        port = self._get_ax_port()
        if port is None:
            self._log_restore("restore skipped reason=ax_unavailable")
            return
        attempt = port.snapshot_focused()
        raw = attempt.snapshot if isinstance(attempt, SnapshotAttempt) else attempt
        fail_detail = attempt.failure if isinstance(attempt, SnapshotAttempt) else ""
        if raw is None:
            suffix = f" {fail_detail}".rstrip()
            self._log_restore(f"restore skipped reason=unreadable{suffix}")
            return
        match_name = str(profile.get("match") or "")
        if not is_safe_snapshot_target(raw, match_name):
            self._log_restore(
                f"restore skipped reason=unsafe_target role={raw.role} subrole={raw.subrole}"
            )
            return
        self._cancel_snapshot = raw
        vlen = len(raw.value) if raw.value is not None else 0
        if not raw.value_readable:
            self._log_restore(f"snapshot len=- role={raw.role} subrole={raw.subrole} readable=0")
        else:
            self._log_restore(
                f"snapshot len={vlen} role={raw.role} subrole={raw.subrole}"
            )

    def _schedule_cancel_restore(self, profile: dict, snap: FocusSnapshot) -> None:
        port = self._get_ax_port()
        if port is None:
            return
        match_name = str(profile.get("match") or "")
        timeout_ms = int(getattr(self.config, "cancel_restore_timeout_ms", 4000))
        settle_ms = int(getattr(self.config, "cancel_restore_settle_ms", 400))

        def run() -> None:
            engine = CancelRestoreEngine(
                port,
                timeout_ms=timeout_ms,
                settle_ms=settle_ms,
            )
            engine.restore_after_cancel(snap, match_name, self._log_restore)

        self._delay(350, lambda: self._dispatch(run))

    def _schedule_cancel_fallback(self, profile: dict, *, stop_delay_ms: int = 0) -> None:
        fallback = str(profile.get("cancel_fallback") or "none").strip().lower()
        if fallback in ("", "none"):
            return
        if fallback == "undo":
            delay_ms = int(
                profile.get("cancel_fallback_delay_ms")
                or getattr(self.config, "cancel_fallback_delay_ms", 1500)
            )
            total = stop_delay_ms + delay_ms

            def undo() -> None:
                self.router.dictation_tap(["cmd", "z"])
                self._log_keys("cmd+z (cancel_fallback)")

            self._delay(total, lambda: self._dispatch(undo))

    def _begin_session(self) -> None:
        front = self._front()
        profile = match_profile(front, self.config.dictation_profiles or [])
        if profile is None:
            self._gate.reset()
            self._on_event("dictation", "no_profile")
            self._log_dictation(f"start skipped reason=no_profile front={front[:40]}")
            return
        self._session_gen += 1
        self._gate.activate()
        self._profile = profile
        mode = profile.get("mode", "toggle")
        keys = list(profile.get("keys") or [])
        match_name = profile.get("match", "")
        self._on_event("dictation", ("start", match_name, mode))
        self._log_dictation(
            f"start profile={match_name} mode={mode} knob_source={self.knob_source}")
        label = self._keys_label(keys)
        self._snapshot_for_cancel(profile)

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
        restore_snap = self._cancel_snapshot
        self._cancel_snapshot = None
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
        cancel_method = resolve_cancel_method(profile) if cancelled else ""

        def stop_keys() -> None:
            if cancelled and cancel_method == "stop" and profile.get("cancel_escape"):
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
            f"stop reason={reason} profile={match_name} knob_source={self.knob_source}"
        )

        if cancelled and cancel_method == "escape":
            self._session_toggle_tap_at = None

            def do_escape_cancel() -> None:
                self.router.dictation_grey_cancel_escape(
                    profile,
                    self.config,
                    log_fn=self._log_keys,
                )

            self._dispatch(do_escape_cancel)
            return

        if cancelled and cancel_method == "stop_then_undo":
            self._session_toggle_tap_at = None
            had_speech = self._speech_detected
            session_front = self._front()
            gen = self._session_gen
            undo_ms = int(profile.get("undo_delay_ms", 1500))

            def stop_phase() -> None:
                self.router.dictation_cancel_stop(profile, self._log_keys)

            def undo_phase() -> None:
                if not had_speech:
                    self._log_keys("undo skipped reason=no_speech")
                    return
                if gen != self._session_gen:
                    self._log_keys("undo skipped reason=new_session")
                    return
                if not self._front_matches_session(session_front):
                    self._log_keys("undo skipped reason=app_changed")
                    return
                self.router.dictation_cancel_undo(self._log_keys)

            self._dispatch(stop_phase)
            self._delay(undo_ms, lambda: self._dispatch(undo_phase))
            return

        stop_delay_ms = 0
        if mode == "toggle" and self._session_toggle_tap_at is not None and not cancelled:
            elapsed_ms = (self._mono() - self._session_toggle_tap_at) * 1000.0
            wait_ms = int(max(0, min_gap_ms - elapsed_ms))
            stop_delay_ms = wait_ms
            if wait_ms > 0:
                self._delay(wait_ms, dispatch_stop)
            else:
                dispatch_stop()
        else:
            dispatch_stop()

        if cancelled and cancel_method == "restore" and restore_snap is not None:
            self._schedule_cancel_restore(profile, restore_snap)

        if cancelled and cancel_method == "restore":
            fallback = str(profile.get("cancel_fallback") or "none").strip().lower()
            if fallback == "undo":
                self._schedule_cancel_fallback(profile, stop_delay_ms=stop_delay_ms)

        if post_action and post_action not in ("noop", "unmapped") and not cancelled:
            def post() -> None:
                self.router.fire_named_action(post_action)
                self._log_keys(post_action.replace("_", "+"))

            self._delay(delay_ms, lambda: self._dispatch(post))
