from __future__ import annotations
from pathlib import Path
import atexit
import subprocess
import sys
import time
import rumps
from .config import Config, DEFAULT_PATH
from .audio import AudioCapture, DeviceNotFound, reinitialize as audio_reinitialize
from .detector import ToneDetector, VoiceGate
from .transcribe import Transcriber, resolve_stt
from .actions import ActionRouter
from .engine import Engine, _default_frontmost
from .dictation import DictationController, DictationDebugLogger
from .activity_log import ActivityLogger, DEFAULT_LOG
from . import launchagent

_REPO = Path(__file__).resolve().parent.parent
_ASSETS = _REPO / "assets"
ICON_IDLE = str(_ASSETS / "tink_icon.png")
ICON_ACTIVE = str(_ASSETS / "tink_icon_active.png")


def _main_thread_dispatch(fn):
    """Schedule fn to run on the AppKit main thread/run loop. Keyboard actions
    fire from the audio callback and transcription worker threads, but pynput's
    macOS layout query (HIToolbox TSM) must run on the main thread or the
    process traps. callAfter is thread-safe to call from any thread."""
    from PyObjCTools import AppHelper
    AppHelper.callAfter(fn)


def _main_thread_delay(ms: int, fn):
    """Run fn on the main thread after ms milliseconds (dictation send delay)."""
    import threading

    def _fire():
        _main_thread_dispatch(fn)

    threading.Timer(max(0, ms) / 1000.0, _fire).start()


class TinkAgentApp(rumps.App):
    def __init__(self):
        super().__init__("Tink Agent", icon=ICON_IDLE, template=True)
        self.config = Config.load()
        self.engine = self._build_engine()
        self.capture = None
        self._settings = None  # lazily-created native Settings window controller
        self._button_actions = None  # lazily-created Button Actions window

        # State stashed from background threads (atomic attribute writes); the
        # main-thread timer renders it, keeping all UI mutation on the main thread.
        self._last_heard = "—"
        self._last_action = "—"
        self._status = "stopped"
        self._icon_active = False

        # Verify-step plumbing: the audio thread stashes the last detected tone;
        # the onboarding window reads these (never touches the audio thread).
        self._last_tone_slot = None
        self._last_tone_at = 0.0
        self._onboarding = None       # lazily-created OnboardingController
        self._onboarding_shown = False

        # Slim menu: a quick status line + Enabled safety toggle; everything else
        # lives in the Settings window. (rumps adds Quit automatically.)
        self.status_item = rumps.MenuItem("status: stopped")
        self.enabled_item = rumps.MenuItem("Enabled", callback=self._menu_toggle_enabled)
        self.enabled_item.state = self.config.enabled
        self.menu = [
            self.status_item,
            None,
            self.enabled_item,
            rumps.MenuItem("Settings…", callback=self.open_settings),
            None,
        ]

        # Start audio AFTER the run loop is up (opening the mic can block/prompt;
        # doing it in __init__ would delay run() and the status item appearing).
        self._autostart_done = False
        self._timer = rumps.Timer(self._tick, 0.2)
        self._timer.start()
        atexit.register(self._cleanup_dictation)

    def _cleanup_dictation(self):
        try:
            self.engine.release_dictation("quit")
        except Exception:  # noqa: BLE001
            pass

    def _build_engine(self) -> Engine:
        c = self.config
        det = ToneDetector(c.tones, c.tone_rms_min, c.tone_dominance_min,
                           c.tone_debounce_ms, c.sample_rate,
                           tonality_min=c.tone_tonality_min)
        vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                       c.min_utterance_ms, c.sample_rate, c.block_size,
                       c.max_utterance_ms)
        tr = self._make_transcriber()
        router = ActionRouter(c.slot_actions, dispatch=_main_thread_dispatch)
        self.activity_log = ActivityLogger(
            enabled=c.log_activity, path=(c.log_path or None))
        self._dictation_debug = DictationDebugLogger(
            enabled=c.dictation_debug_log,
            path=(c.dictation_debug_path or None),
        )
        dictation = DictationController(
            c, router, frontmost_fn=_default_frontmost,
            dispatch=_main_thread_dispatch,
            delay_fn=_main_thread_delay,
            on_event=self._on_event,
            logger=self.activity_log,
            debug_logger=self._dictation_debug,
        )
        return Engine(c, det, vg, tr, router, on_event=self._on_event,
                      logger=self.activity_log, dictation=dictation)

    def _make_transcriber(self) -> Transcriber:
        cmd, mode = resolve_stt(self.config)
        return Transcriber(cmd, mode)

    @property
    def log_path(self):
        return self.config.log_path or DEFAULT_LOG

    # --- called from background threads: only stash state, never touch UI ---
    def _on_event(self, kind, payload):
        if kind == "transcript":
            self._last_heard = str(payload)[:40]
        elif kind == "tone":
            self._last_action = f"slot {payload}"
            self._last_tone_slot = payload
            self._last_tone_at = time.monotonic()
        elif kind == "blocked":
            self._last_action = f"blocked (wrong app): {payload}"
        elif kind == "dictation":
            self._last_action = f"dictation: {payload}"
        elif kind == "error":
            self._status = f"error: {str(payload)[:34]}"

    def _on_audio_status(self, status):
        # Audio callback thread: PortAudio reported input overflow / a device
        # drop (which truncates speech). Only stash + write to the run log here;
        # never touch AppKit off the main thread.
        import sys
        print(f"[audio] stream status: {status}", file=sys.stderr, flush=True)
        self.engine.release_dictation("audio_glitch")
        self._status = f"audio glitch: {str(status)[:24]}"

    # --- main-thread rendering ---
    def _tick(self, _):
        if not self._autostart_done:
            self._autostart_done = True
            self.start_listening(None)
        if not self.config.onboarding_done and not self._onboarding_shown:
            self._onboarding_shown = True
            self.open_onboarding(None)
        capturing = self.capture is not None and self.engine.is_capturing
        if capturing != self._icon_active:
            self._icon_active = capturing
            self.icon = ICON_ACTIVE if capturing else ICON_IDLE
        listening = self.capture is not None and self.capture.is_running
        base = "listening" if listening else "stopped"
        shown = self._status if self._status.startswith("error") else base
        if not self.engine.enabled:
            shown = f"{base} (disabled)"
        self.status_item.title = f"status: {shown}{' • capturing' if capturing else ''}"
        if self._settings is not None and self.window_is_visible():
            self._settings.refresh()

    def window_is_visible(self) -> bool:
        try:
            return bool(self._settings.window.isVisible())
        except Exception:  # noqa: BLE001
            return False

    # --- shared setters (called by both the menu and the Settings window) ---
    def set_enabled(self, value: bool):
        if not value:
            self.engine.release_dictation("disabled")
        self.engine.enabled = bool(value)
        self.config.enabled = bool(value)
        self.config.save()
        self.enabled_item.state = bool(value)

    def set_dictation_enabled(self, value: bool):
        self.config.dictation_enabled = bool(value)
        self.config.save()
        if self.engine.dictation is not None:
            if not value:
                self.engine.release_dictation("dictation_off")
            self.engine.dictation.reload_config()

    def set_listening(self, value: bool):
        if value:
            self.start_listening(None)
        else:
            self.stop_listening(None)

    def set_start_at_login(self, value: bool) -> bool:
        try:
            launchagent.set_run_at_login(
                bool(value), launchagent.current_python(), launchagent.repo_root())
        except Exception as e:  # noqa: BLE001
            rumps.alert("Could not update Start at login", str(e))
            return False
        self.config.start_at_login = bool(value)
        self.config.save()
        return True

    def toggle_target(self, identifier: str, on: bool):
        identifier = str(identifier or "").strip()
        if not identifier:
            return
        ids = [i for i in (self.config.target_apps or []) if i]
        if on and identifier not in ids:
            ids.append(identifier)
        elif not on:
            ids = [i for i in ids if i != identifier]
        self.config.target_apps = ids
        self.config.save()

    def set_stt_backend(self, name: str):
        # When switching to custom, seed the editable command from whatever was
        # effective so the user starts from a working template.
        if name == "custom" and not self.config.stt_command:
            cmd, mode = resolve_stt(self.config)
            self.config.stt_command = cmd
            self.config.stt_output = mode
        self.config.stt_backend = name
        self.config.save()
        self.engine.transcriber = self._make_transcriber()

    def set_log_activity(self, value: bool):
        self.config.log_activity = bool(value)
        self.config.save()
        self.activity_log.set_enabled(bool(value))

    def set_device_name(self, name: str):
        self.config.device_name = name
        self.config.save()
        # Apply immediately: re-open the input stream on the new device.
        if self.capture and self.capture.is_running:
            self.stop_listening(None)
            self.start_listening(None)

    def set_slot_action(self, slot, action_id: str):
        """Assign the action a slot fires. Persists to config and updates the
        running router so it takes effect without restarting capture."""
        slot = int(slot)
        self.config.slot_actions[slot] = action_id
        self.config.save()
        self.engine.router.slot_actions[slot] = action_id

    def restore_default_slot_actions(self):
        """Reset all 8 slot→action mappings to the factory defaults. Persists and
        applies to the running router immediately."""
        defaults = dict(Config().slot_actions)
        self.config.slot_actions = defaults
        self.config.save()
        self.engine.router.slot_actions = dict(defaults)

    def rescan_devices(self):
        """Re-enumerate audio devices for the Settings dropdown. PortAudio caches
        its device list, so we stop capture, re-init PortAudio, then restart so
        hot-plugged/unplugged devices are reflected."""
        running = bool(self.capture and self.capture.is_running)
        if running:
            self.stop_listening(None)
        try:
            audio_reinitialize()
        except Exception as e:  # noqa: BLE001 — a failed rescan must not crash
            print(f"[audio] device rescan failed: {e}", file=sys.stderr, flush=True)
        if running:
            self.start_listening(None)

    def open_log(self, _):
        from pathlib import Path as _P
        p = _P(self.log_path)
        if not p.exists():
            rumps.alert("No activity log yet",
                        f"Nothing has been logged. It will be created at:\n{p}")
            return
        subprocess.run(["open", str(p)])

    # --- menu callbacks ---
    def _menu_toggle_enabled(self, sender):
        self.set_enabled(not bool(sender.state))

    def open_settings(self, _):
        if self._settings is None:
            from .settings_ui import SettingsController
            self._settings = SettingsController.alloc().initWithApp_(self)
        self._settings.show()

    def open_button_actions(self, _=None):
        if self._button_actions is None:
            from .button_actions import ButtonActionsController
            self._button_actions = ButtonActionsController.alloc().initWithApp_(self)
        self._button_actions.show()

    def open_onboarding(self, _=None):
        if self._onboarding is None:
            from .onboarding import OnboardingController
            self._onboarding = OnboardingController.alloc().initWithApp_(self)
        self._onboarding.show()

    def complete_onboarding(self):
        self.config.onboarding_done = True
        self.config.save()

    # --- capture lifecycle ---
    def start_listening(self, _):
        if self.capture and self.capture.is_running:
            return
        try:
            self.capture = AudioCapture(
                self.config.device_name, self.config.sample_rate,
                self.config.block_size, on_block=self.engine.handle_block,
                on_status=self._on_audio_status)
            self.capture.start()
            self._status = "listening"
        except DeviceNotFound:
            self.capture = None
            self._status = "error: mic not found"
        except Exception as e:  # noqa: BLE001
            self.capture = None
            self._status = f"error: {str(e)[:30]}"

    def stop_listening(self, _):
        self.engine.release_dictation("stop_listening")
        if self.capture:
            self.capture.stop()
            self.capture = None
        self._status = "stopped"

    def open_config(self, _):
        subprocess.run(["open", str(DEFAULT_PATH)])


def _set_regular_policy():
    """Run as a regular app: Dock icon + Cmd+Tab entry, and windows persist
    normally when switching apps (kept alongside the menu-bar item)."""
    try:
        from AppKit import NSApplication
        # NSApplicationActivationPolicyRegular == 0
        NSApplication.sharedApplication().setActivationPolicy_(0)
    except Exception:  # noqa: BLE001
        pass


def main():
    app = TinkAgentApp()
    _set_regular_policy()
    app.run()
