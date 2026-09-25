from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

from .frontmost import FrontmostTracker, is_self_app
from .fx_button import passes_audio_only_button


class Engine:
    def __init__(self, config, button_detector, voicegate, transcriber, router,
                 on_event=None, submit_fn=None, frontmost_fn=None, logger=None,
                 dictation=None):
        self.config = config
        self.button_detector = button_detector
        self.voicegate = voicegate
        self.transcriber = transcriber
        self.router = router
        self.logger = logger
        self.dictation = dictation
        self.enabled = config.enabled
        self._on_event = on_event or (lambda kind, payload: None)
        raw_front = frontmost_fn or _default_frontmost
        self._frontmost = FrontmostTracker(raw_front)
        if submit_fn is not None:
            self._submit = submit_fn
        else:
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._submit = lambda fn: self._pool.submit(fn)

    @property
    def is_capturing(self) -> bool:
        if getattr(self.voicegate, "active", False):
            return True
        d = self.dictation
        return bool(d is not None and d.is_active)

    def _front(self) -> str:
        try:
            return self._frontmost() or ""
        except Exception:  # noqa: BLE001
            return ""

    def _target_ok(self, front: str) -> bool:
        if is_self_app(front):
            return False
        targets = [t.strip().lower()
                   for t in (self.config.target_apps or []) if t and t.strip()]
        if not targets:
            return True
        f = front.lower()
        return any(t in f for t in targets)

    def handle_block(self, block):
        if not self.enabled:
            return
        probe = getattr(self, "_listen_probe", None)
        if getattr(self, "listen_only", False) and probe is not None:
            self.button_detector.process(block)
            probe.log_block(block, self.button_detector.last_metrics)
            return
        slot = self.button_detector.process(block)
        button_active = self.button_detector.button_active
        metrics = self.button_detector.last_metrics
        detection = getattr(self.button_detector, "last_detection", {})
        front = self._front()

        if slot is not None:
            serial_mode = (
                self.dictation is not None and self.dictation.knob_source == "serial"
            )
            if serial_mode:
                if self.logger:
                    self.logger.tone(slot, front, "audio_detected_serial_ignored")
                if (
                    self.dictation.try_serial_grey_audio_action(slot)
                    and self._target_ok(front)
                ):
                    action = self.router.fire_slot(slot)
                    self._on_event("tone", slot)
                    self._on_event("action", action)
                    if self.logger:
                        self.logger.action(slot, action, front)
                        self.logger.tone(slot, front, "serial_grey_audio")
            elif self.dictation and self.dictation.knob_source == "audio_fallback":
                if slot != 1:
                    if self.logger:
                        self.logger.tone(slot, front, f"audio_grey_ignored slot{slot}")
                    slot = None
                elif not passes_audio_only_button(detection, self.config):
                    slot = None
                elif self.dictation.handle_audio_grey(slot, detection):
                    self._on_event("tone", slot)
                    self._on_event("action", "audio_grey")
                    if self.logger:
                        self.logger.tone(slot, front, f"audio_grey slot{slot}")
                else:
                    slot = None
            elif self.logger:
                self.logger.tone(slot, front, "detected")
            if not serial_mode and slot is not None:
                if (
                    self.dictation
                    and self.dictation.knob_source == "audio_fallback"
                ):
                    pass
                elif self.dictation and self.dictation.handle_button_end(
                    slot, detection
                ):
                    self._on_event("tone", slot)
                    self._on_event("action", "dictation_end")
                    if self.logger:
                        self.logger.tone(slot, front, f"dictation_end slot{slot}")
                elif self._target_ok(front):
                    action = self.router.fire_slot(slot)
                    self._on_event("tone", slot)
                    self._on_event("action", action)
                    if self.logger:
                        self.logger.action(slot, action, front)
                else:
                    self._on_event("blocked", slot)
                    if self.logger:
                        self.logger.blocked(f"slot{slot}", front)

        if self.dictation is not None:
            try:
                self.dictation.observe_block(
                    block, button_active, slot=slot, tone_metrics=metrics)
            except Exception:  # noqa: BLE001
                self.dictation.force_release("error")
        utterance = self.voicegate.process(block, button_active)
        if utterance is not None:
            self._submit(lambda u=utterance: self._handle_utterance(u))

    def release_dictation(self, reason: str = "cleanup") -> None:
        if self.dictation is not None:
            self.dictation.force_release(reason)

    def _handle_utterance(self, utterance):
        text = self.transcriber.transcribe(utterance, self.config.sample_rate)
        if not text:
            if self.transcriber.last_error:
                self._on_event("error", self.transcriber.last_error)
            return
        front = self._front()
        if not self._target_ok(front):
            self._on_event("blocked", text)
            if self.logger:
                self.logger.blocked(text, front)
            return
        self.router.type_text(text)
        self._on_event("transcript", text)
        if self.logger:
            self.logger.transcript(text, front)


def _default_frontmost() -> str:
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ""
        return f"{app.localizedName() or ''} {app.bundleIdentifier() or ''}"
    except Exception:  # noqa: BLE001
        return ""
