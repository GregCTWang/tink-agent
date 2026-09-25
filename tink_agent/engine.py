from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor


class Engine:
    def __init__(self, config, detector, voicegate, transcriber, router,
                 on_event=None, submit_fn=None, frontmost_fn=None, logger=None,
                 dictation=None):
        self.config = config
        self.detector = detector
        self.voicegate = voicegate
        self.transcriber = transcriber
        self.router = router
        self.logger = logger
        self.dictation = dictation
        self.enabled = config.enabled
        self._on_event = on_event or (lambda kind, payload: None)
        # Optional safety guard: when config.target_app is set, actions/typing
        # only fire if the frontmost app's name or bundle id contains it.
        # frontmost_fn() -> str is injectable for tests; defaults to NSWorkspace.
        self._frontmost_fn = frontmost_fn or _default_frontmost
        if submit_fn is not None:
            self._submit = submit_fn
        else:
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._submit = lambda fn: self._pool.submit(fn)

    @property
    def is_capturing(self) -> bool:
        """True while an utterance or dictation session is open (menu bar icon)."""
        if getattr(self.voicegate, "active", False):
            return True
        d = self.dictation
        return bool(d is not None and d.is_active)

    def _front(self) -> str:
        try:
            return self._frontmost_fn() or ""
        except Exception:  # noqa: BLE001 — never let the guard crash the path
            return ""

    def _target_ok(self, front: str) -> bool:
        targets = [t.strip().lower()
                   for t in (self.config.target_apps or []) if t and t.strip()]
        if not targets:
            return True
        f = front.lower()
        return any(t in f for t in targets)

    def handle_block(self, block):
        if not self.enabled:
            return
        slot = self.detector.process(block)
        tone_active = self.detector.tone_active
        metrics = getattr(self.detector, "last_metrics", None)
        front = self._front()
        if slot is not None:
            if self.logger:
                self.logger.tone(slot, front, "detected")
            if self.dictation and self.dictation.handle_tone(slot):
                self._on_event("tone", slot)
                self._on_event("action", "dictation_cancel")
                if self.logger:
                    self.logger.tone(slot, front, "dictation_cancel")
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
                    block, tone_active, slot=slot, tone_metrics=metrics)
            except Exception:  # noqa: BLE001
                self.dictation.force_release("error")
        utterance = self.voicegate.process(block, tone_active)
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
    """Frontmost app identity (name + bundle id) via AppKit, or "" if unavailable."""
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ""
        return f"{app.localizedName() or ''} {app.bundleIdentifier() or ''}"
    except Exception:  # noqa: BLE001
        return ""
