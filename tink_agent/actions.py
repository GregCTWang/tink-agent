from __future__ import annotations

import time

# Catalog of selectable per-slot actions: the single source of truth shared by
# the router (what an id does, below) and the Button Actions UI (label + glyph).
# `type:<text>` ids type the literal text then press Enter (submit).
ACTION_CATALOG = [
    {"id": "enter",         "label": "Enter",                  "glyph": "⏎"},
    {"id": "escape",        "label": "Escape",                 "glyph": "⎋"},
    {"id": "esc_esc",       "label": "Double Esc (clear)",     "glyph": "⎋⎋"},
    {"id": "ctrl_c",        "label": "Ctrl + C (interrupt)",   "glyph": "⌃C"},
    {"id": "ctrl_d",        "label": "Ctrl + D",               "glyph": "⌃D"},
    {"id": "tab",           "label": "Tab",                    "glyph": "⇥"},
    {"id": "shift_tab",     "label": "Shift + Tab (mode)",     "glyph": "⇤"},
    {"id": "shift_enter",   "label": "Shift + Enter (newline)","glyph": "⇧⏎"},
    {"id": "up",            "label": "Arrow Up",               "glyph": "↑"},
    {"id": "down",          "label": "Arrow Down",             "glyph": "↓"},
    {"id": "left",          "label": "Arrow Left",             "glyph": "←"},
    {"id": "right",         "label": "Arrow Right",            "glyph": "→"},
    {"id": "space",         "label": "Space",                  "glyph": "␣"},
    {"id": "backspace",     "label": "Backspace",              "glyph": "⌫"},
    {"id": "cmd_v",         "label": "Paste",                  "glyph": "⌘V"},
    {"id": "key_1",         "label": 'Type "1"',     "glyph": "1"},
    {"id": "key_2",         "label": 'Type "2"',     "glyph": "2"},
    {"id": "key_3",         "label": 'Type "3"',     "glyph": "3"},
    {"id": "type:yes",      "label": 'Send "yes"',       "glyph": "y"},
    {"id": "type:continue", "label": 'Send "continue"',  "glyph": "»"},
    {"id": "type:/clear",   "label": "Send /clear",            "glyph": "/"},
    {"id": "type:/compact", "label": "Send /compact",          "glyph": "/"},
    {"id": "dictation_cancel", "label": "Cancel dictation (no send)", "glyph": "✕"},
    {"id": "noop",          "label": "No action",              "glyph": "–"},
]

_CATALOG_BY_ID = {a["id"]: a for a in ACTION_CATALOG}


def action_label(action_id: str) -> str:
    a = _CATALOG_BY_ID.get(action_id)
    return a["label"] if a else (action_id or "No action")


class ActionRouter:
    def __init__(self, slot_actions, keyboard=None, dispatch=None):
        self.slot_actions = dict(slot_actions)
        self._kb = keyboard  # injected (tests) or lazy pynput controller
        # All pynput calls must run on the main thread: on macOS pynput queries
        # the keyboard layout via HIToolbox TSM, which traps (SIGTRAP via
        # libdispatch) when invoked off the main thread under a running NSApp.
        # fire_slot runs on the audio callback thread and type_text on the
        # transcription worker thread, so the menu bar injects a main-thread
        # dispatcher. Default = run inline (tests / non-GUI).
        self._dispatch = dispatch or (lambda fn: fn())
        self.last_error: str | None = None
        # Keys held for hold-mode dictation (Ctrl+M etc.); released on cleanup.
        self._held_keys: list = []
        self._held_tokens: list[str] = []

    @property
    def kb(self):
        if self._kb is None:
            from pynput import keyboard
            ctrl = keyboard.Controller()
            ctrl.Key = keyboard.Key  # attach enum for uniform access
            self._kb = ctrl
        return self._kb

    def fire_slot(self, slot: int) -> str:
        action = self.slot_actions.get(slot)
        if action is None:
            return "unmapped"
        if action == "noop":
            return "noop"
        self.fire_named_action(action)
        return action

    def fire_named_action(self, action: str) -> None:
        if not action or action in ("noop", "unmapped"):
            return
        self._dispatch(lambda a=action: self._run(self._perform, a))

    def _run(self, fn, arg) -> None:
        """Execute a keyboard side-effect (on the main thread), trapping errors
        (e.g. missing Accessibility) so they never crash the calling thread."""
        try:
            fn(arg)
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)

    def _perform(self, action: str) -> None:
        kb = self.kb
        Key = kb.Key
        if action == "enter":
            kb.press(Key.enter); kb.release(Key.enter)
        elif action == "escape":
            kb.press(Key.esc); kb.release(Key.esc)
        elif action == "ctrl_c":
            with kb.pressed(Key.ctrl):
                kb.press("c"); kb.release("c")
        elif action == "tab":
            kb.press(Key.tab); kb.release(Key.tab)
        elif action == "shift_tab":
            with kb.pressed(Key.shift):
                kb.press(Key.tab); kb.release(Key.tab)
        elif action == "up":
            kb.press(Key.up); kb.release(Key.up)
        elif action == "down":
            kb.press(Key.down); kb.release(Key.down)
        elif action == "esc_esc":
            kb.press(Key.esc); kb.release(Key.esc)
            kb.press(Key.esc); kb.release(Key.esc)
        elif action == "ctrl_d":
            with kb.pressed(Key.ctrl):
                kb.press("d"); kb.release("d")
        elif action == "shift_enter":
            with kb.pressed(Key.shift):
                kb.press(Key.enter); kb.release(Key.enter)
        elif action == "left":
            kb.press(Key.left); kb.release(Key.left)
        elif action == "right":
            kb.press(Key.right); kb.release(Key.right)
        elif action == "space":
            kb.press(Key.space); kb.release(Key.space)
        elif action == "backspace":
            kb.press(Key.backspace); kb.release(Key.backspace)
        elif action == "cmd_v":
            with kb.pressed(Key.cmd):
                kb.press("v"); kb.release("v")
        elif action in ("key_1", "key_2", "key_3"):
            ch = action[-1]
            kb.press(ch); kb.release(ch)
        elif action.startswith("type:"):
            self._do_type(action[len("type:"):])
            kb.press(Key.enter); kb.release(Key.enter)

    def type_text(self, text: str) -> None:
        if not text:
            return
        self._dispatch(lambda t=text: self._run(self._do_type, t))

    def _do_type(self, text: str) -> None:
        self.kb.type(text)

    def has_held_keys(self) -> bool:
        return bool(self._held_keys)

    def _key_obj(self, name: str):
        Key = self.kb.Key
        return getattr(Key, name, name)

    def dictation_tap(self, keys: list) -> None:
        """Tap a shortcut (press+release each modifier, then letter, unmodified)."""
        self._run(self._do_dictation_tap, list(keys))

    def _do_dictation_tap(self, keys: list) -> None:
        kb = self.kb
        mods = [self._key_obj(k) for k in keys[:-1]] if len(keys) > 1 else []
        final = self._key_obj(keys[-1]) if keys else None
        for m in mods:
            kb.press(m)
        if final is not None:
            kb.press(final)
            kb.release(final)
        for m in reversed(mods):
            kb.release(m)

    def dictation_press(self, keys: list) -> None:
        """Press and hold a combo until dictation_release_all."""
        self._run(self._do_dictation_press, list(keys))

    def _do_dictation_press(self, keys: list) -> None:
        self.dictation_release_all()
        kb = self.kb
        held = []
        for token in keys:
            k = self._key_obj(token)
            kb.press(k)
            held.append(k)
        self._held_keys = held
        self._held_tokens = list(keys)

    def dictation_grey_cancel_escape(self, profile: dict, config, log_fn=None) -> None:
        """Escape while recording; hold mode releases modifiers with flags cleared."""
        self._dispatch(
            lambda: self._run(
                self._do_grey_cancel_escape,
                (profile, config, log_fn),
            )
        )

    def _do_grey_cancel_escape(self, args) -> None:
        profile, config, log_fn = args
        mode = str(profile.get("mode") or "toggle")
        keys = list(profile.get("keys") or [])
        delay_ms = int(profile.get("cancel_release_delay_ms", 120))
        sequence = str(profile.get("cancel_sequence") or "escape_then_release")
        then_toggle = bool(getattr(config, "cancel_escape_then_toggle", False))
        held = list(self._held_tokens or keys)

        def log(msg: str) -> None:
            if log_fn:
                log_fn(msg)

        try:
            from . import keyboard_cgevent as cg

            if mode == "hold":
                if sequence == "release_then_escape":
                    finals = [held[-1]] if held else []
                    mods = held[:-1] if len(held) > 1 else []
                    if finals:
                        cg.release_tokens_cleared(finals)
                        log("release (cancel) flags=0")
                    time.sleep(delay_ms / 1000.0)
                    fl = cg.post_escape_cleared()
                    log(f"escape (cancel) {fl}")
                    if mods:
                        cg.release_tokens_cleared(mods)
                        log("modifiers up (cancel) flags=0")
                else:
                    fl = cg.post_escape_cleared()
                    log(f"escape (cancel) {fl}")
                    time.sleep(delay_ms / 1000.0)
                    cg.release_tokens_cleared(held)
                    log("release (cancel) flags=0")
                self._held_keys = []
                self._held_tokens = []
                return

            fl = cg.post_escape_cleared()
            log(f"escape (cancel) {fl}")
            if then_toggle and keys:
                self._do_dictation_tap(keys)
                label = "+".join(str(k) for k in keys)
                log(f"{label} (cancel_toggle)")
            self._held_keys = []
            self._held_tokens = []
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            kb = self.kb
            Key = kb.Key
            kb.press(Key.esc)
            kb.release(Key.esc)
            log("escape (cancel) flags=fallback_pynput")
            if mode == "hold":
                time.sleep(delay_ms / 1000.0)
                self._do_dictation_release_all(None)
                log("release (cancel) flags=fallback_pynput")
            elif then_toggle and keys:
                self._do_dictation_tap(keys)

    def dictation_release_all(self) -> None:
        self._run(self._do_dictation_release_all, None)

    def _do_dictation_release_all(self, _ignored) -> None:
        kb = self.kb
        for k in reversed(self._held_keys):
            try:
                kb.release(k)
            except Exception:  # noqa: BLE001
                pass
        self._held_keys = []
        self._held_tokens = []
