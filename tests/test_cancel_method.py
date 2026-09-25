import numpy as np

from tink_agent.actions import ActionRouter
from tink_agent.config import Config
from tink_agent.dictation import DictationController, resolve_cancel_method


class FakeKey:
    enter = "ENTER"
    esc = "ESC"
    ctrl = "CTRL"
    cmd = "CMD"
    m = "m"
    d = "d"


class FakeKeyboard:
    Key = FakeKey

    def __init__(self):
        self.events = []

    def press(self, k):
        self.events.append(("press", k))

    def release(self, k):
        self.events.append(("release", k))

    class _P:
        def __init__(self, kb, k):
            self.kb, self.k = kb, k

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def pressed(self, k):
        return FakeKeyboard._P(self, k)


def test_resolve_cancel_method_defaults_escape():
    assert resolve_cancel_method({}) == "escape"
    assert resolve_cancel_method({"restore_on_cancel": True}) == "restore"
    assert resolve_cancel_method({"cancel_method": "stop"}) == "stop"


def test_grey_cancel_escape_skips_stop_toggle():
    kb = FakeKeyboard()
    c = Config()
    router = ActionRouter(c.slot_actions, keyboard=kb)
    logs: list[str] = []

    def fake_cancel(profile, config, log_fn=None):
        if log_fn:
            log_fn("escape (cancel) flags=0")

    router.dictation_grey_cancel_escape = fake_cancel  # type: ignore[method-assign]
    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(),
        delay_fn=lambda _m, fn: fn(),
    )
    profile = next(p for p in c.dictation_profiles if "Grok" in p["match"])
    d._profile = dict(profile)
    d._gate.activate()
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    assert ("press", "CMD") not in kb.events
    assert ("press", "d") not in kb.events


def test_grey_cancel_stop_still_toggles():
    kb = FakeKeyboard()
    c = Config()
    profiles = [dict(p) for p in c.dictation_profiles]
    for p in profiles:
        if "Grok" in p.get("match", ""):
            p["cancel_method"] = "stop"
    c.dictation_profiles = profiles
    router = ActionRouter(c.slot_actions, keyboard=kb)
    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(),
        delay_fn=lambda _m, fn: fn(),
    )
    p = next(x for x in profiles if "Grok" in x["match"])
    d._profile = p
    d._session_toggle_tap_at = 0.0
    d._gate.activate()
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    assert ("press", "CMD") in kb.events
