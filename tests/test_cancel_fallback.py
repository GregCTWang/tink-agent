import numpy as np

from tink_agent.actions import ActionRouter
from tink_agent.config import Config
from tink_agent.dictation import DictationController


class FakeKey:
    enter = "ENTER"
    esc = "ESC"
    ctrl = "CTRL"
    cmd = "CMD"
    m = "m"
    d = "d"
    z = "z"


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


def test_cancel_fallback_undo_scheduled():
    kb = FakeKeyboard()
    c = Config()
    profiles = [dict(p) for p in c.dictation_profiles]
    for p in profiles:
        if p.get("match") == "Claude":
            p["cancel_fallback"] = "undo"
    c.dictation_profiles = profiles
    router = ActionRouter(c.slot_actions, keyboard=kb)
    sched = []

    def delay_fn(ms, fn):
        sched.append((ms, fn))

    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: "Claude app",
        dispatch=lambda fn: fn(),
        delay_fn=delay_fn,
    )
    d._profile = profiles[1]
    d._gate.activate()
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    undo = [x for x in sched if x[0] >= 1500]
    assert undo
    undo[0][1]()
    assert ("press", "CMD") in kb.events and ("press", "z") in kb.events
