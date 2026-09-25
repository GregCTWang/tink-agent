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


def _speech():
    return np.full(800, 200, dtype=np.int16)


def _setup(kb, frontmost="Grok Bot com.grok"):
    c = Config(dictation_min_speech_ms=100, dictation_restart_cooldown_ms=700)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    sched = []
    front_fn = frontmost if callable(frontmost) else (lambda: frontmost)

    def delay_fn(ms, fn):
        sched.append((ms, fn))

    d = DictationController(
        c,
        router,
        frontmost_fn=front_fn,
        dispatch=lambda fn: fn(),
        delay_fn=delay_fn,
    )
    d.set_serial_link(True, "t")
    return d, sched, router


def test_enter_skipped_when_app_changed():
    kb = FakeKeyboard()
    front = {"v": "Grok Bot com.grok"}
    d, sched, _ = _setup(kb, lambda: front["v"])
    d.on_serial_line("K1")
    sched.pop(0)[1]()
    for _ in range(4):
        d.observe_block(_speech(), False)
    d.on_serial_line("K0")
    front["v"] = "Claude com.anthropic"
    while sched:
        ms, fn = sched.pop(0)
        fn()
    assert ("press", "ENTER") not in kb.events


def test_enter_skipped_when_new_session_started():
    kb = FakeKeyboard()
    d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    sched.pop(0)[1]()
    for _ in range(4):
        d.observe_block(_speech(), False)
    closed = d._session_epoch
    d.on_serial_line("K0")
    d._session_epoch += 1
    while sched:
        sched.pop(0)[1]()
    assert ("press", "ENTER") not in kb.events
    assert closed != d._session_epoch


def test_grey_cancel_cancels_pending_enter():
    kb = FakeKeyboard()
    d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    sched.pop(0)[1]()
    for _ in range(4):
        d.observe_block(_speech(), False)
    d.on_serial_line("G0")
    d.knob_held = True
    while sched:
        sched.pop(0)[1]()
    assert ("press", "ENTER") not in kb.events


def test_short_speech_no_enter():
    kb = FakeKeyboard()
    c = Config(dictation_min_speech_ms=300)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    sched = []
    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(),
        delay_fn=lambda ms, fn: sched.append((ms, fn)),
    )
    d.set_serial_link(True, "t")
    d.on_serial_line("K1")
    sched.pop(0)[1]()
    d.observe_block(_speech(), False)
    d.on_serial_line("K0")
    while sched:
        sched.pop(0)[1]()
    assert ("press", "ENTER") not in kb.events


def test_cooldown_defers_restart():
    kb = FakeKeyboard()
    t = [0.0]

    def mono():
        return t[0]

    c = Config(dictation_restart_cooldown_ms=700)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    sched = []
    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(),
        delay_fn=lambda ms, fn: sched.append((ms, fn)),
        monotonic_fn=mono,
    )
    d.set_serial_link(True, "t")
    d._restart_ready_at = 1.0
    d.knob_held = True
    d._schedule_knob_start()
    assert len(sched) == 1
    t[0] = 1.0
    sched[0][1]()
    assert d.is_active
