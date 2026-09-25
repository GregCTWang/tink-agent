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
    z = "z"


class FakeKeyboard:
    Key = FakeKey

    def __init__(self):
        self.events = []
        self.key_logs: list[str] = []

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


def _ctrl(kb):
    c = Config()
    router = ActionRouter(c.slot_actions, keyboard=kb)
    sched: list[tuple[int, object]] = []
    logs: list[str] = []

    def delay_fn(ms, fn):
        sched.append((ms, fn))

    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(),
        delay_fn=delay_fn,
    )
    d._log_keys = lambda msg: logs.append(msg)  # type: ignore[method-assign]
    return d, sched, logs, router, c


def test_default_profiles_use_stop_then_undo():
    c = Config()
    for p in c.dictation_profiles:
        assert p.get("cancel_method") == "stop_then_undo"
        assert p.get("undo_delay_ms") == 1500


def test_undo_skipped_no_speech():
    kb = FakeKeyboard()
    d, sched, logs, router, c = _ctrl(kb)
    profile = next(dict(p) for p in c.dictation_profiles if "Grok" in p["match"])
    d._profile = profile
    d._gate.activate()
    d._speech_detected = False
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    sched[0][1]()
    assert any("(cancel_stop)" in x for x in logs)
    assert ("press", "ENTER") not in kb.events
    assert not any("cmd+z (cancel_undo)" in x for x in logs)
    assert any("no_speech" in x for x in logs)


def test_undo_skipped_app_changed():
    kb = FakeKeyboard()
    front = {"v": "Grok Bot com.grok"}
    c = Config()
    router = ActionRouter(c.slot_actions, keyboard=kb)
    sched = []
    logs: list[str] = []
    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: front["v"],
        dispatch=lambda fn: fn(),
        delay_fn=lambda ms, fn: sched.append((ms, fn)),
    )
    d._log_keys = lambda msg: logs.append(msg)  # type: ignore[method-assign]
    profile = next(dict(p) for p in c.dictation_profiles if "Grok" in p["match"])
    d._profile = profile
    d._gate.activate()
    d._speech_detected = True
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    front["v"] = "Claude com.anthropic"
    sched[0][1]()
    assert any("app_changed" in x for x in logs)


def test_undo_skipped_new_session():
    kb = FakeKeyboard()
    d, sched, logs, _, c = _ctrl(kb)
    profile = next(dict(p) for p in c.dictation_profiles if "Grok" in p["match"])
    d._profile = profile
    d._gate.activate()
    d._speech_detected = True
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    d._session_gen += 1
    sched[0][1]()
    assert any("new_session" in x for x in logs)


def test_stop_then_undo_toggle_and_hold_modes():
    kb = FakeKeyboard()
    d, sched, logs, router, c = _ctrl(kb)
    router.dictation_cancel_stop = lambda p, lf: lf("cmd+d (cancel_stop)")  # type: ignore
    router.dictation_cancel_undo = lambda lf: lf("cmd+z (cancel_undo) flags=0")  # type: ignore
    profile = next(dict(p) for p in c.dictation_profiles if "Grok" in p["match"])
    d._profile = profile
    d._gate.activate()
    d._speech_detected = True
    d._end_session(reason="grey_cancel", post_action=None, cancelled=True)
    sched[0][1]()
    assert any("(cancel_stop)" in x for x in logs)
    assert any("cancel_undo" in x for x in logs)
    assert resolve_cancel_method({"cancel_method": "escape"}) == "escape"
