import numpy as np

from tink_agent.config import Config
from tink_agent.detector import VoiceGate
from tink_agent.actions import ActionRouter
from tink_agent.dictation import DictationController, DictationOnsetGate, match_profile
from tink_agent.engine import Engine
from tink_agent.audio_buttons import make_button_detector


class FakeKey:
    enter = "ENTER"
    esc = "ESC"
    ctrl = "CTRL"
    cmd = "CMD"
    m = "m"
    d = "d"
    tab = "TAB"
    shift = "SHIFT"


class FakeKeyboard:
    Key = FakeKey

    def __init__(self):
        self.events = []

    def press(self, k):
        self.events.append(("press", k))

    def release(self, k):
        self.events.append(("release", k))

    def type(self, s):
        self.events.append(("type", s))

    class _Pressed:
        def __init__(self, kb, k):
            self.kb, self.k = kb, k

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def pressed(self, k):
        return FakeKeyboard._Pressed(self, k)


def _speech_block(level=200):
    return np.full(800, level, dtype=np.int16)


def _engine_with_dictation(kb, frontmost="Cursor com.todesktop.230313mzl4w4u92",
                           config=None, delays=None):
    c = config or Config(
        vad_rms_start=1e9,
        vad_rms_end=1e9,
        dictation_onset_min_ms=50,
        button_detector="tone",
    )
    buttons = make_button_detector(c)
    vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                   c.min_utterance_ms, c.sample_rate, c.block_size)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    scheduled = delays if delays is not None else []

    def delay_fn(ms, fn):
        scheduled.append((ms, fn))

    dictation = DictationController(
        c, router, frontmost_fn=lambda: frontmost,
        dispatch=lambda fn: fn(),
        delay_fn=delay_fn,
    )
    eng = Engine(c, buttons, vg, object(), router, dictation=dictation,
                 submit_fn=lambda fn: fn(), frontmost_fn=lambda: frontmost)
    return eng, scheduled


def test_match_profile_substring():
    profiles = [{"match": "Cursor", "mode": "hold", "keys": ["ctrl", "m"]}]
    assert match_profile("Cursor com.todesktop", profiles) is profiles[0]


def test_silence_does_not_end_active_session():
    g = DictationOnsetGate(90, 50, 16000, 800, 400)
    g.process(_speech_block(200), button_active=False, knob_held=True, require_knob=True)
    assert g.active
    for _ in range(200):
        assert g.process(
            _speech_block(26), button_active=False, knob_held=True, require_knob=True
        ) is None
        assert g.active


def test_hold_cursor_and_button_end_slot1():
    kb = FakeKeyboard()
    eng, sched = _engine_with_dictation(kb)
    eng.dictation.set_serial_link(False, "no_port")
    tap = {
        "slot": 1,
        "similarity": 0.92,
        "similarity_margin": 0.1,
        "square_score": 0.3,
        "rms": 9000,
        "duration_ms": 150,
    }
    assert eng.dictation.handle_audio_grey(1, tap)
    assert eng.dictation.is_active
    det = {
        "slot": 1,
        "similarity": 0.92,
        "similarity_margin": 0.1,
        "square_score": 0.3,
        "rms": 9000,
        "duration_ms": 150,
    }
    before = len(sched)
    assert eng.dictation.handle_audio_grey(1, det)
    for _, fn in sched[before:]:
        fn()
    assert ("press", "ENTER") in kb.events


def test_idle_cap_no_post_action():
    kb = FakeKeyboard()
    c = Config(
        vad_rms_start=1e9,
        dictation_idle_cap_ms=100,
        dictation_onset_min_ms=50,
        button_detector="tone",
    )
    eng, sched = _engine_with_dictation(kb, config=c)
    eng.dictation.set_serial_link(True, "t")
    eng.dictation.on_serial_line("K1")
    assert sched, "knob debounce"
    sched.pop(0)[1]()
    eng.dictation.knob_held = False
    for _ in range(10):
        eng.handle_block(np.full(800, 26, dtype=np.int16))
    assert sched == []
    assert not eng.dictation.is_active
