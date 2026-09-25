import numpy as np

from tink_agent.config import Config
from tink_agent.detector import ToneDetector, VoiceGate
from tink_agent.actions import ActionRouter
from tink_agent.dictation import (
    DictationController,
    DictationOnsetGate,
    KnobFloorCalibrator,
    match_profile,
)
from tink_agent.engine import Engine


class FakeKey:
    enter = "ENTER"
    esc = "ESC"
    ctrl = "CTRL"
    shift = "SHIFT"
    cmd = "CMD"
    tab = "TAB"
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

    def type(self, s):
        self.events.append(("type", s))

    class _Pressed:
        def __init__(self, kb, k):
            self.kb, self.k = kb, k

        def __enter__(self):
            self.kb.events.append(("hold", self.k))
            return self

        def __exit__(self, *a):
            self.kb.events.append(("unhold", self.k))

    def pressed(self, k):
        return FakeKeyboard._Pressed(self, k)


def _level_block(level: int):
    return np.full(800, level, dtype=np.int16)


def _speech_block(level=200):
    return _level_block(level)


def _released_block():
    return _level_block(28)


def _held_silent_block():
    return _level_block(43)


def _tone_block(freq=1500):
    sr, block = 16000, 800
    t = np.arange(0, block / sr, 1 / sr)[:block]
    return (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)


def _gate(**kwargs):
    c = Config(**kwargs)
    floors = KnobFloorCalibrator(
        released_floor=c.dictation_floor_released_rms,
        held_floor=c.dictation_floor_held_rms,
        auto_floor=c.dictation_auto_floor,
        fixed_release_rms=c.dictation_release_rms,
    )
    return DictationOnsetGate(
        onset_rms=c.dictation_onset_rms,
        onset_min_ms=c.dictation_onset_min_ms,
        hangover_ms=c.dictation_hangover_ms,
        sample_rate=c.sample_rate,
        block_size=c.block_size,
        onset_window_samples=400,
        max_session_ms=c.dictation_max_session_ms,
        floors=floors,
    )


def _engine_with_dictation(kb, frontmost="Cursor com.todesktop.230313mzl4w4u92",
                           config=None, delays=None):
    c = config or Config(
        vad_rms_start=1e9,
        vad_rms_end=1e9,
        dictation_onset_min_ms=50,
        dictation_hangover_ms=400,
        dictation_onset_rms=90,
        dictation_auto_floor=True,
    )
    det = ToneDetector(c.tones, c.tone_rms_min, c.tone_dominance_min,
                       c.tone_debounce_ms, c.sample_rate)
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
    eng = Engine(c, det, vg, object(), router, dictation=dictation,
                 submit_fn=lambda fn: fn(), frontmost_fn=lambda: frontmost)
    return eng, scheduled


def test_match_profile_substring():
    profiles = [{"match": "Cursor", "mode": "hold", "keys": ["ctrl", "m"]}]
    assert match_profile("Cursor com.todesktop", profiles) is profiles[0]
    assert match_profile("Safari", profiles) is None


def test_release_threshold_between_floors():
    cal = KnobFloorCalibrator(released_floor=28, held_floor=43, auto_floor=True)
    thr = cal.release_threshold()
    assert 28 < thr < 43
    assert 43 > thr  # knob-held-silent (~43) stays above release cutoff


def test_held_silent_rms_does_not_end_session():
    g = _gate(dictation_onset_min_ms=50, dictation_hangover_ms=400)
    assert g.process(_speech_block(200), tone_active=False) == "start"
    for _ in range(30):
        assert g.process(_held_silent_block(), tone_active=False) is None
        assert g.active is True


def test_released_rms_ends_after_hangover():
    g = _gate(dictation_onset_min_ms=50, dictation_hangover_ms=200)
    g.process(_speech_block(200), tone_active=False)
    assert g.active
    # 200 ms hangover @ 50 ms blocks -> 4 blocks below threshold
    for _ in range(3):
        assert g.process(_released_block(), tone_active=False) is None
    assert g.process(_released_block(), tone_active=False) == "stop"
    assert g.last_stop_reason == "hangover"


def test_onset_gate_ignores_tone_active():
    g = _gate(dictation_onset_min_ms=100)
    loud = _speech_block(500)
    assert g.process(loud, tone_active=False) is None
    assert g.process(loud, tone_active=False) == "start"
    assert g.process(_tone_block(), tone_active=True) is None
    assert g.active is True


def test_toggle_app_starts_and_stops_with_send():
    kb = FakeKeyboard()
    scheduled = []
    eng, scheduled = _engine_with_dictation(
        kb, frontmost="Claude com.anthropic.claude", delays=scheduled)
    for _ in range(2):
        eng.handle_block(_speech_block())
    assert ("press", "CMD") in kb.events and ("press", "d") in kb.events
    for _ in range(10):
        eng.handle_block(_released_block())
    assert len(scheduled) == 1
    scheduled[0][1]()
    assert ("press", "ENTER") in kb.events


def test_hold_cursor_presses_ctrl_m_and_releases():
    kb = FakeKeyboard()
    eng, _ = _engine_with_dictation(kb)
    for _ in range(2):
        eng.handle_block(_speech_block())
    assert kb.events[:2] == [("press", "CTRL"), ("press", "m")]
    for _ in range(10):
        eng.handle_block(_released_block())
    assert ("release", "m") in kb.events and ("release", "CTRL") in kb.events


def test_no_profile_does_not_send_keys():
    kb = FakeKeyboard()
    eng, _ = _engine_with_dictation(kb, frontmost="Notes com.apple.Notes")
    for _ in range(3):
        eng.handle_block(_speech_block())
    assert kb.events == []


def test_cancel_on_any_tone_during_session():
    kb = FakeKeyboard()
    eng, _ = _engine_with_dictation(kb)
    for _ in range(2):
        eng.handle_block(_speech_block())
    kb.events.clear()
    eng.handle_block(_tone_block(1500))  # slot 1 = enter, must not fire
    assert ("press", "ENTER") not in kb.events
    assert ("release", "m") in kb.events


def test_dictation_cancel_hold_sends_escape_then_release():
    kb = FakeKeyboard()
    c = Config(
        vad_rms_start=1e9,
        slot_actions={**Config().slot_actions, 3: "dictation_cancel"},
        dictation_onset_min_ms=50,
    )
    eng, _ = _engine_with_dictation(kb, config=c)
    for _ in range(2):
        eng.handle_block(_speech_block())
    kb.events.clear()
    eng.handle_block(_tone_block(3100))
    assert ("press", "ESC") in kb.events
    assert ("release", "m") in kb.events


def test_dictation_cancel_toggle_skips_enter():
    kb = FakeKeyboard()
    scheduled = []
    c = Config(
        vad_rms_start=1e9,
        dictation_onset_min_ms=50,
    )
    eng, scheduled = _engine_with_dictation(
        kb, frontmost="Claude com.anthropic", config=c, delays=scheduled)
    for _ in range(2):
        eng.handle_block(_speech_block())
    kb.events.clear()
    eng.handle_block(_tone_block(3100))
    for _ in range(10):
        eng.handle_block(_released_block())
    assert scheduled == []


def test_force_release_clears_held_keys():
    kb = FakeKeyboard()
    eng, _ = _engine_with_dictation(kb)
    for _ in range(2):
        eng.handle_block(_speech_block())
    eng.release_dictation("test")
    assert ("release", "m") in kb.events
    assert eng.router.has_held_keys() is False


def test_max_session_ends_with_cleanup():
    kb = FakeKeyboard()
    c = Config(
        vad_rms_start=1e9,
        dictation_max_session_ms=100,
        dictation_onset_min_ms=50,
        block_size=800,
    )
    eng, _ = _engine_with_dictation(kb, config=c)
    for _ in range(2):
        eng.handle_block(_speech_block())
    eng.handle_block(_speech_block())
    assert ("release", "m") in kb.events


def test_beeps_still_fire_outside_dictation_session():
    kb = FakeKeyboard()
    eng, _ = _engine_with_dictation(kb)
    eng.handle_block(_tone_block(1500))
    assert ("press", "ENTER") in kb.events


def test_is_capturing_true_during_dictation():
    kb = FakeKeyboard()
    eng, _ = _engine_with_dictation(kb)
    assert eng.is_capturing is False
    eng.handle_block(_speech_block())
    eng.handle_block(_speech_block())
    assert eng.is_capturing is True


def test_loud_beep_does_not_end_active_session():
    g = _gate(dictation_onset_min_ms=50)
    g.process(_speech_block(200), tone_active=False)
    assert g.process(_level_block(16000), tone_active=True) is None
    assert g.active is True
