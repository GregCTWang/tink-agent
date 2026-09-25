"""Grey sample-light-1 only when knob_source=audio_fallback (3.5 mm)."""
from __future__ import annotations

import numpy as np

from tink_agent.actions import ActionRouter
from tink_agent.audio_buttons import make_button_detector
from tink_agent.config import Config
from tink_agent.detector import VoiceGate
from tink_agent.dictation import DictationController
from tink_agent.engine import Engine
from tink_agent.fx_button import (
    default_fx_templates,
    FxButtonDetector,
    passes_audio_only_button,
    read_pcm_int16,
    replay_detections,
)


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
        self.events: list = []
        self._held: set = set()

    def press(self, k):
        self.events.append(("press", k))
        self._held.add(k)

    def release(self, k):
        self.events.append(("release", k))
        self._held.discard(k)

    class _P:
        def __init__(self, kb, k):
            self.kb, self.k = kb, k

        def __enter__(self):
            self.kb.press(self.k)
            return self

        def __exit__(self, *a):
            self.kb.release(self.k)

    def pressed(self, k):
        return FakeKeyboard._P(self, k)


def _fx_detector(c: Config) -> FxButtonDetector:
    block_ms = 1000 * c.block_size / c.sample_rate
    return FxButtonDetector(
        templates=default_fx_templates(),
        sample_rate=c.sample_rate,
        block_size=c.block_size,
        similarity_min=c.fx_button_similarity_min,
        rms_min=c.fx_button_rms_min,
        lockout_ms=c.fx_button_lockout_ms,
        slot1_rms_min=c.fx_button_slot1_rms_min,
        slot1_min_blocks=max(
            1, int(c.fx_button_slot1_min_ms / block_ms)
        ),
        slot1_square_min=c.fx_button_slot1_square_min,
    )


def _engine(
    kb: FakeKeyboard,
    front: str = "Cursor com.todesktop.230313mzl4w4u92",
):
    c = Config(
        vad_rms_start=1e9,
        vad_rms_end=1e9,
        dictation_grey_tap_max_ms=350,
        dictation_grey_hold_min_ms=700,
        dictation_audio_post_button_ms=900,
        dictation_audio_session_cooldown_ms=800,
    )
    buttons = make_button_detector(c)
    vg = VoiceGate(
        c.vad_rms_start,
        c.vad_rms_end,
        c.vad_hangover_ms,
        c.min_utterance_ms,
        c.sample_rate,
        c.block_size,
    )
    router = ActionRouter(c.slot_actions, keyboard=kb)
    t = {"t": 0.0}
    scheduled: list = []

    def mono_fn():
        return t["t"]

    def delay_fn(ms, fn):
        scheduled.append((ms, fn))

    dictation = DictationController(
        c,
        router,
        frontmost_fn=lambda: front,
        dispatch=lambda fn: fn(),
        delay_fn=delay_fn,
        monotonic_fn=mono_fn,
    )
    dictation.set_serial_link(False, "no_port")
    eng = Engine(
        c,
        buttons,
        vg,
        object(),
        router,
        dictation=dictation,
        submit_fn=lambda fn: fn(),
        frontmost_fn=lambda: front,
    )
    return eng, c, t, scheduled


def _advance(scheduled, t, dt=0.05):
    t["t"] += dt
    for ms, fn in list(scheduled):
        if ms <= dt * 1000:
            fn()
            scheduled.remove((ms, fn))


def _replay_pcm(eng, pcm: np.ndarray, t, scheduled, *, block_ms: float = 50.0):
    bs = eng.config.block_size
    n = len(pcm) // bs
    for _ in range(n):
        eng.handle_block(pcm[:bs])
        pcm = pcm[bs:]
        t["t"] += block_ms / 1000.0
        due = [(ms, fn) for ms, fn in scheduled if ms <= block_ms]
        for ms, fn in due:
            fn()
            scheduled.remove((ms, fn))


def test_beeps3_slot1_durations_match_hold_length():
    pcm = read_pcm_int16("tests/fixtures/fxmic/beeps3-0924.wav")
    c = Config()
    det = _fx_detector(c)
    slot1 = [
        d for d in replay_detections(pcm, det, c.block_size) if d["slot"] == 1
    ]
    durs = sorted(int(d["duration_ms"]) for d in slot1)
    assert any(d <= 350 for d in durs), durs
    assert any(d >= 700 for d in durs), durs


def test_tap_idle_starts_cursor_hold_mode():
    kb = FakeKeyboard()
    eng, c, t, scheduled = _engine(kb)
    pcm = read_pcm_int16("tests/fixtures/fxmic/beeps3-0924.wav")
    bs = c.block_size
    started = False
    for i in range(len(pcm) // bs):
        block = pcm[i * bs : (i + 1) * bs]
        eng.handle_block(block)
        t["t"] += 0.05
        for ms, fn in list(scheduled):
            fn()
            scheduled.clear()
        if eng.dictation.is_active:
            started = True
            break
    assert started
    assert ("press", "CTRL") in kb.events
    assert ("press", "m") in kb.events
    assert "CTRL" in kb._held and "m" in kb._held


def test_tap_active_stop_and_enter():
    kb = FakeKeyboard()
    eng, c, t, scheduled = _engine(kb)
    tap = {
        "slot": 1,
        "similarity": 0.98,
        "similarity_margin": 0.12,
        "square_score": 0.31,
        "rms": 9000,
        "duration_ms": 150,
    }
    assert eng.dictation.handle_audio_grey(1, tap)
    for _, fn in scheduled:
        fn()
    assert eng.dictation.is_active
    det = {
        "slot": 1,
        "similarity": 0.98,
        "similarity_margin": 0.12,
        "square_score": 0.31,
        "rms": 9000,
        "duration_ms": 150,
    }
    assert eng.dictation.handle_audio_grey(1, det)
    for _, fn in scheduled:
        fn()
    releases = [e for e in kb.events if e[0] == "release"]
    assert releases
    assert ("press", "ENTER") in kb.events
    assert not eng.dictation.is_active


def test_hold_active_cancels_once_no_enter():
    kb = FakeKeyboard()
    eng, c, t, scheduled = _engine(kb)
    eng.dictation._gate.activate()
    eng.dictation._profile = c.dictation_profiles[2]
    kb.press("CTRL")
    kb.press("m")
    pcm = read_pcm_int16("tests/fixtures/fxmic/beeps3-0924.wav")
    bs = c.block_size
    # Second slot-1 press in beeps3 is the long hold (~1250 ms).
    det = _fx_detector(c)
    all_d = replay_detections(pcm, det, bs)
    long_start_block = None
    seen_short = 0
    for d in all_d:
        if d["slot"] == 1 and d["duration_ms"] <= 350:
            seen_short += 1
        if d["slot"] == 1 and d["duration_ms"] >= 700 and long_start_block is None:
            long_start_block = int(d["time_s"] * c.sample_rate / bs)
    assert long_start_block is not None
    cancel_events = 0
    for i in range(long_start_block, long_start_block + 40):
        block = pcm[i * bs : (i + 1) * bs]
        eng.handle_block(block)
        t["t"] += 0.05
        for ms, fn in list(scheduled):
            fn()
            scheduled.remove((ms, fn))
        if not eng.dictation.is_active and cancel_events == 0:
            esc = [e for e in kb.events if e == ("press", "ESC")]
            if esc:
                cancel_events = 1
    assert cancel_events == 1
    assert ("press", "ENTER") not in kb.events
    # Tail blocks after cancel must not restart session
    for i in range(long_start_block + 40, long_start_block + 55):
        if i * bs >= len(pcm):
            break
        eng.handle_block(pcm[i * bs : (i + 1) * bs])
        t["t"] += 0.05
    assert not eng.dictation.is_active


def test_speech_blocks_do_not_start_dictation():
    kb = FakeKeyboard()
    eng, c, t, scheduled = _engine(kb, front="Grok Bot com.test")
    speech = np.full(800, 400, dtype=np.int16)
    for _ in range(20):
        eng.handle_block(speech)
    assert not eng.dictation.is_active
    assert not kb.events


def test_click_like_block_no_action():
    kb = FakeKeyboard()
    eng, c, t, scheduled = _engine(kb)
    click = np.full(800, 800, dtype=np.int16)
    for _ in range(4):
        eng.handle_block(click)
    assert not eng.dictation.is_active


def test_non_slot1_detection_ignored():
    kb = FakeKeyboard()
    eng, c, t, scheduled = _engine(kb)
    det = {
        "slot": 2,
        "similarity": 0.97,
        "similarity_margin": 0.1,
        "square_score": 0.05,
        "rms": 5000,
        "duration_ms": 1000,
    }
    assert not passes_audio_only_button(det, c)
    eng.handle_block(np.zeros(800, dtype=np.int16))
    assert not eng.dictation.is_active
