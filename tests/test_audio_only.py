import re
from pathlib import Path

import numpy as np

from tink_agent.actions import ActionRouter
from tink_agent.config import Config
from tink_agent.dictation import DictationController
from tink_agent.fx_button import (
    is_grey_tap_detection,
    knob_squeeze_detection,
    knob_squeeze_onset_block,
    passes_audio_only_button,
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


def _dictation(kb, front="Grok Bot com.test"):
    c = Config(
        dictation_onset_rms=300,
        dictation_audio_session_cooldown_ms=800,
        dictation_audio_post_button_ms=900,
    )
    router = ActionRouter(c.slot_actions, keyboard=kb)
    t = {"t": 0.0}
    scheduled: list = []

    def mono_fn():
        return t["t"]

    def delay_fn(ms, fn):
        scheduled.append((ms, fn))

    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: front,
        dispatch=lambda fn: fn(),
        delay_fn=delay_fn,
        monotonic_fn=mono_fn,
    )
    d.set_serial_link(False, "no_port")
    return d, c, t, scheduled


def _squeeze_block_metrics(rms: float = 9800.0) -> dict:
    return {
        "rms": rms,
        "best_slot": 1,
        "similarity": 0.985,
        "square_score": 0.29,
        "similarity_margin": 0.2,
    }


def _quiet_block_metrics() -> dict:
    return {
        "rms": 7.5,
        "best_slot": None,
        "similarity": 0.75,
        "square_score": 0.03,
    }


def test_knob_squeeze_vs_grey_tap_duration():
    c = Config()
    short = {
        "slot": 1,
        "duration_ms": 150,
        "rms": 8104,
        "similarity": 0.98,
        "square_score": 0.31,
        "similarity_margin": 0.2,
    }
    long = {
        "slot": 1,
        "duration_ms": 1250,
        "rms": 9154,
        "similarity": 0.98,
        "square_score": 0.29,
        "similarity_margin": 0.21,
    }
    assert not knob_squeeze_detection(short, c)
    assert is_grey_tap_detection(short, c)
    assert knob_squeeze_detection(long, c)
    assert not is_grey_tap_detection(long, c)


def test_greg_log_pattern_squeeze_then_silence_stays_active():
    """Replay Greg d0aacaf failure: loud squeeze tone then rms~7 must not stop session."""
    kb = FakeKeyboard()
    d, c, t, scheduled = _dictation(kb)
    loud = np.zeros(800, dtype=np.int16)
    quiet = np.full(800, 7, dtype=np.int16)
    for _ in range(16):
        d.observe_block(loud, True, tone_metrics=_squeeze_block_metrics())
    assert d.is_active
    assert ("press", "CMD") in kb.events
    presses = sum(1 for ev in kb.events if ev == ("press", "d"))
    for _ in range(12):
        d.observe_block(quiet, False, tone_metrics=_quiet_block_metrics())
    assert d.is_active
    assert sum(1 for ev in kb.events if ev == ("press", "d")) == presses


def test_squeeze_while_active_is_ignored():
    kb = FakeKeyboard()
    d, c, t, scheduled = _dictation(kb)
    d._gate.activate()
    d._profile = c.dictation_profiles[0]
    d.observe_block(np.zeros(800, dtype=np.int16), True, tone_metrics=_squeeze_block_metrics())
    assert d.is_active
    assert len([e for e in kb.events if e[0] == "press" and e[1] == "CMD"]) <= 1


def test_passes_audio_only_rejects_voice_like_margin():
    c = Config()
    ok = {
        "slot": 2,
        "similarity": 0.79,
        "similarity_margin": 0.02,
        "rms": 5000,
        "square_score": 0.05,
    }
    assert not passes_audio_only_button(ok, c)


def test_greg_debug_log_quiet_voice_blocks_fail_audio_gate():
    log_path = Path("/home/ubuntu/.cursor/projects/workspace/uploads/dictation-debug-tail_2d94.log")
    if not log_path.exists():
        return
    c = Config()
    pattern = re.compile(r"block rms=([\d.]+).*sim=([\d.]+).*session=1")
    false_pos = 0
    total = 0
    for line in log_path.read_text().splitlines():
        if not line.startswith("block "):
            continue
        m = pattern.search(line)
        if not m:
            continue
        rms = float(m.group(1))
        sim = float(m.group(2))
        if rms >= 500:
            continue
        total += 1
        det = {
            "slot": 2,
            "similarity": sim,
            "similarity_margin": 0.02,
            "rms": rms,
            "square_score": 0.04,
            "duration_ms": 50,
        }
        if is_grey_tap_detection(det, c):
            false_pos += 1
    assert total > 100
    assert false_pos == 0


def test_serial_path_unchanged_still_uses_k1():
    kb = FakeKeyboard()
    d, c, t, scheduled = _dictation(kb)
    d.set_serial_link(True, "port")
    d.on_serial_line("K1")
    assert scheduled
    scheduled[0][1]()
    assert d.is_active
    d.on_serial_line("K0")
    assert not d.is_active
