import re
from pathlib import Path

import numpy as np

from tink_agent.actions import ActionRouter
from tink_agent.config import Config
from tink_agent.dictation import DictationController
from tink_agent.engine import Engine
from tink_agent.fx_button import passes_audio_only_button
from tink_agent.audio_ptt import AudioPttTracker
from tink_agent.detector import VoiceGate


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


class FakeButtons:
    def __init__(self):
        self.button_active = False
        self.last_metrics = {}
        self.last_detection = {}

    def process(self, block):
        return None


def _dictation(kb, front="Grok Bot com.test", mono=None):
    c = Config(
        dictation_onset_rms=300,
        dictation_onset_min_ms=50,
        dictation_auto_floor=True,
        dictation_floor_released_rms=28,
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


def test_audio_ptt_press_and_release():
    ptt = AudioPttTracker(
        onset_rms=300,
        onset_min_blocks=2,
        release_rms=40,
        hangover_blocks=3,
        auto_floor=True,
        floor_released_rms=28,
        floor_ema_alpha=0.08,
    )
    quiet = 25.0
    loud = 400.0
    assert ptt.update(quiet) is None
    assert ptt.update(loud) is None
    assert ptt.update(loud) == "down"
    assert ptt.held
    for _ in range(2):
        assert ptt.update(quiet) is None
    assert ptt.update(quiet) == "up"


def test_audio_fallback_ptt_starts_dictation_not_voice_alone():
    kb = FakeKeyboard()
    d, c, t, scheduled = _dictation(kb)
    loud = np.full(800, 400, dtype=np.int16)
    for _ in range(3):
        d.observe_block(loud, False)
    assert scheduled
    scheduled[0][1]()
    assert d.is_active
    assert ("press", "CMD") in kb.events


def test_audio_grey_send_and_cancel_no_immediate_restart():
    kb = FakeKeyboard()
    d, c, t, _scheduled = _dictation(kb)
    d._gate.activate()
    d._profile = c.dictation_profiles[0]
    d._speech_detected = True
    assert d.handle_audio_grey(1, {"slot": 1, "similarity": 0.92, "similarity_margin": 0.1, "square_score": 0.3, "rms": 4000})
    assert not d.is_active
    t["t"] += 0.05
    d.observe_block(np.full(800, 400, dtype=np.int16), False)
    assert not d.is_active


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
    good = {
        "slot": 2,
        "similarity": 0.91,
        "similarity_margin": 0.09,
        "rms": 5000,
        "square_score": 0.05,
    }
    assert passes_audio_only_button(good, c)


def test_greg_debug_log_quiet_voice_blocks_fail_audio_gate():
    log_path = Path("/home/ubuntu/.cursor/projects/workspace/uploads/dictation-debug-tail_2d94.log")
    if not log_path.exists():
        return
    c = Config()
    pattern = re.compile(
        r"block rms=([\d.]+).*sim=([\d.]+).*session=1"
    )
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
        }
        if passes_audio_only_button(det, c):
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
