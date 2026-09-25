import re
from pathlib import Path

from tink_agent.config import Config
from tink_agent.dictation import DictationController
from tink_agent.actions import ActionRouter
from tink_agent.fx_button import passes_audio_only_button


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
        dictation_audio_session_cooldown_ms=800,
        dictation_audio_post_button_ms=900,
    )
    router = ActionRouter(c.slot_actions, keyboard=kb)
    t = {"t": 0.0}

    d = DictationController(
        c,
        router,
        frontmost_fn=lambda: front,
        dispatch=lambda fn: fn(),
        delay_fn=lambda _ms, fn: fn(),
        monotonic_fn=lambda: t["t"],
    )
    d.set_serial_link(False, "no_port")
    return d, c, t


def test_audio_grey_tap_start_and_send():
    kb = FakeKeyboard()
    d, c, _t = _dictation(kb)
    tap = {
        "slot": 1,
        "similarity": 0.98,
        "similarity_margin": 0.1,
        "square_score": 0.3,
        "rms": 9000,
        "duration_ms": 150,
    }
    assert d.handle_audio_grey(1, tap)
    assert d.is_active
    assert ("press", "CMD") in kb.events
    d._speech_detected = True
    assert d.handle_audio_grey(1, tap)
    assert not d.is_active
    assert ("press", "ENTER") in kb.events


def test_passes_audio_only_rejects_non_slot1_and_weak_margin():
    c = Config()
    weak = {
        "slot": 1,
        "similarity": 0.79,
        "similarity_margin": 0.02,
        "rms": 9000,
        "square_score": 0.31,
    }
    assert not passes_audio_only_button(weak, c)
    other_slot = {
        "slot": 2,
        "similarity": 0.91,
        "similarity_margin": 0.09,
        "rms": 5000,
        "square_score": 0.05,
    }
    assert not passes_audio_only_button(other_slot, c)
    good = {
        "slot": 1,
        "similarity": 0.91,
        "similarity_margin": 0.09,
        "rms": 9000,
        "square_score": 0.31,
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
            "slot": 1,
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
    d, c, t = _dictation(kb)
    scheduled: list = []

    def delay_fn(ms, fn):
        scheduled.append((ms, fn))

    d._delay = delay_fn
    d.set_serial_link(True, "port")
    d.on_serial_line("K1")
    assert scheduled
    scheduled[0][1]()
    assert d.is_active
    d.on_serial_line("K0")
    assert not d.is_active
