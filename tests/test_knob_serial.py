import time

import numpy as np

from tink_agent.config import Config
from tink_agent.detector import VoiceGate
from tink_agent.actions import ActionRouter
from tink_agent.dictation import DictationController
from tink_agent.engine import Engine
from tink_agent.audio_buttons import make_button_detector
from tink_agent.knob_serial import KnobSerialMonitor, resolve_knob_port, _exec_payload


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


class FakeSerial:
    def __init__(self):
        self.writes: list[bytes] = []
        self.read_queue: list[bytes] = []
        self.closed = False

    def read(self, n):
        if self.read_queue:
            return self.read_queue.pop(0)
        return b""

    def write(self, data):
        self.writes.append(bytes(data))

    def flush(self):
        pass

    def close(self):
        self.closed = True


def _speech():
    return np.full(800, 200, dtype=np.int16)


def _setup(kb, serial_linked=True):
    c = Config(
        vad_rms_start=1e9,
        dictation_onset_min_ms=50,
        knob_serial_enabled=True,
        button_detector="tone",
    )
    router = ActionRouter(c.slot_actions, keyboard=kb)
    scheduled = []

    def delay_fn(ms, fn):
        scheduled.append((ms, fn))

    dictation = DictationController(
        c, router, frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(), delay_fn=delay_fn,
    )
    if serial_linked:
        dictation.set_serial_link(True, "test")
        dictation.knob_held = True
    buttons = make_button_detector(c)
    vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                   c.min_utterance_ms, c.sample_rate, c.block_size)
    eng = Engine(c, buttons, vg, object(), router, dictation=dictation,
                 submit_fn=lambda fn: fn())
    return eng, dictation, scheduled, router


def test_hold_speak_release_sends_enter_once():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    for _ in range(2):
        eng.handle_block(_speech())
    assert d.is_active
    d.on_serial_line("K0")
    assert not d.is_active
    assert len(sched) == 1
    sched[0][1]()
    assert ("press", "ENTER") in kb.events


def test_grey_cancel_then_release_no_send():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    for _ in range(2):
        eng.handle_block(_speech())
    d.on_serial_line("G0")
    assert not d.is_active
    d.on_serial_line("K0")
    assert sched == []


def test_no_start_without_knob_held():
    kb = FakeKeyboard()
    eng, d, _, _ = _setup(kb, serial_linked=True)
    d.knob_held = False
    for _ in range(5):
        eng.handle_block(_speech())
    assert not d.is_active


def test_long_silent_hold_never_ends():
    kb = FakeKeyboard()
    eng, d, _, _ = _setup(kb)
    d.on_serial_line("K1")
    for _ in range(2):
        eng.handle_block(_speech())
    for _ in range(500):
        eng.handle_block(np.full(800, 26, dtype=np.int16))
    assert d.is_active


def test_idle_cap_when_released_and_silent():
    kb = FakeKeyboard()
    c = Config(
        vad_rms_start=1e9,
        dictation_onset_min_ms=50,
        dictation_idle_cap_ms=200,
        knob_serial_enabled=True,
        button_detector="tone",
    )
    router = ActionRouter(c.slot_actions, keyboard=kb)
    d = DictationController(
        c, router, frontmost_fn=lambda: "Grok Bot com.grok",
        dispatch=lambda fn: fn(), delay_fn=lambda _m, fn: fn(),
    )
    d.set_serial_link(True, "t")
    d.knob_held = True
    d.on_serial_line("K1")
    for _ in range(2):
        d.observe_block(_speech(), False)
    d.knob_held = False
    for _ in range(10):
        d.observe_block(np.full(800, 26, dtype=np.int16), False)
    assert not d.is_active


def test_audio_fallback_button_end():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb, serial_linked=False)
    for _ in range(2):
        eng.handle_block(_speech())
    assert d.is_active
    assert d.uses_audio_button_end()
    consumed = d.handle_button_end(2, {"similarity": 0.9})
    assert consumed
    sched[0][1]()


def test_exec_payload_is_read_only_poll():
    payload = _exec_payload("import ui\nwhile True: pass")
    assert b"exec(" in payload
    assert b"vfs" not in payload


def test_resolve_port_hint():
    assert resolve_knob_port("/dev/cu.usbmodemEPTEST") == "/dev/cu.usbmodemEPTEST"


def test_monitor_parses_k_g_tokens():
    lines = []
    mon = KnobSerialMonitor(Config(), on_line=lines.append)
    mon._handle_line("noise K1 trailing")
    mon._handle_line("G0")
    assert lines == ["K1", "G0"]
    assert mon.knob_held is True
