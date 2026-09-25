import time

import numpy as np

from tink_agent.config import Config
from tink_agent.detector import VoiceGate
from tink_agent.actions import ActionRouter
from tink_agent.dictation import DictationController
from tink_agent.engine import Engine
from tink_agent.audio_buttons import make_button_detector
from tink_agent.knob_serial import (
    KnobSerialMonitor,
    _exec_payload,
    _poll_loop_source,
    resolve_knob_port,
    unescape_exec_literal,
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


class SilentUntilInterruptSerial:
    """Simulates a device stuck in the poll loop until Ctrl-C."""

    def __init__(self):
        self.writes: list[bytes] = []
        self._prompt_ready = False

    def read(self, n):
        if self._prompt_ready:
            self._prompt_ready = False
            return b">>> "
        return b""

    def write(self, data):
        self.writes.append(bytes(data))
        if b"\x03" in data:
            self._prompt_ready = True

    def flush(self):
        pass


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


def _run_debounce(sched):
    assert sched, "knob debounce not scheduled"
    ms, fn = sched.pop(0)
    fn()
    return ms


def _drain_sched(sched):
    while sched:
        _, fn = sched.pop(0)
        fn()


def test_hold_speak_release_sends_enter_once():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    _run_debounce(sched)
    assert d.is_active
    for _ in range(8):
        eng.handle_block(_speech())
    d.on_serial_line("K0")
    assert not d.is_active
    d._restart_ready_at = 0.0
    _drain_sched(sched)
    assert ("press", "ENTER") in kb.events


def test_hold_no_speech_release_no_enter():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    _run_debounce(sched)
    assert d.is_active
    for _ in range(5):
        eng.handle_block(np.full(800, 26, dtype=np.int16))
    d.on_serial_line("K0")
    _drain_sched(sched)
    assert ("press", "ENTER") not in kb.events
    assert any(e[0] == "press" for e in kb.events)


def test_knob_tap_shorter_than_debounce_does_nothing():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    c = d.config
    c.knob_start_debounce_ms = 150
    d.on_serial_line("K1")
    d.on_serial_line("K0")
    assert not d.is_active
    assert kb.events == []
    _drain_sched(sched)
    assert not d.is_active


def test_grey_cancel_then_release_no_send():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    _run_debounce(sched)
    d.on_serial_line("G0")
    assert not d.is_active
    d.on_serial_line("K0")
    _drain_sched(sched)
    assert ("press", "ENTER") not in kb.events


def test_no_start_without_knob_held():
    kb = FakeKeyboard()
    eng, d, _, _ = _setup(kb, serial_linked=True)
    d.knob_held = False
    for _ in range(5):
        eng.handle_block(_speech())
    assert not d.is_active


def test_long_silent_hold_never_ends():
    kb = FakeKeyboard()
    eng, d, sched, _ = _setup(kb)
    d.on_serial_line("K1")
    _run_debounce(sched)
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
    sched: list = []

    def delay_fn(ms, fn):
        sched.append((ms, fn))

    d._delay = delay_fn
    d.on_serial_line("K1")
    _run_debounce(sched)
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


def test_exec_payload_single_line_no_raw_lf():
    payload = _exec_payload(_poll_loop_source(10))
    body = payload[:-2]
    assert b"\n" not in body
    assert b"\\n" in body
    src = unescape_exec_literal(payload)
    compile(src, "<poll>", "exec")
    assert "ui.sw(4)" in src
    assert "time.sleep_ms(10)" in src


def test_exec_payload_unescaped_compiles():
    payload = _exec_payload("import ui\nwhile True:\n pass\n")
    src = unescape_exec_literal(payload)
    compile(src, "<snippet>", "exec")


def test_resolve_port_hint():
    assert resolve_knob_port("/dev/cu.usbmodemEPTEST") == "/dev/cu.usbmodemEPTEST"


def test_repl_prompt_sends_ctrl_c_when_silent():
    clock = [0.0]

    def mono():
        return clock[0]

    def sleep(dt):
        clock[0] += dt

    ser = SilentUntilInterruptSerial()
    mon = KnobSerialMonitor(
        Config(),
        on_line=lambda _t: None,
        sleep_fn=sleep,
        monotonic_fn=mono,
    )
    buf = mon._wait_for_repl_prompt(ser, b"", deadline=10.0)
    assert b">>> " in buf
    joined = b"".join(ser.writes)
    assert joined.count(b"\x03") == 2
    assert b"\r" in joined


def test_monitor_parses_k_g_tokens():
    lines = []
    mon = KnobSerialMonitor(Config(), on_line=lines.append)
    buf, _ = mon._drain_lines(b"noise K1 trailing\nG0\n")
    assert buf == b""
    assert lines == ["K1", "G0"]
    assert mon.knob_held is True


def _tone_block(freq=1500):
    sr, block = 16000, 800
    t = np.arange(0, block / sr, 1 / sr)[:block]
    return (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)


def test_serial_linked_audio_tone_ignored_without_grey():
    kb = FakeKeyboard()
    eng, d, _, _ = _setup(kb)
    d.knob_held = False
    eng.handle_block(_tone_block(2300))
    assert ("press", "ESC") not in kb.events
    assert ("press", "ENTER") not in kb.events


def test_serial_grey_then_audio_fires_slot_action():
    kb = FakeKeyboard()
    eng, d, _, _ = _setup(kb)
    d.knob_held = False
    d.on_serial_line("G0")
    eng.handle_block(_tone_block(1500))
    assert ("press", "ENTER") in kb.events
