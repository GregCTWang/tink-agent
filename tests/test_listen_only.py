import numpy as np

from tink_agent.actions import ActionRouter
from tink_agent.config import Config
from tink_agent.detector import VoiceGate
from tink_agent.dictation import DictationController
from tink_agent.engine import Engine
from tink_agent.listen_only import ListenProbeLogger
from tink_agent.audio_buttons import make_button_detector


class FakeKey:
    enter = "ENTER"
    esc = "ESC"
    cmd = "CMD"
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

    class _P:
        def __init__(self, kb, k):
            self.kb, self.k = kb, k

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def pressed(self, k):
        return FakeKeyboard._P(self, k)


def _engine_with_probe(kb, log_path, flag_path):
    c = Config(vad_rms_start=1e9, vad_rms_end=1e9, dictation_onset_min_ms=50)
    buttons = make_button_detector(c)
    vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                   c.min_utterance_ms, c.sample_rate, c.block_size)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    dictation = DictationController(
        c, router, frontmost_fn=lambda: "Grok Bot com.test",
        dispatch=lambda fn: fn(), delay_fn=lambda _m, fn: fn(),
    )
    dictation.set_serial_link(False, "no_port")
    eng = Engine(c, buttons, vg, object(), router, dictation=dictation,
                 submit_fn=lambda fn: fn())
    probe = ListenProbeLogger(log_path=log_path, flag_path=flag_path, sample_rate=16000)
    probe.write_header()
    eng.listen_only = True
    eng._listen_probe = probe
    return eng, probe


def test_listen_only_skips_dictation_and_keys(tmp_path):
    flag = tmp_path / "LISTEN_ONLY"
    flag.write_text("")
    log = tmp_path / "listen.log"
    kb = FakeKeyboard()
    eng, probe = _engine_with_probe(kb, log, flag)
    metrics = {
        "rms": 9800,
        "best_slot": 1,
        "similarity": 0.985,
        "square_score": 0.29,
    }
    loud = np.full(800, 4000, dtype=np.int16)
    for _ in range(5):
        eng.button_detector._det.last_metrics = metrics
        eng.button_detector._det.button_active = True
        eng.handle_block(loud)
    assert not eng.dictation.is_active
    assert kb.events == []
    text = log.read_text()
    assert "LISTEN_ONLY active" in text
    assert "rms=" in text


def test_engine_without_probe_calls_dictation_observe():
    kb = FakeKeyboard()
    c = Config(vad_rms_start=1e9, vad_rms_end=1e9)
    buttons = make_button_detector(c)
    vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                   c.min_utterance_ms, c.sample_rate, c.block_size)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    dictation = DictationController(
        c, router, frontmost_fn=lambda: "Grok Bot com.test",
        dispatch=lambda fn: fn(), delay_fn=lambda _m, fn: fn(),
    )
    seen: list[int] = []
    dictation.observe_block = lambda *a, **k: seen.append(1)  # type: ignore[method-assign]
    eng = Engine(c, buttons, vg, object(), router, dictation=dictation,
                 submit_fn=lambda fn: fn())
    assert not getattr(eng, "listen_only", False)
    eng.handle_block(np.zeros(800, dtype=np.int16))
    assert seen == [1]
