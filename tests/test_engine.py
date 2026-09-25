import numpy as np
from tink_agent.config import Config
from tink_agent.detector import ToneDetector, VoiceGate
from tink_agent.actions import ActionRouter
from tink_agent.engine import Engine


class FakeKeyboard:
    class Key:
        enter = "ENTER"; esc = "ESC"; ctrl = "CTRL"
        tab = "TAB"; shift = "SHIFT"; up = "UP"; down = "DOWN"
    def __init__(self): self.events = []
    def press(self, k): self.events.append(("press", k))
    def release(self, k): self.events.append(("release", k))
    def type(self, s): self.events.append(("type", s))
    class _P:
        def __init__(s, kb, k): s.kb, s.k = kb, k
        def __enter__(s): return s
        def __exit__(s, *a): pass
    def pressed(self, k): return FakeKeyboard._P(self, k)


class FakeTranscriber:
    last_error = None
    def transcribe(self, samples, sr): return "approve this"


class FakeLogger:
    def __init__(self):
        self.calls = []
    def transcript(self, text, app=None): self.calls.append(("voice", text, app))
    def action(self, slot, action, app=None): self.calls.append(("action", slot, action, app))
    def blocked(self, what, app=None): self.calls.append(("blocked", what, app))
    def tone(self, slot, app=None, detail=""): self.calls.append(("tone", slot, app, detail))
    def dictation(self, app, detail): self.calls.append(("dictation", app, detail))


def _engine(kb, events, target_app="", frontmost="Terminal com.apple.Terminal",
            logger=None):
    c = Config(target_apps=[target_app] if target_app else [])
    det = ToneDetector(c.tones, c.tone_rms_min, c.tone_dominance_min,
                       c.tone_debounce_ms, c.sample_rate)
    vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                   c.min_utterance_ms, c.sample_rate, c.block_size)
    router = ActionRouter(c.slot_actions, keyboard=kb)
    return Engine(c, det, vg, FakeTranscriber(), router,
                  on_event=lambda k, p: events.append((k, p)),
                  submit_fn=lambda fn: fn(),  # synchronous
                  frontmost_fn=lambda: frontmost, logger=logger)


def _tone_block(freq=1500):
    sr, block = 16000, 800
    t = np.arange(0, block / sr, 1 / sr)[:block]
    return (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)


def test_tone_block_fires_keystroke_and_event():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events)
    sr, block = 16000, 800
    t = np.arange(0, block / sr, 1 / sr)[:block]
    tone = (np.sin(2 * np.pi * 1500 * t) * 12000).astype(np.int16)
    eng.handle_block(tone)
    assert ("press", "ENTER") in kb.events
    assert ("tone", 1) in events


def test_speech_then_silence_types_transcript():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events)
    rng = np.random.default_rng(0)
    for _ in range(10):
        eng.handle_block((rng.standard_normal(800) * 3000).astype(np.int16))
    for _ in range(20):
        eng.handle_block((rng.standard_normal(800) * 20).astype(np.int16))
    assert ("type", "approve this") in kb.events
    assert any(k == "transcript" for k, _ in events)


def test_disabled_engine_does_nothing():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events)
    eng.enabled = False
    eng.handle_block(_tone_block())
    assert kb.events == []


def test_target_app_gate_allows_matching_frontmost():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events, target_app="Terminal", frontmost="Terminal com.apple.Terminal")
    eng.handle_block(_tone_block())
    assert ("press", "ENTER") in kb.events
    assert ("tone", 1) in events


def test_target_app_gate_blocks_nonmatching_frontmost():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events, target_app="Terminal", frontmost="Safari com.apple.Safari")
    eng.handle_block(_tone_block())
    assert kb.events == []  # no keystroke into the wrong app
    assert any(k == "blocked" for k, _ in events)


def test_target_apps_matches_any_in_list():
    from tink_agent.config import Config as Cfg
    kb, events = FakeKeyboard(), []
    c = Cfg(target_apps=["com.apple.Terminal", "com.googlecode.iterm2"])
    det = ToneDetector(c.tones, c.tone_rms_min, c.tone_dominance_min,
                       c.tone_debounce_ms, c.sample_rate)
    vg = VoiceGate(c.vad_rms_start, c.vad_rms_end, c.vad_hangover_ms,
                   c.min_utterance_ms, c.sample_rate, c.block_size)
    eng = Engine(c, det, vg, FakeTranscriber(), ActionRouter(c.slot_actions, keyboard=kb),
                 on_event=lambda k, p: events.append((k, p)), submit_fn=lambda fn: fn(),
                 frontmost_fn=lambda: "iTerm com.googlecode.iterm2")
    eng.handle_block(_tone_block())
    assert ("press", "ENTER") in kb.events  # iTerm is in the allowed list


def test_target_app_gate_blocks_transcript_typing():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events, target_app="Terminal", frontmost="Notes com.apple.Notes")
    rng = np.random.default_rng(0)
    for _ in range(10):
        eng.handle_block((rng.standard_normal(800) * 3000).astype(np.int16))
    for _ in range(20):
        eng.handle_block((rng.standard_normal(800) * 20).astype(np.int16))
    assert not any(e[0] == "type" for e in kb.events)
    assert any(k == "blocked" for k, _ in events)


def test_is_capturing_reflects_voicegate_state():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events)
    assert eng.is_capturing is False
    rng = np.random.default_rng(0)
    eng.handle_block((rng.standard_normal(800) * 3000).astype(np.int16))
    assert eng.is_capturing is True  # utterance open
    for _ in range(20):
        eng.handle_block((rng.standard_normal(800) * 20).astype(np.int16))
    assert eng.is_capturing is False  # closed after silence


def test_logs_action_with_frontmost_app():
    kb, events, lg = FakeKeyboard(), [], FakeLogger()
    eng = _engine(kb, events, frontmost="Terminal com.apple.Terminal", logger=lg)
    eng.handle_block(_tone_block())  # slot 1 -> enter
    assert ("action", 1, "enter", "Terminal com.apple.Terminal") in lg.calls


def test_logs_transcript_with_frontmost_app():
    kb, events, lg = FakeKeyboard(), [], FakeLogger()
    eng = _engine(kb, events, frontmost="Notes com.apple.Notes", logger=lg)
    rng = np.random.default_rng(0)
    for _ in range(10):
        eng.handle_block((rng.standard_normal(800) * 3000).astype(np.int16))
    for _ in range(20):
        eng.handle_block((rng.standard_normal(800) * 20).astype(np.int16))
    assert ("voice", "approve this", "Notes com.apple.Notes") in lg.calls


def test_logs_blocked_when_wrong_app():
    kb, events, lg = FakeKeyboard(), [], FakeLogger()
    eng = _engine(kb, events, target_app="Terminal",
                  frontmost="Safari com.apple.Safari", logger=lg)
    eng.handle_block(_tone_block())
    assert any(c[0] == "blocked" for c in lg.calls)


def test_slot4_shift_tab():
    kb, events = FakeKeyboard(), []
    eng = _engine(kb, events)
    eng.handle_block(_tone_block(3900))  # slot 4 = shift_tab
    assert ("press", "SHIFT") not in kb.events  # shift is held via context mgr
    assert ("press", "TAB") in kb.events
    assert ("tone", 4) in events
