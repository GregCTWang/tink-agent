import numpy as np

from tink_agent.fx_button import (
    FxButtonDetector,
    default_fx_templates,
    read_pcm_int16,
    replay_detections,
)


def _detector():
    return FxButtonDetector(
        templates=default_fx_templates(),
        sample_rate=16000,
        block_size=800,
        lockout_ms=700,
    )


def test_beeps3_detects_all_four_positions():
    pcm = read_pcm_int16("tests/fixtures/fxmic/beeps3-0924.wav")
    dets = replay_detections(pcm, _detector(), 800)
    slots = [d["slot"] for d in dets]
    assert slots.count(1) >= 3
    assert any(s == 2 for s in slots)
    assert any(s == 3 for s in slots)
    assert any(s == 4 for s in slots)
    # Rough chronological grouping: pos1 before pos2 before pos3 before pos4
    first = {1: min(d["time_s"] for d in dets if d["slot"] == 1),
             2: min(d["time_s"] for d in dets if d["slot"] == 2),
             3: min(d["time_s"] for d in dets if d["slot"] == 3),
             4: min(d["time_s"] for d in dets if d["slot"] == 4)}
    assert first[1] < first[2] < first[3] < first[4]


def test_beeps5_detects_positions_in_order():
    pcm = read_pcm_int16("tests/fixtures/fxmic/beeps5-0924.wav")
    dets = replay_detections(pcm, _detector(), 800)
    slots = [d["slot"] for d in dets]
    # Recording started late: often only 2 presses at position 1
    assert slots.count(1) >= 2
    assert any(s == 2 for s in slots)
    assert any(s == 3 for s in slots)
    assert slots.count(4) >= 2


def test_speech_fixture_no_button_detections():
    from pathlib import Path
    wav = Path("tests/fixtures/speech.wav")
    if not wav.exists():
        return
    pcm = read_pcm_int16(wav)
    dets = replay_detections(pcm, _detector(), 800)
    assert dets == []


def test_synthetic_speech_rms_no_detection():
    det = _detector()
    speech = np.full(800, 350, dtype=np.int16)
    fired = []
    for _ in range(40):
        s = det.process(speech)
        if s:
            fired.append(s)
    assert fired == []
