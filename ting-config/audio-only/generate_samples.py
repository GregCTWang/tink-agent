#!/usr/bin/env python3
"""Generate pure-tone WAV cues for audio-only FX-MIC mode (16 kHz mono int16)."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
DURATION_S = 0.18
LEVEL = 0.85

TONES = {
    "1.wav": 5600.0,
    "2.wav": 7400.0,
    "3.wav": 4800.0,
    "4.wav": 6200.0,
}


def write_tone(path: Path, freq: float) -> None:
    n = int(SAMPLE_RATE * DURATION_S)
    frames = bytearray()
    for i in range(n):
        t = i / SAMPLE_RATE
        env = 1.0
        if i < SAMPLE_RATE * 0.01:
            env = i / (SAMPLE_RATE * 0.01)
        elif i > n - SAMPLE_RATE * 0.02:
            env = (n - i) / (SAMPLE_RATE * 0.02)
        sample = int(32767 * LEVEL * env * math.sin(2 * math.pi * freq * t))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(frames)


def main() -> None:
    out = Path(__file__).resolve().parent / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for name, freq in TONES.items():
        write_tone(out / name, freq)
        print(f"wrote {out / name} @ {freq:.0f} Hz")


if __name__ == "__main__":
    main()
