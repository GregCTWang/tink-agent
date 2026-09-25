# Audio-only (3.5 mm) cue samples

Use these **instead of** the default `1.wav`–`4.wav` when running tink-agent with **only**
the line-out → USB audio adapter (no USB-C serial link). They are narrow pure tones so
voice is unlikely to match the FX grey-button detector.

## Flash onto TINGDISK (one-time setup, USB-C required once)

1. Copy `ting-config/audio-only/config.json` → `TINGDISK/config.json`
2. Copy `ting-config/audio-only/samples/*.wav` → `TINGDISK/` (same filenames as in config)
3. Power-cycle the mic (button above USB-C off, then handle on)

## Mapping (green slot → white press)

| Green position | File   | Nominal tone | App meaning (audio-only branch) |
|----------------|--------|--------------|----------------------------------|
| 0              | 1.wav  | 5600 Hz      | Grey **send** (end session + Enter) |
| 1              | 2.wav  | 7400 Hz      | Grey **cancel** (Escape + stop) |
| 2              | 3.wav  | 4800 Hz      | Reserved |
| 3              | 4.wav  | 6200 Hz      | Reserved |

Generate or regenerate WAVs:

```bash
python3 ting-config/audio-only/generate_samples.py
```

Do **not** copy onto Greg's mic until he approves.
