"""CLI: learn FX-MIC button templates from live mic or WAV recordings."""
from __future__ import annotations

import sys
from pathlib import Path

from tink_agent.config import Config, DEFAULT_PATH
from tink_agent.fx_button import (
    learn_templates_from_pcm,
    learn_templates_from_wav,
    merge_template_dicts,
    read_pcm_int16,
    templates_from_dict,
    templates_to_dict,
)


def _save_templates(config: Config, templates: dict) -> None:
    config.fx_button_templates = templates_to_dict(templates)
    config.button_detector = "fxmic"
    config.save()
    print(f"Saved templates for slots {sorted(templates.keys())} to {DEFAULT_PATH}")


def learn_from_wav(wav_path: Path, config: Config) -> None:
    templates = learn_templates_from_wav(wav_path, sample_rate=config.sample_rate, block_size=config.block_size)
    existing = templates_from_dict(config.fx_button_templates) if config.fx_button_templates else {}
    merged = merge_template_dicts(existing, templates)
    _save_templates(config, merged)
    print(f"Learned from {wav_path} ({len(templates)} slots in this file)")


def learn_interactive(config: Config) -> None:
    try:
        import sounddevice as sd
    except ImportError as e:
        print("sounddevice required for interactive learn", file=sys.stderr)
        raise SystemExit(1) from e

    sr = config.sample_rate
    print("Interactive learn: at each prompt, press the grey button 3 times at that white position.")
    merged: dict = templates_from_dict(config.fx_button_templates) if config.fx_button_templates else {}
    for slot in (1, 2, 3, 4):
        input(f"\nPress ENTER when ready to record position {slot} (3 presses)...")
        print("Recording 4 s...")
        pcm = sd.rec(int(4 * sr), samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        pcm = pcm.reshape(-1)
        part = learn_templates_from_pcm(pcm, sample_rate=sr, block_size=config.block_size)
        if slot in part:
            merged = merge_template_dicts(merged, {slot: part[slot]})
            print(f"  captured template for slot {slot}")
        else:
            print(f"  warning: could not derive slot {slot} from recording", file=sys.stderr)
    _save_templates(config, merged)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    config = Config.load()
    if len(argv) >= 2 and argv[0] == "--wav":
        learn_from_wav(Path(argv[1]), config)
        return
    if not argv:
        learn_interactive(config)
        return
    print("usage: tink-agent learn-buttons [--wav FILE]", file=sys.stderr)
    raise SystemExit(2)
