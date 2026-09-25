"""Button detection: FX-MIC spectral templates or legacy pure-tone Goertzel."""
from __future__ import annotations

from tink_agent.detector import ToneDetector
from tink_agent.fx_button import FxButtonDetector, default_fx_templates, templates_from_dict


class ToneButtonAdapter:
    def __init__(self, detector: ToneDetector):
        self._det = detector

    def process(self, block):
        return self._det.process(block)

    @property
    def button_active(self) -> bool:
        return bool(self._det.tone_active)

    @property
    def last_metrics(self) -> dict:
        return getattr(self._det, "last_metrics", {})


class FxButtonAdapter:
    def __init__(self, detector: FxButtonDetector):
        self._det = detector

    def process(self, block):
        return self._det.process(block)

    @property
    def button_active(self) -> bool:
        return bool(self._det.button_active)

    @property
    def last_metrics(self) -> dict:
        return self._det.last_metrics

    @property
    def last_detection(self) -> dict:
        return self._det.last_detection


def make_button_detector(config):
    mode = getattr(config, "button_detector", "fxmic") or "fxmic"
    if mode == "tone":
        det = ToneDetector(
            config.tones,
            config.tone_rms_min,
            config.tone_dominance_min,
            config.tone_debounce_ms,
            config.sample_rate,
            tonality_min=config.tone_tonality_min,
        )
        return ToneButtonAdapter(det)
    templates = templates_from_dict(config.fx_button_templates) if config.fx_button_templates else default_fx_templates()
    det = FxButtonDetector(
        templates=dict(templates),
        sample_rate=config.sample_rate,
        block_size=config.block_size,
        similarity_min=config.fx_button_similarity_min,
        rms_min=config.fx_button_rms_min,
        lockout_ms=config.fx_button_lockout_ms,
        slot1_rms_min=config.fx_button_slot1_rms_min,
        slot1_min_blocks=max(1, int(config.fx_button_slot1_min_ms / (1000 * config.block_size / config.sample_rate))),
        slot1_square_min=config.fx_button_slot1_square_min,
    )
    return FxButtonAdapter(det)
