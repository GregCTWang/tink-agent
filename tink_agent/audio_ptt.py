"""Handle-squeeze PTT proxy when the serial knob link is unavailable (3.5 mm only)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioPttTracker:
    """Infer mic handle down/up from RMS vs an adaptive noise floor."""

    onset_rms: float
    onset_min_blocks: int
    release_rms: float
    hangover_blocks: int
    auto_floor: bool
    floor_released_rms: float
    floor_ema_alpha: float

    _held: bool = field(default=False, init=False)
    _press_streak: int = field(default=0, init=False)
    _release_streak: int = field(default=0, init=False)
    _ema: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._ema = self.floor_released_rms

    @property
    def held(self) -> bool:
        return self._held

    def reset(self) -> None:
        self._held = False
        self._press_streak = 0
        self._release_streak = 0

    def _press_threshold(self) -> float:
        base = max(self.onset_rms, self.floor_released_rms * 2.0)
        if self.auto_floor:
            base = max(base, self._ema * 2.5)
        return base

    def _release_threshold(self) -> float:
        if self.auto_floor:
            return max(self.release_rms, self._ema * 1.35)
        return self.release_rms

    def update(self, rms: float) -> str | None:
        """Return 'down' or 'up' on edge transitions, else None."""
        if self.auto_floor and rms < self._press_threshold():
            alpha = self.floor_ema_alpha
            self._ema = (1.0 - alpha) * self._ema + alpha * rms

        if not self._held:
            if rms >= self._press_threshold():
                self._press_streak += 1
                if self._press_streak >= self.onset_min_blocks:
                    self._held = True
                    self._press_streak = 0
                    self._release_streak = 0
                    return "down"
            else:
                self._press_streak = 0
            return None

        if rms < self._release_threshold():
            self._release_streak += 1
            if self._release_streak >= self.hangover_blocks:
                self._held = False
                self._release_streak = 0
                self._press_streak = 0
                return "up"
        else:
            self._release_streak = 0
        return None
