from __future__ import annotations
import math
import numpy as np


def goertzel(samples: np.ndarray, freq: float, sample_rate: int) -> float:
    k = 2.0 * math.cos(2.0 * math.pi * freq / sample_rate)
    s1 = s2 = 0.0
    for x in samples:
        s0 = float(x) + k * s1 - s2
        s2 = s1
        s1 = s0
    return s1 * s1 + s2 * s2 - k * s1 * s2


class ToneDetector:
    def __init__(self, tones, rms_min, dominance_min, debounce_ms, sample_rate,
                 tonality_min=0.5):
        self.tones = dict(tones)
        self.rms_min = rms_min
        self.dominance_min = dominance_min
        # Minimum share of the block's TOTAL energy that must sit in the winning
        # tone bin. A pure on-bin sinusoid scores ~1.0; broadband speech scores
        # near 0. This is the gate that stops speech from firing button actions.
        self.tonality_min = tonality_min
        self.sample_rate = sample_rate
        self._debounce_blocks = max(1, int(debounce_ms / (1000 * 800 / sample_rate)))
        self._cooldown = 0
        self.tone_active = False
        self._last_slot = None
        self._was_tone_active = False
        self.last_metrics: dict = {}

    def process(self, block: np.ndarray) -> int | None:
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
        self.last_metrics = {"rms": rms, "best_freq": None, "dominance": None, "tonality": None}

        if rms < self.rms_min:
            self.tone_active = False
            if self._cooldown > 0:
                self._cooldown -= 1
            self._was_tone_active = False
            self._last_slot = None
            return None

        mags = {slot: goertzel(block, f, self.sample_rate) for slot, f in self.tones.items()}
        total = sum(mags.values()) + 1e-9
        best_slot = max(mags, key=mags.get)
        dominance = mags[best_slot] / total
        # Tonality: winning-bin power relative to total block energy. For an
        # N-sample on-bin sinusoid, goertzel power ~ (A*N/2)^2 and block energy
        # ~ N*A^2/2, so the ratio normalises to ~1.0; speech spreads energy and
        # scores ~0. Requiring both dominance AND tonality rejects speech that
        # merely happens to peak near a tone frequency.
        n = block.size
        block_energy = float(np.sum(block * block)) + 1e-9
        tonality = mags[best_slot] / (block_energy * n / 2.0)
        self.last_metrics = {
            "rms": rms,
            "best_freq": self.tones.get(best_slot),
            "best_slot": best_slot,
            "dominance": dominance,
            "tonality": tonality,
        }
        self.tone_active = (dominance >= self.dominance_min
                            and tonality >= self.tonality_min)

        if not self.tone_active:
            # Soft fail: dominance/tonality dipped this block (typical at the
            # attack/decay edges of an otherwise-held tone). Do NOT treat this as
            # a new-press boundary — only real silence (the rms < rms_min branch
            # above) resets the press state. Otherwise a single button press whose
            # tonality briefly wobbles would re-fire the same slot.
            if self._cooldown > 0:
                self._cooldown -= 1
            return None

        # Fire only on transitions:
        # (1) New slot detected (different from last slot), OR
        # (2) Tone just became active (was silent last block)
        if self._cooldown > 0:
            # In cooldown, don't fire
            self._cooldown -= 1
            return None

        # Cooldown is 0, check if we should fire
        is_new_slot = best_slot != self._last_slot
        is_tone_reactivation = not self._was_tone_active

        if is_new_slot or is_tone_reactivation:
            self._cooldown = self._debounce_blocks
            self._last_slot = best_slot
            self._was_tone_active = True
            return best_slot

        # Same slot, was already active, no fire
        return None


class VoiceGate:
    def __init__(self, rms_start, rms_end, hangover_ms,
                 min_utterance_ms, sample_rate, block_size,
                 max_utterance_ms=30000):
        self.rms_start = rms_start
        self.rms_end = rms_end
        block_ms = 1000 * block_size / sample_rate
        self._hangover_blocks = max(1, int(hangover_ms / block_ms))
        self._min_blocks = max(1, int(min_utterance_ms / block_ms))
        self._max_blocks = max(1, int(max_utterance_ms / block_ms))
        self._active = False
        self._silence = 0
        self._buf: list[np.ndarray] = []

    @property
    def active(self) -> bool:
        """True while an utterance is open (drives the menu bar capture indicator)."""
        return self._active

    def _close(self):
        blocks, self._buf = self._buf, []
        self._active = False
        self._silence = 0
        if len(blocks) < self._min_blocks:
            return None
        return np.concatenate(blocks).astype(np.int16)

    def process(self, block: np.ndarray, tone_active: bool):
        block = np.asarray(block).reshape(-1)
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if block.size else 0.0
        if not self._active:
            if rms >= self.rms_start:
                self._active = True
                self._silence = 0
                if not tone_active:
                    self._buf.append(block.astype(np.int16))
            return None
        # active
        if not tone_active and rms >= self.rms_end:
            self._buf.append(block.astype(np.int16))
        if len(self._buf) >= self._max_blocks:
            return self._close()
        if rms < self.rms_end:
            self._silence += 1
            if self._silence >= self._hangover_blocks:
                return self._close()
        else:
            self._silence = 0
        return None
