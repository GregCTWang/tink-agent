"""FX-MIC grey-button recognition via normalized log-band spectral templates."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_BANDS = 64
PCM_HEADER_SKIP = 44


def read_pcm_int16(path: Path | str) -> np.ndarray:
    data = Path(path).read_bytes()
    if len(data) > PCM_HEADER_SKIP and data[:4] == b"RIFF":
        return np.frombuffer(data[PCM_HEADER_SKIP:], dtype=np.int16)
    return np.frombuffer(data, dtype=np.int16)


def block_rms(block: np.ndarray) -> float:
    x = np.asarray(block, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def spectral_feature_vector(block: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(block, dtype=np.float64).reshape(-1)
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    edges = np.linspace(0, len(spec), N_BANDS + 1, dtype=int)
    bands = np.array([spec[edges[i] : edges[i + 1]].mean() + 1e-12 for i in range(N_BANDS)])
    v = np.log(bands)
    v -= float(v.mean())
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _freq_bin_energy(block: np.ndarray, sample_rate: int, freq: float) -> float:
    x = np.asarray(block, dtype=np.float64).reshape(-1)
    n = len(x)
    k = int(round(freq * n / sample_rate))
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    if k >= len(spec):
        return 0.0
    lo = max(0, k - 2)
    hi = min(len(spec), k + 3)
    return float(spec[lo:hi].sum())


def slot1_square_wave_score(block: np.ndarray, sample_rate: int) -> float:
    """380 Hz square: strong fundamental + odd harmonics vs total low-mid energy."""
    f0 = _freq_bin_energy(block, sample_rate, 380)
    h3 = _freq_bin_energy(block, sample_rate, 1140)
    h5 = _freq_bin_energy(block, sample_rate, 1900)
    total = float(np.sum(np.abs(np.fft.rfft(block.astype(np.float64) * np.hanning(len(block)))))) + 1e-9
    odd = f0 + h3 + h5
    return odd / total


def templates_to_dict(templates: dict[int, np.ndarray]) -> dict:
    return {str(k): [float(x) for x in v.tolist()] for k, v in templates.items()}


def templates_from_dict(d: dict) -> dict[int, np.ndarray]:
    return {int(k): np.array(v, dtype=np.float64) for k, v in d.items()}


def _high_band_ratio(block: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(block, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    n = len(spec)
    mid = spec[n // 4 : n // 2].sum() + 1e-9
    high = spec[n // 2 :].sum() + 1e-9
    return float(high / (mid + high))


def _split_run_on_valleys(
    pcm: np.ndarray,
    start: int,
    end: int,
    block_size: int,
    valley_ratio: float = 0.82,
) -> list[tuple[int, int]]:
    if start >= end:
        return [(start, end)]
    subruns: list[tuple[int, int]] = []
    sub_start = start
    peak = 0.0
    quiet_streak = 0
    for bi in range(start, end + 1):
        b = pcm[bi * block_size : (bi + 1) * block_size]
        r = block_rms(b)
        peak = max(peak, r)
        if peak > 0 and r < peak * valley_ratio:
            quiet_streak += 1
        else:
            quiet_streak = 0
        if quiet_streak >= 2 and bi - quiet_streak > sub_start:
            subruns.append((sub_start, bi - quiet_streak))
            sub_start = bi - 1
            peak = r
            quiet_streak = 0
    subruns.append((sub_start, end))
    return subruns


def _loud_runs(pcm: np.ndarray, sample_rate: int, block_size: int, rms_min: float):
    runs: list[tuple[int, int, float]] = []
    n_blocks = len(pcm) // block_size
    i = 0
    while i < n_blocks:
        while i < n_blocks:
            b = pcm[i * block_size : (i + 1) * block_size]
            if block_rms(b) >= rms_min:
                break
            i += 1
        if i >= n_blocks:
            break
        start = i
        peak = 0.0
        while i < n_blocks:
            b = pcm[i * block_size : (i + 1) * block_size]
            r = block_rms(b)
            if r < rms_min * 0.45:
                break
            peak = max(peak, r)
            i += 1
        for s0, s1 in _split_run_on_valleys(pcm, start, i - 1, block_size):
            pk = max(
                block_rms(pcm[bi * block_size : (bi + 1) * block_size])
                for bi in range(s0, s1 + 1)
            )
            runs.append((s0, s1, pk))
    return runs


def _run_feature(pcm: np.ndarray, start: int, end: int, block_size: int, sample_rate: int) -> np.ndarray:
    vecs = []
    for bi in range(start, end + 1):
        b = pcm[bi * block_size : (bi + 1) * block_size]
        vecs.append(spectral_feature_vector(b, sample_rate))
    return np.mean(np.stack(vecs, axis=0), axis=0)


def _classify_run(
    pcm: np.ndarray,
    start: int,
    end: int,
    block_size: int,
    sample_rate: int,
) -> int:
    dur_ms = (end - start + 1) * block_size * 1000 / sample_rate
    blocks = [pcm[bi * block_size : (bi + 1) * block_size] for bi in range(start, end + 1)]
    peak_rms = max(block_rms(b) for b in blocks)
    sq = np.mean([slot1_square_wave_score(b, sample_rate) for b in blocks])
    hi = np.mean([_high_band_ratio(b, sample_rate) for b in blocks])
    if peak_rms >= 9000 and dur_ms <= 350:
        return 4
    if peak_rms >= 3500 and sq >= 0.22:
        return 1
    if hi >= 0.42 and 300 <= dur_ms <= 2000:
        return 3
    if dur_ms >= 600:
        return 2
    return 3


def learn_templates_from_pcm(
    pcm: np.ndarray,
    sample_rate: int = 16000,
    block_size: int = 800,
    rms_min: float = 2000.0,
) -> dict[int, np.ndarray]:
    runs = _loud_runs(pcm, sample_rate, block_size, rms_min)
    by_slot: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: []}
    for start, end, _peak in runs:
        slot = _classify_run(pcm, start, end, block_size, sample_rate)
        if slot in by_slot:
            by_slot[slot].append(_run_feature(pcm, start, end, block_size, sample_rate))
    out: dict[int, np.ndarray] = {}
    for slot, vecs in by_slot.items():
        if vecs:
            m = np.mean(np.stack(vecs, axis=0), axis=0)
            m /= np.linalg.norm(m) + 1e-12
            out[slot] = m
    return out


def learn_templates_from_wav(path: Path | str, **kwargs) -> dict[int, np.ndarray]:
    pcm = read_pcm_int16(path)
    return learn_templates_from_pcm(pcm, **kwargs)


def merge_template_dicts(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out = dict(a)
    for slot, vec in b.items():
        if slot not in out:
            out[slot] = vec
        else:
            m = (out[slot] + vec) / 2.0
            out[slot] = m / (np.linalg.norm(m) + 1e-12)
    return out


def default_fx_templates() -> dict[int, np.ndarray]:
    bundled = Path(__file__).resolve().parent / "data" / "fx_button_templates.json"
    if bundled.exists():
        return templates_from_dict(json.loads(bundled.read_text()))
    root = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "fxmic"
    t: dict[int, np.ndarray] = {}
    for name in ("beeps3-0924.wav", "beeps5-0924.wav"):
        p = root / name
        if p.exists():
            t = merge_template_dicts(t, learn_templates_from_wav(p))
    return t


@dataclass
class FxButtonDetector:
    templates: dict[int, np.ndarray]
    sample_rate: int
    block_size: int
    similarity_min: float = 0.72
    rms_min: float = 2000.0
    lockout_ms: int = 1000
    slot1_rms_min: float = 3500.0
    slot1_min_blocks: int = 3
    slot1_square_min: float = 0.22

    _lockout_blocks: int = field(init=False)
    _cooldown: int = field(init=False, default=0)
    _acc_slot: int | None = field(init=False, default=None)
    _acc_blocks: int = field(init=False, default=0)
    _acc_vec: np.ndarray | None = field(init=False, default=None)
    _acc_rms: float = field(init=False, default=0.0)
    _acc_sq: float = field(init=False, default=0.0)
    _acc_peak_rms: float = field(init=False, default=0.0)
    _valley_streak: int = field(init=False, default=0)
    button_active: bool = field(init=False, default=False)
    last_metrics: dict = field(init=False, default_factory=dict)
    last_detection: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        block_ms = 1000.0 * self.block_size / self.sample_rate
        self._lockout_blocks = max(1, int(round(self.lockout_ms / block_ms)))
        for slot, vec in self.templates.items():
            n = np.linalg.norm(vec)
            if n > 0:
                self.templates[slot] = vec / n

    def _best_match(self, feat: np.ndarray, rms: float) -> tuple[int | None, float, float]:
        best_slot = None
        best_sim = -1.0
        second_sim = -1.0
        for slot, tmpl in self.templates.items():
            sim = cosine_similarity(feat, tmpl)
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_slot = slot
            elif sim > second_sim:
                second_sim = sim
        if best_slot is None or best_sim < self.similarity_min:
            return None, best_sim, second_sim
        if best_slot == 4 and rms < 8500:
            return None, best_sim, second_sim
        return best_slot, best_sim, second_sim

    def _reset_acc(self) -> None:
        self._acc_slot = None
        self._acc_blocks = 0
        self._acc_vec = None
        self._acc_rms = 0.0
        self._acc_sq = 0.0
        self._acc_peak_rms = 0.0
        self._valley_streak = 0

    def _finalize_acc(self) -> int | None:
        if self._acc_blocks == 0 or self._acc_slot is None or self._acc_vec is None:
            self._reset_acc()
            return None
        slot = self._acc_slot
        avg_rms = self._acc_rms / self._acc_blocks
        avg_sq = self._acc_sq / self._acc_blocks
        blocks = self._acc_blocks
        dur_ms = blocks * self.block_size * 1000 / self.sample_rate
        feat = self._acc_vec.copy()
        self._reset_acc()
        if slot == 1:
            if avg_rms < self.slot1_rms_min or blocks < self.slot1_min_blocks or avg_sq < self.slot1_square_min:
                return None
        elif slot == 4:
            if avg_rms < 8500:
                return None
        elif avg_rms < self.rms_min:
            return None
        if self._cooldown > 0:
            return None
        block_ms = 1000.0 * self.block_size / self.sample_rate
        if slot == 4:
            self._cooldown = max(1, int(round(400 / block_ms)))
        else:
            self._cooldown = self._lockout_blocks
        sim = cosine_similarity(feat, self.templates[slot]) if slot in self.templates else 0.0
        second = -1.0
        for s, tmpl in self.templates.items():
            if s == slot:
                continue
            second = max(second, cosine_similarity(feat, tmpl))
        self.last_detection = {
            "slot": slot,
            "similarity": sim,
            "rms": avg_rms,
            "duration_ms": dur_ms,
            "square_score": avg_sq,
            "similarity_margin": sim - second if second >= 0 else sim,
        }
        return slot

    def process(self, block: np.ndarray) -> int | None:
        block = np.asarray(block).reshape(-1)
        rms = block_rms(block)
        feat = spectral_feature_vector(block, self.sample_rate)
        slot_guess, sim, second_sim = self._best_match(feat, rms)
        sq = slot1_square_wave_score(block, self.sample_rate)
        margin = sim - second_sim if second_sim >= 0 else 0.0
        self.last_metrics = {
            "rms": rms,
            "best_slot": slot_guess,
            "similarity": sim,
            "similarity_second": second_sim,
            "similarity_margin": margin,
            "square_score": sq,
        }
        self.button_active = self._acc_blocks > 0

        if self._cooldown > 0:
            self._cooldown -= 1

        quiet = rms < self.rms_min * 0.45
        if quiet:
            fired = self._finalize_acc()
            if fired is not None:
                return fired
            return None

        if slot_guess is None:
            return None

        if self._acc_slot is None or slot_guess != self._acc_slot:
            prev = self._finalize_acc()
            self._acc_slot = slot_guess
            self._acc_blocks = 1
            self._acc_vec = feat.copy()
            self._acc_rms = rms
            self._acc_sq = sq
            self._acc_peak_rms = rms
            self._valley_streak = 0
            if prev is not None:
                return prev

        self._acc_blocks += 1
        self._acc_rms += rms
        self._acc_sq += sq
        self._acc_peak_rms = max(self._acc_peak_rms, rms)
        self._acc_vec = (self._acc_vec * (self._acc_blocks - 1) + feat) / self._acc_blocks

        if self._acc_peak_rms > 0 and rms < self._acc_peak_rms * 0.82:
            self._valley_streak += 1
        else:
            self._valley_streak = 0
        if self._valley_streak >= 2 and self._acc_blocks >= self.slot1_min_blocks:
            fired = self._finalize_acc()
            if fired is not None:
                return fired
        return None


def knob_squeeze_onset_block(metrics: dict, config) -> bool:
    """First block of a handle squeeze: loud slot-1-like tone (not speech)."""
    if int(metrics.get("best_slot") or 0) != 1:
        return False
    rms = float(metrics.get("rms") or 0.0)
    sim = float(metrics.get("similarity") or 0.0)
    sq = float(metrics.get("square_score") or 0.0)
    rms_min = float(getattr(config, "dictation_audio_knob_squeeze_rms_min", 7500.0))
    sim_min = float(getattr(config, "dictation_audio_knob_squeeze_sim_min", 0.92))
    sq_min = float(getattr(config, "dictation_audio_knob_squeeze_sq_min", 0.20))
    return rms >= rms_min and sim >= sim_min and sq >= sq_min


def knob_squeeze_detection(detection: dict, config) -> bool:
    """Finalized accumulator burst from holding the knob (long slot-1 tone)."""
    slot = int(detection.get("slot") or 0)
    if slot != 1:
        return False
    dur = float(detection.get("duration_ms") or 0.0)
    dur_min = float(getattr(config, "dictation_audio_knob_squeeze_min_ms", 400.0))
    rms = float(detection.get("rms") or 0.0)
    sim = float(detection.get("similarity") or 0.0)
    sq = float(detection.get("square_score") or 0.0)
    rms_min = float(getattr(config, "dictation_audio_knob_squeeze_rms_min", 7500.0))
    sim_min = float(getattr(config, "dictation_audio_knob_squeeze_sim_min", 0.92))
    sq_min = float(getattr(config, "dictation_audio_knob_squeeze_sq_min", 0.20))
    return dur >= dur_min and rms >= rms_min and sim >= sim_min and sq >= sq_min


def is_grey_tap_detection(detection: dict, config) -> bool:
    """Short grey-key beep (slots 1–2); not a knob squeeze."""
    slot = int(detection.get("slot") or 0)
    if slot not in (1, 2):
        return False
    if knob_squeeze_detection(detection, config):
        return False
    dur_max = float(getattr(config, "dictation_audio_grey_max_ms", 350.0))
    if float(detection.get("duration_ms") or 0.0) > dur_max:
        return False
    return passes_audio_only_button(detection, config)


def passes_audio_only_button(detection: dict, config) -> bool:
    """Stricter acceptance for 3.5 mm-only mode (no serial knob)."""
    slot = int(detection.get("slot") or 0)
    sim = float(detection.get("similarity") or 0.0)
    margin = float(detection.get("similarity_margin") or 0.0)
    sq = float(detection.get("square_score") or 0.0)
    rms = float(detection.get("rms") or 0.0)
    sim_min = float(getattr(config, "fx_button_similarity_min_audio", 0.88))
    margin_min = float(getattr(config, "fx_button_similarity_margin_min", 0.06))
    if sim < sim_min or margin < margin_min:
        return False
    if slot == 1:
        sq_min = float(getattr(config, "fx_button_slot1_square_min_audio", 0.28))
        rms_min = float(getattr(config, "fx_button_slot1_rms_min", 3500.0))
        if sq < sq_min or rms < rms_min:
            return False
    else:
        rms_min = float(getattr(config, "fx_button_rms_min_audio", 2500.0))
        if rms < rms_min:
            return False
    return True


def replay_detections(
    pcm: np.ndarray,
    detector: FxButtonDetector,
    block_size: int,
) -> list[dict]:
    out: list[dict] = []
    n = len(pcm) // block_size
    for i in range(n):
        slot = detector.process(pcm[i * block_size : (i + 1) * block_size])
        if slot is not None:
            out.append({
                "time_s": i * block_size / detector.sample_rate,
                **detector.last_detection,
                "slot": slot,
            })
    return out
