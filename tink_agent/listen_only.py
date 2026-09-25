"""Diagnostic capture-only mode when ~/.tink-agent/LISTEN_ONLY exists."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LISTEN_FLAG_PATH = Path.home() / ".tink-agent" / "LISTEN_ONLY"
DEFAULT_LOG_PATH = Path.home() / "Library" / "Logs" / "TinkAgent-listen.log"


def listen_flag_present(flag_path: Path | None = None) -> bool:
    path = flag_path if flag_path is not None else LISTEN_FLAG_PATH
    return path.is_file()


def spectral_centroid_hz(block: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(block, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    total = float(mag.sum())
    if total < 1e-12:
        return 0.0
    return float((freqs * mag).sum() / total)


class ListenProbeLogger:
    def __init__(
        self,
        *,
        log_path: Path | None = None,
        sample_rate: int = 16000,
        flag_path: Path | None = None,
    ) -> None:
        self.log_path = log_path if log_path is not None else DEFAULT_LOG_PATH
        self.sample_rate = sample_rate
        self.flag_path = flag_path if flag_path is not None else LISTEN_FLAG_PATH
        self._lock = threading.Lock()
        self._fh = None
        self._knob_source = "audio_fallback"
        self._serial_link = 0

    @classmethod
    def try_activate(
        cls,
        sample_rate: int = 16000,
        flag_path: Path | None = None,
    ) -> ListenProbeLogger | None:
        if not listen_flag_present(flag_path):
            return None
        return cls(sample_rate=sample_rate, flag_path=flag_path)

    def refresh_active(self) -> bool:
        return listen_flag_present(self.flag_path)

    def write_header(self) -> None:
        self._write("LISTEN_ONLY active")

    def _write(self, line: str) -> None:
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._fh is None:
                self._fh = open(self.log_path, "a", buffering=1, encoding="utf-8")
            self._fh.write(line + "\n")

    def log_knob_link(self, connected: bool, detail: str = "") -> None:
        link = int(connected)
        src = "serial" if connected else "audio_fallback"
        if src == self._knob_source and link == self._serial_link:
            return
        self._knob_source = src
        self._serial_link = link
        msg = f"knob_source={src} link={link} {detail}".strip()
        self._write(msg)

    def log_serial_token(self, token: str) -> None:
        self._write(f"serial {token}")

    def log_block(self, block: np.ndarray, metrics: dict | None) -> None:
        block = np.asarray(block).reshape(-1)
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if block.size else 0.0
        peak = int(np.max(np.abs(block))) if block.size else 0
        m = metrics or {}
        best = m.get("best_slot", "-")
        sim = m.get("similarity", "-")
        sq = m.get("square_score", "-")
        centroid = spectral_centroid_hz(block, self.sample_rate)
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
        self._write(
            f"{ts} rms={rms:.1f} peak={peak} slot={best} sim={sim} sq={sq} "
            f"centroid_hz={centroid:.1f} knob_src={self._knob_source}"
        )

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
