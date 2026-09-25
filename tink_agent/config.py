from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .dictation import DEFAULT_PROFILES

DEFAULT_PATH = Path.home() / ".tink-agent" / "config.json"
MW_BINARY = "/Applications/MacWhisper.app/Contents/MacOS/mw"


@dataclass
class Config:
    device_name: str = "USB Audio Device"
    sample_rate: int = 16000
    block_size: int = 800  # 50 ms at 16 kHz
    tones: dict = field(default_factory=lambda: {
        1: 1500, 2: 2300, 3: 3100, 4: 3900,      # mode A (orange pos 0)
        5: 2751, 6: 4218, 7: 5685, 8: 7153,      # mode B (orange pos 1, pitch +10.5)
    })
    tone_rms_min: float = 1000
    tone_dominance_min: float = 0.9
    tone_tonality_min: float = 0.5
    tone_debounce_ms: int = 300
    vad_rms_start: float = 800
    vad_rms_end: float = 500
    vad_hangover_ms: int = 800
    min_utterance_ms: int = 400
    max_utterance_ms: int = 30000
    stt_model: str = ""  # "" = mw's currently selected model (Large v3 Turbo)
    mw_binary: str = MW_BINARY
    # Transcription backend. "macwhisper" | "whisper-cpp" | "openai-whisper" use
    # built-in command templates; "custom" uses stt_command/stt_output below.
    stt_backend: str = "macwhisper"
    stt_command: list = field(default_factory=list)  # custom: tokens with {file}
    stt_output: str = "last_line"  # custom: last_line | all | file | dir
    slot_actions: dict = field(default_factory=lambda: {
        1: "enter", 2: "escape", 3: "ctrl_c", 4: "shift_tab",   # mode A
        5: "up", 6: "down", 7: "tab", 8: "backspace",           # mode B
    })
    # Push-to-talk dictation (app-native STT shortcuts; independent of VoiceGate/STT).
    dictation_enabled: bool = True
    dictation_onset_rms: float = 90.0
    # Fixed release cutoff when dictation_auto_floor is false; otherwise mid-point hint.
    dictation_release_rms: float = 35.0
    dictation_onset_min_ms: int = 80
    dictation_hangover_ms: int = 450
    dictation_onset_window_ms: int = 25
    dictation_send_delay_ms: int = 200
    dictation_max_session_ms: int = 60000
    dictation_profiles: list = field(default_factory=lambda: list(DEFAULT_PROFILES))
    # Release when RMS drops below threshold between released (~28) and held (~43) floors.
    dictation_auto_floor: bool = True
    dictation_floor_released_rms: float = 28.0
    dictation_floor_held_rms: float = 43.0
    dictation_floor_ema_alpha: float = 0.08
    dictation_cancel_on_any_tone: bool = True
    dictation_debug_log: bool = False
    dictation_debug_path: str = ""
    dictation_debug_rms_min: float = 500.0
    # When non-empty, keystrokes/typing only fire if the frontmost app's name or
    # bundle id matches one of these entries (substrings, e.g. bundle ids like
    # "com.apple.Terminal"). Empty list = act in any app.
    target_apps: list = field(default_factory=list)
    target_app: str = ""  # legacy single value; migrated into target_apps on load
    output_mode: str = "type"
    enabled: bool = True
    start_at_login: bool = False
    # First-run onboarding ("Set up TINK") completed at least once.
    onboarding_done: bool = False
    # Activity log: transcripts, target app, and actions written to a TSV file.
    log_activity: bool = False
    log_path: str = ""  # empty -> ~/Library/Logs/TinkAgent-activity.log

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON object keys must be strings; normalise int-keyed maps.
        d["tones"] = {str(k): v for k, v in self.tones.items()}
        d["slot_actions"] = {str(k): v for k, v in self.slot_actions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "tones" in d:
            d["tones"] = {int(k): int(v) for k, v in d["tones"].items()}
        if "slot_actions" in d:
            d["slot_actions"] = {int(k): str(v) for k, v in d["slot_actions"].items()}
        # Backfill mode-B slots (5-8) for configs written before they existed,
        # without clobbering any slot the user has set.
        defaults = cls()
        if "tones" in d:
            d["tones"] = {**defaults.tones, **d["tones"]}
        if "slot_actions" in d:
            d["slot_actions"] = {**defaults.slot_actions, **d["slot_actions"]}
        if "dictation_profiles" not in d:
            d["dictation_profiles"] = list(defaults.dictation_profiles)
        # Migrate legacy single target_app -> target_apps list.
        if not d.get("target_apps") and d.get("target_app"):
            d["target_apps"] = [d["target_app"]]
        known = {f for f in cls().__dict__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path=None) -> None:
        path = Path(path or DEFAULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path=None) -> "Config":
        path = Path(path or DEFAULT_PATH)
        if not path.exists():
            c = cls()
            c.save(path)
            return c
        return cls.from_dict(json.loads(path.read_text()))
