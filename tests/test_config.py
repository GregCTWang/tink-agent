import json
from pathlib import Path
from tink_agent.config import Config


def test_defaults_have_validated_values():
    c = Config()
    assert c.device_name == "USB Audio Device"
    assert c.sample_rate == 16000
    assert c.block_size == 800
    assert {k: c.tones[k] for k in (1, 2, 3, 4)} == {1: 1500, 2: 2300, 3: 3100, 4: 3900}
    assert c.tone_rms_min == 1000
    assert c.tone_dominance_min == 0.9
    assert c.tone_debounce_ms == 300
    assert {k: c.slot_actions[k] for k in (1, 2, 3, 4)} == \
        {1: "enter", 2: "escape", 3: "ctrl_c", 4: "shift_tab"}


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    c = Config(tone_rms_min=1234)
    c.save(p)
    assert json.loads(p.read_text())["tone_rms_min"] == 1234
    loaded = Config.load(p)
    assert loaded.tone_rms_min == 1234
    assert {k: loaded.tones[k] for k in (1, 2, 3, 4)} == {1: 1500, 2: 2300, 3: 3100, 4: 3900}


def test_load_missing_writes_defaults(tmp_path):
    p = tmp_path / "config.json"
    loaded = Config.load(p)
    assert p.exists()
    assert loaded.device_name == "USB Audio Device"
    assert loaded.target_apps == []


def test_legacy_target_app_migrates_to_list(tmp_path):
    import json as _j
    p = tmp_path / "config.json"
    p.write_text(_j.dumps({"target_app": "com.apple.Terminal"}))
    loaded = Config.load(p)
    assert loaded.target_apps == ["com.apple.Terminal"]


def test_target_apps_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    Config(target_apps=["com.apple.Terminal", "com.googlecode.iterm2"]).save(p)
    loaded = Config.load(p)
    assert loaded.target_apps == ["com.apple.Terminal", "com.googlecode.iterm2"]


def test_defaults_have_eight_slots():
    c = Config()
    assert c.tones == {1: 1500, 2: 2300, 3: 3100, 4: 3900,
                       5: 2751, 6: 4218, 7: 5685, 8: 7153}
    assert c.slot_actions == {1: "enter", 2: "escape", 3: "ctrl_c", 4: "shift_tab",
                              5: "up", 6: "down", 7: "tab", 8: "backspace"}


def test_legacy_four_slot_config_backfills_modeB(tmp_path):
    # A config saved before mode B existed has only slots 1-4.
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "tones": {"1": 1500, "2": 2300, "3": 3100, "4": 3900},
        "slot_actions": {"1": "enter", "2": "escape", "3": "ctrl_c", "4": "shift_tab"},
    }))
    loaded = Config.load(p)
    assert loaded.tones[5] == 2751 and loaded.tones[8] == 7153
    assert loaded.slot_actions[5] == "up" and loaded.slot_actions[6] == "down"
    assert loaded.slot_actions[7] == "tab" and loaded.slot_actions[8] == "backspace"
    # User's mode-A choices are preserved, not overwritten.
    assert loaded.tones[1] == 1500 and loaded.slot_actions[1] == "enter"


def test_user_overrides_in_modeB_are_kept(tmp_path):
    p = tmp_path / "config.json"
    Config(slot_actions={**Config().slot_actions, 7: "tab"}).save(p)
    loaded = Config.load(p)
    assert loaded.slot_actions[7] == "tab"  # not clobbered by backfill


def test_eight_slot_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    Config().save(p)
    loaded = Config.load(p)
    assert loaded.tones == Config().tones
    assert loaded.slot_actions == Config().slot_actions


def test_onboarding_done_defaults_false():
    assert Config().onboarding_done is False


def test_onboarding_done_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    Config(onboarding_done=True).save(p)
    assert Config.load(p).onboarding_done is True


def test_dictation_defaults():
    c = Config()
    assert c.dictation_enabled is True
    assert c.dictation_onset_rms == 90.0
    assert c.button_detector == "fxmic"
    assert c.knob_serial_enabled is True
    assert c.dictation_idle_cap_ms == 180000
    assert c.dictation_min_toggle_gap_ms == 400
    assert c.knob_start_debounce_ms == 150
    assert c.dictation_release_action == "enter"
    assert c.fx_button_similarity_min == 0.72
    assert c.dictation_debug_log is False
    assert len(c.dictation_profiles) >= 3
    assert any(p.get("match") == "Cursor" for p in c.dictation_profiles)


def test_dictation_profiles_backfill_on_load(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"device_name": "USB Advanced Audio Device"}))
    loaded = Config.load(p)
    assert loaded.dictation_profiles


def test_onboarding_done_absent_loads_false(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"device_name": "USB Audio Device"}))
    assert Config.load(p).onboarding_done is False
