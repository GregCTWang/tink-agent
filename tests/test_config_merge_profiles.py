from tink_agent.config import Config, merge_dictation_profiles
from tink_agent.dictation import DEFAULT_PROFILES


def test_merge_backfills_restore_on_cancel_by_match():
    existing = [
        {
            "match": "Claude",
            "mode": "toggle",
            "keys": ["cmd", "d"],
        },
    ]
    merged = merge_dictation_profiles(existing, DEFAULT_PROFILES)
    claude = next(p for p in merged if p["match"] == "Claude")
    assert claude["restore_on_cancel"] is True
    assert claude["cancel_fallback"] == "none"
    assert claude["send_delay_ms"] == 200


def test_merge_does_not_overwrite_user_restore_flag():
    existing = [{"match": "Claude", "mode": "toggle", "keys": ["cmd", "d"], "restore_on_cancel": False}]
    merged = merge_dictation_profiles(existing, DEFAULT_PROFILES)
    claude = next(p for p in merged if p["match"] == "Claude")
    assert claude["restore_on_cancel"] is False


def test_config_load_merges_profiles(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"dictation_profiles":[{"match":"Grok Bot","mode":"toggle","keys":["cmd","d"]}]}'
    )
    loaded = Config.load(path)
    grok = next(p for p in loaded.dictation_profiles if "Grok" in p["match"])
    assert "restore_on_cancel" in grok
    assert grok["cancel_fallback"] == "none"
