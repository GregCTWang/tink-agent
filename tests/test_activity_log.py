from tink_agent.activity_log import ActivityLogger


def _logger(tmp_path, enabled=True):
    return ActivityLogger(enabled=enabled, path=tmp_path / "activity.log",
                          now_fn=lambda: "2026-06-27T12:00:00")


def test_disabled_writes_nothing(tmp_path):
    lg = _logger(tmp_path, enabled=False)
    lg.transcript("hello", "Terminal")
    lg.action(1, "enter", "Terminal")
    assert not (tmp_path / "activity.log").exists()  # no file even created


def test_logs_transcript_action_blocked_as_tsv(tmp_path):
    lg = _logger(tmp_path)
    lg.transcript("approve this", "Terminal com.apple.Terminal")
    lg.action(1, "enter", "Terminal com.apple.Terminal")
    lg.blocked("secret text", "Safari com.apple.Safari")
    lg.close()
    lines = (tmp_path / "activity.log").read_text().splitlines()
    assert lines[0] == "2026-06-27T12:00:00\tvoice\tTerminal com.apple.Terminal\tapprove this"
    assert lines[1] == "2026-06-27T12:00:00\taction\tTerminal com.apple.Terminal\tslot1:enter"
    assert lines[2] == "2026-06-27T12:00:00\tblocked\tSafari com.apple.Safari\tsecret text"


def test_missing_app_becomes_dash(tmp_path):
    lg = _logger(tmp_path)
    lg.transcript("hi", None)
    lg.close()
    assert (tmp_path / "activity.log").read_text().split("\t")[2] == "-"


def test_tabs_and_newlines_sanitized(tmp_path):
    lg = _logger(tmp_path)
    lg.transcript("a\tb\nc", "App")
    lg.close()
    line = (tmp_path / "activity.log").read_text().rstrip("\n")
    assert line.count("\t") == 3  # only the field separators
    assert line.endswith("a b c")


def test_set_enabled_toggles(tmp_path):
    lg = _logger(tmp_path, enabled=False)
    lg.transcript("nope", "App")
    lg.set_enabled(True)
    lg.transcript("yes", "App")
    lg.close()
    text = (tmp_path / "activity.log").read_text()
    assert "yes" in text and "nope" not in text


def test_dictation_and_tone_log_kinds(tmp_path):
    lg = _logger(tmp_path)
    lg.dictation("Cursor app", "start profile=Cursor mode=hold")
    lg.tone(2, "Cursor app", "detected")
    lg.close()
    text = (tmp_path / "activity.log").read_text()
    assert "\tdictation\t" in text
    assert "start profile=Cursor" in text
    assert "\ttone\t" in text and "slot2" in text


def test_keys_log_kind(tmp_path):
    lg = _logger(tmp_path)
    lg.keys("Grok Bot com.grok", "cmd+d (start)")
    lg.keys("Grok Bot com.grok", "enter")
    lg.close()
    text = (tmp_path / "activity.log").read_text()
    assert "\tkeys\tGrok Bot com.grok\tcmd+d (start)" in text
    assert "\tkeys\tGrok Bot com.grok\tenter" in text
