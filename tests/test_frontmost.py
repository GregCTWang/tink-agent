from tink_agent.frontmost import FrontmostTracker, is_self_app


def test_is_self_app():
    assert is_self_app("Python org.python.python")
    assert not is_self_app("Grok Bot com.x")


def test_tracker_ignores_self():
    cur = ["Grok Bot com.grok"]

    def raw():
        return cur[0]

    t = FrontmostTracker(raw)
    assert "Grok" in t()
    cur[0] = "Python org.python.python"
    assert "Grok" in t()
