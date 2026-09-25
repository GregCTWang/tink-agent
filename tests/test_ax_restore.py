from tink_agent.ax_restore import (
    CancelRestoreEngine,
    FocusSnapshot,
    insertion_span,
    is_editor_like,
    is_safe_snapshot_target,
)


class FakeAxPort:
    def __init__(self):
        self.focused: FocusSnapshot | None = None
        self.set_value_ok = True
        self.set_selection_ok = True
        self.deleted = False
        self.select_all = False

    def snapshot_focused(self):
        return self.focused

    def read_focused(self):
        return self.focused

    def same_element(self, a: FocusSnapshot, b: FocusSnapshot) -> bool:
        return a.identity == b.identity

    def try_set_value(self, snap: FocusSnapshot, text: str) -> bool:
        if not self.set_value_ok:
            return False
        if self.focused and self.focused.identity == snap.identity:
            self.focused = FocusSnapshot(
                snap.pid,
                snap.identity,
                snap.role,
                snap.subrole,
                text,
                True,
                snap.selection,
                snap.description,
            )
        return True

    def try_set_selection(self, snap: FocusSnapshot, location: int, length: int) -> bool:
        return self.set_selection_ok

    def send_delete_selection(self) -> None:
        self.deleted = True
        if self.focused and self.focused.value:
            s, e = 0, len(self.focused.value)
            if self.focused.selection:
                s, ln = self.focused.selection
                e = s + ln
            self.focused = FocusSnapshot(
                self.focused.pid,
                self.focused.identity,
                self.focused.role,
                self.focused.subrole,
                self.focused.value[:s] + self.focused.value[e:],
                True,
                (s, 0),
                self.focused.description,
            )

    def send_select_all_delete(self) -> None:
        self.select_all = True
        if self.focused:
            self.focused = FocusSnapshot(
                self.focused.pid,
                self.focused.identity,
                self.focused.role,
                self.focused.subrole,
                "",
                True,
                (0, 0),
                self.focused.description,
            )


def _snap(value: str, ident: str = "1:AXTextArea::0,0,100,20", sel=(0, 0)):
    return FocusSnapshot(1, ident, "AXTextArea", "", value, True, sel)


def test_insertion_span_middle():
    assert insertion_span("hello world", "hello big world", 6) == (6, 10)


def test_empty_before_select_all_restore():
    ax = FakeAxPort()
    start = _snap("", ident="a")
    ax.focused = _snap("transcript only", ident="a")
    logs: list[str] = []

    def mono():
        return 0.0

    engine = CancelRestoreEngine(ax, settle_ms=0, timeout_ms=1000, monotonic_fn=mono)
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert out.ok and out.method in ("select_all", "axvalue")
    assert ax.focused and ax.focused.value == ""


def test_preexisting_text_axvalue_restore():
    ax = FakeAxPort()
    start = _snap("prefix SUFFIX", ident="b", sel=(7, 0))
    ax.focused = _snap("prefix inserted SUFFIX", ident="b")
    engine = CancelRestoreEngine(ax, settle_ms=0, timeout_ms=500, monotonic_fn=lambda: 0.0)
    logs: list[str] = []
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert out.ok and out.method == "axvalue"
    assert ax.focused and ax.focused.value == "prefix SUFFIX"


def test_focus_moved_skips():
    ax = FakeAxPort()
    start = _snap("", ident="a")
    ax.focused = _snap("only", ident="b")
    engine = CancelRestoreEngine(ax, settle_ms=0, timeout_ms=500, monotonic_fn=lambda: 0.0)
    logs: list[str] = []
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert not out.ok
    assert any("focus_moved" in x for x in logs)


def test_non_text_role_skips():
    ax = FakeAxPort()
    start = _snap("", ident="a")
    ax.focused = FocusSnapshot(1, "a", "AXButton", "", "", True, (0, 0), "")
    engine = CancelRestoreEngine(ax, settle_ms=0, timeout_ms=500, monotonic_fn=lambda: 0.0)
    logs: list[str] = []
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert not out.ok
    assert any("non_text_role" in x for x in logs)


def test_unreadable_prefilled_skips():
    ax = FakeAxPort()
    start = FocusSnapshot(1, "a", "AXTextArea", "", None, False, None)
    ax.focused = _snap("already here plus new", ident="a")
    engine = CancelRestoreEngine(ax, settle_ms=0, timeout_ms=500, monotonic_fn=lambda: 0.0)
    logs: list[str] = []
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert not out.ok
    assert any("unreadable_prefilled" in x for x in logs)


def test_unreadable_empty_then_transcript_select_all():
    start = FocusSnapshot(1, "a", "AXTextArea", "", None, False, None)

    class QueuePort(FakeAxPort):
        def __init__(self, values: list[str]):
            super().__init__()
            self._values = values
            self._i = 0

        def read_focused(self):
            idx = min(self._i, len(self._values) - 1)
            self._i += 1
            self.focused = _snap(self._values[idx], ident="a")
            return self.focused

    seq = QueuePort(["", "dictated", "dictated"])
    engine = CancelRestoreEngine(seq, settle_ms=0, timeout_ms=500, monotonic_fn=lambda: 0.0)
    logs: list[str] = []
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert out.ok and out.method == "select_all"


def test_timeout_skips():
    ax = FakeAxPort()
    start = _snap("hi", ident="a")
    ax.focused = _snap("hi", ident="a")
    t = [0.0]

    def mono():
        return t[0]

    def sleep(_s):
        t[0] += 1.0

    engine = CancelRestoreEngine(
        ax, settle_ms=400, timeout_ms=500, monotonic_fn=mono, sleep_fn=sleep
    )
    logs: list[str] = []
    out = engine.restore_after_cancel(start, "Claude", logs.append)
    assert not out.ok
    assert any("timeout" in x for x in logs)


def test_editor_like_not_safe():
    snap = FocusSnapshot(1, "x", "AXTextArea", "AXCodeEditor", "code", True, (0, 0), "Source Code")
    assert not is_safe_snapshot_target(snap, "Cursor")


def test_is_editor_like_cursor():
    assert is_editor_like("AXTextArea", "", "Editor pane", "Cursor")
