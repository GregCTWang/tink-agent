"""macOS Accessibility backend for dictation restore (pyobjc ApplicationServices)."""
from __future__ import annotations

from typing import Callable

from .ax_restore import FocusSnapshot, element_identity, is_text_input_role

_ELECTRON_PIDS: set[int] = set()


def _ax_call(element, attr: str):
    from ApplicationServices import AXUIElementCopyAttributeValue

    err, val = AXUIElementCopyAttributeValue(element, attr, None)
    if err != 0:
        return None
    return val


def _ax_set(element, attr: str, value) -> bool:
    from ApplicationServices import AXUIElementSetAttributeValue

    err = AXUIElementSetAttributeValue(element, attr, value)
    return err == 0


def _position_size(element) -> tuple[float, float, float, float]:
    pos = _ax_call(element, "AXPosition") or None
    size = _ax_call(element, "AXSize") or None
    if pos is None or size is None:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        return (float(pos.x), float(pos.y), float(size.width), float(size.height))
    except Exception:  # noqa: BLE001
        return (0.0, 0.0, 0.0, 0.0)


def _ensure_electron_ax(app_element, pid: int) -> None:
    if pid in _ELECTRON_PIDS:
        return
    _ax_set(app_element, "AXManualAccessibility", True)
    _ax_set(app_element, "AXEnhancedUserInterface", True)
    _ELECTRON_PIDS.add(pid)


def _focused_element():
    from ApplicationServices import (
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
    )

    sys_wide = AXUIElementCreateSystemWide()
    app_ref = _ax_call(sys_wide, "AXFocusedApplication")
    if app_ref is None:
        return None, 0
    pid_ref = _ax_call(app_ref, "AXPID")
    pid = int(pid_ref) if pid_ref is not None else 0
    app_el = AXUIElementCreateApplication(pid)
    _ensure_electron_ax(app_el, pid)
    focused = _ax_call(app_el, "AXFocusedUIElement")
    if focused is None:
        focused = _ax_call(sys_wide, "AXFocusedUIElement")
    return focused, pid


def _snapshot_from_element(element, pid: int) -> FocusSnapshot | None:
    if element is None:
        return None
    role = str(_ax_call(element, "AXRole") or "")
    subrole = str(_ax_call(element, "AXSubrole") or "")
    desc = str(_ax_call(element, "AXDescription") or "")
    val = _ax_call(element, "AXValue")
    value_readable = val is not None and isinstance(val, str)
    value = val if value_readable else None
    sel = _ax_call(element, "AXSelectedTextRange")
    selection = None
    if sel is not None:
        try:
            selection = (int(sel.location), int(sel.length))
        except Exception:  # noqa: BLE001
            selection = None
    pos = _position_size(element)
    ident = element_identity(pid, role, subrole, pos)
    return FocusSnapshot(
        pid=pid,
        identity=ident,
        role=role,
        subrole=subrole,
        value=value,
        value_readable=value_readable,
        selection=selection,
        description=desc,
    )


class MacAxPort:
    def __init__(
        self,
        router,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
    ):
        self._router = router
        self._dispatch = dispatch or (lambda fn: fn())
        self._element_by_identity: dict[str, object] = {}

    def snapshot_focused(self) -> FocusSnapshot | None:
        element, pid = _focused_element()
        snap = _snapshot_from_element(element, pid)
        if snap is not None:
            self._element_by_identity[snap.identity] = element
        return snap

    def read_focused(self) -> FocusSnapshot | None:
        element, pid = _focused_element()
        snap = _snapshot_from_element(element, pid)
        if snap is not None and element is not None:
            self._element_by_identity[snap.identity] = element
        return snap

    def same_element(self, a: FocusSnapshot, b: FocusSnapshot) -> bool:
        return a.identity == b.identity and a.pid == b.pid

    def _element(self, snap: FocusSnapshot):
        return self._element_by_identity.get(snap.identity)

    def try_set_value(self, snap: FocusSnapshot, text: str) -> bool:
        el = self._element(snap)
        if el is None:
            return False
        if not is_text_input_role(snap.role):
            return False
        if not _ax_set(el, "AXValue", text):
            return False
        read_back = _ax_call(el, "AXValue")
        return isinstance(read_back, str) and read_back == text

    def try_set_selection(self, snap: FocusSnapshot, location: int, length: int) -> bool:
        el = self._element(snap)
        if el is None:
            return False
        try:
            from ApplicationServices import AXValueCreate, kAXValueCFRangeType
        except ImportError:
            return False
        rng = AXValueCreate(kAXValueCFRangeType, (int(location), int(length)))
        if rng is None:
            return False
        return _ax_set(el, "AXSelectedTextRange", rng)

    def send_delete_selection(self) -> None:
        self._router.fire_named_action("backspace")

    def send_select_all_delete(self) -> None:
        self._dispatch(lambda: self._router.dictation_tap(["cmd", "a"]))
        self._dispatch(lambda: self._router.fire_named_action("backspace"))


def probe_focused() -> str:
    """Human-readable probe of focused AX element (for ax-probe CLI)."""
    try:
        element, pid = _focused_element()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    if element is None:
        return "no focused element"
    snap = _snapshot_from_element(element, pid)
    if snap is None:
        return "could not build snapshot"
    sel = snap.selection if snap.selection else (-1, -1)
    settable = "unknown"
    try:
        from ApplicationServices import AXUIElementIsAttributeSettable

        err, ok = AXUIElementIsAttributeSettable(element, "AXValue", None)
        settable = "yes" if err == 0 and ok else "no"
    except Exception:  # noqa: BLE001
        pass
    vlen = len(snap.value) if snap.value is not None else -1
    return (
        f"pid={pid} role={snap.role} subrole={snap.subrole}\n"
        f"description={snap.description[:80]!r}\n"
        f"value_readable={snap.value_readable} value_len={vlen} "
        f"AXValue_settable={settable}\n"
        f"selection=({sel[0]},{sel[1]}) identity={snap.identity}"
    )


def create_mac_ax_port(router, dispatch=None) -> MacAxPort | None:
    try:
        import ApplicationServices  # noqa: F401
    except ImportError:
        return None
    return MacAxPort(router, dispatch=dispatch)
