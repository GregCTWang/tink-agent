"""macOS Accessibility backend for dictation restore (pyobjc ApplicationServices)."""
from __future__ import annotations

import sys
import time
from typing import Callable

from .ax_restore import FocusSnapshot, SnapshotAttempt, element_identity, is_text_input_role

_ELECTRON_PIDS: set[int] = set()
_AX_TRUST_LOGGED = False


def _ax_get(element, attr: str) -> tuple[int, object | None]:
    from ApplicationServices import AXUIElementCopyAttributeValue

    err, val = AXUIElementCopyAttributeValue(element, attr, None)
    return int(err), val


def _ax_set(element, attr: str, value) -> tuple[int, bool]:
    from ApplicationServices import AXUIElementSetAttributeValue

    err = int(AXUIElementSetAttributeValue(element, attr, value))
    return err, err == 0


def prepare_electron_ax_for_pid(pid: int) -> None:
    if not pid or pid in _ELECTRON_PIDS:
        return
    try:
        from ApplicationServices import AXUIElementCreateApplication
    except ImportError:
        return
    app_el = AXUIElementCreateApplication(pid)
    _ax_set(app_el, "AXManualAccessibility", True)
    _ax_set(app_el, "AXEnhancedUserInterface", True)
    _ELECTRON_PIDS.add(pid)


def log_ax_trust_at_startup(logger) -> None:
    global _AX_TRUST_LOGGED
    if _AX_TRUST_LOGGED:
        return
    _AX_TRUST_LOGGED = True
    exe = sys.executable
    detail = f"ax trusted=unknown exe={exe}"
    try:
        from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions
        from Foundation import NSDictionary

        trusted = bool(AXIsProcessTrusted())
        detail = f"ax trusted={int(trusted)} exe={exe}"
        if logger is not None:
            if hasattr(logger, "dictation"):
                logger.dictation("-", detail)
            elif hasattr(logger, "restore"):
                logger.restore("-", detail)
        if not trusted:
            opts = NSDictionary.dictionaryWithDictionary_(
                {"AXTrustedCheckOptionPrompt": True}
            )
            AXIsProcessTrustedWithOptions(opts)
            if logger is not None and hasattr(logger, "dictation"):
                logger.dictation("-", "ax trusted=0 prompt=shown")
    except ImportError:
        if logger is not None and hasattr(logger, "dictation"):
            logger.dictation("-", detail + " (no ApplicationServices)")


def _position_size(element) -> tuple[float, float, float, float]:
    err_p, pos = _ax_get(element, "AXPosition")
    err_s, size = _ax_get(element, "AXSize")
    if err_p != 0 or err_s != 0 or pos is None or size is None:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        return (float(pos.x), float(pos.y), float(size.width), float(size.height))
    except Exception:  # noqa: BLE001
        return (0.0, 0.0, 0.0, 0.0)


def _lookup_focused(hint_pid: int = 0) -> tuple[object | None, int, str]:
    try:
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCreateSystemWide,
        )
    except ImportError:
        return None, 0, "step=import err=-1"

    fail_pid = ""
    if hint_pid:
        app_el = AXUIElementCreateApplication(hint_pid)
        prepare_electron_ax_for_pid(hint_pid)
        err, focused = _ax_get(app_el, "AXFocusedUIElement")
        if err == 0 and focused is not None:
            return focused, hint_pid, ""
        fail_pid = f"step=app_focused_ui pid={hint_pid} err={err}"

    sys_wide = AXUIElementCreateSystemWide()
    err, app_ref = _ax_get(sys_wide, "AXFocusedApplication")
    if err != 0 or app_ref is None:
        extra = fail_pid if hint_pid else ""
        base = f"step=focused_app err={err}"
        return None, 0, f"{base} {extra}".strip()

    err, pid_ref = _ax_get(app_ref, "AXPID")
    pid = int(pid_ref) if err == 0 and pid_ref is not None else hint_pid
    if pid:
        app_el = AXUIElementCreateApplication(pid)
        prepare_electron_ax_for_pid(pid)
        err_f, focused = _ax_get(app_el, "AXFocusedUIElement")
        if err_f == 0 and focused is not None:
            return focused, pid, ""
        err2, focused = _ax_get(sys_wide, "AXFocusedUIElement")
        if err2 == 0 and focused is not None:
            return focused, pid, ""
        parts = [f"step=focused_ui pid={pid} err={err_f}"]
        if hint_pid:
            parts.append(fail_pid)
        return None, pid, " ".join(parts)

    err2, focused = _ax_get(sys_wide, "AXFocusedUIElement")
    if err2 == 0 and focused is not None:
        return focused, 0, ""
    return None, 0, f"step=focused_ui err={err2}"


def _snapshot_from_element(element, pid: int) -> tuple[FocusSnapshot | None, str]:
    if element is None:
        return None, "step=element err=-1"
    err, role_val = _ax_get(element, "AXRole")
    if err != 0 or role_val is None:
        return None, f"step=role err={err}"
    role = str(role_val)
    _, subrole_val = _ax_get(element, "AXSubrole")
    subrole = str(subrole_val or "")
    _, desc_val = _ax_get(element, "AXDescription")
    desc = str(desc_val or "")
    err_v, val = _ax_get(element, "AXValue")
    value_readable = err_v == 0 and isinstance(val, str)
    value = val if value_readable else None
    _, sel = _ax_get(element, "AXSelectedTextRange")
    selection = None
    if sel is not None:
        try:
            selection = (int(sel.location), int(sel.length))
        except Exception:  # noqa: BLE001
            selection = None
    if err_v != 0 and not value_readable:
        # Still return snapshot for identity/unreadable restore path.
        pass
    pos = _position_size(element)
    ident = element_identity(pid, role, subrole, pos)
    return (
        FocusSnapshot(
            pid=pid,
            identity=ident,
            role=role,
            subrole=subrole,
            value=value,
            value_readable=value_readable,
            selection=selection,
            description=desc,
        ),
        "" if role else f"step=role_empty err={err_v}",
    )


class MacAxPort:
    def __init__(
        self,
        router,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        frontmost_pid_fn: Callable[[], int] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self._router = router
        self._dispatch = dispatch or (lambda fn: fn())
        self._frontmost_pid_fn = frontmost_pid_fn or (lambda: 0)
        self._sleep = sleep_fn or time.sleep
        self._element_by_identity: dict[str, object] = {}

    def snapshot_focused(self) -> SnapshotAttempt:
        last_fail = "step=unknown err=-1"
        hint = int(self._frontmost_pid_fn() or 0)
        for attempt in range(3):
            if attempt:
                self._sleep(0.1)
            element, pid, fail = _lookup_focused(hint)
            if element is None:
                last_fail = fail or last_fail
                continue
            snap, fail2 = _snapshot_from_element(element, pid)
            if snap is None:
                last_fail = fail2 or fail or last_fail
                continue
            self._element_by_identity[snap.identity] = element
            if not snap.value_readable and fail2:
                return SnapshotAttempt(snap, "")
            return SnapshotAttempt(snap, "")
        return SnapshotAttempt(None, last_fail)

    def read_focused(self) -> FocusSnapshot | None:
        attempt = self.snapshot_focused()
        return attempt.snapshot

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
        err, ok = _ax_set(el, "AXValue", text)
        if not ok:
            return False
        err_r, read_back = _ax_get(el, "AXValue")
        return err_r == 0 and isinstance(read_back, str) and read_back == text

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
        _, ok = _ax_set(el, "AXSelectedTextRange", rng)
        return ok

    def send_delete_selection(self) -> None:
        self._router.fire_named_action("backspace")

    def send_select_all_delete(self) -> None:
        self._dispatch(lambda: self._router.dictation_tap(["cmd", "a"]))
        self._dispatch(lambda: self._router.fire_named_action("backspace"))


def probe_focused() -> str:
    """Human-readable probe of focused AX element (for ax-probe CLI)."""
    log_ax_trust_at_startup(None)
    try:
        element, pid, fail = _lookup_focused(0)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    if element is None:
        return f"no focused element ({fail})"
    snap, _ = _snapshot_from_element(element, pid)
    if snap is None:
        return "could not build snapshot"
    sel = snap.selection if snap.selection else (-1, -1)
    err_s, settable = _ax_get(element, "AXValue")
    settable_s = "n/a"
    try:
        from ApplicationServices import AXUIElementIsAttributeSettable

        err2, ok = AXUIElementIsAttributeSettable(element, "AXValue", None)
        settable_s = "yes" if err2 == 0 and ok else f"no err={err2}"
    except Exception:  # noqa: BLE001
        settable_s = f"read_err={err_s}"
    vlen = len(snap.value) if snap.value is not None else -1
    return (
        f"pid={pid} role={snap.role} subrole={snap.subrole}\n"
        f"description={snap.description[:80]!r}\n"
        f"value_readable={snap.value_readable} value_len={vlen} "
        f"AXValue_settable={settable_s}\n"
        f"selection=({sel[0]},{sel[1]}) identity={snap.identity}"
    )


def create_mac_ax_port(router, dispatch=None, frontmost_pid_fn=None) -> MacAxPort | None:
    try:
        import ApplicationServices  # noqa: F401
    except ImportError:
        return None
    return MacAxPort(router, dispatch=dispatch, frontmost_pid_fn=frontmost_pid_fn)
