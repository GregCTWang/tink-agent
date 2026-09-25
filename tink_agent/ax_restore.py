"""Accessibility snapshot/restore for dictation grey-cancel (Claude/Cursor)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

TEXT_INPUT_ROLES = frozenset(
    {"AXTextArea", "AXTextField", "AXComboBox", "AXSearchField"}
)


@dataclass(frozen=True)
class FocusSnapshot:
    pid: int
    identity: str
    role: str
    subrole: str
    value: str | None
    value_readable: bool
    selection: tuple[int, int] | None
    description: str = ""


@dataclass(frozen=True)
class SnapshotAttempt:
    snapshot: FocusSnapshot | None
    failure: str = ""


def element_identity(pid: int, role: str, subrole: str, pos: tuple[float, float, float, float]) -> str:
    x, y, w, h = pos
    return f"{pid}:{role}:{subrole}:{x:.0f},{y:.0f},{w:.0f},{h:.0f}"


def is_text_input_role(role: str) -> bool:
    return role in TEXT_INPUT_ROLES


def is_editor_like(role: str, subrole: str, description: str, profile_match: str) -> bool:
    blob = f"{role} {subrole} {description}".lower()
    if "source code" in blob or "code editor" in blob:
        return True
    if profile_match.lower() == "cursor":
        if "editor" in blob and "chat" not in blob and "composer" not in blob:
            return True
    return False


def is_safe_snapshot_target(snap: FocusSnapshot, profile_match: str) -> bool:
    if not is_text_input_role(snap.role):
        return False
    if is_editor_like(snap.role, snap.subrole, snap.description, profile_match):
        return False
    return True


def insertion_span(before: str, after: str, cursor: int) -> tuple[int, int] | None:
    """If after equals before with text inserted at cursor, return (start, end) of insertion."""
    if cursor < 0 or cursor > len(before):
        return None
    prefix = before[:cursor]
    suffix = before[cursor:]
    if not after.startswith(prefix):
        return None
    mid = after[len(prefix) : len(after) - len(suffix)] if suffix else after[len(prefix) :]
    if prefix + mid + suffix != after:
        return None
    if mid == "":
        return None
    return (cursor, cursor + len(mid))


def value_stable(prev: str | None, cur: str | None, stable_ms: float, since_mono: float | None, now: float) -> tuple[bool, float | None]:
    if prev != cur:
        return False, now
    if since_mono is None:
        return False, now
    return (now - since_mono) * 1000.0 >= stable_ms, since_mono


class AxPort(Protocol):
    def snapshot_focused(self) -> FocusSnapshot | None: ...

    def read_focused(self) -> FocusSnapshot | None: ...

    def same_element(self, a: FocusSnapshot, b: FocusSnapshot) -> bool: ...

    def try_set_value(self, snap: FocusSnapshot, text: str) -> bool: ...

    def try_set_selection(self, snap: FocusSnapshot, location: int, length: int) -> bool: ...

    def send_delete_selection(self) -> None: ...

    def send_select_all_delete(self) -> None: ...


@dataclass
class RestoreOutcome:
    ok: bool
    method: str = ""
    reason: str = ""


class CancelRestoreEngine:
    def __init__(
        self,
        ax: AxPort,
        *,
        poll_ms: int = 100,
        timeout_ms: int = 4000,
        settle_ms: int = 400,
        sleep_fn: Callable[[float], None] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ):
        self._ax = ax
        self._poll_s = poll_ms / 1000.0
        self._timeout_s = timeout_ms / 1000.0
        self._settle_ms = settle_ms
        self._sleep = sleep_fn or time.sleep
        self._mono = monotonic_fn or time.monotonic

    def restore_after_cancel(
        self,
        snap: FocusSnapshot,
        profile_match: str,
        log: Callable[[str], None],
    ) -> RestoreOutcome:
        if not snap.value_readable or snap.value is None:
            return self._restore_without_snapshot_value(snap, profile_match, log)
        before = snap.value
        cursor = snap.selection[0] if snap.selection else len(before)
        deadline = self._mono() + self._timeout_s
        last = before
        stable_since: float | None = None
        changed = False

        while self._mono() < deadline:
            cur_snap = self._ax.read_focused()
            if cur_snap is None:
                self._sleep(self._poll_s)
                continue
            cur_val = cur_snap.value if cur_snap.value_readable else None
            if cur_val is None:
                self._sleep(self._poll_s)
                continue
            if not is_text_input_role(cur_snap.role):
                log("restore skipped reason=non_text_role")
                return RestoreOutcome(False, reason="non_text_role")
            if is_editor_like(cur_snap.role, cur_snap.subrole, cur_snap.description, profile_match):
                log("restore skipped reason=editor_like")
                return RestoreOutcome(False, reason="editor_like")
            if not changed:
                if cur_val != before:
                    changed = True
                    last = cur_val
                    stable_since = self._mono()
            else:
                if cur_val != last:
                    last = cur_val
                    stable_since = self._mono()
                else:
                    ok_stable, _ = value_stable(last, cur_val, self._settle_ms, stable_since, self._mono())
                    if ok_stable:
                        return self._try_restore(
                            snap, cur_snap, before, cur_val, cursor, profile_match, log
                        )
            self._sleep(self._poll_s)

        log("restore skipped reason=timeout")
        return RestoreOutcome(False, reason="timeout")

    def _try_restore(
        self,
        start: FocusSnapshot,
        current: FocusSnapshot,
        before: str,
        after: str,
        cursor: int,
        profile_match: str,
        log: Callable[[str], None],
    ) -> RestoreOutcome:
        if not self._ax.same_element(start, current):
            log("restore skipped reason=focus_moved")
            return RestoreOutcome(False, reason="focus_moved")
        if not is_text_input_role(current.role):
            log("restore skipped reason=non_text_role")
            return RestoreOutcome(False, reason="non_text_role")
        if is_editor_like(current.role, current.subrole, current.description, profile_match):
            log("restore skipped reason=editor_like")
            return RestoreOutcome(False, reason="editor_like")
        span = insertion_span(before, after, cursor)
        if span is None:
            log("restore skipped reason=not_pure_insertion")
            return RestoreOutcome(False, reason="not_pure_insertion")
        ins_start, ins_end = span
        if self._ax.try_set_value(start, before):
            if start.selection and self._ax.try_set_selection(start, start.selection[0], start.selection[1]):
                log("restore done method=axvalue")
                return RestoreOutcome(True, method="axvalue")
        if self._ax.try_set_selection(current, ins_start, ins_end - ins_start):
            self._ax.send_delete_selection()
            log("restore done method=range_delete")
            return RestoreOutcome(True, method="range_delete")
        if before == "" and self._ax.same_element(start, current):
            self._ax.send_select_all_delete()
            log("restore done method=select_all")
            return RestoreOutcome(True, method="select_all")
        log("restore skipped reason=ax_failed")
        return RestoreOutcome(False, reason="ax_failed")

    def _restore_without_snapshot_value(
        self,
        start: FocusSnapshot,
        profile_match: str,
        log: Callable[[str], None],
    ) -> RestoreOutcome:
        deadline = self._mono() + self._timeout_s
        last: str | None = None
        stable_since: float | None = None
        saw_empty = False
        while self._mono() < deadline:
            cur = self._ax.read_focused()
            if cur is None or not cur.value_readable or cur.value is None:
                self._sleep(self._poll_s)
                continue
            if not self._ax.same_element(start, cur):
                log("restore skipped reason=focus_moved")
                return RestoreOutcome(False, reason="focus_moved")
            if not is_text_input_role(cur.role):
                log("restore skipped reason=non_text_role")
                return RestoreOutcome(False, reason="non_text_role")
            if is_editor_like(cur.role, cur.subrole, cur.description, profile_match):
                log("restore skipped reason=editor_like")
                return RestoreOutcome(False, reason="editor_like")
            if cur.value == "":
                saw_empty = True
            ok_stable, stable_since = value_stable(last, cur.value, self._settle_ms, stable_since, self._mono())
            last = cur.value
            if ok_stable and cur.value:
                if not saw_empty:
                    log("restore skipped reason=unreadable_prefilled")
                    return RestoreOutcome(False, reason="unreadable_prefilled")
                self._ax.send_select_all_delete()
                log("restore done method=select_all")
                return RestoreOutcome(True, method="select_all")
            self._sleep(self._poll_s)
        log("restore skipped reason=timeout")
        return RestoreOutcome(False, reason="timeout")
