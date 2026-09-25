"""Post keyboard events with explicit modifier flags (macOS Quartz)."""
from __future__ import annotations

# USB keycodes for CGEventCreateKeyboardEvent (macOS)
_VK = {
    "esc": 0x35,
    "escape": 0x35,
    "ctrl": 0x3B,
    "control": 0x3B,
    "cmd": 0x37,
    "command": 0x37,
    "m": 0x2E,
    "d": 0x02,
    "z": 0x06,
}


def token_vk(token: str) -> int | None:
    return _VK.get(str(token).lower())


def post_key(vk: int, down: bool, flags: int = 0) -> str:
    """Post one key event; return flags detail for logging."""
    try:
        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventPost,
            CGEventSetFlags,
            kCGHIDEventTap,
        )
    except ImportError:
        raise RuntimeError("Quartz unavailable")
    ev = CGEventCreateKeyboardEvent(None, vk, down)
    CGEventSetFlags(ev, flags)
    CGEventPost(kCGHIDEventTap, ev)
    return f"flags={flags}"


def post_escape_cleared() -> str:
    detail = post_key(_VK["esc"], True, 0)
    post_key(_VK["esc"], False, 0)
    return detail


def post_cmd_z_cleared() -> str:
    cmd, z = _VK["cmd"], _VK["z"]
    post_key(cmd, True, 0)
    post_key(z, True, 0)
    post_key(z, False, 0)
    post_key(cmd, False, 0)
    return "flags=0"


def release_tokens_cleared(tokens: list[str]) -> str:
    """Release keys in reverse order with flags cleared."""
    vks = [token_vk(t) for t in tokens]
    vks = [v for v in vks if v is not None]
    detail = "flags=0"
    for vk in reversed(vks):
        detail = post_key(vk, False, 0)
    return detail
