"""Read-only EP-2350 knob/grey state over USB CDC (MicroPython REPL polling)."""
from __future__ import annotations

import ast
import glob
import re
import threading
import time
from typing import Callable

TE_VID = 0x2367
TE_PID = 0x0620


def _poll_loop_source(sleep_ms: int) -> str:
    ms = max(5, min(int(sleep_ms), 50))
    return (
        "import ui,time\n"
        "lk=ui.sw(4);lg=ui.sw(0);print('HB')\n"
        "while True:\n"
        " k=ui.sw(4);g=ui.sw(0)\n"
        " if k!=lk:print('K'+str(k));lk=k\n"
        " if g!=lg:print('G'+str(g));lg=g\n"
        f" time.sleep_ms({ms})\n"
    )


def _exec_payload(code: str) -> bytes:
    escaped = (
        code.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace("'", "\\'")
    )
    line = f"exec('{escaped}')\r\n"
    if "\n" in line[:-2]:
        raise ValueError("exec payload must be a single line")
    return line.encode("utf-8")


def unescape_exec_literal(payload: bytes) -> str:
    """Recover Python source from exec('...') bytes (for tests)."""
    text = payload.decode("utf-8").strip()
    if not text.startswith("exec('") or not text.endswith("')"):
        raise ValueError("not an exec payload")
    inner = text[6:-2]
    return ast.literal_eval(f"'{inner}'")


def resolve_knob_port(port_hint: str = "", list_ports_fn=None) -> str | None:
    hint = (port_hint or "").strip()
    if hint:
        return hint
    matches = sorted(glob.glob("/dev/cu.usbmodemEP*"))
    if matches:
        return matches[0]
    list_ports_fn = list_ports_fn or _default_list_ports
    for dev, vid, pid in list_ports_fn():
        if vid == TE_VID and pid == TE_PID:
            return dev
    return None


def _default_list_ports() -> list[tuple[str, int, int]]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    out: list[tuple[str, int, int]] = []
    for p in list_ports.comports():
        vid = int(p.vid) if p.vid is not None else -1
        pid = int(p.pid) if p.pid is not None else -1
        dev = p.device or ""
        if dev:
            out.append((dev, vid, pid))
    return out


class KnobSerialMonitor:
    """Background reader; calls on_line('K1'|'K0'|'G0'|'G1'|'HB') and on_link(bool)."""

    def __init__(
        self,
        config,
        on_line: Callable[[str], None],
        on_link: Callable[[bool, str], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
        serial_factory=None,
        list_ports_fn=None,
        sleep_fn=None,
        monotonic_fn=None,
    ):
        self.config = config
        self._on_line = on_line
        self._on_link = on_link or (lambda _ok, _msg: None)
        self._on_debug = on_debug or (lambda _s: None)
        self._serial_factory = serial_factory or _open_serial
        self._list_ports_fn = list_ports_fn
        self._sleep = sleep_fn or time.sleep
        self._mono = monotonic_fn or time.monotonic
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ser = None
        self.connected = False
        self.port: str | None = None
        self.knob_held = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="knob-serial", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        ser = self._ser
        if ser is not None:
            try:
                ser.write(b"\x03")
                ser.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                ser.close()
            except Exception:  # noqa: BLE001
                pass
        self._ser = None
        if self._thread:
            self._thread.join(timeout=2.0)
        self._set_connected(False, "stopped")

    def _set_connected(self, ok: bool, msg: str) -> None:
        self.connected = ok
        self._on_link(ok, msg)

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            if not getattr(self.config, "knob_serial_enabled", True):
                self._set_connected(False, "disabled")
                self._sleep(1.0)
                continue
            port = resolve_knob_port(
                getattr(self.config, "knob_serial_port", "") or "",
                self._list_ports_fn,
            )
            if not port:
                self._set_connected(False, "no_port")
                self._sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 5.0)
                continue
            try:
                self._session(port)
                backoff = 0.5
            except Exception as exc:  # noqa: BLE001
                self._set_connected(False, f"error:{exc}")
                self._sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 5.0)

    def _drain_lines(self, buf: bytes) -> tuple[bytes, bool]:
        got_hb = False
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._on_debug(line)
            for token in self._tokens_from_line(line):
                if token == "HB":
                    got_hb = True
                if token.startswith("K"):
                    self.knob_held = token == "K1"
                self._on_line(token)
        return buf, got_hb

    @staticmethod
    def _tokens_from_line(line: str) -> list[str]:
        if line.startswith(">>>") or line.startswith("MicroPython"):
            return []
        return re.findall(r"(K[01]|G[01]|HB)", line)

    def _session(self, port: str) -> None:
        ser = self._serial_factory(port, timeout=0.1)
        self._ser = ser
        self.port = port
        buf = b""
        # Interrupt any poll loop left running on the mic by a previous session
        # (e.g. after tink-agent was restarted), so the REPL prompt comes back.
        ser.write(b"\x03\x03")
        self._sleep(0.1)
        deadline = self._mono() + 8.0
        while self._mono() < deadline and not self._stop.is_set():
            buf += ser.read(256)
            buf, _ = self._drain_lines(buf)
            if b">>> " in buf:
                break
            if not buf:
                ser.write(b"\r")
                self._sleep(0.05)
        if b">>> " not in buf:
            raise RuntimeError("repl prompt timeout")

        poll_ms = getattr(self.config, "knob_poll_ms", 10)
        ser.write(_exec_payload(_poll_loop_source(poll_ms)))
        ser.flush()

        got_hb = False
        hb_deadline = self._mono() + 2.0
        while self._mono() < hb_deadline and not self._stop.is_set():
            buf += ser.read(256)
            buf, saw = self._drain_lines(buf)
            if saw:
                got_hb = True
                break
            self._sleep(0.02)

        if not got_hb:
            tail = buf.decode("utf-8", errors="replace").strip()
            if tail:
                self._on_debug(f"repl_no_hb: {tail}")
            try:
                ser.write(b"\x03")
                ser.flush()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError("poll loop HB timeout")

        self._set_connected(True, port)
        while not self._stop.is_set():
            chunk = ser.read(256)
            if chunk:
                buf += chunk
                buf, _ = self._drain_lines(buf)
            self._sleep(0.02)
        self._set_connected(False, "session_end")


def _open_serial(port: str, timeout: float = 0.1):
    import serial
    return serial.Serial(port, baudrate=115200, timeout=timeout)
