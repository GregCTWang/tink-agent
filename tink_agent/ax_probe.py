"""CLI: inspect focused Accessibility element for dictation restore debugging."""
from __future__ import annotations

import sys
import time


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    delay = 3.0
    if argv and argv[0].isdigit():
        delay = float(argv[0])
    print(f"Switch to the target app; probing focused element in {delay:.0f}s…", flush=True)
    time.sleep(delay)
    try:
        from .ax_macos import probe_focused
    except ImportError:
        print("ApplicationServices unavailable (macOS only).", file=sys.stderr)
        sys.exit(1)
    print(probe_focused())


if __name__ == "__main__":
    main()
