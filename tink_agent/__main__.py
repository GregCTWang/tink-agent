from __future__ import annotations

import sys

from . import __version__


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in {"--version", "-V"}:
        print(f"tink-agent {__version__}")
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "learn-buttons":
        from .learn_buttons import main as learn_main
        learn_main(sys.argv[2:])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "ax-probe":
        from .ax_probe import main as ax_probe_main
        ax_probe_main(sys.argv[2:])
        return

    from .menubar import main as menubar_main
    menubar_main()

if __name__ == "__main__":
    main()
