import os
import sys

# Started without a console (pythonw.exe on Windows, as the installed app is),
# sys.stdout/sys.stderr are None and any library that writes progress or
# warnings to them would crash. This runs in the worker processes too, since
# they import this package to find their entry point.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))


def main() -> None:
    from .app import main as _main

    _main()
