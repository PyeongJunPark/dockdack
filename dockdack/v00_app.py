"""Compatibility import; implementation: dockdack.ui.v00_app."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.ui.v00_app")
_sys.modules[__name__] = _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
