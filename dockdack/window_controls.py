"""Compatibility import; implementation: dockdack.ui.window_controls."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.ui.window_controls")
_sys.modules[__name__] = _implementation
