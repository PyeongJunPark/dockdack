"""Compatibility import; implementation: dockdack.ui.v00_widgets."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.ui.v00_widgets")
_sys.modules[__name__] = _implementation
