"""Compatibility import; implementation: dockdack.ui.dashboard_theme."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.ui.dashboard_theme")
_sys.modules[__name__] = _implementation
