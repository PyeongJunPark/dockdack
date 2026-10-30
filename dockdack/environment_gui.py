"""Compatibility import; implementation: dockdack.ui.environment_gui."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.ui.environment_gui")
_sys.modules[__name__] = _implementation
