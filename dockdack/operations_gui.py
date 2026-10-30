"""Compatibility import; implementation: dockdack.ui.operations_gui."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.ui.operations_gui")
_sys.modules[__name__] = _implementation
