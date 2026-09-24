"""Compatibility import; implementation: dockdack.trading.fill_recovery."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.trading.fill_recovery")
_sys.modules[__name__] = _implementation
