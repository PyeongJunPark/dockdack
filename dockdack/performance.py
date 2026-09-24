"""Compatibility import; implementation: dockdack.trading.performance."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.trading.performance")
_sys.modules[__name__] = _implementation
