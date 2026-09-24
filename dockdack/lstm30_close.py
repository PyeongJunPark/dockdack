"""Compatibility import; implementation: dockdack.trading.close_liquidation."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.trading.close_liquidation")
_sys.modules[__name__] = _implementation
