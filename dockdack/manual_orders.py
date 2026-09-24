"""Compatibility import; implementation: dockdack.trading.manual_orders."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.trading.manual_orders")
_sys.modules[__name__] = _implementation
