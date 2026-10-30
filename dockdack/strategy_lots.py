"""Compatibility import; implementation: dockdack.trading.strategy_lots."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.trading.strategy_lots")
_sys.modules[__name__] = _implementation
