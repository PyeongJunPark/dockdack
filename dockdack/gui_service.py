"""Compatibility import; implementation: dockdack.application.trading_service."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.application.trading_service")
_sys.modules[__name__] = _implementation
