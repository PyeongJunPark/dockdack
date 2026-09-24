"""Compatibility import; implementation: dockdack.trading.execution_policy."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.trading.execution_policy")
_sys.modules[__name__] = _implementation
