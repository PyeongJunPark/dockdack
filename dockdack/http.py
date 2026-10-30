"""Compatibility import; implementation: dockdack.broker.http."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.broker.http")
_sys.modules[__name__] = _implementation
