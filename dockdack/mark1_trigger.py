"""Compatibility import; implementation: dockdack.signals.mark1_trigger."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.signals.mark1_trigger")
_sys.modules[__name__] = _implementation
