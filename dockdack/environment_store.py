"""Compatibility import; implementation: dockdack.persistence.environment_store."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("dockdack.persistence.environment_store")
_sys.modules[__name__] = _implementation
