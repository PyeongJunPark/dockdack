"""Import each packaged command from a wheel without opening accounts or models.

This is a fast archive smoke, not an installed-environment or inference test.
Pass a wheel copied outside the checkout so runtime path discovery cannot use
the source tree by accident.
"""

from __future__ import annotations

import argparse
import configparser
import importlib
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile


ENTRY_POINTS = {
    "dockdack": "dockdack.cli:main",
    "dockdack-gui": "dockdack.v00_app:main",
    "dockdack-prototype-worker": "dockdack.signals.worker:main",
    "dockdack-maintenance": "dockdack.maintenance:main",
    "dockdack-research": "dockdack.research_tools:main",
}
ASSETS = {"dockdack/assets/dockdack.ico", "dockdack/assets/dockdack-mark.svg"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("--wheel must point to a built .whl archive")

    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata) != 1 or not ASSETS.issubset(names):
            raise RuntimeError("Wheel entry points or GUI assets are missing")
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(metadata[0]).decode("utf-8"))
        if not entries.has_section("console_scripts"):
            raise RuntimeError("Wheel has no console script entry points")
        for command, target in ENTRY_POINTS.items():
            if entries["console_scripts"].get(command) != target:
                raise RuntimeError(f"Wheel entry point differs: {command}")

    # Import from the wheel archive ahead of this interpreter's editable
    # checkout. A fresh temporary home ensures even an accidental path lookup
    # cannot open the user's operating ledger.
    with tempfile.TemporaryDirectory(prefix="dockdack-wheel-imports-") as home:
        os.environ["DOCKDACK_HOME"] = home
        os.environ["DOCKDACK_MODEL_ROOT"] = str(Path(home) / "models")
        sys.path.insert(0, str(wheel))
        import dockdack
        origin = Path(dockdack.__file__).as_posix().casefold()
        if not origin.startswith(wheel.as_posix().casefold() + "/dockdack/"):
            raise RuntimeError("Imported the checkout instead of the built wheel")
        from dockdack.runtime_paths import checkout_root
        if checkout_root() is not None:
            raise RuntimeError("Wheel unexpectedly resolved a source checkout")
        for target in ENTRY_POINTS.values():
            module_name, callable_name = target.split(":", 1)
            module = importlib.import_module(module_name)
            if not callable(getattr(module, callable_name, None)):
                raise RuntimeError(f"Wheel command is not callable: {target}")

    print(f"Wheel archive imports: {len(ENTRY_POINTS)} commands, {len(ASSETS)} GUI assets; no account, model or order access")


if __name__ == "__main__":
    main()
