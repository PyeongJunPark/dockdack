"""Explicit application/artifact locations for checkouts and installed wheels."""
from pathlib import Path
import os


def checkout_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "DockDack.vbs").is_file():
            return candidate
    return None


def app_home():
    configured = os.environ.get("DOCKDACK_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = checkout_root()
    if checkout is not None:
        return checkout
    return (Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "DockDack").resolve()


def model_root():
    configured = os.environ.get("DOCKDACK_MODEL_ROOT")
    return Path(configured).expanduser().resolve() if configured else app_home() / "models"


def model_bundle(name):
    if name not in {"mark1_prototype", "mark1_1_prototype", "mark1_2_prototype", "mark1_0504", "mark1", "lstm30"}:
        raise ValueError("Unknown model bundle")
    return model_root() / name
