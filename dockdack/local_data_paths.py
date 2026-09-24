"""Read-only defaults for optional local datasets after checkout consolidation."""
from pathlib import Path


CLEAN_DAILY_RELATIVE = Path("data/kiwoom_daily/clean-20260916-v1")


def default_clean_database_dir(root=None):
    """Prefer this checkout, then canonical/legacy sibling data when present.

    This selects a default only: explicit CLI paths still win. It never creates
    directories, copies a DB, rewrites historical manifests or changes a model.
    """
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    candidates = (root / CLEAN_DAILY_RELATIVE,
                  root.parent / "dockdack" / CLEAN_DAILY_RELATIVE,
                  root.parent / "dockdack-data-collection" / CLEAN_DAILY_RELATIVE)
    return next((path for path in candidates if path.is_dir()), candidates[0])
