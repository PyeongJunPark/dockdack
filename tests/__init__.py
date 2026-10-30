"""Make root unittest discovery find the suite, including historical helper imports."""
from pathlib import Path
import sys

# Existing test modules intentionally share fixture helpers by short names.
# This path contains tests only and is never part of the distributed package.
_fixture_path = str(Path(__file__).resolve().parent)
if _fixture_path not in sys.path:
    sys.path.insert(0, _fixture_path)
