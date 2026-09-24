"""One offline developer/CI check command; rejects accidentally empty suites."""
from pathlib import Path
import argparse
import ast
import os
import sys
import unittest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shard", help="Optional independent module partition, e.g. 1/4 (one-based)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Parse without writing bytecode into source directories.
    for folder in ("dockdack", "tests", "scripts", "examples"):
        for path in (root / folder).rglob("*.py"):
            ast.parse(path.read_bytes(), filename=str(path))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=args.pattern)
    if args.shard:
        try:
            index, total = map(int, args.shard.split("/"))
            if not 1 <= index <= total <= 32:
                raise ValueError()
        except ValueError:
            parser.error("--shard must be index/total with 1 <= index <= total <= 32")
        def cases(group):
            for item in group:
                if isinstance(item, unittest.TestSuite):
                    yield from cases(item)
                else:
                    yield item
        all_cases = list(cases(suite))
        modules = sorted({case.__class__.__module__ for case in all_cases})
        assigned = {module for position, module in enumerate(modules) if position % total == index - 1}
        suite = unittest.TestSuite(case for case in all_cases if case.__class__.__module__ in assigned)
    count = suite.countTestCases()
    if count == 0:
        raise SystemExit("ERROR: no tests discovered")
    print(f"Discovered {count} tests" + (f" (shard {args.shard})" if args.shard else ""), flush=True)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
