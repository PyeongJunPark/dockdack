"""Compatibility entry point; installed apps use dockdack-prototype-worker."""
from dockdack.signals.worker import deny_network, main


if __name__ == "__main__":
    raise SystemExit(main())
