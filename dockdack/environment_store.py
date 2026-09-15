"""Environment/account-scoped local data, never containing plaintext API keys."""
from hashlib import sha256
import json
from pathlib import Path

from dockdack.config import KiwoomConfig
from dockdack.exceptions import ConfigurationError
from dockdack.models import Market, TradingMode
from dockdack.watchlist import WatchStore, default_store


def selected_mode(service) -> TradingMode:
    return TradingMode(getattr(service, "mode", TradingMode.DEMO))


def environment_base(store) -> Path:
    folder = store.path.parent
    if store.mode is TradingMode.REAL and folder.parent.name == 'real' and folder.name == store.storage_scope:
        return folder.parent.parent
    return folder


def scope_for_configs(configs: dict) -> str:
    configured = [(market.value, configs[market].app_key if market in configs else None) for market in Market]
    if not any(key for _, key in configured):
        return "unconfigured"
    return sha256(json.dumps(configured, separators=(",", ":")).encode()).hexdigest()


def real_storage_scope(service=None) -> str:
    frozen_scope = getattr(service, 'storage_scope', None)
    if frozen_scope is not None:
        return frozen_scope
    configured = {}
    for market in Market:
        try:
            config = KiwoomConfig.from_env(TradingMode.REAL, market=market)
        except ConfigurationError:
            pass
        else:
            configured[market] = config
    # The full digest is an opaque scope, not an account identifier. Rotating
    # app keys deliberately opens a new ledger; never blend unknown accounts.
    return scope_for_configs(configured)


def store_for_service(service, *, base_folder: Path | None = None) -> WatchStore:
    mode = selected_mode(service)
    scope = real_storage_scope(service) if mode is TradingMode.REAL else "unconfigured"
    if base_folder is None:
        return default_store(mode, scope=scope)
    folder = Path(base_folder)
    if mode is TradingMode.REAL:
        folder = folder / "real" / scope
    return WatchStore(folder / "watchlist.sqlite3", seed_defaults=True, mode=mode,
                      storage_scope=scope if mode is TradingMode.REAL else "demo")
