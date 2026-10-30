"""Environment/account-scoped local data, never containing plaintext API keys."""
from hashlib import sha256
import json
import re
from pathlib import Path

from dockdack.config import KiwoomConfig
from dockdack.exceptions import ConfigurationError
from dockdack.models import Market, TradingMode
from dockdack.watchlist import WatchStore


def selected_mode(service) -> TradingMode:
    return TradingMode(getattr(service, "mode", TradingMode.DEMO))


def environment_base(store) -> Path:
    folder = store.path.parent
    if folder.parent.name == store.mode.value and folder.name == store.storage_scope:
        return folder.parent.parent
    return folder


def scope_for_configs(configs: dict) -> str:
    configured = [(market.value, configs[market].app_key if market in configs else None) for market in Market]
    if not any(key for _, key in configured):
        return "unconfigured"
    return sha256(json.dumps(configured, separators=(",", ":")).encode()).hexdigest()


def demo_scope(key_scope, generation="0"):
    """An explicit generation separates a broker-reset paper account as well."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", generation):
        raise ConfigurationError("DOCKDACK_DEMO_GENERATION은 1~64자리 영숫자/_/-여야 합니다.")
    if key_scope == "unconfigured":
        return key_scope
    return sha256(f"demo:{key_scope}:{generation}".encode()).hexdigest()


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


def scoped_store_path(service, *, base_folder: Path | None = None) -> Path:
    mode = selected_mode(service)
    scope = real_storage_scope(service) if mode is TradingMode.REAL else getattr(service, "storage_scope", "demo")
    if base_folder is None:
        from dockdack.runtime_paths import app_home
        base_folder = app_home() / ".dockdack"
    folder = Path(base_folder).resolve()
    if mode is TradingMode.REAL or scope != "demo":
        if scope != "unconfigured" and not re.fullmatch(r"[0-9a-f]{64}", scope):
            raise ValueError("계정 저장소 범위 식별자가 올바르지 않습니다.")
        folder = folder / mode.value / scope
    return folder / "watchlist.sqlite3"


def store_for_service(service, *, base_folder: Path | None = None) -> WatchStore:
    mode = selected_mode(service)
    scope = real_storage_scope(service) if mode is TradingMode.REAL else getattr(service, "storage_scope", "demo")
    path = scoped_store_path(service, base_folder=base_folder)
    return WatchStore(path, seed_defaults=True, mode=mode, storage_scope=scope)
