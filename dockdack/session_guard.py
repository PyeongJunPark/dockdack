"""Cross-checkout execution locks; optional shared OS-locking filesystem."""
from hashlib import sha256
import os
from pathlib import Path
from dockdack.lstm30_runtime import SessionLock


class _HeldLock:
    def __init__(self, path):
        self.lock = SessionLock(path)
        self.lock.acquire()
        self.references = 1

    def release(self):
        self.references -= 1
        if not self.references:
            self.lock.release()


class AccountSessionLock:
    def __init__(self, path, service, *, previous=None):
        self.path = Path(path).resolve()
        self.previous = previous
        self.held = {}
        mode = service.mode.value
        keys = {sha256(f'{mode}:{config.app_key}'.encode()).hexdigest()
                for config in getattr(service, '_frozen_configs', {}).values()}
        account_id = os.environ.get('DOCKDACK_ACCOUNT_LOCK_ID', '').strip()
        if account_id:
            keys.add(sha256(f'{mode}:account:{account_id}'.encode()).hexdigest())
        roots = [Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share')) / 'DockDack' / 'execution-locks']
        if os.environ.get('DOCKDACK_SHARED_LOCK_DIR'):
            roots.append(Path(os.environ['DOCKDACK_SHARED_LOCK_DIR']).expanduser().resolve())
        self.paths = sorted({self.path, *(root / (key + '.lock') for root in roots for key in keys)}, key=str)

    @property
    def stream(self):
        """Expose the acquired ledger descriptor to the desktop ownership check."""
        held = self.held.get(self.path)
        return held.lock.stream if held is not None else None

    def acquire(self):
        if self.held:
            raise RuntimeError('실행 잠금이 이미 설정되어 있습니다.')
        try:
            for path in self.paths:
                inherited = getattr(self.previous, 'held', {}).get(path)
                if inherited is not None:
                    inherited.references += 1
                    self.held[path] = inherited
                else:
                    self.held[path] = _HeldLock(path)
        except (OSError, RuntimeError) as exc:
            self.release()
            raise RuntimeError('같은 계정 설정의 앱이 이미 실행 중이거나 실행 잠금 경로에 접근할 수 없습니다.') from exc

    def release(self):
        for lock in reversed(tuple(self.held.values())):
            lock.release()
        self.held.clear()
        self.previous = None
