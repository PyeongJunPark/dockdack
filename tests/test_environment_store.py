from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dockdack.environment_store import real_storage_scope, store_for_service
from dockdack.models import TradingMode
from dockdack.watchlist import WatchStore


class EnvironmentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'watchlist.sqlite3'

    def test_mode_binding_persists_and_blocks_relabel(self):
        demo = WatchStore(self.path)
        self.assertEqual(demo.mode, TradingMode.DEMO)
        with self.assertRaises(ValueError):
            WatchStore(self.path, mode=TradingMode.REAL)
        self.assertEqual(WatchStore(self.path).mode, TradingMode.DEMO)

    def test_legacy_nonempty_store_cannot_be_real(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('CREATE TABLE legacy(value TEXT)')
        with self.assertRaisesRegex(ValueError, '기존 기록'):
            WatchStore(self.path, mode=TradingMode.REAL)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='app_environment'").fetchone())

    def test_unknown_schema_is_not_tagged_or_migrated(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('PRAGMA user_version=99')
            db.execute('CREATE TABLE future_schema(value TEXT)')
        with self.assertRaisesRegex(ValueError, 'DB 버전'):
            WatchStore(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 99)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='app_environment'").fetchone())

    def test_scoped_real_and_demo_have_independent_events(self):
        with patch.dict('os.environ', {}, clear=True), patch('dotenv.load_dotenv', return_value=False), patch('dockdack.config.load_dotenv', return_value=False):
            demo = store_for_service(SimpleNamespace(mode=TradingMode.DEMO), base_folder=self.path.parent)
            real = store_for_service(SimpleNamespace(mode=TradingMode.REAL), base_folder=self.path.parent)
        demo.event('SYSTEM', 'demo-only')
        self.assertNotEqual(demo.path, real.path)
        self.assertEqual(real.events(), ())
        self.assertEqual(real.mode, TradingMode.REAL)

    def test_scope_hash_contains_no_plaintext_and_changes_with_keys(self):
        def config(mode, *, market):
            return SimpleNamespace(app_key='private-' + market.value)
        with patch('dockdack.environment_store.KiwoomConfig.from_env', side_effect=config):
            first = real_storage_scope()
        with patch('dockdack.environment_store.KiwoomConfig.from_env', return_value=SimpleNamespace(app_key='rotated')):
            second = real_storage_scope()
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)
        self.assertNotIn('private', first)


if __name__ == '__main__':
    unittest.main()
