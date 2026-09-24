"""Relocation/storage tests use disposable databases and arrays, never live data."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dockdack.research_artifacts import ArtifactResolver, cache_key, sha256_file, verify_protocol_sources
from dockdack.research_arrays import (ARRAY_NAMES, _close_mmap, load_mmap_cache,
                                       unpack_frozen_cache, write_feature_chunks)
from dockdack.research_compat import bind_recorded_artifact, inspect_frozen_cache, load_frozen_cache_compatible
from dockdack.research_tools import (RUNNERS, _fresh_output, compatibility_context,
                                     project_hash_maps, run_legacy)


class ResearchFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.resolver = ArtifactResolver(self.root, relocations={"C:/old/research": self.root})
        self.database = self.root / "clean.sqlite3"
        with sqlite3.connect(self.database) as db:
            db.executescript("""CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE daily_bars(dummy); CREATE TABLE training_samples(dummy);
                CREATE TABLE sessions(dummy); CREATE TABLE instruments(dummy);""")
            db.executemany("INSERT INTO metadata VALUES (?,?)", [("market", '"domestic"'), ("build_status", '"complete"')])
        db.close()
        self.source = {"version": 2, "market": "domestic", "database_path": "C:\\old\\research\\clean.sqlite3",
                       "database_sha256": sha256_file(self.database), "seed": 42}
        self.cache = self.root / f"domestic-{cache_key(self.source)}.npz"
        bars = np.empty((80, 5), np.float32)
        close = 100 + np.arange(80) * .1
        bars[:, 0], bars[:, 1], bars[:, 2], bars[:, 3], bars[:, 4] = close, close + 1, close - 1, close, 10000
        self.arrays = {"bars": bars, "starts": np.arange(7, dtype=np.int64),
                       "target_dates": np.arange(7, dtype=np.int64), "symbol_ids": np.zeros(7, np.int64),
                       "target_ohlc": np.tile([102., 104., 101., 103.], (7, 1)),
                       **{name: np.array([0, 1], np.int64) for name in ARRAY_NAMES[5:]}}
        np.savez(self.cache, **self.arrays, manifest=json.dumps({"market": "domestic", "symbols": []}),
                 cache_config=json.dumps(self.source))
        self.cache_hash = sha256_file(self.cache)

    def tearDown(self):
        self.temp.cleanup()

    def test_resolver_preserves_logical_contract_and_cache_key(self):
        original = copy.deepcopy(self.source)
        artifact = self.resolver.database(self.source, "domestic")
        self.assertEqual(artifact.physical_path, self.database)
        self.assertEqual(self.source, original)
        self.assertNotEqual(cache_key(self.source), cache_key({**self.source, "database_path": str(self.database)}))

    def test_resolver_rejects_changed_wrong_market_unknown_root_traversal(self):
        with self.assertRaises(ValueError):
            self.resolver.database(self.source, "us")
        with self.assertRaises(ValueError):
            self.resolver.location("C:/unapproved/clean.sqlite3")
        with self.assertRaises(ValueError):
            self.resolver.location("C:/old/research/../outside")
        with self.assertRaises(ValueError):
            self.resolver.location("C:drive-relative")
        with self.database.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            self.resolver.database(self.source, "domestic")

    def test_nonempty_wal_is_refused(self):
        Path(str(self.database) + "-wal").write_bytes(b"not immutable")
        with self.assertRaisesRegex(ValueError, "WAL"):
            self.resolver.database(self.source, "domestic")

    def test_cache_inspection_reads_headers_and_preserves_contract(self):
        receipt = inspect_frozen_cache(self.source, "domestic", self.root, resolver=self.resolver,
                                        expected_cache_sha256=self.cache_hash)
        self.assertEqual(receipt["arrays"]["bars"]["shape"], [80, 5])
        self.assertEqual(receipt["source_contract"], self.source)

    def test_npz_conversion_is_exact_readonly_mmap_and_never_overwrites(self):
        destination = self.root / "mmap"
        manifest = unpack_frozen_cache(self.source, "domestic", self.root, destination,
                                       resolver=self.resolver, expected_cache_sha256=self.cache_hash)
        self.assertEqual(manifest["source_contract"], self.source)
        dataset = load_mmap_cache(destination, self.source)
        try:
            self.assertIsInstance(dataset.bars, np.memmap)
            self.assertFalse(dataset.bars.flags.writeable)
            for name in ARRAY_NAMES[:5]:
                np.testing.assert_array_equal(getattr(dataset, name), self.arrays[name])
        finally:
            for array in (*[getattr(dataset, name) for name in ARRAY_NAMES[:5]], *dataset.splits.values()):
                _close_mmap(array)
        with self.assertRaises(FileExistsError):
            unpack_frozen_cache(self.source, "domestic", self.root, destination,
                                resolver=self.resolver, expected_cache_sha256=self.cache_hash)
        self.assertEqual(sha256_file(self.cache), self.cache_hash)

    def test_npz_conversion_refuses_destination_outside_workspace(self):
        destination = self.root.parent / (self.root.name + "-outside")
        with self.assertRaises(ValueError):
            unpack_frozen_cache(self.source, "domestic", self.root, destination,
                                resolver=self.resolver, expected_cache_sha256=self.cache_hash)
        self.assertFalse(destination.exists())

    def test_changed_mmap_member_and_wrong_source_are_refused(self):
        destination = self.root / "mmap"
        unpack_frozen_cache(self.source, "domestic", self.root, destination,
                            resolver=self.resolver, expected_cache_sha256=self.cache_hash)
        with self.assertRaises(ValueError):
            load_mmap_cache(destination, {**self.source, "seed": 1})
        with (destination / "bars.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            load_mmap_cache(destination, self.source)

    def test_compatible_loader_returns_original_dataclass_without_new_cache(self):
        dataset, actual, receipt = load_frozen_cache_compatible(self.source, "domestic", self.root,
            resolver=self.resolver, expected_cache_sha256=self.cache_hash)
        from dataclasses import replace
        self.assertEqual(replace(dataset, manifest={}).bars.shape, (80, 5))
        self.assertEqual(actual, self.source)
        self.assertEqual(receipt["cache_key"], cache_key(self.source))
        self.assertEqual(len(list(self.root.glob("*.npz"))), 1)

    def test_gateway_context_routes_explicit_mmap_root_to_verified_cache(self):
        destination = self.root / f"domestic-{cache_key(self.source)}"
        unpack_frozen_cache(self.source, "domestic", self.root, destination,
                            resolver=self.resolver, expected_cache_sha256=self.cache_hash)
        module_name = "examples.fixture_cache_runner"
        source = self.root / "examples/fixture_cache_runner.py"
        source.parent.mkdir(exist_ok=True)
        source.write_text("# Fixture source\n", encoding="utf-8")
        module = SimpleNamespace(__name__=module_name, __file__=str(source), Path=Path,
                                 load_frozen_cache=lambda *_: None)
        dataset = None
        try:
            with patch("dockdack.research_tools.ArtifactResolver", return_value=self.resolver), \
                    patch("dockdack.research_tools.COMPAT_MODULES", (module_name,)), \
                    patch("dockdack.research_tools.importlib.import_module", return_value=module):
                with compatibility_context(self.root, expected_caches={str(self.cache): self.cache_hash}, mmap_root=self.root) as state:
                    dataset = module.load_frozen_cache(self.source, "domestic", self.root)
                    self.assertIsInstance(dataset.bars, np.memmap)
                    self.assertFalse(dataset.bars.flags.writeable)
                    self.assertEqual(state["receipts"][-1]["mmap_directory"], str(destination))
        finally:
            if dataset is not None:
                for array in (*[getattr(dataset, name) for name in ARRAY_NAMES[:5]], *dataset.splits.values()):
                    _close_mmap(array)

    def test_wrong_npz_receipt_or_embedded_contract_is_refused(self):
        with self.assertRaises(ValueError):
            inspect_frozen_cache(self.source, "domestic", self.root,
                                 resolver=self.resolver, expected_cache_sha256="0" * 64)
        np.savez(self.cache, **self.arrays, manifest="{}", cache_config=json.dumps({**self.source, "seed": 2}))
        with self.assertRaisesRegex(ValueError, "contract"):
            inspect_frozen_cache(self.source, "domestic", self.root,
                                 resolver=self.resolver, expected_cache_sha256=sha256_file(self.cache))

    def test_verified_alias_preserves_original_key_and_rejects_wrong_content(self):
        target = self.root / "summary.json"
        target.write_text('{"completed":true}', encoding="utf-8")
        original = {"C:/old/research/summary.json": sha256_file(target)}
        bound, _ = bind_recorded_artifact(original, target, resolver=self.resolver)
        self.assertEqual(set(original), {"C:/old/research/summary.json"})
        self.assertEqual(bound[str(target)], original["C:/old/research/summary.json"])
        target.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            bind_recorded_artifact(original, target, resolver=self.resolver)

    def test_projection_changes_only_verified_hash_keys_not_logical_source(self):
        value = {"source": self.source, "artifact_sha256": {self.source["database_path"]: self.source["database_sha256"]}}
        projected = project_hash_maps(value, self.resolver, [])
        self.assertEqual(projected["source"], self.source)
        self.assertEqual(projected["artifact_sha256"], {str(self.database): self.source["database_sha256"]})
        self.assertIn(self.source["database_path"], value["artifact_sha256"])

    def test_source_seal_verification_rejects_modified_file(self):
        (self.root / "dockdack").mkdir()
        source = self.root / "dockdack/mark1_example.py"
        source.write_text("pass\n", encoding="utf-8")
        protocol = self.root / "protocol.json"
        protocol.write_text(json.dumps({"code_sha256": {source.name: sha256_file(source)}}), encoding="utf-8")
        self.assertEqual(len(verify_protocol_sources(protocol, self.root)), 1)
        source.write_text("pass # changed\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_protocol_sources(protocol, self.root)

    def test_legacy_gateway_rejects_training_unknown_commands_and_old_outputs(self):
        with self.assertRaises(ValueError):
            run_legacy("train-selective", [], workspace=self.root)
        with self.assertRaises(ValueError):
            _fresh_output([], "backtest-deep", self.resolver)
        with self.assertRaises(FileExistsError):
            _fresh_output(["--output-dir", str(self.root)], "backtest-deep", self.resolver)
        with self.assertRaises(ValueError):
            _fresh_output(["--output-dir", "../escape"], "backtest-deep", self.resolver)
        with self.assertRaises(ValueError):
            _fresh_output(["--output-dir", "one", "--output-dir", "two"], "backtest-deep", self.resolver)
        with self.assertRaises(ValueError):
            _fresh_output(["--output-dir", "checked", "--output-di", "actual"], "backtest-deep", self.resolver)

    def test_missing_source_checkout_refuses_execution_and_restores_import_path(self):
        import sys
        before = list(sys.path)
        with self.assertRaises(FileNotFoundError):
            with compatibility_context(self.root, expected_caches={}):
                self.fail("A missing research checkout must not execute")
        self.assertEqual(sys.path, before)

    def test_chunked_features_match_both_frozen_feature_functions_exactly(self):
        from dockdack.mark1_selective_features import features_from_history as selective
        from dockdack.mark1_0504_data import features_from_history as half
        dataset = SimpleNamespace(**{name: self.arrays[name] for name in ARRAY_NAMES[:5]})
        indices = np.array([6, 0, 3, 1, 4], np.int64)
        for name, function in (("selective", selective), ("half", half)):
            batches = []

            def counted(history, entry):
                batches.append(len(history))
                return function(history, entry)

            expected = function(dataset.bars[dataset.starts[indices, None] + np.arange(30)], dataset.target_ohlc[indices, 0]).astype(np.float32)
            write_feature_chunks(dataset, indices, self.root / name, feature_function=counted,
                                  feature_count=184, feature_identity={"fixture": name}, batch_size=2)
            actual = np.load(self.root / name / "features.npy", mmap_mode="r")
            try:
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(batches, [2, 2, 1])
            finally:
                _close_mmap(actual)

    def test_interrupted_feature_write_keeps_partial_unpublished(self):
        dataset = SimpleNamespace(**{name: self.arrays[name] for name in ARRAY_NAMES[:5]})
        destination = self.root / "features"
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            write_feature_chunks(dataset, np.arange(3), destination,
                feature_function=lambda *_: (_ for _ in ()).throw(RuntimeError("interrupted")),
                feature_count=184, feature_identity={"fixture": True}, batch_size=2)
        self.assertFalse(destination.exists())
        self.assertEqual(len(list(self.root.glob(".features-building-*"))), 1)
        self.assertEqual(list(self.root.glob(".features-building-*/completed.json")), [])

    def test_context_restores_module_globals_even_on_error(self):
        import sys
        from examples import train_mark1
        original_path, original_json = train_mark1.Path, train_mark1.json
        original_import_path = list(sys.path)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with compatibility_context(Path(__file__).resolve().parents[1], expected_caches={}):
                self.assertEqual(train_mark1.jsonable(self.root), str(self.root))
                with self.assertRaises(ValueError):
                    train_mark1.Path("../outside-source")
                self.assertFalse(train_mark1.Path("relative-member.json").is_absolute())
                self.assertIsNot(train_mark1.Path, original_path)
                raise RuntimeError("fixture")
        self.assertIs(train_mark1.Path, original_path)
        self.assertIs(train_mark1.json, original_json)
        self.assertEqual(sys.path, original_import_path)

    def test_failed_cwd_restore_still_restores_import_path_and_module_globals(self):
        import sys
        from examples import train_mark1
        original_path, original_json, original_import_path = train_mark1.Path, train_mark1.json, list(sys.path)
        with patch("dockdack.research_tools.os.chdir", side_effect=[None, FileNotFoundError("old cwd removed")]):
            with self.assertRaises(FileNotFoundError):
                with compatibility_context(Path(__file__).resolve().parents[1], expected_caches={}):
                    self.assertIsNot(train_mark1.Path, original_path)
        self.assertIs(train_mark1.Path, original_path)
        self.assertIs(train_mark1.json, original_json)
        self.assertEqual(sys.path, original_import_path)

    def test_all_five_gateways_dispatch_to_original_main_with_no_training(self):
        # No actual runner body, model execution or account call is used here.
        for command, module_name in RUNNERS.items():
            source = self.root.joinpath(*module_name.split(".")).with_suffix(".py")
            source.parent.mkdir(exist_ok=True)
            source.write_text("# Fixture runner, no original code execution\n"
                              "parser.add_argument('--output-dir')\nparser.add_argument('--output')\n", encoding="utf-8")
            output = self.root / command
            option = "--output" if command.startswith("export-") else "--output-dir"

            def fake_main(*args):
                output.mkdir()
                return 0

            module = SimpleNamespace(__name__=module_name, __file__=str(source), Path=Path, json=json, main=fake_main)
            with self.subTest(command=command), patch.object(module, "main", side_effect=fake_main) as called, \
                    patch("dockdack.research_tools.cache_receipts", return_value={}), \
                    patch("dockdack.research_tools.COMPAT_MODULES", (module_name,)), \
                    patch("dockdack.research_tools.importlib.import_module", return_value=module):
                self.assertEqual(run_legacy(command, [option, str(output)], workspace=self.root), 0)
                called.assert_called_once()
                receipt = json.loads((output / "relocation-compatibility.json").read_text())
                self.assertFalse(receipt["training_started"])
                self.assertFalse(receipt["model_math_changed"])


if __name__ == "__main__":
    unittest.main()
