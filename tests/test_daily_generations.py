"""Offline collector regressions: every database is a disposable fixture."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from dockdack.daily_dataset import DailyDatasetStore, Instrument, _bar_row, _collect_instrument, _validated_page
from dockdack.daily_generations import DailyGenerationStore, GenerationConflict
from dockdack.models import DailyBar, Market, StockInfo


INSTRUMENT = Instrument("005930", "KRX", "fixture")


def record(day="20260924", price="50", **extra):
    return dict(dt=day, open_pric=price, high_pric=price, low_pric=price,
                cur_prc=price, trde_qty="100", **extra)


def page(*records, token=None, body=None):
    return SimpleNamespace(body=body if body is not None else {"stk_dt_pole_chart_qry": list(records)},
                           cont_yn="Y" if token else "N", next_key=token)


class FakeBroker:
    def __init__(self, *pages):
        self.pages, self.requests = list(pages), []

    def _http_for(self, market):
        return self

    def request(self, **kwargs):
        self.requests.append(kwargs)
        result = self.pages.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def collect(store, *pages, max_pages=None):
    broker = FakeBroker(*pages)
    result = _collect_instrument(broker, Market.DOMESTIC, store, INSTRUMENT,
                                 cont_yn="Y", next_key="unsafe-legacy", known_latest="2026-09-23",
                                 max_pages=max_pages, retries=0)
    return result, broker


class DailyGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "fixture.sqlite3"
        self.store = DailyDatasetStore(self.path, Market.DOMESTIC)
        self.store.save_catalog([StockInfo(Market.DOMESTIC, "005930", "fixture", "KOSPI")])

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def seed(self, price="100"):
        self.store.save_page(INSTRUMENT, [DailyBar(Market.DOMESTIC, "005930", "KRX", date(2026, 9, day),
            *(Decimal(price) for _ in range(4)), Decimal(100), "KRW") for day in (21, 22, 23)],
            has_next=False, cont_yn=None, next_key=None)

    def prices(self):
        return [tuple(row) for row in self.store.connection.execute("SELECT trade_date,close FROM daily_bars ORDER BY trade_date")]

    def test_split_refresh_replaces_entire_history_and_archives_previous(self):
        self.seed()
        first = page(record(), record("20260923"), token="older")
        completed, broker = collect(self.store, first, page(record("20260922"), record("20260921")), first)
        self.assertTrue(completed)
        self.assertEqual(len(broker.requests), 3)
        self.assertTrue(all(price == "50" for _, price in self.prices()))
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM daily_generation_previous").fetchone()[0], 3)
        self.assertEqual(self.store.progress(INSTRUMENT)["rows_seen"], 4)

    def test_reverse_split_refetches_older_rows_in_the_same_basis(self):
        self.seed()
        first = page(record(price="200"), record("20260923", "200"), token="older")
        collect(self.store, first, page(record("20260922", "200"), record("20260921", "200")), first)
        self.assertEqual([price for _, price in self.prices()], ["200"] * 4)

    def test_failure_inside_publish_rolls_back_bars_progress_and_archive(self):
        self.seed()
        before = self.prices()
        self.store.connection.execute("""CREATE TRIGGER fixture_publication_failure BEFORE INSERT ON daily_bars
            BEGIN SELECT RAISE(ABORT, 'fixture interrupted publication'); END""")
        self.store.connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            collect(self.store, page(record()), page(record()))
        self.assertEqual(self.prices(), before)
        self.assertEqual(self.store.progress(INSTRUMENT)["status"], "complete")
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM daily_generation_previous").fetchone()[0], 0)

    def test_stale_error_writer_cannot_relabel_a_new_complete_publication(self):
        self.seed()
        self.store.mark_error(INSTRUMENT, "late error from an earlier attempt")
        self.assertEqual(self.store.progress(INSTRUMENT)["status"], "complete")
        self.assertIsNone(self.store.progress(INSTRUMENT)["error"])

    def test_partial_refresh_preserves_published_history_then_resumes_after_reopen(self):
        self.seed()
        before = self.prices()
        first = page(record(), record("20260923"), token="older")
        self.assertFalse(collect(self.store, first, max_pages=1)[0])
        self.assertEqual(self.prices(), before)
        self.assertEqual(self.store.progress(INSTRUMENT)["status"], "complete")
        self.store.close()
        self.store = DailyDatasetStore(self.path, Market.DOMESTIC)
        completed, broker = collect(self.store, first, page(record("20260922"), record("20260921")), first, max_pages=1)
        self.assertTrue(completed)
        self.assertEqual(broker.requests[1]["next_key"], "older")
        self.assertEqual(len(self.prices()), 4)

    def test_revision_on_resume_archives_old_partial_and_starts_clean(self):
        first = page(record(price="50"), token="older")
        collect(self.store, first, max_pages=1)
        revised = page(record(price="25"), token="new-older")
        collect(self.store, revised, page(record("20260923", "25")), revised)
        self.assertEqual([price for _, price in self.prices()], ["25", "25"])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM daily_generations WHERE status='superseded'").fetchone()[0], 1)

    def test_revision_before_publish_is_rejected_without_erasing_old_data(self):
        self.seed()
        before = self.prices()
        with self.assertRaises(GenerationConflict):
            collect(self.store, page(record()), page(record(price="25")))
        self.assertEqual(self.prices(), before)
        self.assertTrue(self.store.generations.pending("005930", "KRX"))

    def test_network_failure_preserves_staging_and_published_rows(self):
        self.seed()
        before = self.prices()
        first = page(record(), token="older")
        with self.assertRaisesRegex(RuntimeError, "offline"):
            collect(self.store, first, RuntimeError("offline"))
        self.assertEqual(self.prices(), before)
        self.assertEqual(self.store.stats()["staged_error"], 1)
        collect(self.store, first, page(record("20260923")), first)
        self.assertEqual(len(self.prices()), 2)

    def test_keyboard_interrupt_releases_lease_and_preserves_stage(self):
        first = page(record(), token="older")
        with self.assertRaises(KeyboardInterrupt):
            collect(self.store, first, KeyboardInterrupt())
        collect(self.store, first, page(record("20260923")), first)
        self.assertEqual(len(self.prices()), 2)

    def test_empty_response_cannot_erase_prior_data(self):
        self.seed()
        before = self.prices()
        with self.assertRaises(GenerationConflict):
            collect(self.store, page(), page())
        self.assertEqual(self.prices(), before)

    def test_explicit_empty_list_can_publish_first_empty_history(self):
        self.assertTrue(collect(self.store, page(), page())[0])
        self.assertEqual(self.store.progress(INSTRUMENT)["status"], "complete")

    def test_identical_overlap_allowed_conflicting_overlap_refused(self):
        first = page(record(), record("20260923"), token="older")
        collect(self.store, first, page(record("20260923"), record("20260922")), first)
        before = self.prices()
        with self.assertRaises(GenerationConflict):
            collect(self.store, first, page(record("20260923", "25")))
        self.assertEqual(self.prices(), before)

    def test_continuation_cycle_is_rejected_before_publication(self):
        first = page(record(), token="same")
        with self.assertRaises(GenerationConflict):
            collect(self.store, first, page(record("20260923"), token="same"))
        self.assertEqual(self.prices(), [])

    def test_concurrent_claim_is_blocked_and_expired_owner_is_fenced(self):
        clock = [100.0]
        one = DailyGenerationStore(self.store.connection, clock=lambda: clock[0], lease_seconds=10)
        generation = one.claim("005930", "KRX")
        with DailyDatasetStore(self.path, Market.DOMESTIC) as other:
            two = DailyGenerationStore(other.connection, clock=lambda: clock[0], lease_seconds=10)
            with self.assertRaises(GenerationConflict):
                two.claim("005930", "KRX")
            clock[0] = 111
            reclaimed = two.claim("005930", "KRX")
            self.assertEqual(generation["id"], reclaimed["id"])
            with self.assertRaises(GenerationConflict):
                one.renew(generation["id"])
            one.release(generation["id"])
            two.renew(generation["id"])

    def test_direct_canonical_writer_invalidates_publication(self):
        self.seed()
        generation = self.store.generations.claim("005930", "KRX")
        rows = [_bar_row(bar, "fixture") for bar in _validated_page(page(record()).body, "stk_dt_pole_chart_qry", INSTRUMENT, Market.DOMESTIC)]
        self.store.generations.append(generation["id"], rows, anchor="anchor", request_json="{}", next_key=None)
        self.store.connection.execute("UPDATE daily_bars SET close='101'")
        self.store.connection.commit()
        with self.assertRaises(GenerationConflict):
            self.store.generations.publish(generation["id"], verified_anchor="anchor", timestamp="fixture")
        self.assertEqual(len(self.prices()), 3)

    def test_malformed_response_never_marks_complete_or_writes_rows(self):
        cases = [{}, {"stk_dt_pole_chart_qry": None}, {"stk_cd": "999999", "stk_dt_pole_chart_qry": [record()]},
                 {"stk_dt_pole_chart_qry": [None]}, {"stk_dt_pole_chart_qry": [{"dt": "20260924"}]}]
        for body in cases:
            with self.subTest(body=body), self.assertRaises(ValueError):
                collect(self.store, page(body=body))
        self.assertEqual(self.prices(), [])
        self.assertIsNone(self.store.progress(INSTRUMENT))


class ChartValidationTests(unittest.TestCase):
    def validate(self, row):
        return _validated_page({"stk_dt_pole_chart_qry": [row]}, "stk_dt_pole_chart_qry", INSTRUMENT, Market.DOMESTIC)

    def test_bad_records(self):
        cases = [("dt", "20260230"), ("dt", "2026-09-24"), ("open_pric", ""), ("high_pric", "NaN"),
                 ("trde_qty", "-1"), ("trde_qty", "1.5"), ("trde_qty", str(2**63)),
                 ("low_pric", "51"), ("stk_cd", "999999"), ("stex_tp", "NY"), ("pred_pre", "Infinity")]
        for field, value in cases:
            row = record()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, ArithmeticError)):
                self.validate(row)

    def test_signed_prices_and_valid_domestic_prefix_are_supported(self):
        row = record(price="-50", stk_cd="A005930")
        self.assertEqual(self.validate(row)[0].close, Decimal(50))

    def test_us_identity_and_volume_are_validated(self):
        row = record(stk_cd="AAPL", stex_tp="ND", acc_trde_qty="100")
        bars = _validated_page({"result_list": [row]}, "result_list", Instrument("AAPL", "ND", "Apple"), Market.US)
        self.assertEqual(bars[0].currency, "USD")
        row["stex_tp"] = "NY"
        with self.assertRaises(ValueError):
            _validated_page({"result_list": [row]}, "result_list", Instrument("AAPL", "ND", "Apple"), Market.US)


if __name__ == "__main__":
    unittest.main()
