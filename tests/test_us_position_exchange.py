"""Offline US balance venue regression tests; no network or order requests."""

from decimal import Decimal
import unittest

from dockdack import KiwoomBroker, Market, USExchange
from dockdack.gui_service import Instrument
from dockdack.kiwoom import _US_POSITION_KEYS, _us_position
from dockdack.watchlist import instrument_key
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


def position_row(**overrides):
    row = {
        "stex_nm": "나스닥", "stk_cd": "LITE", "frgn_stk_nm": "Lumentum",
        "crnc_code": "USD", "poss_qty": "3", "sell_alowq": "2",
        "frgn_stk_book_uv": "100.25", "now_pric": "102.50",
        "evlt_amt": "307.50", "pl_amt": "6.75", "pl_rt": "2.2444",
    }
    row.update(overrides)
    return row


class USPositionExchangeTests(unittest.TestCase):
    def test_known_codes_and_exchange_names_are_orderable(self):
        aliases = {
            "ND": "ND", "NASDAQ": "ND", "나스닥": "ND", " nasdaq ": "ND",
            "NY": "NY", "NYSE": "NY", "뉴욕": "NY",
            "NA": "NA", "AMEX": "NA", "아멕스": "NA",
        }
        for label, code in aliases.items():
            with self.subTest(label=label):
                position = _us_position(position_row(stex_nm=label))
                self.assertEqual(position.exchange, code)
                self.assertEqual(instrument_key(Instrument(position.market, position.symbol, code)),
                                 f"us:{code}:LITE")

    def test_explicit_code_can_resolve_country_display_but_country_cannot(self):
        for field in ("stex_code", "stex_tp"):
            with self.subTest(field=field):
                position = _us_position(position_row(stex_nm="미국", **{field: "ND"}))
                self.assertEqual(position.exchange, "ND")

    def test_unknown_country_and_ambiguous_venues_are_not_orderable(self):
        rows = [
            position_row(stex_nm=label)
            for label in ("", None, "미국", "USA", "US", "%", "ALL", "NASDAQ/NYSE", "OTHER")
        ]
        rows.extend([
            position_row(stex_nm="나스닥", stex_code="NY"),
            position_row(stex_nm="", stex_code="ND", stex_tp="NA"),
            position_row(stex_nm="미국", natn_nm="미국", default_exchange="ND"),
        ])
        for row in rows:
            with self.subTest(row=row):
                position = _us_position(row)
                self.assertEqual(position.exchange, "")
                self.assertIs(position.raw, row)
                with self.assertRaises(ValueError):
                    instrument_key(Instrument(position.market, position.symbol, position.exchange))

    def test_normalization_preserves_raw_and_financial_values(self):
        row = position_row(stk_cd=" lite ")
        original = dict(row)
        position = _us_position(row)
        self.assertEqual(position.symbol, "LITE")
        self.assertEqual(position.market, Market.US)
        self.assertEqual(position.currency, "USD")
        self.assertEqual(position.quantity, Decimal("3"))
        self.assertEqual(position.sellable_quantity, Decimal("2"))
        self.assertEqual(position.average_price, Decimal("100.25"))
        self.assertEqual(position.current_price, Decimal("102.50"))
        self.assertEqual(position.evaluation_amount, Decimal("307.50"))
        self.assertEqual(position.profit_loss, Decimal("6.75"))
        self.assertEqual(position.profit_rate, Decimal("2.2444"))
        self.assertEqual(row, original)
        self.assertIs(position.raw, row)

    def test_share_class_suffix_is_not_uppercased(self):
        self.assertEqual(_us_position(position_row(stk_cd=" BRKb ", stex_nm="뉴욕")).symbol, "BRKb")

    def test_malformed_symbol_does_not_prevent_account_display(self):
        self.assertEqual(_us_position(position_row(stk_cd=None)).symbol, "")

    def test_account_normalizes_dict_and_array_rows_without_defaulting_unknown(self):
        known = position_row()
        array_row = position_row(stk_cd="SWKS", stex_nm="NASDAQ")
        unknown = position_row(stk_cd="UNKNOWN", stex_nm="미국")
        balance = {
            "return_code": 0, "tot_prch_amt": "1000", "tot_evlt_amt": "1100",
            "tot_pl_amt": "100", "tot_pl_rt": "10",
            "result_list": [known, [array_row.get(key, "") for key in _US_POSITION_KEYS], unknown],
        }
        transport = QueueTransport(
            token_response(), FakeResponse(balance),
            FakeResponse({"return_code": 0, "result_list": [
                {"crnc_code": "USD", "fc_entra": "500", "fc_ord_alowa": "450"}]}),
        )
        account = KiwoomBroker(config(), transport=transport).account_us(exchange=USExchange.NASDAQ)
        self.assertEqual([position.exchange for position in account.positions], ["ND", "ND", ""])
        self.assertEqual([position.symbol for position in account.positions], ["LITE", "SWKS", "UNKNOWN"])
        self.assertEqual(account.cash, Decimal("500"))
        self.assertEqual(account.available_to_order, Decimal("450"))
        self.assertEqual(account.total_evaluation, Decimal("1100"))
        self.assertEqual(account.raw["balance"][0], balance)
        self.assertEqual([call["headers"].get("api-id") for call in transport.calls[1:]],
                         ["ust21070", "ust21110"])
        self.assertFalse(transport.responses)


if __name__ == "__main__":
    unittest.main()
