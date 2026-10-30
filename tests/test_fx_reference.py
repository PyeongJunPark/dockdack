"""Official rate parsing is pure; HTTP tests use a fake response only."""
from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal as D
from unittest.mock import patch

from dockdack.fx_reference import ECB_DAILY_XML, fetch_ecb_usd_krw, parse_ecb_usd_krw


XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01"
                 xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
  <Cube><Cube time="2026-09-25">
    <Cube currency="USD" rate="1.25"/>
    <Cube currency="KRW" rate="1500"/>
  </Cube></Cube>
</gesmes:Envelope>"""


class FakeResponse:
    def __init__(self, payload=XML, url=ECB_DAILY_XML):
        self.payload, self.url = payload, url

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def geturl(self):
        return self.url

    def read(self, count):
        return self.payload[:count]


class EcbFxReferenceTests(unittest.TestCase):
    def test_same_date_eur_quotes_cross_to_usd_krw_on_weekend(self):
        rate = parse_ecb_usd_krw(XML, today=date(2026, 9, 27))
        self.assertEqual((rate.published_on, rate.eur_usd, rate.eur_krw, rate.krw_per_usd),
                         (date(2026, 9, 25), D("1.25"), D("1500"), D("1200")))

    def test_missing_duplicate_nonpositive_or_stale_rates_fail_closed(self):
        samples = (XML.replace(b'currency="KRW"', b'currency="JPY"'),
                   XML.replace(b'currency="KRW"', b'currency="USD"'),
                   XML.replace(b'rate="1.25"', b'rate="0"'),
                   XML.replace(b'rate="1500"', b'rate="NaN"'),
                   XML.replace(b'time="2026-09-25"', b'time="bad-date"'),
                   XML.replace(b'<Cube><Cube time="2026-09-25">',
                               b'<Cube><Cube time="2026-09-25"><Cube time="2026-09-24"/>'))
        for payload in samples:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_ecb_usd_krw(payload, today=date(2026, 9, 27))
        with self.assertRaises(ValueError):
            parse_ecb_usd_krw(XML, today=date(2026, 10, 4))
        with self.assertRaises(ValueError):
            parse_ecb_usd_krw(XML, today=date(2026, 9, 24))

    def test_bounded_official_https_fetch_and_redirect_validation(self):
        with patch("dockdack.fx_reference.urlopen", return_value=FakeResponse()) as open_mock, \
                patch("dockdack.fx_reference.parse_ecb_usd_krw",
                      return_value="validated-rate") as parser:
            self.assertEqual(fetch_ecb_usd_krw(timeout=2), "validated-rate")
            self.assertEqual(open_mock.call_args.kwargs["timeout"], 2)
            self.assertEqual(open_mock.call_args.args[0].full_url, ECB_DAILY_XML)
            parser.assert_called_once_with(XML)
        with patch("dockdack.fx_reference.urlopen",
                   return_value=FakeResponse(url="https://example.com/fake.xml")):
            with self.assertRaisesRegex(ValueError, "공식 HTTPS"):
                fetch_ecb_usd_krw()
        with self.assertRaises(ValueError):
            fetch_ecb_usd_krw(timeout=0)


if __name__ == "__main__":
    unittest.main()
