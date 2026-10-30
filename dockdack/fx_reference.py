"""Read-only ECB reference-rate conversion for display, never settlement."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ECB_DAILY_XML = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
MAX_RESPONSE_BYTES = 128_000
MAX_RATE_AGE_DAYS = 7


@dataclass(frozen=True)
class UsdKrwReference:
    """Current-display reference, not a trade-date or executable FX rate."""

    published_on: date
    krw_per_usd: Decimal
    eur_usd: Decimal
    eur_krw: Decimal
    source_url: str = ECB_DAILY_XML


def parse_ecb_usd_krw(xml_data: bytes, *, today: date | None = None) -> UsdKrwReference:
    """Require one ECB publication containing both EUR-base quotes on one day."""
    if not isinstance(xml_data, bytes) or not xml_data or len(xml_data) > MAX_RESPONSE_BYTES:
        raise ValueError("ECB 환율 응답 크기를 확인할 수 없습니다.")
    try:
        root = ElementTree.fromstring(xml_data)
    except ElementTree.ParseError as exc:
        raise ValueError("ECB 환율 XML 형식이 올바르지 않습니다.") from exc
    dated = [node for node in root.iter() if "time" in node.attrib]
    if len(dated) != 1:
        raise ValueError("ECB 환율 고시일을 하나로 확인할 수 없습니다.")
    try:
        published = date.fromisoformat(dated[0].attrib["time"])
    except ValueError as exc:
        raise ValueError("ECB 환율 고시일 형식이 올바르지 않습니다.") from exc
    today = today or datetime.now(timezone.utc).date()
    age = (today - published).days
    if not 0 <= age <= MAX_RATE_AGE_DAYS:
        raise ValueError("ECB 환율 고시일이 미래이거나 너무 오래되었습니다.")
    rates: dict[str, Decimal] = {}
    for node in dated[0]:
        currency = node.attrib.get("currency")
        if currency not in {"USD", "KRW"}:
            continue
        if currency in rates:
            raise ValueError("ECB 환율 통화가 중복되었습니다.")
        try:
            rate = Decimal(node.attrib["rate"])
        except (KeyError, InvalidOperation) as exc:
            raise ValueError("ECB 환율 숫자를 확인할 수 없습니다.") from exc
        if not rate.is_finite() or rate <= 0:
            raise ValueError("ECB 환율은 유효한 양수여야 합니다.")
        rates[currency] = rate
    if set(rates) != {"USD", "KRW"}:
        raise ValueError("ECB의 같은 고시일 USD·KRW 환율이 모두 필요합니다.")
    return UsdKrwReference(published, rates["KRW"] / rates["USD"],
                           rates["USD"], rates["KRW"])


def fetch_ecb_usd_krw(*, timeout: float = 3.0) -> UsdKrwReference:
    """Fetch a small official XML snapshot with a bounded network wait."""
    if not 0 < timeout <= 10:
        raise ValueError("환율 조회 제한 시간은 0초 초과 10초 이하여야 합니다.")
    request = Request(ECB_DAILY_XML, headers={
        "User-Agent": "DockDack/1.0 (read-only reference-rate display)",
        "Accept": "application/xml,text/xml",
    })
    with urlopen(request, timeout=timeout) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != "www.ecb.europa.eu":
            raise ValueError("ECB 공식 HTTPS 출처를 확인할 수 없습니다.")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    return parse_ecb_usd_krw(payload)
