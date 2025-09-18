#!/usr/bin/env python3
"""Utility to fetch recent Nikkei 225 (^N225) quotes via Yahoo Finance."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    import requests
except ImportError as exc:  # pragma: nocover - runtime guard
    raise SystemExit("'requests' 패키지가 필요합니다. 'pip install requests'로 설치해 주세요.") from exc

YAHOO_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/%5EN225"
DEFAULT_INTERVAL = "1d"
DEFAULT_PERIOD = "5d"
DEFAULT_TIMEOUT = 10.0
DEFAULT_VERIFY_SSL = False


@dataclass(frozen=True)
class Quote:
    timestamp: dt.datetime
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[int]

    def as_csv_row(self) -> str:
        return ",".join(
            [
                self.timestamp.isoformat(),
                _format_float_csv(self.open),
                _format_float_csv(self.high),
                _format_float_csv(self.low),
                _format_float_csv(self.close),
                _format_int_csv(self.volume),
            ]
        )

    def as_readable_row(self) -> str:
        open_str = _format_float_human(self.open)
        high_str = _format_float_human(self.high)
        low_str = _format_float_human(self.low)
        close_str = _format_float_human(self.close)
        volume_str = _format_int_human(self.volume)
        return (
            f"{self.timestamp.isoformat()}  "
            f"시가:{open_str:>10} "
            f"고가:{high_str:>10} "
            f"저가:{low_str:>10} "
            f"종가:{close_str:>10} "
            f"거래량:{volume_str:>12}"
        )


def fetch_nikkei225(
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
    timeout: float = DEFAULT_TIMEOUT,
    verify_ssl: bool = DEFAULT_VERIFY_SSL,
) -> List[Quote]:
    params = {"interval": interval, "range": period}
    response = requests.get(
        YAHOO_CHART_ENDPOINT,
        params=params,
        timeout=timeout,
        verify=verify_ssl,
    )
    response.raise_for_status()
    payload = response.json()
    return _parse_chart_payload(payload)


def _parse_chart_payload(payload: object) -> List[Quote]:
    if not isinstance(payload, dict):
        raise ValueError("예상치 못한 응답 형식입니다: payload가 dict가 아닙니다.")

    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("예상치 못한 응답 형식입니다: chart 필드가 없습니다.")

    error_block = chart.get("error")
    if error_block:
        raise RuntimeError(f"야후 파이낸스에서 오류를 반환했습니다: {error_block}")

    results = chart.get("result")
    if not results:
        return []

    if not isinstance(results, list):
        raise ValueError("예상치 못한 응답 형식입니다: result 필드가 리스트가 아닙니다.")

    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("예상치 못한 응답 형식입니다: result 항목이 dict가 아닙니다.")

    meta = result.get("meta", {})
    offset_seconds = 0
    if isinstance(meta, dict):
        offset_seconds = int(meta.get("gmtoffset", 0) or 0)

    tz = dt.timezone(dt.timedelta(seconds=offset_seconds))
    timestamps = result.get("timestamp") or []
    if not isinstance(timestamps, list):
        raise ValueError("예상치 못한 응답 형식입니다: timestamp 필드가 리스트가 아닙니다.")

    indicators = result.get("indicators", {})
    if not isinstance(indicators, dict):
        raise ValueError("예상치 못한 응답 형식입니다: indicators 필드가 dict가 아닙니다.")

    quote_blocks = indicators.get("quote") or []
    if not isinstance(quote_blocks, list) or not quote_blocks:
        return []

    quote_block = quote_blocks[0]
    if not isinstance(quote_block, dict):
        raise ValueError("예상치 못한 응답 형식입니다: quote 블록이 dict가 아닙니다.")

    opens = quote_block.get("open")
    highs = quote_block.get("high")
    lows = quote_block.get("low")
    closes = quote_block.get("close")
    volumes = quote_block.get("volume")

    quotes: List[Quote] = []
    for index, raw_timestamp in enumerate(timestamps):
        if not isinstance(raw_timestamp, (int, float)):
            continue
        timestamp = dt.datetime.fromtimestamp(raw_timestamp, tz=tz)
        quotes.append(
            Quote(
                timestamp=timestamp,
                open=_safe_float(opens, index),
                high=_safe_float(highs, index),
                low=_safe_float(lows, index),
                close=_safe_float(closes, index),
                volume=_safe_int(volumes, index),
            )
        )

    return quotes


def _safe_float(values: Optional[Sequence[Optional[float]]], index: int) -> Optional[float]:
    value = _value_at(values, index)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(values: Optional[Sequence[Optional[float]]], index: int) -> Optional[int]:
    value = _value_at(values, index)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _value_at(values: Optional[Sequence], index: int):  # type: ignore[override]
    if values is None:
        return None
    try:
        return values[index]
    except (IndexError, TypeError):
        return None


def _format_float_csv(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_int_csv(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def _format_float_human(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:,.2f}"


def _format_int_human(value: Optional[int]) -> str:
    return "--" if value is None else f"{value:,}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="야후 파이낸스에서 니케이225 (^N225) 지수 시세를 조회합니다.",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="조회 간격 (예: 1d, 1h, 30m, 5m).",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="조회 기간 (예: 1d, 5d, 1mo, 6mo, 1y, max).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="표시할 최근 행 수 (0이면 전부 표시).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="사람이 읽기 쉬운 형식 대신 CSV로 출력합니다.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP 타임아웃(초).",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="SSL 인증서를 검증합니다 (기본값: 검증하지 않음).",
    )

    args = parser.parse_args(argv)

    try:
        quotes = fetch_nikkei225(
            interval=args.interval,
            period=args.period,
            timeout=args.timeout,
            verify_ssl=args.verify_ssl,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"네트워크 요청에 실패했습니다: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"응답을 해석하지 못했습니다: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not quotes:
        print("니케이225 데이터가 반환되지 않았습니다.")
        return 0

    if args.limit and args.limit > 0:
        quotes = quotes[-args.limit :]

    if args.csv:
        print("일시,시가,고가,저가,종가,거래량")
        for quote in quotes:
            print(quote.as_csv_row())
    else:
        for quote in quotes:
            print(quote.as_readable_row())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
