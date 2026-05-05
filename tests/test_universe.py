from __future__ import annotations

from datetime import datetime

import pandas as pd

from universe import (
    closed_candle_window,
    filter_low_volatility_tickers,
    get_krw_tickers,
    price_noise_pct,
    quote_volume_krw,
    rank_by_quote_volume,
    select_top_volume_tickers,
    universe_refresh_due,
)


def _ohlcv(close: float, volume: float, rows: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="min")
    return pd.DataFrame({"high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume}, index=idx)


def test_quote_volume_uses_close_times_volume():
    df = pd.DataFrame({"close": [100.0, 110.0], "volume": [2.0, 3.0]})

    assert quote_volume_krw(df) == 530.0


def test_price_noise_pct_uses_high_low_range_over_close():
    df = pd.DataFrame({"high": [110.0], "low": [90.0], "close": [100.0]})

    assert price_noise_pct(df) == 0.2


def test_filter_low_volatility_tickers_preserves_volume_order():
    ranked = [
        ("KRW-BTC", 1_000.0, 0.001),
        ("KRW-XRP", 900.0, 0.020),
        ("KRW-ETH", 800.0, 0.010),
        ("KRW-SOL", 700.0, 0.030),
    ]

    selected = filter_low_volatility_tickers(ranked, exclude_ratio=0.25)

    assert selected == ["KRW-XRP", "KRW-ETH", "KRW-SOL"]


def test_closed_candle_window_drops_newest_when_extra_row_exists():
    df = _ohlcv(100.0, 1.0, rows=4)

    out = closed_candle_window(df, lookback_minutes=3)

    assert list(out.index) == list(df.index[:3])


def test_select_top_volume_tickers_filters_krw_and_ranks_by_quote_volume():
    def ticker_fetcher(fiat: str):
        assert fiat == "KRW"
        return ["KRW-BTC", "BTC-ETH", "KRW-XRP", "KRW-ETH"]

    frames = {
        "KRW-BTC": _ohlcv(100.0, 1.0),
        "KRW-XRP": _ohlcv(10.0, 50.0),
        "KRW-ETH": _ohlcv(200.0, 2.0),
    }

    ranked = rank_by_quote_volume(
        get_krw_tickers(ticker_fetcher),
        ohlcv_fetcher=lambda ticker, **_: frames[ticker],
        request_sleep_sec=0,
    )
    selected = select_top_volume_tickers(
        limit=2,
        ticker_fetcher=ticker_fetcher,
        ohlcv_fetcher=lambda ticker, **_: frames[ticker],
        request_sleep_sec=0,
    )

    assert ranked == [("KRW-XRP", 1500.0), ("KRW-ETH", 1200.0), ("KRW-BTC", 300.0)]
    assert selected == ["KRW-XRP", "KRW-ETH"]


def test_select_top_volume_tickers_excludes_usdt_by_default():
    def ticker_fetcher(fiat: str):
        assert fiat == "KRW"
        return ["KRW-USDT", "KRW-BTC", "KRW-XRP"]

    frames = {
        "KRW-USDT": _ohlcv(1_400.0, 10_000.0),
        "KRW-BTC": _ohlcv(100.0, 1.0),
        "KRW-XRP": _ohlcv(10.0, 50.0),
    }

    selected = select_top_volume_tickers(
        limit=2,
        ticker_fetcher=ticker_fetcher,
        ohlcv_fetcher=lambda ticker, **_: frames[ticker],
        request_sleep_sec=0,
    )

    assert selected == ["KRW-XRP", "KRW-BTC"]


def test_rank_by_quote_volume_retries_empty_ohlcv_response():
    calls = 0

    def flaky_fetcher(ticker: str, **_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return _ohlcv(100.0, 2.0, rows=4)

    ranked = rank_by_quote_volume(
        ["KRW-BTC"],
        ohlcv_fetcher=flaky_fetcher,
        lookback_minutes=3,
        request_sleep_sec=0,
    )

    assert ranked == [("KRW-BTC", 600.0)]
    assert calls == 2


def test_universe_refresh_due_defaults_to_every_loop():
    now = datetime(2026, 1, 1, 1, 0, 0)

    assert universe_refresh_due(None, now=now)
    assert universe_refresh_due(now.isoformat(), now=now)
    assert universe_refresh_due("bad timestamp", now=now)
