"""
universe.py - Upbit KRW market dynamic ticker universe helpers.

The live bot and backtest use this module so ticker selection logic does not
drift between trading and research.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Optional

import pandas as pd
import pyupbit

UNIVERSE_SIZE = 3
UNIVERSE_REFRESH_MINUTES = 0
UNIVERSE_VOLUME_LOOKBACK_MINUTES = 60
UNIVERSE_DAILY_LOOKBACK_MINUTES = 24 * 60
UNIVERSE_MIN_DAILY_QUOTE_KRW = 50_000_000_000
UNIVERSE_INTERVAL = "minute1"
EXCLUDED_TICKERS = {"KRW-USDT"}
LOW_VOLATILITY_EXCLUDE_RATIO = 0.30
# Strategy v7.0: 유니버스에서 ATR% 상위 10% 제외를 고정값으로 사용
UNIVERSE_ATR_EXCLUDE_RATIO = 0.10
UNIVERSE_NOISE_THRESHOLD = 0.55
UNIVERSE_BETA_TOP_N = 10
UNIVERSE_BETA_MIN = 1.2

TickerFetcher = Callable[..., list[str]]
OhlcvFetcher = Callable[..., Optional[pd.DataFrame]]


def get_krw_tickers(fetcher: TickerFetcher = pyupbit.get_tickers) -> list[str]:
    tickers = fetcher(fiat="KRW")
    return sorted(ticker for ticker in tickers if isinstance(ticker, str) and ticker.startswith("KRW-"))


def quote_volume_krw(df: pd.DataFrame) -> float:
    if df is None or df.empty or not {"close", "volume"}.issubset(df.columns):
        return 0.0
    quote_volume = (df["close"].astype(float) * df["volume"].astype(float)).sum()
    return float(quote_volume) if pd.notna(quote_volume) else 0.0


def price_noise_pct(df: pd.DataFrame) -> float:
    """최근 확정 구간의 평균 변동폭을 가격 대비 비율로 계산한다."""
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0
    close = df["close"].astype(float).replace(0, pd.NA)
    noise = ((df["high"].astype(float) - df["low"].astype(float)).abs() / close).dropna()
    if noise.empty:
        return 0.0
    return float(noise.mean())


def filter_low_volatility_tickers(
    ranked: list[tuple[str, float, float]],
    *,
    exclude_ratio: float = LOW_VOLATILITY_EXCLUDE_RATIO,
) -> list[str]:
    """거래대금 순위를 유지하되 변동성 하위권 종목을 제외한다."""
    if not ranked:
        return []
    scored = [(ticker, volume, volatility) for ticker, volume, volatility in ranked if volatility > 0]
    if not scored:
        return [ticker for ticker, _, _ in ranked]

    remove_count = int(len(scored) * exclude_ratio)
    if remove_count <= 0:
        return [ticker for ticker, _, _ in scored]

    low_volatility = {
        ticker
        for ticker, _, _ in sorted(scored, key=lambda item: item[2])[:remove_count]
    }
    return [ticker for ticker, _, _ in scored if ticker not in low_volatility]


def closed_candle_window(df: pd.DataFrame, lookback_minutes: int) -> pd.DataFrame:
    """Return the last N completed candles, excluding the newest in-progress candle."""
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) <= lookback_minutes:
        return df.tail(lookback_minutes)
    return df.iloc[-(lookback_minutes + 1):-1]


def rank_by_quote_volume(
    tickers: Iterable[str],
    *,
    ohlcv_fetcher: OhlcvFetcher = pyupbit.get_ohlcv,
    interval: str = UNIVERSE_INTERVAL,
    lookback_minutes: int = UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    request_sleep_sec: float = 0.08,
    max_retries: int = 3,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for ticker in tickers:
        df = None
        for attempt in range(max_retries):
            try:
                df = ohlcv_fetcher(ticker, interval=interval, count=lookback_minutes + 1)
            except Exception:
                df = None
            if df is not None and not df.empty:
                break
            if request_sleep_sec > 0:
                time.sleep(request_sleep_sec * (attempt + 1))
        if df is None or df.empty:
            continue
        closed = closed_candle_window(df, lookback_minutes) if df is not None else None
        volume = quote_volume_krw(closed) if closed is not None else 0.0
        if volume > 0:
            ranked.append((ticker, volume))
        if request_sleep_sec > 0:
            time.sleep(request_sleep_sec)
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def select_top_volume_tickers(
    *,
    limit: int = UNIVERSE_SIZE,
    lookback_minutes: int = UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    min_quote_volume_krw: float = 0.0,
    excluded_tickers: set[str] | None = EXCLUDED_TICKERS,
    ticker_fetcher: TickerFetcher = pyupbit.get_tickers,
    ohlcv_fetcher: OhlcvFetcher = pyupbit.get_ohlcv,
    request_sleep_sec: float = 0.08,
) -> list[str]:
    excluded = excluded_tickers or set()
    tickers = [ticker for ticker in get_krw_tickers(ticker_fetcher) if ticker not in excluded]
    ranked = rank_by_quote_volume(
        tickers,
        ohlcv_fetcher=ohlcv_fetcher,
        lookback_minutes=lookback_minutes,
        request_sleep_sec=request_sleep_sec,
    )
    filtered = [
        (ticker, volume)
        for ticker, volume in ranked
        if volume >= min_quote_volume_krw
    ]
    return [ticker for ticker, _ in filtered[:limit]]


def universe_refresh_due(
    updated_at: str | None,
    *,
    now: datetime | None = None,
    refresh_minutes: int = UNIVERSE_REFRESH_MINUTES,
) -> bool:
    if refresh_minutes <= 0:
        return True
    if not updated_at:
        return True
    now = now or datetime.now()
    try:
        last_updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return True
    return (now - last_updated).total_seconds() >= refresh_minutes * 60
