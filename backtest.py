#!/usr/bin/env python3
"""
backtest.py - 1-hour multi-coin walk-forward backtest.

The backtest deliberately calls strategy.py for all entry/exit decisions so
live trading and research cannot drift apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyupbit

import strategy
from bot import (
    EXCLUDED_TICKERS,
    MAX_CONCURRENT,
    PER_TRADE_RATIO,
    PER_TRADE_RATIO_SPIKE,
    UNIVERSE_TOP_N,
)
from strategy import (
    DEFAULT_SLIPPAGE,
    StrategyConfig,
    activate_final_tuned_profile,
    add_indicators,
    adaptive_stop_price,
    cooldown_until_after_loss,
    entry_signal,
    exit_signal,
    get_strategy_config,
)
from universe import (
    UNIVERSE_ATR_EXCLUDE_RATIO,
    UNIVERSE_BETA_MIN,
    UNIVERSE_BETA_TOP_N,
    UNIVERSE_DAILY_LOOKBACK_MINUTES,
    UNIVERSE_MIN_DAILY_QUOTE_KRW,
    UNIVERSE_NOISE_THRESHOLD,
    UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    blend_universe_candidates,
    exclude_high_atr_candidates,
    filter_by_noise_threshold,
    get_krw_tickers,
    price_noise_pct,
    prioritize_by_beta,
    quote_volume_krw,
)

DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]
INTERVAL = "minute60"
INITIAL_KRW = 1_000_000
FEE = 0.001  # Upbit 0.05% buy + 0.05% sell
SLIPPAGE = DEFAULT_SLIPPAGE
INDICATOR_CACHE_VERSION = 2
SIGNAL_TICKER_CACHE_VERSION = 1


def _cache_path(ticker: str) -> Path:
    return Path(f"data_cache_{ticker.replace('-', '_')}_{INTERVAL}.pkl")


def _indicator_cache_path(ticker: str) -> Path:
    return Path(f"indicator_cache_{ticker.replace('-', '_')}_{INTERVAL}_v{INDICATOR_CACHE_VERSION}.pkl")


def _signal_ticker_cache_path(cache_key: str) -> Path:
    return Path(f"signal_tickers_{cache_key}_v{SIGNAL_TICKER_CACHE_VERSION}.pkl")


def _signal_ticker_cache_key(
    *,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    universe_mode: str,
    universe_limit: int,
    universe_lookback: int,
    min_quote_volume_krw: float,
    noise_threshold: float,
    atr_exclude_ratio: float,
    beta_min: float,
    beta_top_n: int,
) -> str:
    payload = {
        "start": str(pd.Timestamp(start_at)),
        "end": str(pd.Timestamp(end_at)),
        "universe_mode": universe_mode,
        "universe_limit": universe_limit,
        "universe_lookback": universe_lookback,
        "min_quote_volume_krw": min_quote_volume_krw,
        "noise_threshold": noise_threshold,
        "atr_exclude_ratio": atr_exclude_ratio,
        "beta_min": beta_min,
        "beta_top_n": beta_top_n,
        "strategy": {
            "SECONDARY_RSI_LEVEL": strategy.SECONDARY_RSI_LEVEL,
            "PRIMARY_VOLUME_SPIKE_MULTIPLIER": strategy.PRIMARY_VOLUME_SPIKE_MULTIPLIER,
            "ENTRY_RSI_MAX": strategy.ENTRY_RSI_MAX,
            "VOLUME_PULLBACK_MULTIPLIER": strategy.VOLUME_PULLBACK_MULTIPLIER,
            "MARKET_RSI_MIN": strategy.MARKET_RSI_MIN,
            "ATR_STOP_MULTIPLIER": strategy.ATR_STOP_MULTIPLIER,
            "TAKE_PROFIT_ARM_PNL": strategy.TAKE_PROFIT_ARM_PNL,
            "TRAILING_TAKE_PROFIT_DRAWDOWN": strategy.TRAILING_TAKE_PROFIT_DRAWDOWN,
            "TRAILING_STEP_UP_PNL": strategy.TRAILING_STEP_UP_PNL,
            "TRAILING_STEP_UP_DRAWDOWN": strategy.TRAILING_STEP_UP_DRAWDOWN,
            "VWAP_EXIT_BUFFER": strategy.VWAP_EXIT_BUFFER,
            "VWAP_EXIT_LOCKED_PROFIT_BUFFER": strategy.VWAP_EXIT_LOCKED_PROFIT_BUFFER,
            "TIME_CUT_BARS": strategy.TIME_CUT_BARS,
            "TIME_CUT_MIN_PNL": strategy.TIME_CUT_MIN_PNL,
            "FAST_TIME_CUT_BARS": strategy.FAST_TIME_CUT_BARS,
            "FAST_TIME_CUT_MIN_PNL": strategy.FAST_TIME_CUT_MIN_PNL,
            "DEAD_CROSS_EXIT_ENABLED": strategy.DEAD_CROSS_EXIT_ENABLED,
            "BB_REV_PARTIAL_EXIT_ENABLED": strategy.BB_REV_PARTIAL_EXIT_ENABLED,
        },
        "portfolio": {
            "MAX_CONCURRENT": MAX_CONCURRENT,
            "PER_TRADE_RATIO": PER_TRADE_RATIO,
            "PER_TRADE_RATIO_SPIKE": PER_TRADE_RATIO_SPIKE,
        },
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def build_signal_ticker_cache(
    frames: dict[str, pd.DataFrame],
    configs: dict[str, StrategyConfig],
    *,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    universe_mode: str,
    universe_limit: int,
    universe_lookback: int = UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    use_cache: bool = True,
) -> set[str]:
    cache_key = _signal_ticker_cache_key(
        start_at=start_at,
        end_at=end_at,
        universe_mode=universe_mode,
        universe_limit=universe_limit,
        universe_lookback=universe_lookback,
        min_quote_volume_krw=UNIVERSE_MIN_DAILY_QUOTE_KRW,
        noise_threshold=UNIVERSE_NOISE_THRESHOLD,
        atr_exclude_ratio=UNIVERSE_ATR_EXCLUDE_RATIO,
        beta_min=UNIVERSE_BETA_MIN,
        beta_top_n=UNIVERSE_BETA_TOP_N,
    )
    cache_path = _signal_ticker_cache_path(cache_key)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, set):
                return cached
        except Exception:
            pass

    signal_tickers: set[str] = {"KRW-BTC"}
    daily_universe_cache: dict[pd.Timestamp, set[str]] = {}
    hourly_universe_cache: dict[pd.Timestamp, set[str]] = {}

    for ticker, df in frames.items():
        if ticker == "KRW-BTC":
            continue
        cfg = configs.get(ticker, get_strategy_config(ticker))
        for i in range(1, len(df) - 1):
            ts = df.index[i]
            if ts < start_at or ts > end_at:
                continue

            if universe_mode == "daily":
                day = pd.Timestamp(ts).normalize()
                if day not in daily_universe_cache:
                    daily_universe_cache[day] = set(
                        select_daily_universe_from_frames(
                            frames,
                            day,
                            limit=universe_limit,
                            min_quote_volume_krw=UNIVERSE_MIN_DAILY_QUOTE_KRW,
                            noise_threshold=UNIVERSE_NOISE_THRESHOLD,
                            atr_exclude_ratio=UNIVERSE_ATR_EXCLUDE_RATIO,
                            beta_min=UNIVERSE_BETA_MIN,
                            beta_top_n=UNIVERSE_BETA_TOP_N,
                        )
                    )
                active_universe = daily_universe_cache[day]
            else:
                candle_ts = pd.Timestamp(ts)
                if candle_ts not in hourly_universe_cache:
                    hourly_universe_cache[candle_ts] = set(
                        select_universe_from_frames(
                            frames,
                            candle_ts,
                            limit=universe_limit,
                            lookback=universe_lookback,
                            min_quote_volume_krw=UNIVERSE_MIN_DAILY_QUOTE_KRW,
                            noise_threshold=UNIVERSE_NOISE_THRESHOLD,
                            atr_exclude_ratio=UNIVERSE_ATR_EXCLUDE_RATIO,
                            beta_min=UNIVERSE_BETA_MIN,
                            beta_top_n=UNIVERSE_BETA_TOP_N,
                        )
                    )
                active_universe = hourly_universe_cache[candle_ts]

            if ticker not in active_universe:
                continue

            ok, _ = entry_signal(
                df.iloc[i],
                df.iloc[i - 1],
                ticker=ticker,
                config=cfg,
                now=ts,
                market_current=_market_row(frames, ts),
            )
            if ok:
                signal_tickers.add(ticker)
                break

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(signal_tickers, f)
    return signal_tickers


def fetch_year_of_data(ticker: str, *, refresh_cache: bool = False) -> pd.DataFrame:
    cache = _cache_path(ticker)
    cached_df: pd.DataFrame | None = None
    if cache.exists():
        age_h = (time.time() - cache.stat().st_mtime) / 3600
        with open(cache, "rb") as f:
            cached_df = pickle.load(f)
        if not refresh_cache or age_h < 6:
            freshness = "fresh" if age_h < 6 else "stale"
            print(f"[{ticker}] cache hit ({freshness}, {age_h:.1f}h old)")
            return cached_df
        print(f"[{ticker}] cache expired - downloading")

    cutoff = pd.Timestamp(datetime.now() - timedelta(days=365))
    all_frames: list[pd.DataFrame] = []
    to: str | None = None

    print(f"[{ticker}] downloading 1y of 1h candles...")
    while True:
        kwargs: dict = {"count": 200}
        if to:
            kwargs["to"] = to

        df = None
        for attempt in range(4):
            try:
                df = pyupbit.get_ohlcv(ticker, interval=INTERVAL, **kwargs)
                break
            except Exception as exc:
                wait = 2**attempt
                print(f"\n  fetch error ({attempt + 1}): {exc}; retrying in {wait}s")
                time.sleep(wait)

        if df is None or df.empty:
            break

        all_frames.append(df)
        print("." if len(all_frames) % 100 else f"\n  {len(all_frames)*200:,} candles", end="", flush=True)
        if df.index[0] <= cutoff:
            break

        to = (df.index[0] - pd.Timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(0.13)

    print(f"\n[{ticker}] download complete")
    if not all_frames:
        if cached_df is not None and not cached_df.empty:
            print(f"[{ticker}] refresh failed - using stale cache")
            return cached_df
        return pd.DataFrame()

    combined = pd.concat(all_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    result = combined[combined.index >= cutoff].copy()
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result


def load_indicator_frame(ticker: str, *, refresh_cache: bool = False) -> pd.DataFrame:
    raw = fetch_year_of_data(ticker, refresh_cache=refresh_cache)
    if raw.empty:
        return pd.DataFrame()

    raw_cache = _cache_path(ticker)
    indicator_cache = _indicator_cache_path(ticker)
    raw_mtime = raw_cache.stat().st_mtime if raw_cache.exists() else None

    if indicator_cache.exists() and raw_mtime is not None:
        try:
            with open(indicator_cache, "rb") as f:
                payload = pickle.load(f)
            if (
                isinstance(payload, dict)
                and payload.get("raw_mtime") == raw_mtime
                and isinstance(payload.get("frame"), pd.DataFrame)
            ):
                return payload["frame"]
        except Exception:
            pass

    enriched = add_indicators(raw.sort_index())
    enriched = add_backtest_universe_metrics(enriched)
    if raw_mtime is not None:
        with open(indicator_cache, "wb") as f:
            pickle.dump({"raw_mtime": raw_mtime, "frame": enriched}, f)
    return enriched


def add_backtest_universe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    quote_value = out["close"].astype(float) * out["volume"].astype(float)
    out["quote_volume_24h"] = quote_value.rolling(24).sum()
    prev_quote_volume_24h = out["quote_volume_24h"].shift(24)
    out["surge_24h"] = (out["quote_volume_24h"] - prev_quote_volume_24h) / prev_quote_volume_24h
    out["rsi_momentum_4"] = out["rsi"] - out["rsi"].shift(3)
    out["noise_24h"] = ((out["high"].astype(float) - out["low"].astype(float)).abs() / out["close"].astype(float)).rolling(24).mean()
    out["atr_pct_24h"] = out["atr_pct"].rolling(24).mean()
    out["ret_24h"] = out["close"].astype(float) / out["close"].astype(float).shift(24) - 1
    return out


def simulate_ticker(
    df: pd.DataFrame,
    *,
    ticker: str,
    config: StrategyConfig,
    fee: float = FEE,
    slippage: float = SLIPPAGE,
    start_at: pd.Timestamp | None = None,
) -> tuple[list[dict], pd.Series]:
    in_position = False
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    highest_price = 0.0
    trailing_active = False
    partial_taken = False
    bb_break_seen = False
    overbought_seen = False
    position_fraction = 1.0
    signal_type = ""
    cooldown_until: pd.Timestamp | None = None

    trades: list[dict] = []
    equity = 1.0
    eq_ts: dict[pd.Timestamp, float] = {}

    for i in range(1, len(df) - 1):
        c = df.iloc[i]
        p = df.iloc[i - 1]
        if start_at is not None and c.name < start_at:
            continue
        eq_ts[c.name] = equity

        if in_position:
            exit_price, reason, trailing_active, highest_price, overbought_seen, bb_break_seen = exit_signal(
                c,
                entry_price=entry_price,
                previous=p,
                entry_time=entry_time,
                current_time=c.name,
                partial_taken=partial_taken,
                bb_break_seen=bb_break_seen,
                overbought_seen=overbought_seen,
                config=config,
                use_ohlc=True,
                take_profit_armed=trailing_active,
                highest_price=highest_price,
            )
            if exit_price is not None:
                if reason.startswith("PARTIAL_TAKE_PROFIT_50"):
                    net_pnl = (exit_price / entry_price - 1) - fee - (2 * slippage)
                    equity *= 1 + (net_pnl * 0.5)
                    partial_taken = True
                    position_fraction = 0.5
                    trailing_active = True
                    continue
                if reason.startswith("PARTIAL_TAKE_PROFIT_25_BB_REV"):
                    net_pnl = (exit_price / entry_price - 1) - fee - (2 * slippage)
                    equity *= 1 + (net_pnl * 0.25)
                    position_fraction = 0.25
                    partial_taken = True
                    continue
                net_pnl = (exit_price / entry_price - 1) - fee - (2 * slippage)
                equity *= 1 + (net_pnl * position_fraction)
                reason_code = reason.split()[0]
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": c.name,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": net_pnl,
                        "reason": reason_code,
                        "signal": signal_type,
                    }
                )
                in_position = False
                if net_pnl < 0:
                    cooldown_until = pd.Timestamp(cooldown_until_after_loss(c.name, config))
            continue

        ok, reason = entry_signal(
            c,
            p,
            ticker=ticker,
            config=config,
            cooldown_until=cooldown_until,
            now=c.name,
            market_current=c if ticker == "KRW-BTC" else None,
        )
        if ok:
            nc = df.iloc[i + 1]
            entry_price = float(nc["open"]) * (1 + slippage)
            entry_time = nc.name
            highest_price = entry_price
            trailing_active = False
            partial_taken = False
            bb_break_seen = False
            position_fraction = 1.0
            overbought_seen = False
            signal_type = reason.split("]", 1)[0].replace("[", "")
            in_position = True

    if in_position and entry_time is not None:
        final = df.iloc[-1]
        exit_price = float(final["close"])
        net_pnl = (exit_price / entry_price - 1) - fee - (2 * slippage)
        equity *= 1 + net_pnl
        eq_ts[final.name] = equity
        trades.append(
            {
                "ticker": ticker,
                "entry_time": entry_time,
                "exit_time": final.name,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": net_pnl,
                "reason": "EOD",
                "signal": signal_type,
            }
        )

    return trades, pd.Series(eq_ts, name="equity", dtype=float)


def simulate_portfolio(
    frames: dict[str, pd.DataFrame],
    configs: dict[str, StrategyConfig],
    *,
    initial_krw: float = INITIAL_KRW,
    fee: float = FEE,
    slippage: float = SLIPPAGE,
    dynamic_universe: bool = False,
    universe_mode: str = "hourly",
    universe_limit: int = UNIVERSE_TOP_N,
    universe_lookback: int = UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    start_at: pd.Timestamp | None = None,
    hourly_universe_cache: dict[pd.Timestamp, set[str]] | None = None,
    event_tickers: set[str] | None = None,
) -> tuple[list[dict], pd.Series]:
    events: list[tuple[pd.Timestamp, str, int]] = []
    for ticker, df in frames.items():
        if event_tickers is not None and ticker not in event_tickers:
            continue
        for i in range(1, len(df) - 1):
            events.append((df.index[i], ticker, i))
    events.sort(key=lambda x: x[0])

    cash = initial_krw
    positions: dict[str, dict] = {}
    cooldowns: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    equity_ts: dict[pd.Timestamp, float] = {}
    last_close: dict[str, float] = {}
    active_universe = set(frames.keys())
    daily_universe_cache: dict[pd.Timestamp, set[str]] = {}
    hourly_cache = hourly_universe_cache or {}

    for ts, ticker, i in events:
        if start_at is not None and ts < start_at:
            continue
        df = frames[ticker]
        c = df.iloc[i]
        p = df.iloc[i - 1]
        cfg = configs.get(ticker, get_strategy_config(ticker))
        last_close[ticker] = float(c["close"])

        if dynamic_universe and universe_mode == "daily":
            day = pd.Timestamp(ts).normalize()
            if day not in daily_universe_cache:
                daily_universe_cache[day] = set(
                    select_daily_universe_from_frames(
                        frames,
                        day,
                        limit=universe_limit,
                        min_quote_volume_krw=UNIVERSE_MIN_DAILY_QUOTE_KRW,
                        noise_threshold=UNIVERSE_NOISE_THRESHOLD,
                        atr_exclude_ratio=UNIVERSE_ATR_EXCLUDE_RATIO,
                        beta_min=UNIVERSE_BETA_MIN,
                        beta_top_n=UNIVERSE_BETA_TOP_N,
                    )
                )
            active_universe = daily_universe_cache[day]
        elif dynamic_universe:
            candle_ts = pd.Timestamp(ts)
            if candle_ts not in hourly_cache:
                hourly_cache[candle_ts] = set(
                    select_universe_from_frames(
                        frames,
                        candle_ts,
                        limit=universe_limit,
                        lookback=universe_lookback,
                        min_quote_volume_krw=UNIVERSE_MIN_DAILY_QUOTE_KRW,
                        noise_threshold=UNIVERSE_NOISE_THRESHOLD,
                        atr_exclude_ratio=UNIVERSE_ATR_EXCLUDE_RATIO,
                        beta_min=UNIVERSE_BETA_MIN,
                        beta_top_n=UNIVERSE_BETA_TOP_N,
                    )
                )
            active_universe = hourly_cache[candle_ts]

        equity = cash + sum(pos["qty"] * last_close.get(t, pos["entry_price"]) for t, pos in positions.items())
        equity_ts[ts] = equity

        if ticker in positions:
            pos = positions[ticker]
            exit_price, reason, armed, highest, overbought_seen, bb_break_seen = exit_signal(
                c,
                entry_price=pos["entry_price"],
                previous=p,
                entry_time=pos["entry_time"],
                current_time=ts,
                partial_taken=pos.get("partial_taken", False),
                bb_break_seen=pos.get("bb_break_seen", False),
                overbought_seen=pos.get("overbought_seen", False),
                config=cfg,
                use_ohlc=True,
                take_profit_armed=pos.get("take_profit_armed", False),
                highest_price=pos.get("highest_price", pos["entry_price"]),
            )
            pos["take_profit_armed"] = armed
            pos["highest_price"] = highest
            pos["overbought_seen"] = overbought_seen
            pos["bb_break_seen"] = bb_break_seen
            if exit_price is None:
                continue

            if reason.startswith("PARTIAL_TAKE_PROFIT_50"):
                fill_price = exit_price * (1 - slippage)
                sell_qty = pos["qty"] * 0.5
                proceeds = sell_qty * fill_price
                sell_fee = proceeds * (fee / 2)
                cash += proceeds - sell_fee
                realized_invest = pos["invest_krw"] * 0.5
                pnl_krw = proceeds - sell_fee - realized_invest
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "entry_price": pos["entry_price"],
                        "exit_price": fill_price,
                        "invest_krw": realized_invest,
                        "pnl": pnl_krw / realized_invest if realized_invest > 0 else 0.0,
                        "pnl_krw": pnl_krw,
                        "reason": "PARTIAL_TAKE_PROFIT_50",
                        "signal": pos["signal"],
                    }
                )
                pos["qty"] = pos["qty"] - sell_qty
                pos["invest_krw"] = realized_invest
                pos["partial_taken"] = True
                pos["take_profit_armed"] = True
                pos["overbought_seen"] = overbought_seen
                continue
            if reason.startswith("PARTIAL_TAKE_PROFIT_25_BB_REV"):
                fill_price = exit_price * (1 - slippage)
                sell_qty = pos["qty"] * 0.5
                proceeds = sell_qty * fill_price
                sell_fee = proceeds * (fee / 2)
                cash += proceeds - sell_fee
                realized_invest = pos["invest_krw"] * 0.5
                pnl_krw = proceeds - sell_fee - realized_invest
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "entry_price": pos["entry_price"],
                        "exit_price": fill_price,
                        "invest_krw": realized_invest,
                        "pnl": pnl_krw / realized_invest if realized_invest > 0 else 0.0,
                        "pnl_krw": pnl_krw,
                        "reason": "PARTIAL_TAKE_PROFIT_25_BB_REV",
                        "signal": pos["signal"],
                    }
                )
                pos["qty"] = pos["qty"] - sell_qty
                pos["invest_krw"] = realized_invest
                continue

            fill_price = exit_price * (1 - slippage)
            proceeds = pos["qty"] * fill_price
            sell_fee = proceeds * (fee / 2)
            cash += proceeds - sell_fee
            pnl_krw = proceeds - sell_fee - pos["invest_krw"]
            pnl = pnl_krw / pos["invest_krw"]
            reason_code = reason.split()[0]
            trades.append(
                {
                    "ticker": ticker,
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_price": pos["entry_price"],
                    "exit_price": fill_price,
                    "invest_krw": pos["invest_krw"],
                    "pnl": pnl,
                    "pnl_krw": pnl_krw,
                    "reason": reason_code,
                    "signal": pos["signal"],
                }
            )
            del positions[ticker]
            if pnl < 0:
                cooldowns[ticker] = pd.Timestamp(cooldown_until_after_loss(ts, cfg))
            continue

        if ticker not in active_universe:
            continue

        if len(positions) >= MAX_CONCURRENT or not cfg.enabled:
            continue
        ok, reason = entry_signal(
            c,
            p,
            ticker=ticker,
            config=cfg,
            cooldown_until=cooldowns.get(ticker),
            now=ts,
            market_current=_market_row(frames, ts),
        )
        if not ok:
            continue

        next_open = float(df.iloc[i + 1]["open"]) * (1 + slippage)
        stop_price = adaptive_stop_price(next_open, c, cfg)
        trade_ratio = PER_TRADE_RATIO_SPIKE if "ENTRY:PRIMARY_SPIKE" in reason else PER_TRADE_RATIO
        invest_krw = min(cash, equity * trade_ratio)
        qty = invest_krw / next_open if next_open > 0 else 0.0
        if invest_krw < 5_000:
            continue
        buy_fee = invest_krw * (fee / 2)
        qty = max((invest_krw - buy_fee) / next_open, 0.0)
        cash -= invest_krw
        positions[ticker] = {
            "entry_price": next_open,
            "entry_time": df.index[i + 1],
            "invest_krw": invest_krw,
            "qty": qty,
            "signal": reason.split("]", 1)[0].replace("[", ""),
            "stop_price": stop_price,
            "highest_price": next_open,
            "take_profit_armed": False,
            "partial_taken": False,
            "bb_break_seen": False,
            "overbought_seen": False,
        }

    if positions:
        final_ts = max(df.index[-1] for df in frames.values())
        for ticker, pos in list(positions.items()):
            if ticker in frames:
                last_close[ticker] = float(frames[ticker].iloc[-1]["close"])
            fill_price = last_close.get(ticker, pos["entry_price"])
            proceeds = pos["qty"] * fill_price
            sell_fee = proceeds * (fee / 2)
            cash += proceeds - sell_fee
            pnl_krw = proceeds - sell_fee - pos["invest_krw"]
            trades.append(
                {
                    "ticker": ticker,
                    "entry_time": pos["entry_time"],
                    "exit_time": final_ts,
                    "entry_price": pos["entry_price"],
                    "exit_price": fill_price,
                    "invest_krw": pos["invest_krw"],
                    "pnl": pnl_krw / pos["invest_krw"],
                    "pnl_krw": pnl_krw,
                    "reason": "EOD",
                    "signal": pos["signal"],
                }
            )
        equity_ts[final_ts] = cash

    return trades, pd.Series(equity_ts, name="equity", dtype=float).sort_index()


def build_hourly_universe_cache(
    frames: dict[str, pd.DataFrame],
    *,
    start_at: pd.Timestamp | None = None,
    universe_limit: int = UNIVERSE_TOP_N,
    universe_lookback: int = UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    min_quote_volume_krw: float = UNIVERSE_MIN_DAILY_QUOTE_KRW,
    noise_threshold: float = UNIVERSE_NOISE_THRESHOLD,
    atr_exclude_ratio: float = UNIVERSE_ATR_EXCLUDE_RATIO,
    beta_min: float = UNIVERSE_BETA_MIN,
    beta_top_n: int = UNIVERSE_BETA_TOP_N,
) -> tuple[dict[pd.Timestamp, set[str]], set[str]]:
    timestamps = sorted({idx for df in frames.values() for idx in df.index if start_at is None or idx >= start_at})
    hourly_cache: dict[pd.Timestamp, set[str]] = {}
    union_tickers: set[str] = {"KRW-BTC"}
    for ts in timestamps:
        selected = set(
            select_universe_from_frames(
                frames,
                pd.Timestamp(ts),
                limit=universe_limit,
                lookback=universe_lookback,
                min_quote_volume_krw=min_quote_volume_krw,
                noise_threshold=noise_threshold,
                atr_exclude_ratio=atr_exclude_ratio,
                beta_min=beta_min,
                beta_top_n=beta_top_n,
            )
        )
        hourly_cache[pd.Timestamp(ts)] = selected
        union_tickers.update(selected)
    return hourly_cache, union_tickers


def select_universe_from_frames(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    limit: int = UNIVERSE_TOP_N,
    lookback: int = UNIVERSE_DAILY_LOOKBACK_MINUTES,
    min_quote_volume_krw: float = 0.0,
    noise_threshold: float | None = None,
    atr_exclude_ratio: float = 0.0,
    beta_min: float = 0.0,
    beta_top_n: int | None = None,
) -> list[str]:
    if all(
        required in df.columns
        for df in frames.values()
        for required in ("quote_volume_24h", "surge_24h", "rsi_momentum_4", "noise_24h", "atr_pct_24h", "ret_24h")
    ):
        return _select_universe_from_precomputed_rows(
            frames,
            ts,
            limit=limit,
            min_quote_volume_krw=min_quote_volume_krw,
            noise_threshold=noise_threshold,
            atr_exclude_ratio=atr_exclude_ratio,
            beta_min=beta_min,
            beta_top_n=beta_top_n,
        )

    ranked: list[tuple[str, float]] = []
    for ticker, df in frames.items():
        if ticker in EXCLUDED_TICKERS:
            continue
        window = df.loc[df.index <= ts].tail(lookback)
        volume = quote_volume_krw(window)
        if volume >= min_quote_volume_krw:
            ranked.append((ticker, volume))
    ranked.sort(key=lambda item: item[1], reverse=True)
    top30 = [ticker for ticker, _ in ranked[:30]]
    top_volume_ranked = ranked[:30]
    surge_ranked = _volume_surge_from_frames(frames, ts, candidates=top30)
    rsi_momentum_ranked = _rsi_momentum_from_frames(frames, ts, candidates=top30)
    mixed = blend_universe_candidates(top_volume_ranked, surge_ranked, rsi_momentum_ranked, limit=30, rsi_bonus_count=5)
    noise_filtered = mixed
    if noise_threshold is not None:
        noise_map = _noise_ratio_map_from_frames(frames, ts, mixed)
        noise_filtered = filter_by_noise_threshold(mixed, noise_map, threshold=noise_threshold)
    atr_filtered = noise_filtered
    if atr_exclude_ratio > 0:
        atr_filtered = exclude_high_atr_candidates(
            noise_filtered,
            _atr_pct_map_from_frames(frames, ts, noise_filtered),
            exclude_ratio=atr_exclude_ratio,
        )
    beta_top = prioritize_by_beta(
        atr_filtered,
        _beta_map_from_frames(frames, ts, atr_filtered),
        topn=beta_top_n or limit,
        min_beta=beta_min,
    )
    if beta_top:
        return beta_top[:limit]
    return atr_filtered[:limit]


def _select_universe_from_precomputed_rows(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    limit: int,
    min_quote_volume_krw: float,
    noise_threshold: float | None,
    atr_exclude_ratio: float,
    beta_min: float,
    beta_top_n: int | None,
) -> list[str]:
    top_volume_ranked: list[tuple[str, float]] = []
    surge_ranked: list[tuple[str, float]] = []
    rsi_ranked: list[tuple[str, float]] = []
    noise_map: dict[str, float] = {}
    atr_pct_map: dict[str, float] = {}
    ret_map: dict[str, float] = {}

    btc_ret = 0.0
    btc_row = _row_at_or_before(frames.get("KRW-BTC"), ts)
    if btc_row is not None and pd.notna(btc_row.get("ret_24h")):
        btc_ret = float(btc_row["ret_24h"])

    for ticker, df in frames.items():
        if ticker in EXCLUDED_TICKERS:
            continue
        row = _row_at_or_before(df, ts)
        if row is None:
            continue
        quote_volume = row.get("quote_volume_24h")
        if pd.notna(quote_volume) and float(quote_volume) >= min_quote_volume_krw:
            top_volume_ranked.append((ticker, float(quote_volume)))
        if pd.notna(row.get("surge_24h")):
            surge_ranked.append((ticker, float(row["surge_24h"])))
        if pd.notna(row.get("rsi_momentum_4")):
            rsi_ranked.append((ticker, float(row["rsi_momentum_4"])))
        if pd.notna(row.get("noise_24h")):
            noise_map[ticker] = float(row["noise_24h"])
        if pd.notna(row.get("atr_pct_24h")):
            atr_pct_map[ticker] = float(row["atr_pct_24h"])
        if pd.notna(row.get("ret_24h")):
            ret = float(row["ret_24h"])
            ret_map[ticker] = ret / btc_ret if abs(btc_ret) > 1e-9 else 0.0

    top_volume_ranked.sort(key=lambda item: item[1], reverse=True)
    surge_ranked.sort(key=lambda item: item[1], reverse=True)
    rsi_ranked.sort(key=lambda item: item[1], reverse=True)

    mixed = blend_universe_candidates(top_volume_ranked[:30], surge_ranked, rsi_ranked, limit=30, rsi_bonus_count=5)
    if noise_threshold is not None:
        mixed = filter_by_noise_threshold(mixed, noise_map, threshold=noise_threshold)
    if atr_exclude_ratio > 0:
        mixed = exclude_high_atr_candidates(mixed, atr_pct_map, exclude_ratio=atr_exclude_ratio)
    beta_top = prioritize_by_beta(mixed, ret_map, min_beta=beta_min, topn=beta_top_n or limit)
    if beta_top:
        return beta_top[:limit]
    return mixed[:limit]


def select_daily_universe_from_frames(
    frames: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    *,
    limit: int = UNIVERSE_TOP_N,
    min_quote_volume_krw: float = 0.0,
    noise_threshold: float | None = None,
    atr_exclude_ratio: float = 0.0,
    beta_min: float = 0.0,
    beta_top_n: int | None = None,
) -> list[str]:
    """Select TOP N by previous completed day's quote volume to avoid lookahead."""
    current_day = pd.Timestamp(day).normalize()
    previous_day = current_day - pd.Timedelta(days=1)
    ranked: list[tuple[str, float]] = []
    for ticker, df in frames.items():
        if ticker in EXCLUDED_TICKERS:
            continue
        window = df.loc[(df.index >= previous_day) & (df.index < current_day)]
        volume = quote_volume_krw(window)
        if volume >= min_quote_volume_krw:
            ranked.append((ticker, volume))
    ranked.sort(key=lambda item: item[1], reverse=True)
    top30 = [ticker for ticker, _ in ranked[:30]]
    top_volume_ranked = ranked[:30]
    surge_ranked = _volume_surge_from_day(frames, current_day, candidates=top30)
    rsi_momentum_ranked = _rsi_momentum_from_day(frames, current_day, candidates=top30)
    mixed = blend_universe_candidates(top_volume_ranked, surge_ranked, rsi_momentum_ranked, limit=30, rsi_bonus_count=5)
    noise_filtered = mixed
    if noise_threshold is not None:
        noise_map = _noise_ratio_map_from_day(frames, current_day, mixed)
        noise_filtered = filter_by_noise_threshold(mixed, noise_map, threshold=noise_threshold)
    atr_filtered = noise_filtered
    if atr_exclude_ratio > 0:
        atr_filtered = exclude_high_atr_candidates(
            noise_filtered,
            _atr_pct_map_from_day(frames, current_day, noise_filtered),
            exclude_ratio=atr_exclude_ratio,
        )
    beta_top = prioritize_by_beta(
        atr_filtered,
        _beta_map_from_day(frames, current_day, atr_filtered),
        topn=beta_top_n or limit,
        min_beta=beta_min,
    )
    if beta_top:
        return beta_top[:limit]
    return atr_filtered[:limit]


def count_daily_universe_membership(
    frames: dict[str, pd.DataFrame],
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    *,
    limit: int = UNIVERSE_TOP_N,
) -> dict[str, int]:
    counts = {ticker: 0 for ticker in frames}
    for day in pd.date_range(pd.Timestamp(start_at).normalize(), pd.Timestamp(end_at).normalize(), freq="D"):
        for ticker in select_daily_universe_from_frames(frames, day, limit=limit):
            counts[ticker] = counts.get(ticker, 0) + 1
    return {ticker: count for ticker, count in counts.items() if count > 0}


def _market_row(frames: dict[str, pd.DataFrame], ts: pd.Timestamp) -> pd.Series | None:
    btc = frames.get("KRW-BTC")
    return _row_at_or_before(btc, ts)


def _row_at_or_before(df: pd.DataFrame | None, ts: pd.Timestamp) -> pd.Series | None:
    if df is None or df.empty:
        return None
    pos = df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[pos]


def _volume_surge_from_frames(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    candidates: list[str],
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    end = pd.Timestamp(ts)
    split = end - pd.Timedelta(hours=24)
    start = end - pd.Timedelta(hours=48)
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        prev_window = df.loc[(df.index > start) & (df.index <= split)]
        curr_window = df.loc[(df.index > split) & (df.index <= end)]
        prev_vol = quote_volume_krw(prev_window)
        curr_vol = quote_volume_krw(curr_window)
        if prev_vol <= 0:
            continue
        scored.append((ticker, (curr_vol - prev_vol) / prev_vol))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _volume_surge_from_day(
    frames: dict[str, pd.DataFrame],
    current_day: pd.Timestamp,
    *,
    candidates: list[str],
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    prev_day = current_day - pd.Timedelta(days=1)
    prev2_day = current_day - pd.Timedelta(days=2)
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        prev2_window = df.loc[(df.index >= prev2_day) & (df.index < prev_day)]
        prev_window = df.loc[(df.index >= prev_day) & (df.index < current_day)]
        prev2_vol = quote_volume_krw(prev2_window)
        prev_vol = quote_volume_krw(prev_window)
        if prev2_vol <= 0:
            continue
        scored.append((ticker, (prev_vol - prev2_vol) / prev2_vol))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _beta_map_from_frames(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    candidates: list[str],
) -> dict[str, float]:
    btc = frames.get("KRW-BTC")
    if btc is None:
        return {}
    btc_window = btc.loc[btc.index <= ts].tail(25)
    if len(btc_window) < 25:
        return {}
    btc_ret = float(btc_window["close"].iloc[-1] / btc_window["close"].iloc[0] - 1)
    beta_map: dict[str, float] = {}
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[df.index <= ts].tail(25)
        if len(window) < 25:
            continue
        ret = float(window["close"].iloc[-1] / window["close"].iloc[0] - 1)
        beta_map[ticker] = ret / btc_ret if abs(btc_ret) > 1e-9 else 0.0
    return beta_map


def _beta_map_from_day(
    frames: dict[str, pd.DataFrame],
    current_day: pd.Timestamp,
    candidates: list[str],
) -> dict[str, float]:
    btc = frames.get("KRW-BTC")
    if btc is None:
        return {}
    prev_day = current_day - pd.Timedelta(days=1)
    btc_window = btc.loc[(btc.index >= prev_day) & (btc.index < current_day)]
    if len(btc_window) < 2:
        return {}
    btc_ret = float(btc_window["close"].iloc[-1] / btc_window["close"].iloc[0] - 1)
    beta_map: dict[str, float] = {}
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[(df.index >= prev_day) & (df.index < current_day)]
        if len(window) < 2:
            continue
        ret = float(window["close"].iloc[-1] / window["close"].iloc[0] - 1)
        beta_map[ticker] = ret / btc_ret if abs(btc_ret) > 1e-9 else 0.0
    return beta_map


def _rsi_momentum_from_frames(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    candidates: list[str],
    topn: int = 5,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[df.index <= ts]
        if len(window) < 4 or "rsi" not in window:
            continue
        now_rsi = float(window.iloc[-1]["rsi"])
        prev_rsi = float(window.iloc[-4]["rsi"])
        if pd.notna(now_rsi) and pd.notna(prev_rsi):
            scored.append((ticker, now_rsi - prev_rsi))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:topn]


def _rsi_momentum_from_day(
    frames: dict[str, pd.DataFrame],
    current_day: pd.Timestamp,
    *,
    candidates: list[str],
    topn: int = 5,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    prev_day = current_day - pd.Timedelta(days=1)
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[(df.index >= prev_day) & (df.index < current_day)]
        if len(window) < 4 or "rsi" not in window:
            continue
        now_rsi = float(window.iloc[-1]["rsi"])
        prev_rsi = float(window.iloc[-4]["rsi"])
        if pd.notna(now_rsi) and pd.notna(prev_rsi):
            scored.append((ticker, now_rsi - prev_rsi))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:topn]


def _atr_pct_map_from_frames(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    candidates: list[str],
) -> dict[str, float]:
    atr_pct_map: dict[str, float] = {}
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[df.index <= ts].tail(24)
        if window.empty or "atr_pct" not in window:
            continue
        atr_pct = float(window["atr_pct"].mean())
        if pd.notna(atr_pct) and atr_pct > 0:
            atr_pct_map[ticker] = atr_pct
    return atr_pct_map


def _atr_pct_map_from_day(
    frames: dict[str, pd.DataFrame],
    current_day: pd.Timestamp,
    candidates: list[str],
) -> dict[str, float]:
    atr_pct_map: dict[str, float] = {}
    prev_day = current_day - pd.Timedelta(days=1)
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[(df.index >= prev_day) & (df.index < current_day)]
        if window.empty or "atr_pct" not in window:
            continue
        atr_pct = float(window["atr_pct"].mean())
        if pd.notna(atr_pct) and atr_pct > 0:
            atr_pct_map[ticker] = atr_pct
    return atr_pct_map


def _noise_ratio_map_from_frames(
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    candidates: list[str],
) -> dict[str, float]:
    noise_map: dict[str, float] = {}
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[df.index <= ts].tail(24)
        if window.empty:
            continue
        noise_map[ticker] = price_noise_pct(window)
    return noise_map


def _noise_ratio_map_from_day(
    frames: dict[str, pd.DataFrame],
    current_day: pd.Timestamp,
    candidates: list[str],
) -> dict[str, float]:
    noise_map: dict[str, float] = {}
    prev_day = current_day - pd.Timedelta(days=1)
    for ticker in candidates:
        df = frames.get(ticker)
        if df is None:
            continue
        window = df.loc[(df.index >= prev_day) & (df.index < current_day)]
        if window.empty:
            continue
        noise_map[ticker] = price_noise_pct(window)
    return noise_map


def metrics(trades: list[dict], equity: pd.Series, df: pd.DataFrame | None = None) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
            "mdd": 0.0,
            "return": 0.0,
            "max_consecutive_losses": 0,
        }
    tdf = pd.DataFrame(trades)
    pnls = tdf["pnl"]
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    gross_win = winners.sum() if len(winners) else 0.0
    gross_loss = abs(losers.sum()) if len(losers) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min() if len(equity) else 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) and equity.iloc[0] else 0.0
    losses = (pnls <= 0).astype(int)
    streak = losses.groupby((losses != losses.shift()).cumsum()).sum().max()
    return {
        "trades": len(trades),
        "win_rate": len(winners) / len(trades),
        "avg_pnl": pnls.mean(),
        "profit_factor": pf,
        "mdd": mdd,
        "return": total_ret,
        "max_consecutive_losses": int(streak),
    }


def monthly_returns(trades: list[dict]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    tdf = pd.DataFrame(trades)
    tdf["month"] = pd.to_datetime(tdf["exit_time"]).dt.to_period("M")
    return tdf.groupby("month")["pnl"].sum()


def print_ticker_report(ticker: str, trades: list[dict], equity: pd.Series, df: pd.DataFrame) -> None:
    m = metrics(trades, equity, df)
    bnh_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    print("\n" + "=" * 72)
    print(f"{ticker} - 1h trend pullback signal-engine backtest")
    print("=" * 72)
    print(f"period         : {df.index[0].date()} -> {df.index[-1].date()} ({len(df):,} candles)")
    print(f"trades         : {m['trades']}")
    print(f"return         : {m['return']*100:+.2f}%   BnH {bnh_ret*100:+.2f}%")
    print(f"MDD            : {m['mdd']*100:.2f}%")
    print(f"profit factor  : {m['profit_factor']:.2f}")
    print(f"win rate       : {m['win_rate']*100:.1f}%")
    print(f"avg pnl/trade  : {m['avg_pnl']*100:+.3f}%")
    print(f"max loss streak: {m['max_consecutive_losses']}")
    if trades:
        tdf = pd.DataFrame(trades)
        print("exit reasons   : " + ", ".join(f"{k}={v}" for k, v in tdf["reason"].value_counts().items()))


def print_portfolio_report(
    trades: list[dict],
    equity: pd.Series,
    *,
    daily_membership_counts: dict[str, int] | None = None,
) -> None:
    print("\n" + "=" * 72)
    print("Portfolio backtest - bot sizing/concurrency")
    print("=" * 72)
    m = metrics(trades, equity)
    final_krw = equity.iloc[-1] if len(equity) else INITIAL_KRW
    print(f"initial KRW     : {INITIAL_KRW:,.0f}")
    print(f"final KRW       : {final_krw:,.0f} ({(final_krw/INITIAL_KRW-1)*100:+.2f}%)")
    print(f"trades          : {m['trades']}")
    print(f"MDD             : {m['mdd']*100:.2f}%")
    print(f"profit factor   : {m['profit_factor']:.2f}")
    print(f"win rate        : {m['win_rate']*100:.1f}%")
    print(f"avg pnl/trade   : {m['avg_pnl']*100:+.3f}%")
    if trades:
        tdf = pd.DataFrame(trades)
        print("by ticker       :")
        for ticker, group in tdf.groupby("ticker"):
            print(
                f"  {ticker:<8} trades={len(group):4d} "
                f"avg={group['pnl'].mean()*100:+.3f}% pnl={group['pnl_krw'].sum():+,.0f} KRW"
            )
        mr = monthly_returns(trades)
        if not mr.empty:
            print("monthly pnl sum : " + ", ".join(f"{idx}={val*100:+.2f}%" for idx, val in mr.items()))
        print("exit reasons    : " + ", ".join(f"{k}={v}" for k, v in tdf["reason"].value_counts().items()))
        print("entry signals   : " + ", ".join(f"{k}={v}" for k, v in tdf["signal"].value_counts().items()))
    if daily_membership_counts:
        ranked = sorted(daily_membership_counts.items(), key=lambda item: item[1], reverse=True)
        print("daily TOP10 days: " + ", ".join(f"{ticker}={count}" for ticker, count in ranked[:20]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Use only the last N candles per ticker for a fast smoke run.")
    parser.add_argument("--days", type=int, default=0, help="Evaluate only the most recent N days after indicator warmup.")
    parser.add_argument(
        "--universe-mode",
        choices=["hourly", "daily"],
        default="daily",
        help="Dynamic universe ranking cadence. daily uses previous completed day quote-volume TOP N.",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh OHLCV cache from Upbit before running.")
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated KRW tickers to backtest. Defaults to every Upbit KRW market.",
    )
    parser.add_argument(
        "--static-default",
        action="store_true",
        help="Use the legacy 5 fixed tickers instead of dynamic KRW-market discovery.",
    )
    args = parser.parse_args()
    activate_final_tuned_profile()

    frames: dict[str, pd.DataFrame] = {}
    configs: dict[str, StrategyConfig] = {}
    skipped: list[str] = []

    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    elif args.static_default:
        tickers = DEFAULT_TICKERS
    else:
        tickers = get_krw_tickers()

    for ticker in tickers:
        df = load_indicator_frame(ticker, refresh_cache=args.refresh_cache)
        if df.empty or len(df) < 1_000:
            print(f"[{ticker}] insufficient data - skipped")
            skipped.append(ticker)
            continue
        if args.sample:
            df = df.tail(args.sample)
        frames[ticker] = df
        configs[ticker] = get_strategy_config(ticker)

        start_at = df.index[-1] - pd.Timedelta(days=args.days) if args.days else None
        trades, equity = simulate_ticker(df, ticker=ticker, config=configs[ticker], start_at=start_at)
        print_ticker_report(ticker, trades, equity, df)

    if frames:
        final_ts = max(df.index[-1] for df in frames.values())
        start_at = final_ts - pd.Timedelta(days=args.days) if args.days else None
        event_tickers = None
        if start_at is not None:
            event_tickers = build_signal_ticker_cache(
                frames,
                configs,
                start_at=start_at,
                end_at=final_ts,
                universe_mode=args.universe_mode,
                universe_limit=UNIVERSE_TOP_N,
            )
        trades, equity = simulate_portfolio(
            frames,
            configs,
            dynamic_universe=True,
            universe_mode=args.universe_mode,
            universe_limit=UNIVERSE_TOP_N,
            start_at=start_at,
            event_tickers=event_tickers,
        )
        daily_membership_counts = (
            count_daily_universe_membership(frames, start_at or final_ts, final_ts)
            if args.universe_mode == "daily"
            else None
        )
        print_portfolio_report(
            trades,
            equity,
            daily_membership_counts=daily_membership_counts,
        )
        if skipped:
            print(f"skipped tickers  : {len(skipped)}")


if __name__ == "__main__":
    main()
