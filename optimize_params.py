#!/usr/bin/env python3
"""
Simple parameter search runner for 1-year portfolio backtest.
"""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import pandas as pd

import backtest
import strategy


def load_frames_from_cache() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(".").glob("data_cache_KRW_*_minute60.pkl")):
        ticker = path.stem.replace("data_cache_", "").replace("_minute60", "").replace("_", "-")
        with open(path, "rb") as f:
            raw = pickle.load(f)
        if raw is None or len(raw) < 3000:
            continue
        df = strategy.add_indicators(raw)
        df = df.dropna(subset=["ma20", "bb_lower", "rsi", "atr14", "vwap", "ma120d"])
        if len(df) < 3000:
            continue
        frames[ticker] = df
    return frames


def run_search() -> None:
    frames = load_frames_from_cache()
    if "KRW-BTC" not in frames:
        raise RuntimeError("KRW-BTC cache is required")

    end_ts = min(df.index.max() for df in frames.values())
    start_ts = end_ts - pd.Timedelta(days=365)
    configs = {ticker: strategy.get_strategy_config(ticker) for ticker in frames}

    base = {
        "per_trade_ratio": backtest.PER_TRADE_RATIO,
        "per_trade_ratio_spike": backtest.PER_TRADE_RATIO_SPIKE,
        "max_concurrent": backtest.MAX_CONCURRENT,
        "beta_min": backtest.UNIVERSE_BETA_MIN,
        "beta_top_n": backtest.UNIVERSE_BETA_TOP_N,
        "secondary_rsi": strategy.SECONDARY_RSI_LEVEL,
        "volume_multiplier": strategy.VOLUME_PULLBACK_MULTIPLIER,
        "market_rsi_min": strategy.MARKET_RSI_MIN,
    }

    grid = {
        "per_trade_ratio": [0.9, 1.0],
        "max_concurrent": [1],
        "beta_min": [1.0, 1.2],
        "beta_top_n": [8, 10],
        "secondary_rsi": [42, 45],
        "volume_multiplier": [0.7, 0.85],
        "market_rsi_min": [40, 45],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    results: list[tuple[float, dict[str, float]]] = []

    for idx, combo in enumerate(combos, start=1):
        p = dict(zip(keys, combo))

        backtest.PER_TRADE_RATIO = float(p["per_trade_ratio"])
        backtest.PER_TRADE_RATIO_SPIKE = float(p["per_trade_ratio"])
        backtest.MAX_CONCURRENT = int(p["max_concurrent"])
        backtest.UNIVERSE_BETA_MIN = float(p["beta_min"])
        backtest.UNIVERSE_BETA_TOP_N = int(p["beta_top_n"])

        strategy.SECONDARY_RSI_LEVEL = int(p["secondary_rsi"])
        strategy.VOLUME_PULLBACK_MULTIPLIER = float(p["volume_multiplier"])
        strategy.MARKET_RSI_MIN = int(p["market_rsi_min"])

        trades, equity = backtest.simulate_portfolio(
            frames,
            configs,
            dynamic_universe=True,
            universe_mode="daily",
            universe_limit=backtest.UNIVERSE_TOP_N,
            start_at=start_ts,
        )
        if equity.empty:
            continue
        ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
        results.append((ret, p))
        print(f"[{idx}/{len(combos)}] return={ret*100:.2f}% params={p}")

    for k, v in base.items():
        if k == "per_trade_ratio":
            backtest.PER_TRADE_RATIO = float(v)
            backtest.PER_TRADE_RATIO_SPIKE = float(v)
        elif k == "max_concurrent":
            backtest.MAX_CONCURRENT = int(v)
        elif k == "beta_min":
            backtest.UNIVERSE_BETA_MIN = float(v)
        elif k == "beta_top_n":
            backtest.UNIVERSE_BETA_TOP_N = int(v)
        elif k == "secondary_rsi":
            strategy.SECONDARY_RSI_LEVEL = int(v)
        elif k == "volume_multiplier":
            strategy.VOLUME_PULLBACK_MULTIPLIER = float(v)
        elif k == "market_rsi_min":
            strategy.MARKET_RSI_MIN = int(v)

    results.sort(key=lambda item: item[0], reverse=True)
    top = results[:20]
    print("Top parameter sets:")
    for ret, params in top:
        print(f"return={ret*100:.2f}% params={params}")


if __name__ == "__main__":
    run_search()
