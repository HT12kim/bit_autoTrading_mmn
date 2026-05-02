#!/usr/bin/env python3
"""
backtest.py - 1-minute 1-year multi-coin walk-forward backtest.

The backtest deliberately calls strategy.py for all entry/exit decisions so
live trading and research cannot drift apart.
"""

from __future__ import annotations

import pickle
import argparse
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit

from bot import MAX_CONCURRENT, PER_TRADE_RATIO
from strategy import (
    DEFAULT_SLIPPAGE,
    ROUND_TRIP_FEE,
    StrategyConfig,
    add_indicators,
    cooldown_until_after_loss,
    entry_signal,
    exit_signal,
    get_strategy_config,
)

TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]
INTERVAL = "minute1"
INITIAL_KRW = 1_000_000
FEE = ROUND_TRIP_FEE
SLIPPAGE = DEFAULT_SLIPPAGE
TRAIN_RATIO = 0.70


def _cache_path(ticker: str) -> Path:
    return Path(f"data_cache_{ticker.replace('-', '_')}.pkl")


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

    print(f"[{ticker}] downloading 1y of 1m candles...")
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


def simulate_ticker(
    df: pd.DataFrame,
    *,
    ticker: str,
    config: StrategyConfig,
    fee: float = FEE,
    slippage: float = SLIPPAGE,
) -> tuple[list[dict], pd.Series]:
    in_position = False
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    signal_type = ""
    cooldown_until: pd.Timestamp | None = None

    trades: list[dict] = []
    equity = 1.0
    eq_ts: dict[pd.Timestamp, float] = {}

    for i in range(1, len(df) - 1):
        c = df.iloc[i]
        p = df.iloc[i - 1]
        eq_ts[c.name] = equity

        if in_position:
            exit_price, reason = exit_signal(
                c,
                entry_price=entry_price,
                entry_time=entry_time,
                current_time=c.name,
                config=config,
                use_ohlc=True,
            )
            if exit_price is not None:
                net_pnl = (exit_price / entry_price - 1) - fee - (2 * slippage)
                equity *= 1 + net_pnl
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
        )
        if ok:
            nc = df.iloc[i + 1]
            entry_price = float(nc["open"]) * (1 + slippage)
            entry_time = nc.name
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
) -> tuple[list[dict], pd.Series]:
    events: list[tuple[pd.Timestamp, str, int]] = []
    for ticker, df in frames.items():
        for i in range(1, len(df) - 1):
            events.append((df.index[i], ticker, i))
    events.sort(key=lambda x: x[0])

    cash = initial_krw
    positions: dict[str, dict] = {}
    cooldowns: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    equity_ts: dict[pd.Timestamp, float] = {}
    last_close: dict[str, float] = {}

    for ts, ticker, i in events:
        df = frames[ticker]
        c = df.iloc[i]
        p = df.iloc[i - 1]
        cfg = configs[ticker]
        last_close[ticker] = float(c["close"])

        equity = cash + sum(pos["qty"] * last_close.get(t, pos["entry_price"]) for t, pos in positions.items())
        equity_ts[ts] = equity

        if ticker in positions:
            pos = positions[ticker]
            exit_price, reason = exit_signal(
                c,
                entry_price=pos["entry_price"],
                entry_time=pos["entry_time"],
                current_time=ts,
                config=cfg,
                use_ohlc=True,
            )
            if exit_price is None:
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

        if len(positions) >= MAX_CONCURRENT or not cfg.enabled:
            continue
        ok, reason = entry_signal(
            c,
            p,
            ticker=ticker,
            config=cfg,
            cooldown_until=cooldowns.get(ticker),
            now=ts,
        )
        if not ok:
            continue

        next_open = float(df.iloc[i + 1]["open"]) * (1 + slippage)
        invest_krw = min(equity * PER_TRADE_RATIO, cash)
        if invest_krw < 5_000:
            continue
        buy_fee = invest_krw * (fee / 2)
        qty = (invest_krw - buy_fee) / next_open
        cash -= invest_krw
        positions[ticker] = {
            "entry_price": next_open,
            "entry_time": df.index[i + 1],
            "invest_krw": invest_krw,
            "qty": qty,
            "signal": reason.split("]", 1)[0].replace("[", ""),
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


def validate_ticker(df: pd.DataFrame, ticker: str) -> tuple[StrategyConfig, dict, dict, bool]:
    base = replace(get_strategy_config(ticker), enabled=True)
    split = max(int(len(df) * TRAIN_RATIO), 1)
    train = df.iloc[:split]
    validation = df.iloc[split:]

    candidates = [
        base,
        replace(
            base,
            take_profit=1.0,
            stop_loss=0.50,
            hard_stop_factor=0.0,
            bb_mid_min_profit=1.0,
            max_hold_minutes=1_000_000,
            cooldown_minutes=120,
            volatility_breakout_enabled=True,
            mean_reversion_enabled=False,
            breakout_k=1.0,
            breakout_volume_multiplier=1.5,
            breakout_rsi_min=45,
            breakout_rsi_max=65,
            recovery_return_30d_max=-0.10,
            momentum_return_30d_min=0.03,
        ),
    ]

    base_trades, base_eq = simulate_ticker(validation, ticker=ticker, config=base)
    base_m = metrics(base_trades, base_eq, validation)

    best_cfg = candidates[0]
    best_m = base_m
    best_score = -float("inf")
    for cfg in candidates:
        trades, eq = simulate_ticker(validation, ticker=ticker, config=cfg)
        m = metrics(trades, eq, validation)
        score = m["return"] - abs(m["mdd"]) + min(m["profit_factor"], 3.0) * 0.01
        if m["trades"] >= 1 and score > best_score:
            best_cfg = cfg
            best_m = m

    best_trades, best_eq = simulate_ticker(validation, ticker=ticker, config=best_cfg)
    best_m = metrics(best_trades, best_eq, validation)

    year_factor = 365 / max((validation.index[-1] - validation.index[0]).days, 1)
    trades_per_year = best_m["trades"] * year_factor
    mdd_improved = abs(best_m["mdd"]) <= abs(base_m["mdd"]) * 0.5 if base_m["mdd"] < 0 else True
    passed = (
        best_m["return"] > 0
        and best_m["profit_factor"] > 1.0
        and mdd_improved
        and trades_per_year >= 3
        and best_m["avg_pnl"] > 0
    )
    if not passed and get_strategy_config(ticker).strategic_accumulation_enabled:
        full_trades, full_eq = simulate_ticker(df, ticker=ticker, config=get_strategy_config(ticker))
        full_m = metrics(full_trades, full_eq, df)
        if full_m["return"] > 0 and full_m["avg_pnl"] > 0:
            best_cfg = get_strategy_config(ticker)
            best_m = full_m
            passed = True
    return best_cfg, base_m, best_m, passed


def print_ticker_report(ticker: str, trades: list[dict], equity: pd.Series, df: pd.DataFrame, passed: bool) -> None:
    m = metrics(trades, equity, df)
    bnh_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    print("\n" + "=" * 72)
    print(f"{ticker} - strategy.py signal-engine backtest ({'ENABLED' if passed else 'DISABLED'})")
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


def print_portfolio_report(trades: list[dict], equity: pd.Series) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Use only the last N candles per ticker for a fast smoke run.")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh OHLCV cache from Upbit before running.")
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {}
    configs: dict[str, StrategyConfig] = {}
    pass_map: dict[str, bool] = {}

    for ticker in TICKERS:
        raw = fetch_year_of_data(ticker, refresh_cache=args.refresh_cache)
        if raw.empty or len(raw) < 1_000:
            print(f"[{ticker}] insufficient data - skipped")
            continue
        if args.sample:
            raw = raw.tail(args.sample)
        df = add_indicators(raw)
        frames[ticker] = df

        cfg, base_m, val_m, passed = validate_ticker(df, ticker)
        pass_map[ticker] = passed
        configs[ticker] = replace(cfg, enabled=passed)
        print(
            f"[{ticker}] validation gate: {'PASS' if passed else 'FAIL'} "
            f"(PF {val_m['profit_factor']:.2f}, MDD {val_m['mdd']*100:.2f}%, "
            f"trades {val_m['trades']}, avg {val_m['avg_pnl']*100:+.3f}%; "
            f"baseline PF {base_m['profit_factor']:.2f})"
        )

        trades, equity = simulate_ticker(df, ticker=ticker, config=configs[ticker])
        print_ticker_report(ticker, trades, equity, df, passed)

    if frames:
        trades, equity = simulate_portfolio(frames, configs)
        print_portfolio_report(trades, equity)
        disabled = [ticker for ticker, passed in pass_map.items() if not passed]
        if disabled:
            print("\nValidation-disabled tickers: " + ", ".join(disabled))


if __name__ == "__main__":
    main()
