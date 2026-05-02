#!/usr/bin/env python3
"""
Daily multi-asset momentum strategy with volatility targeting.

Strategy:
- Universe: KRW-BTC, KRW-XRP, KRW-SOL, KRW-ADA
- Daily rebalance at the Upbit daily candle boundary, 09:00 KST
- Eligible when close > MA20 and RSI14 > 50
- Target weight per asset = min(25%, target_vol / asset_vol20 / asset_count)
- Failed filters remain in cash
- Exit on filter failure or a -7% stop from the latest entry price
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit


TICKERS = ["KRW-BTC", "KRW-XRP", "KRW-SOL", "KRW-ADA"]
INITIAL_KRW = 1_000_000.0
TARGET_VOL = 0.02
MA_PERIOD = 20
RSI_PERIOD = 14
VOL_PERIOD = 20
FEE_RATE = 0.0005
SLIPPAGE = 0.001
STOP_LOSS = 0.07


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    cumulative_returns: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    asset_mdd: pd.Series
    portfolio_mdd: float
    final_krw: float
    total_return: float


def _minute_cache_path(ticker: str) -> Path:
    return Path(f"data_cache_{ticker.replace('-', '_')}.pkl")


def fetch_daily_ohlcv(ticker: str, count: int = 365, refresh: bool = False) -> pd.DataFrame:
    """Fetch daily OHLCV. Falls back to the repository's 1-minute cache."""
    cache_path = Path(f"daily_cache_{ticker.replace('-', '_')}.pkl")
    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = None
    if refresh:
        for attempt in range(4):
            try:
                df = pyupbit.get_ohlcv(ticker, interval="day", count=count)
                break
            except Exception:
                time.sleep(2**attempt)

    if df is None or df.empty:
        df = load_daily_from_minute_cache(ticker)

    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: daily data unavailable")

    df = df.sort_index().tail(count).copy()
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def load_daily_from_minute_cache(ticker: str) -> pd.DataFrame:
    """Build 09:00 KST daily candles from the existing minute cache."""
    cache = _minute_cache_path(ticker)
    if not cache.exists():
        return pd.DataFrame()

    with open(cache, "rb") as f:
        minute_df = pickle.load(f)

    df = minute_df.sort_index().copy()
    daily = pd.DataFrame(
        {
            "open": df["open"].resample("1D", offset="9h").first(),
            "high": df["high"].resample("1D", offset="9h").max(),
            "low": df["low"].resample("1D", offset="9h").min(),
            "close": df["close"].resample("1D", offset="9h").last(),
            "volume": df["volume"].resample("1D", offset="9h").sum(),
        }
    )
    return daily.dropna()


def fetch_all_daily(
    tickers: list[str],
    *,
    count: int = 365,
    refresh: bool = False,
    max_workers: int = 4,
) -> dict[str, pd.DataFrame]:
    """Fetch all assets in parallel and return ticker keyed daily frames."""
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_daily_ohlcv, ticker, count, refresh): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            frames[ticker] = future.result()
    return frames


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = out["close"].rolling(MA_PERIOD).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    out["return"] = out["close"].pct_change()
    out["vol20"] = out["return"].rolling(VOL_PERIOD).std()
    out["eligible"] = (out["close"] > out["ma20"]) & (out["rsi14"] > 50)
    return out


def merge_close_panel(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    panel = pd.concat(
        {ticker: frame[column] for ticker, frame in frames.items()},
        axis=1,
    )
    panel.columns = panel.columns.get_level_values(0)
    return panel.dropna(how="all").sort_index()


def generate_target_weights(
    frames: dict[str, pd.DataFrame],
    *,
    target_vol: float = TARGET_VOL,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    tickers = tickers or list(frames)
    base_weight = 1.0 / len(tickers)
    vol = merge_close_panel(frames, "vol20").reindex(columns=tickers)
    eligible = merge_close_panel(frames, "eligible").reindex(columns=tickers).fillna(False)
    raw = (target_vol / vol.replace(0, np.nan)) / len(tickers)
    capped = raw.clip(lower=0.0, upper=base_weight)
    return capped.where(eligible, 0.0).fillna(0.0)


def run_backtest(
    frames: dict[str, pd.DataFrame],
    *,
    tickers: list[str] | None = None,
    initial_krw: float = INITIAL_KRW,
    target_vol: float = TARGET_VOL,
    fee_rate: float = FEE_RATE,
    slippage: float = SLIPPAGE,
    stop_loss: float = STOP_LOSS,
) -> BacktestResult:
    tickers = tickers or list(frames)
    frames = {ticker: add_daily_indicators(frames[ticker]) for ticker in tickers}

    opens = merge_close_panel(frames, "open").reindex(columns=tickers)
    highs = merge_close_panel(frames, "high").reindex(columns=tickers)
    lows = merge_close_panel(frames, "low").reindex(columns=tickers)
    closes = merge_close_panel(frames, "close").reindex(columns=tickers)
    target_weights = generate_target_weights(frames, target_vol=target_vol, tickers=tickers)

    index = opens.index.intersection(closes.index).intersection(target_weights.index)
    opens = opens.loc[index].ffill()
    highs = highs.loc[index].ffill()
    lows = lows.loc[index].ffill()
    closes = closes.loc[index].ffill()
    target_weights = target_weights.loc[index].fillna(0.0)

    cash = initial_krw
    qty = pd.Series(0.0, index=tickers)
    entry_price = pd.Series(np.nan, index=tickers)
    equity_points: dict[pd.Timestamp, float] = {}
    weight_points: dict[pd.Timestamp, pd.Series] = {}
    trades: list[dict] = []

    for i in range(max(MA_PERIOD, RSI_PERIOD, VOL_PERIOD) + 1, len(index)):
        ts = index[i]
        px_open = opens.loc[ts]
        px_low = lows.loc[ts]
        px_close = closes.loc[ts]

        equity = cash + float((qty * px_open).sum())

        for ticker in tickers:
            if qty[ticker] <= 0 or not np.isfinite(entry_price[ticker]):
                continue
            stop_price = entry_price[ticker] * (1 - stop_loss)
            if px_low[ticker] <= stop_price:
                fill = stop_price * (1 - slippage)
                proceeds = qty[ticker] * fill * (1 - fee_rate)
                pnl = proceeds - qty[ticker] * entry_price[ticker]
                cash += proceeds
                trades.append(
                    {
                        "date": ts,
                        "ticker": ticker,
                        "side": "SELL",
                        "reason": "STOP_LOSS",
                        "price": fill,
                        "pnl_krw": pnl,
                    }
                )
                qty[ticker] = 0.0
                entry_price[ticker] = np.nan

        equity = cash + float((qty * px_open).sum())
        desired_value = equity * target_weights.loc[ts]
        current_value = qty * px_open
        trade_value = desired_value - current_value

        for ticker in tickers:
            value = float(trade_value[ticker])
            if abs(value) < 5_000:
                continue

            if value > 0:
                fill = px_open[ticker] * (1 + slippage)
                gross = min(value, cash)
                if gross < 5_000:
                    continue
                bought_qty = gross * (1 - fee_rate) / fill
                previous_value = qty[ticker] * entry_price[ticker] if np.isfinite(entry_price[ticker]) else 0.0
                qty[ticker] += bought_qty
                cash -= gross
                entry_price[ticker] = (
                    (previous_value + bought_qty * fill) / qty[ticker]
                    if qty[ticker] > 0
                    else np.nan
                )
                trades.append(
                    {
                        "date": ts,
                        "ticker": ticker,
                        "side": "BUY",
                        "reason": "REBALANCE",
                        "price": fill,
                        "pnl_krw": 0.0,
                    }
                )
            else:
                sell_qty = min(qty[ticker], abs(value) / px_open[ticker])
                if sell_qty <= 0:
                    continue
                fill = px_open[ticker] * (1 - slippage)
                proceeds = sell_qty * fill * (1 - fee_rate)
                pnl = proceeds - sell_qty * entry_price[ticker] if np.isfinite(entry_price[ticker]) else 0.0
                cash += proceeds
                qty[ticker] -= sell_qty
                if qty[ticker] <= 1e-12:
                    qty[ticker] = 0.0
                    entry_price[ticker] = np.nan
                trades.append(
                    {
                        "date": ts,
                        "ticker": ticker,
                        "side": "SELL",
                        "reason": "REBALANCE",
                        "price": fill,
                        "pnl_krw": pnl,
                    }
                )

        equity = cash + float((qty * px_close).sum())
        equity_points[ts] = equity
        weight_points[ts] = (qty * px_close / equity).reindex(tickers).fillna(0.0)

    equity = pd.Series(equity_points, name="portfolio_equity")
    weights = pd.DataFrame(weight_points).T.reindex(columns=tickers).fillna(0.0)

    asset_cum = closes.loc[equity.index, tickers].div(closes.loc[equity.index, tickers].iloc[0]) - 1
    cumulative_returns = asset_cum.copy()
    cumulative_returns["Portfolio"] = equity / initial_krw - 1

    asset_mdd = asset_cum.add(1).apply(max_drawdown)
    portfolio_mdd = max_drawdown(equity / initial_krw)

    return BacktestResult(
        equity=equity,
        cumulative_returns=cumulative_returns,
        weights=weights,
        trades=pd.DataFrame(trades),
        asset_mdd=asset_mdd,
        portfolio_mdd=portfolio_mdd,
        final_krw=float(equity.iloc[-1]),
        total_return=float(equity.iloc[-1] / initial_krw - 1),
    )


def max_drawdown(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    return float((clean / peak - 1).min())


def plot_cumulative_returns(result: BacktestResult, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError("matplotlib is required for visualization") from exc

    fig, ax = plt.subplots(figsize=(12, 7))
    for column in result.cumulative_returns.columns:
        width = 2.8 if column == "Portfolio" else 1.2
        alpha = 1.0 if column == "Portfolio" else 0.75
        ax.plot(
            result.cumulative_returns.index,
            result.cumulative_returns[column] * 100,
            label=column,
            linewidth=width,
            alpha=alpha,
        )

    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_title("Multi-Asset Momentum with Vol-Targeting")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def print_report(result: BacktestResult, initial_krw: float) -> None:
    print("\n" + "=" * 72)
    print("Multi-Asset Momentum with Vol-Targeting")
    print("=" * 72)
    print(f"period        : {result.equity.index[0].date()} -> {result.equity.index[-1].date()}")
    print(f"initial KRW   : {initial_krw:,.0f}")
    print(f"final KRW     : {result.final_krw:,.0f} ({result.total_return*100:+.2f}%)")
    print(f"portfolio MDD : {result.portfolio_mdd*100:.2f}%")
    print(f"trades        : {len(result.trades)}")
    print("asset MDD     :")
    for ticker, mdd in result.asset_mdd.items():
        print(f"  {ticker:<8} {mdd*100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-cache", action="store_true", help="Fetch daily candles from Upbit.")
    parser.add_argument("--count", type=int, default=365, help="Daily candle count.")
    parser.add_argument("--target-vol", type=float, default=TARGET_VOL, help="Daily target volatility.")
    parser.add_argument("--output", default="multi_asset_vol_target_returns.png", help="Output chart path.")
    args = parser.parse_args()

    frames = fetch_all_daily(TICKERS, count=args.count, refresh=args.refresh_cache)
    result = run_backtest(frames, tickers=TICKERS, target_vol=args.target_vol)
    output = Path(args.output)
    plot_cumulative_returns(result, output)
    print_report(result, INITIAL_KRW)
    print(f"chart         : {output.resolve()}")


if __name__ == "__main__":
    main()
