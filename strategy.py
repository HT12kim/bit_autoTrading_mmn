"""
strategy.py - Long-only Mean Reversion Scalping V2 (1-min candles).

The module owns the signal engine used by both live trading and backtests.
Entries are evaluated on closed candles; backtests can call the candle-level
helpers directly while the bot uses the dataframe wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from typing import Any

import pandas as pd

# Indicator periods
EMA_FAST = 20
EMA_SLOW = 50
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
VOL_LOOKBACK = 3
VOL_LOOKBACK_V2 = 20
ATR_PERIOD = 14
SLOPE_LOOKBACK = 10
RETURN_LOOKBACK = 15
LONG_RETURN_LOOKBACK = 43_200  # 30 days on 1-minute candles
TREND_EMA_LONG = 720

# Legacy constants kept for callers/docs that import them.
RSI_PRIMARY = 30
RSI_AGGRESSIVE = 40
TAKE_PROFIT = 0.003
STOP_LOSS = 0.003
FEE_FILTER_MIN = 0.0035
ROUND_TRIP_FEE = 0.001
DEFAULT_SLIPPAGE = 0.0003


@dataclass(frozen=True)
class StrategyConfig:
    ticker: str | None = None
    enabled: bool = True
    rsi_primary: float = 30.0
    fee_min: float = FEE_FILTER_MIN
    bb_entry_factor: float = 1.0
    volume_multiplier: float = 1.1
    atr_pct_max: float = 0.03
    trend_slope_min: float = 0.0
    take_profit: float = TAKE_PROFIT
    stop_loss: float = STOP_LOSS
    hard_stop_factor: float = 0.998
    bb_mid_min_profit: float = ROUND_TRIP_FEE + DEFAULT_SLIPPAGE
    max_hold_minutes: int = 60
    cooldown_minutes: int = 30
    mean_reversion_enabled: bool = True
    strategic_accumulation_enabled: bool = False
    aggressive_enabled: bool = False
    rsi_aggressive: float = RSI_AGGRESSIVE
    volatility_breakout_enabled: bool = False
    breakout_k: float = 0.5
    breakout_rsi_min: float = 40.0
    breakout_rsi_max: float = 70.0
    breakout_volume_multiplier: float = 0.8
    recovery_return_30d_max: float = -0.10
    momentum_return_30d_min: float = 0.03


DEFAULT_CONFIG = StrategyConfig()

TICKER_CONFIGS: dict[str, StrategyConfig] = {
    ticker: replace(DEFAULT_CONFIG, ticker=ticker)
    for ticker in ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]
}
TICKER_CONFIGS["KRW-ETH"] = replace(
    DEFAULT_CONFIG,
    ticker="KRW-ETH",
    mean_reversion_enabled=False,
    volatility_breakout_enabled=False,
    strategic_accumulation_enabled=True,
    take_profit=1.0,
    stop_loss=0.50,
    hard_stop_factor=0.0,
    bb_mid_min_profit=1.0,
    max_hold_minutes=1_000_000,
)


def get_strategy_config(ticker: str | None = None) -> StrategyConfig:
    if ticker and ticker in TICKER_CONFIGS:
        return TICKER_CONFIGS[ticker]
    return replace(DEFAULT_CONFIG, ticker=ticker)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    df["bb_mid"] = df["close"].rolling(BB_PERIOD).mean()
    bb_std = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_width_ma20"] = df["bb_width"].rolling(VOL_LOOKBACK_V2).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    df["vol_ma3"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    df["vol_ma20"] = df["volume"].rolling(VOL_LOOKBACK_V2).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma20"]
    df["ema_slow_slope10"] = df["ema_slow"].pct_change(SLOPE_LOOKBACK)
    df["return_15m"] = df["close"].pct_change(RETURN_LOOKBACK)
    df["return_30d"] = df["close"].pct_change(LONG_RETURN_LOOKBACK)
    df["ema_trend"] = df["close"].ewm(span=TREND_EMA_LONG, adjust=False).mean()
    df["range_prev"] = (df["high"] - df["low"]).shift(1)
    df["breakout_target"] = df["open"] + DEFAULT_CONFIG.breakout_k * df["range_prev"]

    return df


def is_uptrend(c: pd.Series) -> bool:
    return bool(_finite(c, "ema_fast") and _finite(c, "ema_slow") and c["ema_fast"] > c["ema_slow"])


def fee_filter_ok(
    price: float,
    bb_mid: float,
    config: StrategyConfig | None = None,
) -> tuple[bool, float]:
    """Returns (passes, expected_return_to_bb_mid)."""
    cfg = config or DEFAULT_CONFIG
    if price <= 0 or not isfinite(price) or not isfinite(bb_mid):
        return False, 0.0
    expected = (bb_mid - price) / price
    return expected >= cfg.fee_min, expected


def entry_signal(
    current: pd.Series,
    previous: pd.Series | None = None,
    *,
    ticker: str | None = None,
    config: StrategyConfig | None = None,
    cooldown_until: datetime | pd.Timestamp | str | None = None,
    now: datetime | pd.Timestamp | None = None,
) -> tuple[bool, str]:
    """Evaluate one already-closed candle for a new long entry."""
    cfg = config or get_strategy_config(ticker)
    if not cfg.enabled:
        return False, "DISABLED [validation gate failed or ticker disabled]"

    if _in_cooldown(cooldown_until, now or _series_time(current)):
        return False, "HOLD [cooldown active after losing exit]"

    required = [
        "close",
        "bb_mid",
        "bb_lower",
        "rsi",
        "volume",
        "vol_ma20",
        "volume_ratio",
        "atr_pct",
        "ema_slow_slope10",
        "ema_fast",
        "ema_slow",
        "range_prev",
    ]
    if not all(_finite(current, col) for col in required):
        return False, "insufficient indicators"

    if cfg.volatility_breakout_enabled and previous is not None:
        breakout_required = ["open", "bb_mid", "bb_width", "bb_width_ma20", "return_30d", "ema_trend"]
        if all(_finite(current, col) for col in breakout_required) and _finite(previous, "close"):
            target = float(current["open"]) + cfg.breakout_k * float(current["range_prev"])
            crossed_target = bool(previous["close"] <= target and current["close"] > target)
            recovered_bb_mid = bool(current["close"] >= current["bb_mid"])
            recovered_from_drawdown = bool(current["return_30d"] <= cfg.recovery_return_30d_max)
            momentum_continuation = bool(current["return_30d"] >= cfg.momentum_return_30d_min)
            rsi_recovery = bool(cfg.breakout_rsi_min <= current["rsi"] <= cfg.breakout_rsi_max)
            volume_recovery = bool(current["volume_ratio"] >= cfg.breakout_volume_multiplier)
            trend_reclaim = bool(current["close"] >= current["ema_trend"])
            volatility_expanding = bool(current["bb_width"] >= current["bb_width_ma20"])
            if (
                crossed_target
                and recovered_bb_mid
                and (recovered_from_drawdown or momentum_continuation)
                and rsi_recovery
                and volume_recovery
                and trend_reclaim
                and volatility_expanding
            ):
                signal = "VOL_BREAKOUT" if momentum_continuation else "RECOVERY_BREAKOUT"
                return True, (
                    f"[{signal}] close={current['close']:,.0f} target={target:,.0f} "
                    f"RSI={current['rsi']:.1f} vol={current['volume_ratio']:.2f}x "
                    f"ret30d={current['return_30d']*100:.1f}%"
                )

    if cfg.strategic_accumulation_enabled:
        strategic_required = ["bb_mid", "rsi", "volume_ratio", "ema_fast", "ema_slow"]
        if all(_finite(current, col) for col in strategic_required):
            if (
                current["close"] >= current["bb_mid"]
                and current["ema_fast"] >= current["ema_slow"]
                and 40 <= current["rsi"] <= 70
                and current["volume_ratio"] >= 0.8
            ):
                return True, (
                    f"[STRATEGIC_ACCUMULATION] close={current['close']:,.0f} "
                    f"RSI={current['rsi']:.1f} vol={current['volume_ratio']:.2f}x"
                )

    if not is_uptrend(current):
        return False, (
            f"HOLD [downtrend EMA20={current['ema_fast']:,.0f} "
            f"<= EMA50={current['ema_slow']:,.0f}]"
        )

    if current["ema_slow_slope10"] < cfg.trend_slope_min:
        return False, (
            f"HOLD [trend slope {current['ema_slow_slope10']*100:.3f}% "
            f"< {cfg.trend_slope_min*100:.3f}%]"
        )

    if current["atr_pct"] > cfg.atr_pct_max:
        return False, (
            f"HOLD [ATR {current['atr_pct']*100:.2f}% > {cfg.atr_pct_max*100:.2f}%]"
        )

    ff_ok, expected_ret = fee_filter_ok(float(current["close"]), float(current["bb_mid"]), cfg)
    if not ff_ok:
        return False, (
            f"HOLD [fee filter expected={expected_ret*100:.3f}% "
            f"< {cfg.fee_min*100:.2f}%]"
        )

    if not cfg.mean_reversion_enabled:
        return False, "HOLD [mean reversion disabled; no breakout signal]"

    bb_ok = bool(current["close"] <= current["bb_lower"] * cfg.bb_entry_factor)
    rsi_ok = bool(current["rsi"] < cfg.rsi_primary)
    vol_ok = bool(current["volume"] > current["vol_ma20"] * cfg.volume_multiplier)
    if bb_ok and rsi_ok and vol_ok:
        return True, (
            f"[PRIMARY_V2] close={current['close']:,.0f} "
            f"BB_lower={current['bb_lower']:,.0f} RSI={current['rsi']:.1f} "
            f"vol={current['volume']/current['vol_ma20']:.2f}x exp={expected_ret*100:.2f}%"
        )

    if cfg.aggressive_enabled and previous is not None and _finite(previous, "ema_fast"):
        ema_cross = bool(previous["close"] < previous["ema_fast"] and current["close"] >= current["ema_fast"])
        rsi_agg = bool(current["rsi"] < cfg.rsi_aggressive)
        if ema_cross and rsi_agg and vol_ok:
            return True, (
                f"[AGGRESSIVE] EMA20 recovery RSI={current['rsi']:.1f} "
                f"exp={expected_ret*100:.2f}%"
            )

    return False, (
        f"HOLD [RSI={current['rsi']:.1f} BB={'Y' if bb_ok else 'N'} "
        f"VOL={'Y' if vol_ok else 'N'} exp={expected_ret*100:.2f}%]"
    )


def should_buy(
    df: pd.DataFrame,
    ticker: str | None = None,
    *,
    config: StrategyConfig | None = None,
    cooldown_until: datetime | pd.Timestamp | str | None = None,
) -> tuple[bool, str]:
    """Evaluate the last closed candle (iloc[-2])."""
    min_len = max(EMA_SLOW + SLOPE_LOOKBACK, BB_PERIOD, RSI_PERIOD, VOL_LOOKBACK_V2, ATR_PERIOD) + 5
    if len(df) < min_len:
        return False, "insufficient data"

    return entry_signal(
        df.iloc[-2],
        df.iloc[-3],
        ticker=ticker,
        config=config,
        cooldown_until=cooldown_until,
        now=_series_time(df.iloc[-2]),
    )


def exit_signal(
    current: pd.Series,
    *,
    entry_price: float,
    entry_time: datetime | pd.Timestamp | str | None = None,
    current_time: datetime | pd.Timestamp | None = None,
    config: StrategyConfig | None = None,
    use_ohlc: bool = False,
) -> tuple[float | None, str]:
    """Return (exit_price, reason). None means hold."""
    cfg = config or DEFAULT_CONFIG
    close = float(current["close"])
    high = float(current["high"]) if use_ohlc and "high" in current else close
    low = float(current["low"]) if use_ohlc and "low" in current else close
    bb_mid = float(current["bb_mid"])
    bb_lower = float(current["bb_lower"])

    tp_level = entry_price * (1 + cfg.take_profit)
    if high >= tp_level:
        pnl = tp_level / entry_price - 1
        return tp_level, f"TAKE_PROFIT pnl=+{pnl*100:.2f}%"

    bb_mid_pnl = close / entry_price - 1
    if close >= bb_mid and bb_mid_pnl >= cfg.bb_mid_min_profit:
        return close, f"BB_MID pnl={bb_mid_pnl*100:.2f}%"

    sl_level = entry_price * (1 - cfg.stop_loss)
    if low <= sl_level:
        pnl = sl_level / entry_price - 1
        return sl_level, f"STOP_LOSS pnl={pnl*100:.2f}%"

    hard_stop_level = bb_lower * cfg.hard_stop_factor
    if close < hard_stop_level:
        pnl = close / entry_price - 1
        return close, f"HARD_STOP pnl={pnl*100:.2f}%"

    held_minutes = _held_minutes(entry_time, current_time or _series_time(current))
    if held_minutes is not None and held_minutes >= cfg.max_hold_minutes:
        pnl = close / entry_price - 1
        return close, f"MAX_HOLD pnl={pnl*100:.2f}% held={held_minutes:.0f}m"

    pnl = close / entry_price - 1
    return None, f"HOLD pnl={pnl*100:.2f}%"


def should_sell(
    entry_price: float,
    current_price: float,
    bb_mid: float,
    bb_lower: float,
    *,
    entry_time: datetime | pd.Timestamp | str | None = None,
    current_time: datetime | pd.Timestamp | None = None,
    config: StrategyConfig | None = None,
) -> tuple[bool, str]:
    """Live price wrapper around exit_signal."""
    row = pd.Series(
        {
            "open": current_price,
            "high": current_price,
            "low": current_price,
            "close": current_price,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
        },
        name=current_time,
    )
    exit_price, reason = exit_signal(
        row,
        entry_price=entry_price,
        entry_time=entry_time,
        current_time=current_time,
        config=config,
        use_ohlc=False,
    )
    return exit_price is not None, reason


def cooldown_until_after_loss(
    exit_time: datetime | pd.Timestamp,
    config: StrategyConfig | None = None,
) -> datetime | pd.Timestamp:
    cfg = config or DEFAULT_CONFIG
    return pd.Timestamp(exit_time) + pd.Timedelta(minutes=cfg.cooldown_minutes)


def _finite(series: pd.Series, key: str) -> bool:
    try:
        value = float(series[key])
    except (KeyError, TypeError, ValueError):
        return False
    return isfinite(value)


def _series_time(series: pd.Series) -> pd.Timestamp | None:
    name: Any = getattr(series, "name", None)
    if name is None:
        return None
    try:
        return pd.Timestamp(name)
    except Exception:
        return None


def _in_cooldown(
    cooldown_until: datetime | pd.Timestamp | str | None,
    now: datetime | pd.Timestamp | None,
) -> bool:
    if cooldown_until is None:
        return False
    if now is None:
        now = pd.Timestamp.now()
    try:
        return pd.Timestamp(now) < pd.Timestamp(cooldown_until)
    except Exception:
        return False


def _held_minutes(
    entry_time: datetime | pd.Timestamp | str | None,
    current_time: datetime | pd.Timestamp | None,
) -> float | None:
    if entry_time is None or current_time is None:
        return None
    try:
        delta = pd.Timestamp(current_time) - pd.Timestamp(entry_time)
    except Exception:
        return None
    return delta.total_seconds() / 60.0
