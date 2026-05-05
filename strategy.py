"""
strategy.py - 1-hour trend-following pullback strategy for Upbit KRW markets.

Live trading and backtests share this module so indicator, entry, and exit
rules stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from typing import Any

import pandas as pd

MA_5D = 5 * 24
MA_SHORT = 5
MA_MEDIUM = 60
MA_60D = 60 * 24
MA_120D = 120 * 24
MA_MARKET = 20
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_SIGNAL_PERIOD = 9
ATR_PERIOD = 14
VOL_MA_FAST = 5
VOL_MA_SLOW = 20
PULLBACK_RSI_LEVEL = 30
SECONDARY_RSI_LEVEL = 45
ENTRY_RSI_MAX = 72
ENTRY_MA20_DISPARITY_MAX = 0.05
VOLUME_PULLBACK_MULTIPLIER = 0.7
MARKET_RSI_MIN = 45
STOP_LOSS = 0.04
ATR_STOP_MULTIPLIER = 1.8
TAKE_PROFIT_ARM_PNL = 0.03
BREAKEVEN_PLUS_ARM_PNL = 0.03
TRAILING_TAKE_PROFIT_DRAWDOWN = 0.04
TRAILING_STEP_UP_PNL = 0.07
TRAILING_STEP_UP_DRAWDOWN = 0.03
TIME_CUT_BARS = 24
TIME_CUT_MIN_PNL = 0.002
FAST_TIME_CUT_BARS = 6
FAST_TIME_CUT_MIN_PNL = -0.02
ROUND_TRIP_FEE = 0.001
DEFAULT_SLIPPAGE = 0.0003
MIN_CANDLES = 2_900


@dataclass(frozen=True)
class StrategyConfig:
    ticker: str | None = None
    enabled: bool = True
    volume_multiplier: float = VOLUME_PULLBACK_MULTIPLIER
    stop_loss: float = STOP_LOSS
    atr_stop_multiplier: float = ATR_STOP_MULTIPLIER
    take_profit_arm_pnl: float = TAKE_PROFIT_ARM_PNL
    trailing_take_profit_drawdown: float = TRAILING_TAKE_PROFIT_DRAWDOWN
    trailing_step_up_pnl: float = TRAILING_STEP_UP_PNL
    trailing_step_up_drawdown: float = TRAILING_STEP_UP_DRAWDOWN
    time_cut_bars: int = TIME_CUT_BARS
    time_cut_min_pnl: float = TIME_CUT_MIN_PNL
    cooldown_minutes: int = 30


DEFAULT_CONFIG = StrategyConfig()
TICKER_CONFIGS: dict[str, StrategyConfig] = {}


def get_strategy_config(ticker: str | None = None) -> StrategyConfig:
    if ticker and ticker in TICKER_CONFIGS:
        return TICKER_CONFIGS[ticker]
    return replace(DEFAULT_CONFIG, ticker=ticker)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators for the 1-hour trend/pullback strategy."""
    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["ma5"] = close.rolling(MA_SHORT).mean()
    out["ma60"] = close.rolling(MA_MEDIUM).mean()
    out["ma5d"] = close.rolling(MA_5D).mean()
    out["ma20"] = close.rolling(MA_MARKET).mean()
    out["ma60d"] = close.rolling(MA_60D).mean()
    out["ma120d"] = close.rolling(MA_120D).mean()

    out["bb_mid"] = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()
    out["bb_upper"] = out["bb_mid"] + (bb_std * BB_STD)
    out["bb_lower"] = out["bb_mid"] - (bb_std * BB_STD)

    out["rsi"] = _rsi(close, RSI_PERIOD)
    out["rsi_signal"] = out["rsi"].rolling(RSI_SIGNAL_PERIOD).mean()
    out["atr14"] = _atr(out, ATR_PERIOD)
    out["atr_pct"] = out["atr14"] / close
    out["vol_ma5"] = volume.rolling(VOL_MA_FAST).mean()
    out["vol_ma20"] = volume.rolling(VOL_MA_SLOW).mean()
    # 진입 거래량은 현재 확정봉을 제외한 직전 5개 확정봉 평균과 비교한다.
    out["prev5_volume_avg"] = volume.rolling(VOL_MA_FAST).mean().shift(1)
    # 강세 전환 확인: 현재 봉 종가가 직전 봉 고가를 상향 돌파해야 한다.
    out["prev_high"] = out["high"].astype(float).shift(1)
    out["ma60_slope"] = out["ma60"] - out["ma60"].shift(1)
    out["ma20_slope"] = out["ma20"] - out["ma20"].shift(1)
    # 일중 VWAP: 당일 누적(가격*거래량)/누적거래량
    session = pd.to_datetime(out.index).date
    pv = close * volume
    cum_pv = pv.groupby(session).cumsum()
    cum_vol = volume.groupby(session).cumsum()
    out["vwap"] = cum_pv / cum_vol.replace(0, pd.NA)
    out["vwap_slope"] = out["vwap"] - out["vwap"].shift(1)
    # 거래량 밀집 구역(POC) 근사치: 당일 누적 구간에서 거래량이 가장 많이 몰린 가격 레벨
    # 1시간봉 데이터에서는 일자별 기준가격의 0.2%를 고정 버킷 폭으로 사용한다.
    out["poc"] = close
    for session_date, idx in out.groupby(session).groups.items():
        day_index = list(idx)
        day_ref = float(out.loc[day_index[0], "close"])
        bucket_step = max(day_ref * 0.002, 1e-6)
        vol_by_bucket: dict[float, float] = {}
        current_poc = float(out.loc[day_index[0], "close"])
        for i in day_index:
            price = float(out.loc[i, "close"])
            b = round(price / bucket_step) * bucket_step
            vol_by_bucket[b] = vol_by_bucket.get(b, 0.0) + float(volume.loc[i])
            current_poc = max(vol_by_bucket.items(), key=lambda item: item[1])[0]
            out.at[i, "poc"] = current_poc
    return out


def entry_signal(
    current: pd.Series,
    previous: pd.Series | None = None,
    *,
    ticker: str | None = None,
    config: StrategyConfig | None = None,
    cooldown_until: datetime | pd.Timestamp | str | None = None,
    now: datetime | pd.Timestamp | None = None,
    market_current: pd.Series | None = None,
) -> tuple[bool, str]:
    """Evaluate one already-closed 1-hour candle for a new long entry."""
    cfg = config or get_strategy_config(ticker)
    if not cfg.enabled:
        return False, "DISABLED"

    if _in_cooldown(cooldown_until, now or _series_time(current)):
        return False, "HOLD [cooldown active after losing exit]"

    previous = previous if previous is not None else current
    required = [
        "close",
        "low",
        "volume",
        "ma20",
        "high",
        "bb_lower",
        "rsi",
        "atr14",
        "prev5_volume_avg",
        "prev_high",
        "ma60",
        "ma60_slope",
        "ma20_slope",
        "vwap",
        "vwap_slope",
        "vol_ma20",
        "poc",
    ]
    if not all(_finite(current, col) for col in required) or not _finite(previous, "rsi"):
        return False, "HOLD [insufficient indicators]"

    trend_ok = True
    rsi30_rebound_ok = bool(
        previous["rsi"] <= PULLBACK_RSI_LEVEL
        and current["rsi"] > PULLBACK_RSI_LEVEL
    )
    bb_touch_ok = bool(current["low"] <= current["bb_lower"] and current["close"] > current["bb_lower"])
    volume_spike_primary = bool(current["volume"] > current["vol_ma20"] * 2.0)
    primary_entry_ok = rsi30_rebound_ok and bb_touch_ok and volume_spike_primary
    rsi45_rebound_ok = bool(previous["rsi"] <= SECONDARY_RSI_LEVEL and current["rsi"] > SECONDARY_RSI_LEVEL)
    ma60_uptrend_ok = bool(current["ma60_slope"] > 0)
    ma20_touch_ok = bool(current["low"] <= current["ma20"])
    # B안: SECONDARY는 MA20 재돌파 확인(종가가 MA20 위) + MA20 기울기 양수로 추세 전환 신뢰도를 높인다.
    ma20_reclaim_ok = bool(current["close"] > current["ma20"] and current["ma20_slope"] > 0)
    vwap_slope_ok = bool(current["vwap_slope"] > 0)
    vwap_disparity_ok = bool(current["close"] <= current["vwap"] * 1.03)
    secondary_entry_ok = (
        ma60_uptrend_ok
        and ma20_touch_ok
        and ma20_reclaim_ok
        and vwap_slope_ok
        and vwap_disparity_ok
        and rsi45_rebound_ok
    )
    pullback_ok = primary_entry_ok or secondary_entry_ok
    # 현재 확정봉 거래량이 직전 5개 확정봉 평균보다 커야 눌림목 반등의 수급 확인으로 본다.
    volume_ok = bool(current["volume"] > current["prev5_volume_avg"] * cfg.volume_multiplier)
    market_ok = _market_filter_ok(ticker, market_current, current)
    overbought_ok = bool(current["rsi"] <= ENTRY_RSI_MAX)
    disparity_ok = bool(current["close"] <= current["ma20"] * (1 + ENTRY_MA20_DISPARITY_MAX))
    breakout_ok = bool(current["close"] > current["prev_high"])
    vwap_guard_ok = bool(current["close"] > current["vwap"] * 0.99)
    # POC 위 0.5% 이내는 매물대 근접 구간으로 허용해 진입 기회를 넓힌다.
    poc_ok = bool(current["close"] >= current["poc"] * 0.995)
    vwap_breakout_spike = bool(
        volume_spike_primary
        and current["close"] > current["vwap"]
        and (_finite(previous, "close") and _finite(previous, "vwap") and previous["close"] <= previous["vwap"])
    )
    vwap_gap_ok = bool(current["close"] < current["vwap"] * 1.015) or bool(
        vwap_breakout_spike and current["close"] < current["vwap"] * 1.02
    )

    if trend_ok and pullback_ok and volume_ok and market_ok and overbought_ok and disparity_ok and breakout_ok and vwap_guard_ok and vwap_gap_ok and poc_ok:
        entry_tag = "PRIMARY_SPIKE" if primary_entry_ok and volume_spike_primary else ("PRIMARY" if primary_entry_ok else "SECONDARY")
        return True, (
            f"[ENTRY:{entry_tag}] "
            f"close={current['close']:,.0f} "
            f"rsi={current['rsi']:.1f} atr={current['atr14']:,.2f} "
            f"vol={current['volume']/current['prev5_volume_avg']:.2f}x"
        )

    return False, (
        "HOLD "
        f"[TREND={'Y' if trend_ok else 'N'} "
        f"PULLBACK={'Y' if pullback_ok else 'N'} "
        f"VOL={'Y' if volume_ok else 'N'} "
        f"MARKET={'Y' if market_ok else 'N'} "
        f"RSI_COOL={'Y' if overbought_ok else 'N'} "
        f"DISPARITY={'Y' if disparity_ok else 'N'} "
        f"BREAKOUT={'Y' if breakout_ok else 'N'} "
        f"VWAP={'Y' if vwap_guard_ok else 'N'} "
        f"VWAP_GAP={'Y' if vwap_gap_ok else 'N'} "
        f"POC={'Y' if poc_ok else 'N'}]"
    )


def should_buy(
    df: pd.DataFrame,
    ticker: str | None = None,
    *,
    config: StrategyConfig | None = None,
    cooldown_until: datetime | pd.Timestamp | str | None = None,
    market_df: pd.DataFrame | None = None,
) -> tuple[bool, str]:
    """Evaluate the last closed candle (iloc[-2])."""
    if len(df) < MIN_CANDLES:
        return False, "insufficient data"
    return entry_signal(
        df.iloc[-2],
        df.iloc[-3],
        ticker=ticker,
        config=config,
        cooldown_until=cooldown_until,
        now=_series_time(df.iloc[-2]),
        market_current=market_df.iloc[-2] if market_df is not None and len(market_df) >= 2 else None,
    )


def check_buy_signal(
    df: pd.DataFrame,
    ticker: str | None = None,
    *,
    market_df: pd.DataFrame | None = None,
    config: StrategyConfig | None = None,
) -> tuple[bool, str]:
    """Compatibility wrapper for callers that use check_buy_signal naming."""
    return should_buy(df, ticker=ticker, market_df=market_df, config=config)


def exit_signal(
    current: pd.Series,
    *,
    entry_price: float,
    current_price: float | None = None,
    previous: pd.Series | None = None,
    entry_time: datetime | pd.Timestamp | str | None = None,
    current_time: datetime | pd.Timestamp | None = None,
    bars_held: int | None = None,
    partial_taken: bool = False,
    bb_break_seen: bool = False,
    overbought_seen: bool = False,
    config: StrategyConfig | None = None,
    use_ohlc: bool = False,
    take_profit_armed: bool = False,
    highest_price: float | None = None,
) -> tuple[float | None, str, bool, float, bool, bool]:
    """Return (exit_price, reason) when any exit condition is met."""
    cfg = config or DEFAULT_CONFIG
    close = float(current["close"])
    high = float(current["high"]) if use_ohlc and "high" in current else close
    low = float(current["low"]) if use_ohlc and "low" in current else close
    live_price = float(current_price if current_price is not None else close)
    stop_level = adaptive_stop_price(entry_price, current, cfg)
    if partial_taken:
        stop_level = max(stop_level, entry_price * 1.005)
    highest = max(float(highest_price or entry_price), high, live_price)
    armed = bool(take_profit_armed)
    was_armed = armed

    pnl = live_price / entry_price - 1

    # v10.0: +5% 이후에는 VWAP 기준을 1% 상향해 수익 보존을 강화한다.
    if _finite(current, "vwap"):
        vwap_floor = float(current["vwap"]) * (1.01 if pnl >= 0.05 else 1.0)
        if close < vwap_floor:
            return live_price, "VWAP_BREAKDOWN", armed, highest, overbought_seen, bb_break_seen

    # MA5/MA20 데드크로스는 추세 훼손 신호이므로 손절/익절보다 먼저 즉시 청산한다.
    if _dead_cross_ma5_ma20(current, previous):
        return live_price, "MA5_MA20_DEAD_CROSS", armed, highest, overbought_seen, bb_break_seen

    held_bars = bars_held if bars_held is not None else _held_bars(entry_time, current_time or _series_time(current))
    if held_bars is not None and held_bars >= FAST_TIME_CUT_BARS and pnl <= FAST_TIME_CUT_MIN_PNL:
        return live_price, f"FAST_TIME_EXIT bars={held_bars} pnl={pnl*100:.2f}%", armed, highest, overbought_seen, bb_break_seen
    if held_bars is not None and held_bars >= cfg.time_cut_bars and pnl < cfg.time_cut_min_pnl:
        return live_price, f"TIME_EXIT bars={held_bars} pnl={pnl*100:.2f}%", armed, highest, overbought_seen, bb_break_seen

    if not partial_taken and pnl >= cfg.take_profit_arm_pnl:
        return live_price, "PARTIAL_TAKE_PROFIT_50", True, highest, overbought_seen, bb_break_seen

    if partial_taken and _finite(current, "bb_upper"):
        if not bb_break_seen and close > float(current["bb_upper"]):
            bb_break_seen = True
        elif bb_break_seen and close < float(current["bb_upper"]):
            return live_price, "PARTIAL_TAKE_PROFIT_25_BB_REV", armed, highest, overbought_seen, bb_break_seen

    # v9.0: +3% 달성 후 손절선을 본절+수수료 이상으로 상향.
    if pnl >= BREAKEVEN_PLUS_ARM_PNL and _finite(current, "vwap"):
        stop_level = max(stop_level, entry_price * 1.005)

    if partial_taken:
        if not overbought_seen and current["rsi"] >= 70:
            overbought_seen = True
        if overbought_seen and current["rsi"] < 70:
            return live_price, "RSI_SMART_EXIT", armed, highest, overbought_seen, bb_break_seen

    if pnl >= cfg.take_profit_arm_pnl or (_finite(current, "bb_upper") and close > current["bb_upper"]):
        armed = True

    if was_armed:
        trailing_level = highest * (1 - cfg.trailing_take_profit_drawdown)
        if _finite(current, "atr14"):
            trailing_level = max(trailing_level, highest - (2.5 * float(current["atr14"])))
        if live_price <= trailing_level or low <= trailing_level:
            return trailing_level, "TRAILING_TAKE_PROFIT", armed, highest, overbought_seen, bb_break_seen

    if live_price <= stop_level or low <= stop_level:
        pnl = stop_level / entry_price - 1
        return stop_level, f"STOP_LOSS pnl={pnl*100:.2f}%", armed, highest, overbought_seen, bb_break_seen
    # MA120 이탈 청산은 변동성 장세에서 조기 이탈을 유발할 수 있어 비활성화한다.

    armed_text = " armed" if armed else ""
    return None, f"HOLD{armed_text} pnl={pnl*100:.2f}%", armed, highest, overbought_seen, bb_break_seen


def should_sell(
    entry_price: float,
    current_price: float,
    row: pd.Series | None = None,
    *,
    previous: pd.Series | None = None,
    entry_time: datetime | pd.Timestamp | str | None = None,
    current_time: datetime | pd.Timestamp | None = None,
    bars_held: int | None = None,
    partial_taken: bool = False,
    bb_break_seen: bool = False,
    overbought_seen: bool = False,
    config: StrategyConfig | None = None,
    take_profit_armed: bool = False,
    highest_price: float | None = None,
) -> tuple[bool, str, bool, float, bool, bool]:
    current = row if row is not None else pd.Series(
        {"open": current_price, "high": current_price, "low": current_price, "close": current_price}
    )
    exit_price, reason, armed, highest, overbought_seen_next, bb_break_seen_next = exit_signal(
        current,
        entry_price=entry_price,
        current_price=current_price,
        previous=previous,
        entry_time=entry_time,
        current_time=current_time,
        bars_held=bars_held,
        partial_taken=partial_taken,
        bb_break_seen=bb_break_seen,
        overbought_seen=overbought_seen,
        config=config,
        use_ohlc=row is not None,
        take_profit_armed=take_profit_armed,
        highest_price=highest_price,
    )
    return exit_price is not None, reason, armed, highest, overbought_seen_next, bb_break_seen_next


def check_sell_signal(
    entry_price: float,
    current_price: float,
    row: pd.Series | None = None,
    *,
    previous: pd.Series | None = None,
    entry_time: datetime | pd.Timestamp | str | None = None,
    current_time: datetime | pd.Timestamp | None = None,
    bars_held: int | None = None,
    partial_taken: bool = False,
    bb_break_seen: bool = False,
    overbought_seen: bool = False,
    take_profit_armed: bool = False,
    highest_price: float | None = None,
    config: StrategyConfig | None = None,
) -> tuple[bool, str, bool, float, bool, bool]:
    """Compatibility wrapper for callers that use check_sell_signal naming."""
    return should_sell(
        entry_price,
        current_price,
        row=row,
        previous=previous,
        entry_time=entry_time,
        current_time=current_time,
        bars_held=bars_held,
        partial_taken=partial_taken,
        bb_break_seen=bb_break_seen,
        overbought_seen=overbought_seen,
        take_profit_armed=take_profit_armed,
        highest_price=highest_price,
        config=config,
    )


def adaptive_stop_price(
    entry_price: float,
    current: pd.Series,
    config: StrategyConfig | None = None,
) -> float:
    cfg = config or DEFAULT_CONFIG
    if not _finite(current, "atr14"):
        return entry_price * (1 - cfg.stop_loss)
    atr_stop = entry_price - (float(current["atr14"]) * cfg.atr_stop_multiplier)
    return atr_stop


def risk_position_size(
    *,
    equity: float,
    cash: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float = 0.01,
    max_cash_fraction: float = 0.20,
) -> tuple[float, float]:
    risk_per_unit = entry_price - stop_price
    if equity <= 0 or cash <= 0 or entry_price <= 0 or risk_per_unit <= 0:
        return 0.0, 0.0
    risk_budget = equity * risk_fraction
    risk_qty = risk_budget / risk_per_unit
    max_invest = cash * max_cash_fraction
    qty = min(risk_qty, max_invest / entry_price)
    invest_krw = qty * entry_price
    return invest_krw, qty


def cooldown_until_after_loss(
    exit_time: datetime | pd.Timestamp,
    config: StrategyConfig | None = None,
) -> datetime | pd.Timestamp:
    cfg = config or DEFAULT_CONFIG
    return pd.Timestamp(exit_time) + pd.Timedelta(minutes=cfg.cooldown_minutes)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Upbit 스캘핑 봇에서는 지수 가중 RSI가 급격한 반등을 더 빠르게 반영한다.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.mask(avg_loss == 0, float("nan"))
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _market_filter_ok(
    ticker: str | None,
    market_current: pd.Series | None,
    current: pd.Series | None = None,
) -> bool:
    market = current if ticker == "KRW-BTC" else market_current
    if market is None:
        return False
    return _finite(market, "close") and _finite(market, "ma20") and _finite(market, "rsi") and bool(
        (market["close"] > market["ma20"]) or (market["rsi"] > MARKET_RSI_MIN)
    )


def _dead_cross_ma5_ma20(current: pd.Series, previous: pd.Series | None) -> bool:
    if previous is None:
        return False
    if not all(_finite(row, key) for row in (previous, current) for key in ("ma5", "ma20")):
        return False
    return bool(previous["ma5"] >= previous["ma20"] and current["ma5"] < current["ma20"])


def _held_bars(
    entry_time: datetime | pd.Timestamp | str | None,
    current_time: datetime | pd.Timestamp | None,
) -> int | None:
    if entry_time is None or current_time is None:
        return None
    try:
        delta = pd.Timestamp(current_time) - pd.Timestamp(entry_time)
    except Exception:
        return None
    if delta.total_seconds() < 0:
        return None
    return int(delta.total_seconds() // 3600)


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
