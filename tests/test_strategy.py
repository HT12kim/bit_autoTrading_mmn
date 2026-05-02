from __future__ import annotations

import pandas as pd

from strategy import (
    StrategyConfig,
    add_indicators,
    entry_signal,
    exit_signal,
    should_buy,
)


def _entry_row(**overrides):
    data = {
        "open": 100.0,
        "high": 101.0,
        "low": 98.0,
        "close": 99.0,
        "volume": 220.0,
        "ema_fast": 101.0,
        "ema_slow": 100.0,
        "ema_slow_slope10": 0.001,
        "bb_mid": 100.0,
        "bb_lower": 99.0,
        "rsi": 25.0,
        "vol_ma20": 100.0,
        "volume_ratio": 2.2,
        "atr_pct": 0.01,
        "range_prev": 1.0,
    }
    data.update(overrides)
    return pd.Series(data, name=pd.Timestamp("2026-01-01 00:00:00"))


def test_add_indicators_keeps_legacy_and_adds_v2_columns():
    idx = pd.date_range("2026-01-01", periods=80, freq="min")
    close = pd.Series(range(100, 180), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )

    out = add_indicators(df)

    for column in [
        "vol_ma3",
        "vol_ma20",
        "volume_ratio",
        "atr14",
        "atr_pct",
        "ema_slow_slope10",
        "bb_width",
        "bb_width_ma20",
        "return_15m",
        "return_30d",
        "ema_trend",
        "range_prev",
        "breakout_target",
    ]:
        assert column in out.columns


def test_primary_v2_requires_uptrend_fee_volume_and_slope():
    cfg = StrategyConfig()
    ok, reason = entry_signal(_entry_row(), _entry_row(close=100.5), config=cfg)
    assert ok
    assert "PRIMARY_V2" in reason

    ok, reason = entry_signal(_entry_row(ema_fast=99.0), _entry_row(), config=cfg)
    assert not ok
    assert "downtrend" in reason

    ok, reason = entry_signal(_entry_row(volume=105.0), _entry_row(), config=cfg)
    assert not ok
    assert "VOL=N" in reason

    ok, reason = entry_signal(_entry_row(ema_slow_slope10=-0.001), _entry_row(), config=cfg)
    assert not ok
    assert "trend slope" in reason


def test_should_buy_honors_cooldown():
    idx = pd.date_range("2026-01-01", periods=80, freq="min")
    rows = [_entry_row() for _ in idx]
    df = pd.DataFrame(rows, index=idx)
    df.index.name = "time"

    ok, reason = should_buy(
        df,
        config=StrategyConfig(),
        cooldown_until=pd.Timestamp("2026-01-01 01:30:00"),
    )

    assert not ok
    assert "cooldown" in reason


def test_volatility_breakout_uses_recovery_volume_rsi_and_bb_mid():
    current = _entry_row(
        open=100.0,
        close=101.0,
        bb_mid=100.5,
        bb_lower=98.0,
        rsi=50.0,
        volume_ratio=1.6,
        range_prev=1.0,
        return_30d=-0.12,
        ema_trend=100.0,
        bb_width=0.02,
        bb_width_ma20=0.015,
    )
    previous = _entry_row(close=100.4)
    cfg = StrategyConfig(
        volatility_breakout_enabled=True,
        mean_reversion_enabled=False,
        breakout_k=0.5,
        breakout_volume_multiplier=1.5,
    )

    ok, reason = entry_signal(current, previous, config=cfg)

    assert ok
    assert "RECOVERY_BREAKOUT" in reason


def test_bb_mid_exit_requires_minimum_profit():
    row = _entry_row(close=100.05, bb_mid=100.0)
    exit_price, reason = exit_signal(row, entry_price=100.0, config=StrategyConfig(bb_mid_min_profit=0.001))

    assert exit_price is None
    assert reason.startswith("HOLD")

    row = _entry_row(close=100.20, bb_mid=100.0)
    exit_price, reason = exit_signal(row, entry_price=100.0, config=StrategyConfig(bb_mid_min_profit=0.001))

    assert exit_price == 100.20
    assert "BB_MID" in reason


def test_max_hold_exit():
    row = _entry_row(close=99.8, bb_mid=101.0, low=99.7, high=99.9)
    exit_price, reason = exit_signal(
        row,
        entry_price=100.0,
        entry_time=pd.Timestamp("2026-01-01 00:00:00"),
        current_time=pd.Timestamp("2026-01-01 01:01:00"),
        config=StrategyConfig(max_hold_minutes=60, stop_loss=0.01),
        use_ohlc=True,
    )

    assert exit_price == 99.8
    assert "MAX_HOLD" in reason
