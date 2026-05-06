from __future__ import annotations

import pandas as pd

from strategy import (
    MA_120D,
    MA_60D,
    MA_MARKET,
    MA_SHORT,
    StrategyConfig,
    add_indicators,
    adaptive_stop_price,
    check_buy_signal,
    check_sell_signal,
    entry_signal,
    exit_signal,
    risk_position_size,
    should_buy,
)


def _entry_row(**overrides):
    data = {
        "open": 105.0,
        "high": 108.0,
        "low": 101.0,
        "close": 104.0,
        "volume": 180.0,
        "ma5": 101.0,
        "ma20": 100.0,
        "ma60": 98.0,
        "ma60d": 100.0,
        "ma120d": 95.0,
        "bb_upper": 112.0,
        "bb_lower": 102.0,
        "rsi": 31.0,
        "rsi_signal": 29.0,
        "atr14": 3.0,
        "atr_pct": 3.0 / 104.0,
        "prev5_volume_avg": 100.0,
        "prev_high": 103.0,
        "ma60_slope": 1.0,
        "ma20_slope": 1.0,
        "vwap": 103.0,
        "vwap_slope": 0.5,
        "vol_ma20": 80.0,
        "poc": 103.0,
    }
    data.update(overrides)
    return pd.Series(data, name=pd.Timestamp("2026-01-01 00:00:00"))


def _series_with(series: pd.Series, **overrides) -> pd.Series:
    out = series.copy()
    for key, value in overrides.items():
        out[key] = value
    return out


def test_add_indicators_calculates_hourly_strategy_columns():
    idx = pd.date_range("2026-01-01", periods=3_000, freq="h")
    close = pd.Series(range(100, 3_100), index=idx, dtype=float)
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
        "ma5d",
        "ma5",
        "ma20",
        "ma60d",
        "ma120d",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "rsi",
        "rsi_signal",
        "atr14",
        "atr_pct",
        "vol_ma5",
        "vol_ma20",
        "prev5_volume_avg",
    ]:
        assert column in out.columns
    assert out["ma5"].iloc[-1] == out["close"].tail(MA_SHORT).mean()
    assert out["ma20"].iloc[-1] == out["close"].tail(MA_MARKET).mean()
    assert out["ma60d"].iloc[-1] == out["close"].tail(MA_60D).mean()
    assert out["ma120d"].iloc[-1] == out["close"].tail(MA_120D).mean()


def test_entry_uses_btc_ma20_or_rsi40_market_filter():
    previous = _entry_row(rsi=29.0)

    ok, reason = entry_signal(_entry_row(), previous, ticker="KRW-BTC", config=StrategyConfig())
    assert ok
    assert "ENTRY:PRIMARY_SPIKE" in reason

    ok, reason = entry_signal(
        _entry_row(close=99.0, ma20=100.0, rsi=31.0, prev_high=98.0, vwap=98.0, poc=98.0),
        previous,
        ticker="KRW-BTC",
        config=StrategyConfig(),
    )
    assert not ok
    assert "MARKET=N" in reason

    ok, reason = entry_signal(
        _entry_row(close=99.0, low=97.0, bb_lower=98.0, ma20=100.0, rsi=46.0, prev_high=98.0, vwap=98.0, poc=98.0),
        previous,
        ticker="KRW-BTC",
        config=StrategyConfig(),
    )
    assert ok


def test_entry_requires_bollinger_touch_and_rsi30_reclaim():
    previous = _entry_row(rsi=29.0)

    ok, reason = entry_signal(_entry_row(rsi=31.0, low=101.0), previous, ticker="KRW-BTC")
    assert ok
    assert "ENTRY:PRIMARY_SPIKE" in reason

    ok, reason = entry_signal(_entry_row(rsi=31.0, low=104.0), previous, ticker="KRW-BTC")
    assert not ok
    assert "PULLBACK=N" in reason

    ok, reason = entry_signal(_entry_row(rsi=31.0, low=101.0, close=101.0), previous, ticker="KRW-BTC")
    assert not ok
    assert "PULLBACK=N" in reason

    ok, reason = entry_signal(_entry_row(rsi=29.5, low=101.0), previous, ticker="KRW-BTC")
    assert not ok
    assert "PULLBACK=N" in reason


def test_entry_requires_btc_market_filter_for_alts():
    previous = _entry_row(rsi=29.0)

    ok, reason = entry_signal(
        _entry_row(),
        previous,
        ticker="KRW-ETH",
        market_current=_entry_row(close=90.0, ma20=100.0, rsi=40.0),
    )
    assert not ok
    assert "MARKET=N" in reason

    ok, reason = entry_signal(
        _entry_row(),
        previous,
        ticker="KRW-ETH",
        market_current=_entry_row(close=110.0, ma20=100.0, rsi=35.0),
    )
    assert ok

    ok, reason = entry_signal(
        _entry_row(),
        previous,
        ticker="KRW-ETH",
        market_current=_entry_row(close=90.0, ma20=100.0, rsi=46.0),
    )
    assert ok


def test_entry_requires_current_volume_to_exceed_previous_five_average():
    ok, reason = entry_signal(
        _entry_row(volume=60.0, prev5_volume_avg=100.0, vol_ma20=20.0),
        _entry_row(rsi=29.0),
        ticker="KRW-BTC",
        config=StrategyConfig(),
    )

    assert not ok
    assert "VOL=N" in reason


def test_entry_blocks_overbought_and_ma20_disparity():
    previous = _entry_row(rsi=29.0)

    ok, reason = entry_signal(_entry_row(rsi=73.0), previous, ticker="KRW-BTC")
    assert not ok
    assert "RSI_COOL=N" in reason

    ok, reason = entry_signal(_entry_row(close=106.0, ma20=100.0), previous, ticker="KRW-BTC")
    assert not ok
    assert "DISPARITY=N" in reason


def test_should_buy_honors_minimum_data_and_cooldown():
    idx = pd.date_range("2026-01-01", periods=2_900, freq="h")
    df = pd.DataFrame([_entry_row() for _ in idx], index=idx)
    df.loc[idx[-3], "rsi"] = 29.0

    ok, reason = should_buy(
        df,
        config=StrategyConfig(),
        cooldown_until=pd.Timestamp("2026-06-01 00:00:00"),
    )
    assert not ok
    assert "cooldown" in reason

    ok, reason = should_buy(df.iloc[:-1], config=StrategyConfig())
    assert not ok
    assert "insufficient data" in reason


def test_exit_signal_conditions():
    base = _entry_row(close=105.0, low=105.0, rsi=50.0, bb_upper=120.0, ma120d=90.0, vwap=100.0)

    exit_price, reason, armed, highest, _, _ = exit_signal(_series_with(base, close=102.0), entry_price=100.0, current_price=102.0)
    assert exit_price is None
    assert "HOLD" in reason
    assert not armed
    assert highest == 102.0

    exit_price, reason, armed, highest, _, _ = exit_signal(
        _series_with(base, close=121.0, high=122.0),
        entry_price=100.0,
        current_price=121.0,
        use_ohlc=True,
    )
    assert exit_price == 121.0
    assert reason == "PARTIAL_TAKE_PROFIT_50"
    assert armed
    assert highest == 122.0

    exit_price, reason, _, _, _, _ = exit_signal(
        _series_with(base, low=92.0, atr14=4.0, vwap=90.0),
        entry_price=100.0,
        current_price=96.0,
        use_ohlc=True,
    )
    assert exit_price == 92.8
    assert "STOP_LOSS" in reason


def test_dead_cross_exit_has_top_priority():
    current = _entry_row(close=96.0, low=92.0, ma5=99.0, ma20=100.0, atr14=4.0, vwap=90.0)
    previous = _entry_row(ma5=101.0, ma20=100.0)

    exit_price, reason, _, _, _, _ = exit_signal(
        current,
        previous=previous,
        entry_price=100.0,
        current_price=96.0,
        use_ohlc=True,
    )

    assert exit_price == 96.0
    assert reason == "MA5_MA20_DEAD_CROSS"


def test_time_cut_after_48_bars_when_profit_is_too_small():
    current = _entry_row(close=100.1, high=100.1, low=100.1, ma120d=90.0, vwap=99.0)

    exit_price, reason, _, _, _, _ = exit_signal(
        current,
        entry_price=100.0,
        current_price=100.1,
        bars_held=24,
        use_ohlc=True,
    )

    assert exit_price == 100.1
    assert reason.startswith("TIME_EXIT")


def test_step_up_trailing_take_profit_after_five_percent_profit():
    base = _entry_row(close=110.0, low=108.0, high=112.0, bb_upper=120.0, rsi=50.0, atr14=10.0, vwap=100.0)

    exit_price, reason, armed, highest, _, _ = exit_signal(
        base,
        entry_price=100.0,
        current_price=108.0,
        partial_taken=True,
        take_profit_armed=True,
        highest_price=112.0,
        use_ohlc=True,
    )

    assert exit_price == 112.0 * 0.97
    assert reason == "TRAILING_TAKE_PROFIT"
    assert armed
    assert highest == 112.0


def test_base_trailing_take_profit_before_five_percent_profit():
    base = _entry_row(close=103.0, low=99.5, high=104.0, bb_upper=120.0, rsi=50.0, atr14=10.0, vwap=100.0)

    exit_price, reason, armed, highest, _, _ = exit_signal(
        base,
        entry_price=100.0,
        current_price=99.5,
        take_profit_armed=True,
        highest_price=104.0,
        use_ohlc=True,
    )

    assert exit_price == 104.0 * 0.96
    assert reason == "TRAILING_TAKE_PROFIT"
    assert armed
    assert highest == 104.0


def test_adaptive_stop_and_risk_position_size():
    row = _entry_row(atr14=2.0)

    assert adaptive_stop_price(100.0, row) == 96.4
    assert adaptive_stop_price(100.0, _entry_row(atr14=10.0)) == 82.0

    invest_krw, qty = risk_position_size(
        equity=1_000_000,
        cash=1_000_000,
        entry_price=100.0,
        stop_price=95.0,
    )
    assert invest_krw == 200_000.0
    assert qty == 2_000.0


def test_check_signal_compatibility_wrappers():
    idx = pd.date_range("2026-01-01", periods=2_900, freq="h")
    df = pd.DataFrame([_entry_row() for _ in idx], index=idx)
    df.loc[idx[-3], "rsi"] = 29.0

    ok, reason = check_buy_signal(df, ticker="KRW-BTC")
    assert ok
    assert "ENTRY:PRIMARY_SPIKE" in reason

    sell_ok, sell_reason, armed, highest, _, _ = check_sell_signal(
        100.0,
        99.5,
        row=_entry_row(high=104.0, low=99.5, atr14=10.0, vwap=100.0),
        take_profit_armed=True,
        highest_price=104.0,
    )
    assert sell_ok
    assert sell_reason == "TRAILING_TAKE_PROFIT"
    assert armed
    assert highest == 104.0
