from __future__ import annotations

import inspect

import pandas as pd

import backtest


def test_backtest_uses_strategy_signal_engine():
    simulate_source = inspect.getsource(backtest.simulate_ticker)
    portfolio_source = inspect.getsource(backtest.simulate_portfolio)

    assert "entry_signal(" in simulate_source
    assert "exit_signal(" in simulate_source
    assert "entry_signal(" in portfolio_source
    assert "exit_signal(" in portfolio_source


def test_backtest_selects_universe_from_recent_quote_volume():
    idx = pd.date_range("2026-01-01", periods=3, freq="min")
    frames = {
        "KRW-BTC": pd.DataFrame({"close": [100.0, 100.0, 100.0], "volume": [1.0, 1.0, 1.0]}, index=idx),
        "KRW-XRP": pd.DataFrame({"close": [10.0, 10.0, 10.0], "volume": [50.0, 50.0, 50.0]}, index=idx),
        "KRW-ETH": pd.DataFrame({"close": [200.0, 200.0, 200.0], "volume": [2.0, 2.0, 2.0]}, index=idx),
    }

    selected = backtest.select_universe_from_frames(frames, idx[-1], limit=2, lookback=3)

    assert selected == ["KRW-XRP", "KRW-ETH"]


def test_backtest_excludes_low_volatility_from_top_volume_universe():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    frames = {
        "KRW-BTC": pd.DataFrame({"close": [100.0] * 3, "volume": [100.0] * 3, "atr_pct": [0.001] * 3}, index=idx),
        "KRW-XRP": pd.DataFrame({"close": [10.0] * 3, "volume": [50.0] * 3, "atr_pct": [0.020] * 3}, index=idx),
        "KRW-ETH": pd.DataFrame({"close": [200.0] * 3, "volume": [2.0] * 3, "atr_pct": [0.010] * 3}, index=idx),
        "KRW-SOL": pd.DataFrame({"close": [50.0] * 3, "volume": [5.0] * 3, "atr_pct": [0.030] * 3}, index=idx),
    }

    selected = backtest.select_universe_from_frames(frames, idx[-1], limit=4, lookback=3)

    assert selected == ["KRW-XRP", "KRW-ETH", "KRW-SOL"]


def test_backtest_selects_daily_universe_from_previous_completed_day():
    idx = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 12:00",
            "2026-01-02 00:00",
            "2026-01-02 12:00",
        ]
    )
    frames = {
        "KRW-BTC": pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0], "volume": [1.0, 1.0, 100.0, 100.0]}, index=idx),
        "KRW-XRP": pd.DataFrame({"close": [10.0, 10.0, 10.0, 10.0], "volume": [50.0, 50.0, 1.0, 1.0]}, index=idx),
        "KRW-ETH": pd.DataFrame({"close": [200.0, 200.0, 200.0, 200.0], "volume": [2.0, 2.0, 1.0, 1.0]}, index=idx),
    }

    selected = backtest.select_daily_universe_from_frames(frames, pd.Timestamp("2026-01-02"), limit=2)

    assert selected == ["KRW-XRP", "KRW-ETH"]


def test_backtest_counts_daily_universe_membership():
    idx = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 12:00",
            "2026-01-02 00:00",
            "2026-01-02 12:00",
        ]
    )
    frames = {
        "KRW-BTC": pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0], "volume": [1.0, 1.0, 100.0, 100.0]}, index=idx),
        "KRW-XRP": pd.DataFrame({"close": [10.0, 10.0, 10.0, 10.0], "volume": [50.0, 50.0, 1.0, 1.0]}, index=idx),
        "KRW-ETH": pd.DataFrame({"close": [200.0, 200.0, 200.0, 200.0], "volume": [2.0, 2.0, 1.0, 1.0]}, index=idx),
    }

    counts = backtest.count_daily_universe_membership(
        frames,
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-02"),
        limit=2,
    )

    assert counts == {"KRW-XRP": 1, "KRW-ETH": 1}
