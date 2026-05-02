from __future__ import annotations

import inspect

import backtest


def test_backtest_uses_strategy_signal_engine():
    simulate_source = inspect.getsource(backtest.simulate_ticker)
    portfolio_source = inspect.getsource(backtest.simulate_portfolio)

    assert "entry_signal(" in simulate_source
    assert "exit_signal(" in simulate_source
    assert "entry_signal(" in portfolio_source
    assert "exit_signal(" in portfolio_source
