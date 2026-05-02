from __future__ import annotations

import pandas as pd

from multi_asset_vol_target import generate_target_weights, max_drawdown


def test_generate_target_weights_caps_each_asset_allocation():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    frames = {
        "KRW-BTC": pd.DataFrame(
            {
                "vol20": [0.04, 0.04, 0.04],
                "eligible": [True, True, True],
            },
            index=idx,
        ),
        "KRW-XRP": pd.DataFrame(
            {
                "vol20": [0.01, 0.01, 0.01],
                "eligible": [True, True, True],
            },
            index=idx,
        ),
        "KRW-SOL": pd.DataFrame(
            {
                "vol20": [0.02, 0.02, 0.02],
                "eligible": [False, False, False],
            },
            index=idx,
        ),
        "KRW-ADA": pd.DataFrame(
            {
                "vol20": [0.02, 0.02, 0.02],
                "eligible": [True, True, True],
            },
            index=idx,
        ),
    }

    weights = generate_target_weights(frames, target_vol=0.02)

    assert weights["KRW-BTC"].iloc[-1] == 0.125
    assert weights["KRW-XRP"].iloc[-1] == 0.25
    assert weights["KRW-SOL"].iloc[-1] == 0.0
    assert weights["KRW-ADA"].iloc[-1] == 0.25


def test_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.3])

    assert round(max_drawdown(equity), 4) == -0.25
