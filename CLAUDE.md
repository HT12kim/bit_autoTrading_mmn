# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`btc_inv` is a Python-based multi-coin automated trading bot for the Upbit exchange.
It trades KRW-BTC, KRW-ETH, KRW-XRP, KRW-SOL, KRW-ADA on 1-minute candles using a
Long-only Mean Reversion Scalping strategy with an explicit fee filter.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in UPBIT_ACCESS and UPBIT_SECRET
```

## Running the Bot

```bash
python bot.py
```

## Running the Backtest

```bash
python backtest.py
# First run downloads ~525,600 candles per coin (10-15 min each).
# Subsequent runs use data_cache_{TICKER}.pkl (6-hour TTL).
```

## Common Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_strategy.py

# Lint
ruff check .
```

## Architecture

| File | Responsibility |
|---|---|
| `strategy.py` | Indicator calculation (`add_indicators`) and signal logic (`should_buy`, `should_sell`, `fee_filter_ok`) |
| `bot.py` | Main loop: candle timing, multi-coin order execution, risk manager, state persistence |
| `backtest.py` | 1-min 1-year walk-forward backtest across all 5 tickers |
| `state.json` | Runtime state: daily PnL, consecutive losses, halt timestamp, open positions |
| `data_cache_{TICKER}.pkl` | Per-ticker OHLCV cache (e.g. `data_cache_KRW_BTC.pkl`), 6-hour TTL |

### Strategy (`strategy.py`)

**Market filter** — only enter when EMA20 > EMA50 (uptrend). In downtrend, do nothing.

**Fee filter** — `(BB_mid − close) / close ≥ 0.25%` must pass before any entry. Ensures expected move to mean covers round-trip fee (0.1%) with margin.

**Entry — evaluated on last *closed* 1-min candle (`iloc[-2]`):**

| Priority | Signal | Conditions |
|---|---|---|
| 1 | PRIMARY | `close ≤ BB_lower × 1.001` AND `RSI < 35` AND `volume > vol_ma3` |
| 2 | AGGRESSIVE | EMA20 crossover up (prev below → current at/above) AND `RSI < 40` AND `volume > vol_ma3` |

**Exit — first condition met:**

| Priority | Condition | Description |
|---|---|---|
| 1 | PnL ≥ +0.3% | TAKE_PROFIT |
| 2 | current_price ≥ BB_mid | BB_MID — mean reached, take profit |
| 3 | PnL ≤ −0.3% | STOP_LOSS |
| 4 | close < BB_lower × 0.998 | HARD_STOP — strong breakdown |

### Bot loop (`bot.py`)

Sleeps until **5 seconds after** the next 1-minute candle closes before fetching data.

Supports up to `MAX_CONCURRENT = 3` simultaneous positions across all tickers.
Per-trade size: `PER_TRADE_RATIO = 10%` of total equity (KRW + coin holdings at current price).

State is persisted to `state.json` after every cycle. On restart, existing positions are
resumed from state (no phantom entries from stale data).

**Risk manager:**
- Daily loss ≥ 2% of starting equity → halt all trading for the rest of the day.
- 5 consecutive losing trades → 60-minute trading halt.
- Both limits reset at midnight.

### Key constants (bot.py / strategy.py)

| Constant | Value | Meaning |
|---|---|---|
| `EMA_FAST` | `20` | Fast EMA period |
| `EMA_SLOW` | `50` | Slow EMA period (market filter) |
| `BB_PERIOD` | `20` | Bollinger Band period |
| `BB_STD` | `2.0` | Bollinger Band standard deviations |
| `RSI_PERIOD` | `14` | RSI period |
| `RSI_PRIMARY` | `35` | RSI threshold for primary entry |
| `RSI_AGGRESSIVE` | `40` | RSI threshold for aggressive entry |
| `TAKE_PROFIT` | `0.003` | +0.3% → sell |
| `STOP_LOSS` | `0.003` | −0.3% → sell |
| `FEE_FILTER_MIN` | `0.0025` | Min expected profit before entry (0.25%) |
| `ROUND_TRIP_FEE` | `0.001` | Round-trip fee estimate (0.1%) |
| `PER_TRADE_RATIO` | `0.10` | Fraction of equity per trade |
| `MAX_CONCURRENT` | `3` | Max simultaneous open positions |
| `DAILY_LOSS_LIMIT` | `-0.02` | Daily loss cap (−2%) |
| `CONSECUTIVE_LOSS_HALT` | `5` | Consecutive losses before halt |
| `HALT_DURATION_MIN` | `60` | Halt duration in minutes |
| `MIN_ORDER_KRW` | `5_000` | Upbit minimum order size |

## Credentials

Never commit `.env`. Upbit API keys are loaded from environment variables `UPBIT_ACCESS` and `UPBIT_SECRET` via `python-dotenv`.
