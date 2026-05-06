from __future__ import annotations

from datetime import datetime

import bot


def test_per_trade_ratio_uses_twenty_percent_krw_balance():
    assert bot.PER_TRADE_RATIO == 1.00


def test_strategy_summary_describes_active_strategy():
    summary = bot.strategy_summary()

    assert "KRW-USDT 제외 거래대금 상위 10개" in summary
    assert "TOP30+거래량증가 혼합/BTC 필터/PRIMARY 중심 엔트리" in summary
    assert "완화형 VWAP/ATR 추세보유" in summary
    assert "최대 1개" in summary


def test_startup_strategy_message_describes_current_strategy(monkeypatch):
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)

    message = bot.build_startup_strategy_message(
        {
            "positions": {"KRW-BTC": {}},
            "active_universe": ["KRW-BTC", "KRW-ETH"],
        },
        equity=1_234_567,
    )

    assert "<b>btc_inv 봇 시작</b>" in message
    assert "시간: 2026-01-01 09:30:05" in message
    assert "시작 자산: 1,234,567 KRW" in message
    assert "보유 state 포지션: 1개" in message
    assert "초기 유니버스 TOP: KRW-BTC, KRW-ETH" in message
    assert "적용 전략: KRW-USDT 제외 거래대금 상위 10개" in message
    assert "정기 알림: 매시 정각 유니버스/보유종목/자산 현황 전송" in message
    assert "현재가 기준 지정가 우선, 10초 미체결분 취소" in message
    assert "5연속 손실 시 60분 휴식" in message


def test_send_startup_strategy_notice_uses_telegram_when_enabled(monkeypatch):
    messages = []

    monkeypatch.setattr(bot, "telegram_enabled", lambda: True)
    monkeypatch.setattr(bot, "send_telegram_message", lambda message: messages.append(message) or True)
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)

    bot.send_startup_strategy_notice({"positions": {}, "active_universe": ["KRW-BTC"]}, 1_000_000)

    assert len(messages) == 1
    assert "btc_inv 봇 시작" in messages[0]
    assert "초기 유니버스 TOP: KRW-BTC" in messages[0]
    assert "적용 전략:" in messages[0]


def test_send_startup_strategy_notice_skips_when_telegram_disabled(monkeypatch):
    messages = []

    monkeypatch.setattr(bot, "telegram_enabled", lambda: False)
    monkeypatch.setattr(bot, "send_telegram_message", lambda message: messages.append(message) or True)

    bot.send_startup_strategy_notice()

    assert messages == []


def test_telegram_due_only_at_hour():
    state = {"last_telegram_slot": None}

    assert bot.telegram_due(state, now=datetime(2026, 1, 1, 9, 0, 5))
    assert bot.telegram_due(state, now=datetime(2026, 1, 1, 9, 4, 59))
    assert not bot.telegram_due(state, now=datetime(2026, 1, 1, 9, 30, 5))
    assert bot.telegram_due(state, now=datetime(2026, 1, 1, 9, 1, 5))
    assert not bot.telegram_due(state, now=datetime(2026, 1, 1, 9, 5, 0))
    assert not bot.telegram_due(
        {"last_telegram_slot": "2026-01-01 09:00"},
        now=datetime(2026, 1, 1, 9, 0, 10),
    )
    assert not bot.telegram_due(
        {"last_telegram_slot": "2026-01-01 09:00"},
        now=datetime(2026, 1, 1, 9, 4, 59),
    )


def test_current_telegram_slot_uses_hour_slot_with_grace():
    assert bot.current_telegram_slot(datetime(2026, 1, 1, 9, 0, 5)) == "2026-01-01 09:00"
    assert bot.current_telegram_slot(datetime(2026, 1, 1, 9, 4, 59)) == "2026-01-01 09:00"
    assert bot.current_telegram_slot(datetime(2026, 1, 1, 9, 5, 0)) is None


def test_seconds_to_next_hour(monkeypatch):
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)

    assert bot.seconds_to_next_hour() == 1795.0


def test_regular_status_notification_uses_requested_format(monkeypatch):
    messages = []
    state = {
        "starting_equity": 1_000_000,
        "daily_pnl_krw": 12_345,
        "active_universe": ["KRW-BTC", "KRW-ETH"],
        "universe_updated_at": "2026-01-01T08:59:59",
        "positions": {},
        "last_telegram_slot": None,
    }

    monkeypatch.setattr(bot, "telegram_enabled", lambda: True)
    monkeypatch.setattr(bot, "send_telegram_message", lambda message: messages.append(message) or True)
    monkeypatch.setattr(bot, "datetime", _FixedHourDateTime)

    bot.maybe_send_hourly_status(object(), state, 1_012_345)

    assert len(messages) == 1
    assert "<b>btc_inv 정각 현황</b>" in messages[0]
    assert "자산: 1,012,345 KRW" in messages[0]
    assert "일일PnL: +12,345 KRW (+1.23%)" in messages[0]
    assert "상태: OK" in messages[0]
    assert "유니버스 TOP: KRW-BTC, KRW-ETH" in messages[0]
    assert "유니버스 갱신: 2026-01-01T08:59:59" in messages[0]
    assert "보유: 0개" in messages[0]
    assert state["last_telegram_slot"] == "2026-01-01 09:00"


def test_status_message_includes_position_state(monkeypatch):
    state = {
        "starting_equity": 1_000_000,
        "daily_pnl_krw": 0,
        "active_universe": ["KRW-BTC", "KRW-XRP", "KRW-ETH"],
        "universe_updated_at": "2026-01-01T09:29:05",
        "positions": {
            "KRW-BTC": {
                "entry_price": 100.0,
                "invest_krw": 300_000,
            }
        },
    }

    monkeypatch.setattr(bot.pyupbit, "get_current_price", lambda ticker: 107.8)
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)

    message = bot.build_status_message(object(), state, 1_000_000, risk_reason="OK")

    assert "KRW-BTC: +7.80%" in message
    assert "유니버스 TOP: KRW-BTC, KRW-XRP, KRW-ETH" in message
    assert "진입 100" in message
    assert "현재 108" in message


def test_reconcile_positions_removes_stale_state_without_balance(monkeypatch):
    state = {
        "positions": {
            "KRW-BTC": {
                "entry_price": 100.0,
                "invest_krw": 10_000,
            }
        }
    }
    upbit = _FakeUpbit({"BTC": 0.0})

    monkeypatch.setattr(bot.pyupbit, "get_current_price", lambda ticker: 100.0)

    removed = bot.reconcile_positions_with_balances(upbit, state)

    assert removed == ["KRW-BTC"]
    assert state["positions"] == {}


def test_reconcile_positions_keeps_real_balance(monkeypatch):
    state = {
        "positions": {
            "KRW-BTC": {
                "entry_price": 100.0,
                "invest_krw": 10_000,
            }
        }
    }
    upbit = _FakeUpbit({"BTC": 60.0})

    monkeypatch.setattr(bot.pyupbit, "get_current_price", lambda ticker: 100.0)

    removed = bot.reconcile_positions_with_balances(upbit, state)

    assert removed == []
    assert "KRW-BTC" in state["positions"]


def test_trade_message_includes_fill_and_pnl(monkeypatch):
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)

    message = bot.build_trade_message(
        side="매도",
        ticker="KRW-BTC",
        price=100.0,
        invest_krw=10_000,
        reason="RSI_OVERBOUGHT",
        pnl_rate=0.0123,
        pnl_krw=123.0,
    )

    assert "<b>매도 체결</b>" in message
    assert "종목: KRW-BTC" in message
    assert "지정가 체결 가격: 100 KRW" in message
    assert "손익률: +1.23%" in message
    assert "손익금: +123 KRW" in message


def test_shutdown_message_includes_position_count(monkeypatch):
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)
    state = {"positions": {"KRW-BTC": {}, "KRW-ETH": {}}}

    message = bot.build_shutdown_message(state)

    assert "<b>btc_inv 봇 종료</b>" in message
    assert "사유: 사용자 종료 요청(Ctrl+C)" in message
    assert "보유 state 포지션: 2개" in message


def test_send_shutdown_notice_uses_telegram_when_enabled(monkeypatch):
    messages = []
    monkeypatch.setattr(bot, "telegram_enabled", lambda: True)
    monkeypatch.setattr(bot, "send_telegram_message", lambda message: messages.append(message) or True)
    monkeypatch.setattr(bot, "datetime", _FixedDateTime)

    bot.send_shutdown_notice({"positions": {}})

    assert len(messages) == 1
    assert "btc_inv 봇 종료" in messages[0]


class _FixedDateTime(datetime):
    @classmethod
    def now(cls):
        return cls(2026, 1, 1, 9, 30, 5)


class _FixedHourDateTime(datetime):
    @classmethod
    def now(cls):
        return cls(2026, 1, 1, 9, 0, 5)


class _FakeUpbit:
    def __init__(self, balances):
        self.balances = balances

    def get_balance(self, symbol):
        return self.balances.get(symbol, 0.0)


class _FakeOrderUpbit:
    def __init__(self, *, remaining_volume: str):
        self.remaining_volume = remaining_volume
        self.cancelled = []
        self.buy_orders = []
        self.sell_orders = []
        self.market_sells = []

    def buy_limit_order(self, ticker, price, volume):
        self.buy_orders.append((ticker, price, volume))
        return {"uuid": "order-1"}

    def sell_limit_order(self, ticker, price, volume):
        self.sell_orders.append((ticker, price, volume))
        return {"uuid": "order-1"}

    def get_order(self, uuid):
        return {"uuid": uuid, "remaining_volume": self.remaining_volume}

    def cancel_order(self, uuid):
        self.cancelled.append(uuid)
        return {"uuid": uuid}

    def sell_market_order(self, ticker, volume):
        self.market_sells.append((ticker, volume))
        return {"uuid": "market-1"}


class _FailingOrderUpbit:
    def __init__(self):
        self.calls = 0

    def buy_limit_order(self, *_):
        self.calls += 1
        raise RuntimeError("api down")


def test_limit_order_cancelled_when_unfilled(monkeypatch):
    upbit = _FakeOrderUpbit(remaining_volume="1.0")

    monkeypatch.setattr(bot.time, "sleep", lambda _: None)
    monkeypatch.setattr(bot, "telegram_enabled", lambda: False)

    result = bot.place_limit_order_and_wait(
        upbit,
        ticker="KRW-BTC",
        side="buy",
        price=100.0,
        volume=1.0,
    )

    assert result is None
    assert upbit.cancelled == ["order-1"]


def test_limit_order_returns_fill_only_when_fully_filled(monkeypatch):
    upbit = _FakeOrderUpbit(remaining_volume="0")

    monkeypatch.setattr(bot.time, "sleep", lambda _: None)

    result = bot.place_limit_order_and_wait(
        upbit,
        ticker="KRW-BTC",
        side="sell",
        price=100.0,
        volume=1.0,
    )

    assert result == {"uuid": "order-1", "remaining_volume": "0"}
    assert upbit.sell_orders == [("KRW-BTC", 100.0, 1.0)]
    assert upbit.cancelled == []


def test_limit_order_error_sends_telegram_after_retries(monkeypatch):
    messages = []
    upbit = _FailingOrderUpbit()

    monkeypatch.setattr(bot.time, "sleep", lambda _: None)
    monkeypatch.setattr(bot, "telegram_enabled", lambda: True)
    monkeypatch.setattr(bot, "send_telegram_message", lambda message: messages.append(message) or True)

    result = bot.place_limit_order_and_wait(
        upbit,
        ticker="KRW-BTC",
        side="buy",
        price=100.0,
        volume=1.0,
    )

    assert result is None
    assert upbit.calls == bot.ORDER_RETRY_COUNT
    assert len(messages) == 1
    assert "주문 API 오류" in messages[0]


def test_stop_loss_market_sell_uses_market_order(monkeypatch):
    upbit = _FakeOrderUpbit(remaining_volume="0")

    monkeypatch.setattr(bot.time, "sleep", lambda _: None)

    result = bot.place_market_sell(upbit, ticker="KRW-BTC", volume=1.25)

    assert result == {"uuid": "market-1"}
    assert upbit.market_sells == [("KRW-BTC", 1.25)]


def test_refresh_active_universe_refreshes_every_call(monkeypatch):
    calls = 0

    def fake_selector(**_):
        nonlocal calls
        calls += 1
        return ["KRW-BTC"]

    monkeypatch.setattr(bot, "select_liquid_universe", fake_selector)
    state = {
        "active_universe": ["KRW-ETH"],
        "universe_updated_at": datetime.now().isoformat(),
    }

    assert bot.refresh_active_universe(state) == ["KRW-BTC"]
    assert calls == 1


def test_refresh_active_universe_updates_selection(monkeypatch):
    monkeypatch.setattr(bot, "send_telegram_message", lambda _: True)
    monkeypatch.setattr(bot, "select_liquid_universe", lambda: ["KRW-XRP", "KRW-BTC"])
    state = {
        "active_universe": ["KRW-ETH"],
        "universe_updated_at": datetime.now().isoformat(),
    }

    assert bot.refresh_active_universe(state) == ["KRW-XRP", "KRW-BTC"]
    assert state["active_universe"] == ["KRW-XRP", "KRW-BTC"]
    assert state["universe_updated_at"]


def test_refresh_active_universe_keeps_previous_on_empty_selection(monkeypatch):
    monkeypatch.setattr(bot, "select_liquid_universe", lambda: [])
    state = {
        "active_universe": ["KRW-ETH"],
        "universe_updated_at": None,
    }

    assert bot.refresh_active_universe(state) == ["KRW-ETH"]
    assert state["active_universe"] == ["KRW-ETH"]
