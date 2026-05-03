#!/usr/bin/env python3
"""
bot.py — 멀티코인 롱 온리 평균회귀 스캘핑 봇 (Upbit, 1분봉)

종목: KRW-BTC, KRW-ETH, KRW-XRP, KRW-SOL, KRW-ADA
전략: Long-only Mean Reversion Scalping with Fee Filter (strategy.py 참조)
리스크: 일일 -2% 거래 중지 / 5연속 손실 1시간 휴식
상태: state.json 영속화 (재시작 안전)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from urllib import parse, request

import pyupbit
from dotenv import load_dotenv

from strategy import (
    add_indicators,
    get_strategy_config,
    should_buy,
    should_sell,
    cooldown_until_after_loss,
    ROUND_TRIP_FEE,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS               = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]
INTERVAL              = "minute1"
PER_TRADE_RATIO       = 0.10    # 1회 진입 = 총 자산의 10%
MAX_CONCURRENT        = 3       # 동시 보유 최대 종목 수
DAILY_LOSS_LIMIT      = -0.02   # 일일 손실 -2% 도달 시 당일 거래 중지
CONSECUTIVE_LOSS_HALT = 5       # 연속 손실 N회 시 휴식
HALT_DURATION_MIN     = 60      # 휴식 시간(분)
MIN_ORDER_KRW         = 5_000
STATE_FILE            = "state.json"
TELEGRAM_INTERVAL_MIN = 60


# ── Upbit 인증 ─────────────────────────────────────────────────────────────────

def get_upbit() -> pyupbit.Upbit:
    access = os.getenv("UPBIT_ACCESS")
    secret = os.getenv("UPBIT_SECRET")
    if not access or not secret:
        raise EnvironmentError("UPBIT_ACCESS / UPBIT_SECRET 환경변수가 설정되지 않았습니다.")
    return pyupbit.Upbit(access, secret)


# ── 상태 영속화 ────────────────────────────────────────────────────────────────

def _empty_state(equity: float = 0.0) -> dict:
    return {
        "date":               datetime.now().strftime("%Y-%m-%d"),
        "daily_pnl_krw":      0.0,
        "consecutive_losses": 0,
        "halt_until":         None,
        "starting_equity":    equity,
        "positions":          {},
        "cooldowns":          {},
        "last_telegram_at":   None,
    }


def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return _empty_state()


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def reset_if_new_day(state: dict, current_equity: float) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        log.info(
            "새 거래일 — 리스크 카운터 초기화 (시작 자산: %s KRW)",
            f"{current_equity:,.0f}",
        )
        state.update({
            "date":               today,
            "daily_pnl_krw":      0.0,
            "consecutive_losses": 0,
            "halt_until":         None,
            "starting_equity":    current_equity,
        })


# ── 자산 / 잔고 헬퍼 ───────────────────────────────────────────────────────────

def coin_symbol(ticker: str) -> str:
    return ticker.split("-")[1]


def get_total_equity(upbit: pyupbit.Upbit) -> float:
    try:
        equity = float(upbit.get_balance("KRW") or 0)
        for ticker in TICKERS:
            bal = float(upbit.get_balance(coin_symbol(ticker)) or 0)
            if bal > 0:
                price = pyupbit.get_current_price(ticker) or 0
                equity += bal * price
        return equity
    except Exception as exc:
        log.warning("총 자산 계산 실패: %s", exc)
        return 0.0


# ── 텔레그램 알림 ──────────────────────────────────────────────────────────────

def telegram_enabled() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def send_telegram_message(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    try:
        req = request.Request(url, data=payload, method="POST")
        with request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        log.warning("텔레그램 알림 전송 실패: %s", exc)
        return False


def telegram_due(state: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now()
    last_sent = state.get("last_telegram_at")
    if not last_sent:
        return True
    try:
        return now - datetime.fromisoformat(last_sent) >= timedelta(minutes=TELEGRAM_INTERVAL_MIN)
    except ValueError:
        return True


def build_status_message(
    upbit: pyupbit.Upbit,
    state: dict,
    equity: float,
    *,
    risk_reason: str = "OK",
) -> str:
    lines = [
        "<b>btc_inv 상태 알림</b>",
        f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"총 자산: {equity:,.0f} KRW",
        f"일일 PnL: {float(state.get('daily_pnl_krw', 0.0)):+,.0f} KRW",
        f"리스크 상태: {risk_reason}",
        f"보유 포지션: {len(state.get('positions', {}))}개",
    ]

    positions = state.get("positions", {})
    if positions:
        for ticker, pos in positions.items():
            entry_price = float(pos.get("entry_price", 0) or 0)
            invest_krw = float(pos.get("invest_krw", 0) or 0)
            current_price = float(pyupbit.get_current_price(ticker) or 0)
            pnl_rate = (current_price / entry_price - 1) if entry_price > 0 and current_price > 0 else 0.0
            lines.append(
                f"- {ticker}: 진입 {entry_price:,.0f} / 현재 {current_price:,.0f} "
                f"/ PnL {pnl_rate*100:+.2f}% / 투자 {invest_krw:,.0f} KRW"
            )
    else:
        lines.append("- 보유 포지션 없음")

    halt_until = state.get("halt_until")
    if halt_until:
        lines.append(f"거래 중지 해제 예정: {halt_until}")
    return "\n".join(lines)


def maybe_send_hourly_status(
    upbit: pyupbit.Upbit,
    state: dict,
    equity: float,
    *,
    risk_reason: str = "OK",
) -> None:
    if not telegram_enabled() or not telegram_due(state):
        return

    message = build_status_message(upbit, state, equity, risk_reason=risk_reason)
    if send_telegram_message(message):
        state["last_telegram_at"] = datetime.now().isoformat()
        log.info("텔레그램 상태 알림 전송 완료")


def notify_trade_execution(
    *,
    side: str,
    ticker: str,
    price: float,
    invest_krw: float,
    reason: str,
    pnl_rate: float | None = None,
    pnl_krw: float | None = None,
) -> None:
    if not telegram_enabled():
        return

    lines = [
        f"<b>{side} 체결</b>",
        f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"종목: {ticker}",
        f"가격: {price:,.0f} KRW",
        f"금액: {invest_krw:,.0f} KRW",
        f"사유: {reason}",
    ]
    if pnl_rate is not None:
        lines.append(f"손익률: {pnl_rate*100:+.2f}%")
    if pnl_krw is not None:
        lines.append(f"손익금: {pnl_krw:+,.0f} KRW")

    send_telegram_message("\n".join(lines))


# ── 리스크 매니저 ──────────────────────────────────────────────────────────────

def risk_check(state: dict) -> tuple[bool, str]:
    # 연속 손실 휴식 체크
    if state.get("halt_until"):
        halt_dt = datetime.fromisoformat(state["halt_until"])
        if datetime.now() < halt_dt:
            remaining = int((halt_dt - datetime.now()).total_seconds() / 60)
            return False, f"연속 손실 휴식 중 (잔여 {remaining}분)"
        state["halt_until"] = None  # 만료 자동 해제

    # 일일 손실 한도 체크
    starting = state.get("starting_equity", 0.0)
    if starting > 0:
        loss_rate = state["daily_pnl_krw"] / starting
        if loss_rate <= DAILY_LOSS_LIMIT:
            return False, f"일일 손실 한도 도달 ({loss_rate*100:.2f}% / 한도 {DAILY_LOSS_LIMIT*100:.0f}%)"

    return True, "OK"


def record_trade_result(state: dict, ticker: str, pnl_krw: float) -> None:
    state["daily_pnl_krw"] += pnl_krw

    if pnl_krw < 0:
        state["consecutive_losses"] += 1
        cfg = get_strategy_config(ticker)
        state.setdefault("cooldowns", {})[ticker] = cooldown_until_after_loss(
            datetime.now(), cfg
        ).isoformat()
        if state["consecutive_losses"] >= CONSECUTIVE_LOSS_HALT:
            halt_dt = datetime.now() + timedelta(minutes=HALT_DURATION_MIN)
            state["halt_until"] = halt_dt.isoformat()
            log.warning(
                "연속 손실 %d회 달성 — %d분 거래 중지 (재개: %s)",
                state["consecutive_losses"],
                HALT_DURATION_MIN,
                halt_dt.strftime("%H:%M"),
            )
    else:
        state["consecutive_losses"] = 0
        state.setdefault("cooldowns", {}).pop(ticker, None)


# ── 데이터 수집 ────────────────────────────────────────────────────────────────

def fetch_df(ticker: str):
    for attempt in range(3):
        try:
            df = pyupbit.get_ohlcv(ticker, interval=INTERVAL, count=200)
            if df is not None and not df.empty:
                return add_indicators(df)
        except Exception as exc:
            log.warning("[%s] 데이터 수집 실패 (시도 %d): %s", ticker, attempt + 1, exc)
            time.sleep(3)
    return None


# ── 캔들 타이밍 ────────────────────────────────────────────────────────────────

def seconds_to_next_minute() -> float:
    now = datetime.now()
    elapsed = now.second + now.microsecond / 1e6
    return 60.0 - elapsed


# ── 메인 루프 ──────────────────────────────────────────────────────────────────

def main() -> None:
    upbit = get_upbit()
    log.info("봇 시작 — %s | %s", ", ".join(TICKERS), INTERVAL)

    while True:
        # 1분봉 종가 확정 후 5초 대기
        sleep_sec = seconds_to_next_minute() + 5
        log.info("다음 캔들까지 %.1f초 대기…", sleep_sec)
        time.sleep(sleep_sec)

        # ── 상태 로드 / 일자 리셋 ──────────────────────────────────────────────
        state = load_state()
        equity = get_total_equity(upbit)
        if state.get("starting_equity", 0) == 0 and equity > 0:
            state["starting_equity"] = equity
        reset_if_new_day(state, equity)

        # ── 리스크 체크 ────────────────────────────────────────────────────────
        ok, risk_reason = risk_check(state)
        if not ok:
            log.info("거래 중지: %s", risk_reason)
            maybe_send_hourly_status(upbit, state, equity, risk_reason=risk_reason)
            save_state(state)
            continue

        # ── 보유 포지션 청산 검토 ──────────────────────────────────────────────
        for ticker in list(state["positions"].keys()):
            pos = state["positions"][ticker]
            entry_price = float(pos["entry_price"])

            df = fetch_df(ticker)
            if df is None:
                continue

            try:
                current_price = float(pyupbit.get_current_price(ticker) or 0)
                if current_price <= 0:
                    continue
            except Exception as exc:
                log.warning("[%s] 현재가 조회 실패: %s", ticker, exc)
                continue

            c = df.iloc[-2]
            cfg = get_strategy_config(ticker)
            sell_ok, sell_reason = should_sell(
                entry_price,
                current_price,
                float(c["bb_mid"]),
                float(c["bb_lower"]),
                entry_time=pos.get("entry_time"),
                current_time=datetime.now(),
                config=cfg,
            )
            log.info(
                "[%s] 보유 | 진입=%s 현재=%s | %s",
                ticker, f"{entry_price:,.0f}", f"{current_price:,.0f}", sell_reason,
            )

            if sell_ok:
                try:
                    sym = coin_symbol(ticker)
                    bal = float(upbit.get_balance(sym) or 0)
                    if bal * current_price < MIN_ORDER_KRW:
                        log.warning("[%s] 잔고 부족으로 매도 불가 — 포지션 제거", ticker)
                        del state["positions"][ticker]
                        continue

                    upbit.sell_market_order(ticker, bal)

                    pnl_rate = (current_price / entry_price - 1) - ROUND_TRIP_FEE
                    invest_krw = float(pos.get("invest_krw", 0))
                    pnl_krw = invest_krw * pnl_rate

                    log.info(
                        "[%s] 매도 완료 — %s | PnL=%+.2f%% (%s KRW)",
                        ticker, sell_reason, pnl_rate * 100, f"{pnl_krw:+,.0f}",
                    )
                    notify_trade_execution(
                        side="매도",
                        ticker=ticker,
                        price=current_price,
                        invest_krw=invest_krw,
                        reason=sell_reason,
                        pnl_rate=pnl_rate,
                        pnl_krw=pnl_krw,
                    )
                    record_trade_result(state, ticker, pnl_krw)
                    del state["positions"][ticker]
                except Exception as exc:
                    log.error("[%s] 매도 주문 실패: %s", ticker, exc)

        # ── 신규 진입 검토 ─────────────────────────────────────────────────────
        concurrent = len(state["positions"])
        if concurrent >= MAX_CONCURRENT:
            log.info("동시 보유 한도(%d) 도달 — 신규 진입 건너뜀", MAX_CONCURRENT)
        else:
            try:
                krw_balance = float(upbit.get_balance("KRW") or 0)
            except Exception as exc:
                log.error("KRW 잔고 조회 실패: %s", exc)
                save_state(state)
                continue

            for ticker in TICKERS:
                if len(state["positions"]) >= MAX_CONCURRENT:
                    break
                if ticker in state["positions"]:
                    continue
                cfg = get_strategy_config(ticker)
                if not cfg.enabled:
                    log.info("[%s] validation gate disabled — 신규 진입 건너뜀", ticker)
                    continue

                df = fetch_df(ticker)
                if df is None:
                    continue

                cooldown_until = state.setdefault("cooldowns", {}).get(ticker)
                buy_ok, buy_reason = should_buy(
                    df,
                    ticker=ticker,
                    config=cfg,
                    cooldown_until=cooldown_until,
                )
                log.info("[%s] %s", ticker, buy_reason)

                if not buy_ok:
                    continue

                invest_krw = min(equity * PER_TRADE_RATIO, krw_balance)
                if invest_krw < MIN_ORDER_KRW:
                    log.warning("[%s] KRW 잔고 부족 (%.0f < %d)", ticker, invest_krw, MIN_ORDER_KRW)
                    send_telegram_message(
                        "\n".join(
                            [
                                "<b>매수 실패</b>",
                                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"종목: {ticker}",
                                f"사유: KRW 잔고 부족",
                                f"가용 금액: {invest_krw:,.0f} KRW",
                                f"최소 주문 금액: {MIN_ORDER_KRW:,.0f} KRW",
                            ]
                        )
                    )
                    continue

                try:
                    upbit.buy_market_order(ticker, invest_krw)
                    entry_price = float(
                        pyupbit.get_current_price(ticker) or df.iloc[-1]["close"]
                    )
                    state["positions"][ticker] = {
                        "entry_price": entry_price,
                        "entry_time":  datetime.now().isoformat(),
                        "invest_krw":  invest_krw,
                    }
                    krw_balance -= invest_krw
                    log.info(
                        "[%s] 매수 완료 | 투자=%s KRW | %s",
                        ticker, f"{invest_krw:,.0f}", buy_reason,
                    )
                    notify_trade_execution(
                        side="매수",
                        ticker=ticker,
                        price=entry_price,
                        invest_krw=invest_krw,
                        reason=buy_reason,
                    )
                except Exception as exc:
                    log.error("[%s] 매수 주문 실패: %s", ticker, exc)
                    send_telegram_message(
                        "\n".join(
                            [
                                "<b>매수 실패</b>",
                                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"종목: {ticker}",
                                f"사유: 매수 주문 실패",
                                f"주문 금액: {invest_krw:,.0f} KRW",
                                f"오류: {exc}",
                            ]
                        )
                    )

        # ── 상태 저장 ──────────────────────────────────────────────────────────
        save_state(state)
        log.info(
            "사이클 완료 | 자산=%s KRW | 일일PnL=%s KRW | 보유=%d종목 | 연속손실=%d",
            f"{equity:,.0f}",
            f"{state['daily_pnl_krw']:+,.0f}",
            len(state["positions"]),
            state["consecutive_losses"],
        )
        maybe_send_hourly_status(upbit, state, equity, risk_reason=risk_reason)
        save_state(state)


if __name__ == "__main__":
    main()
