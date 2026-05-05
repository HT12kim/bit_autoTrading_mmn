#!/usr/bin/env python3
"""
bot.py — 실시간 거래대금 TOP 10 1시간봉 눌림목 전략 봇 (Upbit)

종목: KRW-USDT 제외 업비트 KRW 마켓 최근 1시간 거래대금 상위 10개
전략: BTC MA20/RSI 시장 필터 + BB 하단 재진입 + RSI30 재돌파 + 거래량 확인
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
    ROUND_TRIP_FEE,
    add_indicators,
    adaptive_stop_price,
    cooldown_until_after_loss,
    get_strategy_config,
    should_buy,
    should_sell,
)
from universe import (
    UNIVERSE_ATR_EXCLUDE_RATIO,
    UNIVERSE_BETA_MIN,
    UNIVERSE_BETA_TOP_N,
    UNIVERSE_DAILY_LOOKBACK_MINUTES,
    UNIVERSE_MIN_DAILY_QUOTE_KRW,
    UNIVERSE_NOISE_THRESHOLD,
    UNIVERSE_VOLUME_LOOKBACK_MINUTES,
    get_krw_tickers,
    price_noise_pct,
    rank_by_quote_volume,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
INTERVAL              = "minute60"
OHLCV_COUNT           = 3_000
PER_TRADE_RATIO       = 1.00    # 기본 진입 100%
PER_TRADE_RATIO_SPIKE = 1.00    # 거래량 스파이크 Primary 진입 100%
UNIVERSE_TOP_N        = 10
EXCLUDED_TICKERS      = {"KRW-USDT", "KRW-XRP", "KRW-SOL", "KRW-BTC", "KRW-DOGE"}
MAX_CONCURRENT        = 1       # 동시 보유 최대 종목 수
DAILY_LOSS_LIMIT      = -0.02   # 일일 손실 -2% 도달 시 당일 거래 중지
CONSECUTIVE_LOSS_HALT = 5       # 연속 손실 N회 시 휴식
HALT_DURATION_MIN     = 60      # 휴식 시간(분)
MIN_ORDER_KRW         = 5_000
LIMIT_ORDER_TIMEOUT_SEC = 10
ORDER_RETRY_COUNT     = 3
STATE_FILE            = "state.json"


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
        "active_universe":    [],
        "universe_updated_at": None,
        "last_telegram_slot": None,
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


def ensure_state_schema(state: dict) -> dict:
    state.setdefault("positions", {})
    state.setdefault("cooldowns", {})
    state.setdefault("active_universe", [])
    state.setdefault("universe_updated_at", None)
    state.setdefault("last_telegram_slot", state.pop("last_telegram_at", None))
    if not isinstance(state["active_universe"], list):
        state["active_universe"] = []
    return state


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


def get_position_balance(upbit: pyupbit.Upbit, ticker: str) -> float:
    try:
        return float(upbit.get_balance(coin_symbol(ticker)) or 0)
    except Exception as exc:
        log.warning("[%s] 잔고 조회 실패: %s", ticker, exc)
        return 0.0


def reconcile_positions_with_balances(upbit: pyupbit.Upbit, state: dict) -> list[str]:
    """실제 잔고가 없는 state 포지션을 제거해 stale 포지션으로 인한 거래 중단을 막는다."""
    ensure_state_schema(state)
    removed: list[str] = []
    for ticker in list(state["positions"].keys()):
        try:
            current_price = float(pyupbit.get_current_price(ticker) or 0)
        except Exception as exc:
            log.warning("[%s] 현재가 조회 실패 — 포지션 동기화 보류: %s", ticker, exc)
            continue
        if current_price <= 0:
            continue

        balance = get_position_balance(upbit, ticker)
        if balance * current_price >= MIN_ORDER_KRW:
            continue

        del state["positions"][ticker]
        removed.append(ticker)
        log.warning(
            "[%s] 실제 보유 잔고 없음 — stale 포지션을 state에서 제거",
            ticker,
        )
    return removed


def get_total_equity(upbit: pyupbit.Upbit) -> float:
    try:
        equity = 0.0
        balances = upbit.get_balances()
        krw_tickers = set(pyupbit.get_tickers(fiat="KRW") or [])
        current_prices = pyupbit.get_current_price(list(krw_tickers)) or {}
        for item in balances:
            currency = item.get("currency")
            balance = float(item.get("balance") or 0)
            if balance <= 0 or not currency:
                continue
            if currency == "KRW":
                equity += balance
                continue
            ticker = f"KRW-{currency}"
            if ticker not in krw_tickers:
                log.debug("[%s] KRW 마켓 미지원 자산 — 총자산 계산에서 제외", ticker)
                continue
            price = float(current_prices.get(ticker) or 0)
            if price <= 0:
                log.debug("[%s] 현재가 없음 — 총자산 계산에서 제외", ticker)
                continue
            equity += balance * price
        return equity
    except Exception as exc:
        log.warning("총 자산 계산 실패: %s", exc)
        return 0.0


def refresh_active_universe(state: dict) -> list[str]:
    ensure_state_schema(state)
    current_universe = list(state.get("active_universe") or [])

    try:
        selected = select_liquid_universe()
    except Exception as exc:
        log.warning("동적 유니버스 선정 실패 — 직전 목록 재사용: %s", exc)
        return current_universe

    if not selected:
        log.warning("동적 유니버스 선정 결과 없음 — 신규 진입 중단")
        return current_universe

    now = datetime.now().isoformat()
    state["active_universe"] = selected
    state["universe_updated_at"] = now

    if selected != current_universe:
        log.info(
            "유니버스 갱신 — 최근 %d분 거래대금 상위 %d: %s",
            UNIVERSE_VOLUME_LOOKBACK_MINUTES,
            len(selected),
            ", ".join(selected),
        )
    return selected


def select_atr_filtered_universe() -> list[str]:
    """Backward-compatible alias for older tests/scripts."""
    return select_liquid_universe()


def select_liquid_universe() -> list[str]:
    tickers = [ticker for ticker in get_krw_tickers() if ticker not in EXCLUDED_TICKERS]
    ranked_24h = rank_by_quote_volume(
        tickers,
        lookback_minutes=UNIVERSE_DAILY_LOOKBACK_MINUTES,
    )
    liquid = [(ticker, volume) for ticker, volume in ranked_24h if volume >= UNIVERSE_MIN_DAILY_QUOTE_KRW]
    top_volume = liquid[:30]

    surge: list[tuple[str, float]] = []
    for ticker, _ in top_volume:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=48)
        if df is None or len(df) < 48:
            continue
        prev_day = float((df.iloc[-48:-24]["close"] * df.iloc[-48:-24]["volume"]).sum())
        curr_day = float((df.iloc[-24:]["close"] * df.iloc[-24:]["volume"]).sum())
        if prev_day <= 0:
            continue
        surge.append((ticker, (curr_day - prev_day) / prev_day))
    surge.sort(key=lambda item: item[1], reverse=True)

    mix_score: dict[str, float] = {}
    for idx, (ticker, _) in enumerate(top_volume):
        mix_score[ticker] = mix_score.get(ticker, 0.0) + (len(top_volume) - idx)
    for idx, (ticker, _) in enumerate(surge):
        mix_score[ticker] = mix_score.get(ticker, 0.0) + (len(surge) - idx)

    # v8.0: RSI 상승 기울기가 가파른 상위 5종목에 가중치를 부여한다.
    rsi_momentum: list[tuple[str, float]] = []
    for ticker, _ in top_volume:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=64)
        if df is None or len(df) < 20:
            continue
        enriched = add_indicators(df)
        rsi = enriched["rsi"].dropna()
        if len(rsi) < 4:
            continue
        rsi_momentum.append((ticker, float(rsi.iloc[-1] - rsi.iloc[-4])))
    rsi_momentum.sort(key=lambda item: item[1], reverse=True)
    for idx, (ticker, _) in enumerate(rsi_momentum[:5]):
        mix_score[ticker] = mix_score.get(ticker, 0.0) + (5 - idx)
    mixed = [ticker for ticker, _ in sorted(mix_score.items(), key=lambda item: item[1], reverse=True)[:30]]

    # v10.0: Noise Ratio < 0.55 인 종목만 유니버스로 유지
    noise_filtered: list[str] = []
    for ticker in mixed:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=24)
        if df is None or df.empty:
            continue
        if price_noise_pct(df) < UNIVERSE_NOISE_THRESHOLD:
            noise_filtered.append(ticker)
    if noise_filtered:
        mixed = noise_filtered

    # B안: 혼합 후보군에서 ATR% 상위 10%를 제외해 급변동 리스크를 완화한다.
    atr_rank: list[tuple[str, float]] = []
    for ticker in mixed:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=48)
        if df is None or len(df) < 24:
            continue
        enriched = add_indicators(df)
        atr_pct = float(enriched["atr_pct"].tail(24).mean())
        if atr_pct > 0:
            atr_rank.append((ticker, atr_pct))
    if atr_rank:
        remove_count = max(1, int(len(atr_rank) * UNIVERSE_ATR_EXCLUDE_RATIO))
        high_atr = {ticker for ticker, _ in sorted(atr_rank, key=lambda item: item[1], reverse=True)[:remove_count]}
        filtered = [ticker for ticker in mixed if ticker not in high_atr]
        if filtered:
            mixed = filtered

    # v10.0: BTC 대비 24시간 수익률 배수(Beta) 상위 종목 우선
    btc_df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=25)
    btc_ret = 0.0
    if btc_df is not None and len(btc_df) >= 25:
        btc_ret = float(btc_df["close"].iloc[-1] / btc_df["close"].iloc[0] - 1)
    beta_rank: list[tuple[str, float]] = []
    for ticker in mixed:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=25)
        if df is None or len(df) < 25:
            continue
        ret = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
        beta = ret / btc_ret if abs(btc_ret) > 1e-9 else 0.0
        beta_rank.append((ticker, beta))
    beta_rank.sort(key=lambda item: item[1], reverse=True)
    selected = [ticker for ticker, beta in beta_rank if beta >= UNIVERSE_BETA_MIN][:UNIVERSE_BETA_TOP_N]
    if not selected:
        selected = [ticker for ticker, _ in beta_rank[:UNIVERSE_BETA_TOP_N]]
    if selected:
        return selected
    return mixed[:UNIVERSE_TOP_N]


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
    if now.minute not in (0, 30):
        return False

    # 루프가 정각/30분 캔들 이후 여러 번 실행되어도 같은 슬롯에는 1회만 전송한다.
    current_slot = now.strftime("%Y-%m-%d %H:%M")
    return state.get("last_telegram_slot") != current_slot


def build_status_message(
    upbit: pyupbit.Upbit,
    state: dict,
    equity: float,
    *,
    risk_reason: str = "OK",
) -> str:
    starting_equity = float(state.get("starting_equity", 0.0) or 0.0)
    daily_pnl_krw = float(state.get("daily_pnl_krw", 0.0) or 0.0)
    daily_pnl_rate = daily_pnl_krw / starting_equity if starting_equity > 0 else 0.0
    universe = ", ".join(state.get("active_universe") or []) or "없음"

    lines = [
        "<b>btc_inv 30분 현황</b>",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"자산: {equity:,.0f} KRW",
        f"일일PnL: {daily_pnl_krw:+,.0f} KRW ({daily_pnl_rate*100:+.2f}%)",
        f"상태: {risk_reason}",
        f"후보: {universe}",
        f"보유: {len(state.get('positions', {}))}개",
    ]

    positions = state.get("positions", {})
    if positions:
        for ticker, pos in positions.items():
            entry_price = float(pos.get("entry_price", 0) or 0)
            current_price = float(pyupbit.get_current_price(ticker) or 0)
            pnl_rate = (current_price / entry_price - 1) if entry_price > 0 and current_price > 0 else 0.0
            lines.append(
                f"- {ticker}: {pnl_rate*100:+.2f}% / 진입 {entry_price:,.0f} / 현재 {current_price:,.0f}"
            )
    return "\n".join(lines)


def build_startup_strategy_message() -> str:
    """봇 프로세스 시작 시 현재 적용 중인 전략을 1회 안내한다."""
    return "\n".join(
        [
            "<b>btc_inv 봇 시작</b>",
            f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"적용 전략: {strategy_summary()}",
            f"유니버스: KRW-USDT 제외 거래대금 TOP {UNIVERSE_TOP_N} + 전일대비 거래량 증가 혼합 (24h 500억 이상)",
            "지표: 1시간봉 MA20/5/60/120일, BB20/2, RSI14, ATR14",
            "진입: BTC MA20 상회 또는 RSI40 초과 + (BB/RSI30 또는 MA20/RSI45) + 직전 1봉 고가 돌파, RSI65 이하",
            "포지션 크기: 종목당 예수금 12.5% 고정, 최대 8개 분산",
            "주문: 현재가 기준 지정가 우선, 10초 미체결분 취소",
            "청산: +3% 50% 부분익절, 이후 RSI 70 상향후 하향 시 잔량 전량 / 고점대비 -3% 트레일링 / 24봉 손익 +1% 미만 Time-Stop / entry-2*ATR",
            f"동시 보유 한도: 최대 {MAX_CONCURRENT}개",
            f"리스크 제한: 일일 손실 {DAILY_LOSS_LIMIT*100:.0f}% 도달 시 중지 / "
            f"{CONSECUTIVE_LOSS_HALT}연속 손실 시 {HALT_DURATION_MIN}분 휴식",
        ]
    )


def strategy_summary() -> str:
    return (
        f"KRW-USDT 제외 거래대금 상위 {UNIVERSE_TOP_N}개, "
        "TOP30+거래량증가 혼합/BTC 필터/듀얼 엔트리, "
        f"ATR 1% 리스크/예수금 {PER_TRADE_RATIO*100:.0f}% 상한, 최대 {MAX_CONCURRENT}개"
    )


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
        state["last_telegram_slot"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        log.info("텔레그램 정기 알림 전송 완료")


def build_trade_message(
    *,
    side: str,
    ticker: str,
    price: float,
    invest_krw: float,
    reason: str,
    pnl_rate: float | None = None,
    pnl_krw: float | None = None,
) -> str:
    lines = [
        f"<b>{side} 체결</b>",
        f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"종목: {ticker}",
        f"지정가 체결 가격: {price:,.0f} KRW",
        f"금액: {invest_krw:,.0f} KRW",
        f"사유: {reason}",
    ]
    if pnl_rate is not None:
        lines.append(f"손익률: {pnl_rate*100:+.2f}%")
    if pnl_krw is not None:
        lines.append(f"손익금: {pnl_krw:+,.0f} KRW")
    return "\n".join(lines)


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
    send_telegram_message(
        build_trade_message(
            side=side,
            ticker=ticker,
            price=price,
            invest_krw=invest_krw,
            reason=reason,
            pnl_rate=pnl_rate,
            pnl_krw=pnl_krw,
        )
    )


def notify_order_error(ticker: str, action: str, error: Exception | str) -> None:
    log.error("[%s] %s 최종 실패: %s", ticker, action, error)
    if telegram_enabled():
        send_telegram_message(
            "\n".join(
                [
                    "<b>주문 API 오류</b>",
                    f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"종목: {ticker}",
                    f"작업: {action}",
                    f"오류: {error}",
                ]
            )
        )


def build_shutdown_message(state: dict | None = None) -> str:
    positions = (state or {}).get("positions", {}) if isinstance(state, dict) else {}
    return "\n".join(
        [
            "<b>btc_inv 봇 종료</b>",
            f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "사유: 사용자 종료 요청(Ctrl+C)",
            f"보유 state 포지션: {len(positions)}개",
        ]
    )


def send_shutdown_notice(state: dict | None = None) -> None:
    if not telegram_enabled():
        return
    if send_telegram_message(build_shutdown_message(state)):
        log.info("텔레그램 종료 알림 전송 완료")


def send_startup_strategy_notice() -> None:
    if not telegram_enabled():
        return
    if send_telegram_message(build_startup_strategy_message()):
        log.info("텔레그램 시작 전략 안내 전송 완료")


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
            df = pyupbit.get_ohlcv(ticker, interval=INTERVAL, count=OHLCV_COUNT)
            if df is not None and not df.empty:
                return add_indicators(df)
        except Exception as exc:
            log.warning("[%s] 데이터 수집 실패 (시도 %d): %s", ticker, attempt + 1, exc)
            time.sleep(3)
    return None


# ── 지정가 주문 헬퍼 ───────────────────────────────────────────────────────────

def retry_api(action: str, ticker: str, func):
    last_error: Exception | None = None
    for attempt in range(ORDER_RETRY_COUNT):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            log.warning("[%s] %s 실패 (시도 %d): %s", ticker, action, attempt + 1, exc)
            time.sleep(1 + attempt)
    notify_order_error(ticker, action, last_error or "unknown error")
    return None


def _order_uuid(order: dict | str | None) -> str | None:
    if isinstance(order, str):
        return order
    if isinstance(order, dict):
        value = order.get("uuid")
        return str(value) if value else None
    return None


def _remaining_volume(order_info) -> float:
    if isinstance(order_info, list):
        return sum(_remaining_volume(item) for item in order_info)
    if not isinstance(order_info, dict):
        return 0.0
    try:
        return float(order_info.get("remaining_volume") or 0)
    except (TypeError, ValueError):
        return 0.0


def place_limit_order_and_wait(
    upbit: pyupbit.Upbit,
    *,
    ticker: str,
    side: str,
    price: float,
    volume: float,
    timeout_sec: int = LIMIT_ORDER_TIMEOUT_SEC,
) -> dict | None:
    """현재가 지정가 주문을 넣고 10초 뒤 미체결분을 취소한다."""
    if price <= 0 or volume <= 0:
        notify_order_error(ticker, f"{side} 지정가 주문", "invalid price or volume")
        return None

    def submit():
        if side == "buy":
            return upbit.buy_limit_order(ticker, price, volume)
        if side == "sell":
            return upbit.sell_limit_order(ticker, price, volume)
        raise ValueError(f"unsupported side: {side}")

    order = retry_api(f"{side} 지정가 주문", ticker, submit)
    uuid = _order_uuid(order)
    if not uuid:
        if order is not None:
            notify_order_error(ticker, f"{side} 지정가 주문", f"uuid 없음: {order}")
        return None

    time.sleep(timeout_sec)
    info = retry_api(f"{side} 주문 조회", ticker, lambda: upbit.get_order(uuid))
    if info is None:
        return None

    remaining = _remaining_volume(info)
    if remaining > 0:
        cancel_result = retry_api(f"{side} 미체결 취소", ticker, lambda: upbit.cancel_order(uuid))
        if cancel_result is None:
            return None
        log.info("[%s] %s 지정가 주문 미체결 %.8f 취소", ticker, side, remaining)
        return None

    return info if isinstance(info, dict) else {"uuid": uuid, "remaining_volume": "0"}


def place_market_sell(upbit: pyupbit.Upbit, *, ticker: str, volume: float) -> dict | None:
    """손절/데드크로스/타임컷은 체결 우선순위가 높으므로 시장가로 청산한다."""
    if volume <= 0:
        notify_order_error(ticker, "손절 시장가 매도", "invalid volume")
        return None
    result = retry_api("손절 시장가 매도", ticker, lambda: upbit.sell_market_order(ticker, volume))
    return result if isinstance(result, dict) else {"result": result}


# ── 캔들 타이밍 ────────────────────────────────────────────────────────────────

def seconds_to_next_hour() -> float:
    now = datetime.now()
    elapsed = (now.minute * 60) + now.second + now.microsecond / 1e6
    return 3600.0 - elapsed


# ── 메인 루프 ──────────────────────────────────────────────────────────────────

def main() -> None:
    upbit = get_upbit()
    state: dict | None = None
    log.info(
        "봇 시작 — KRW 마켓 최근 %d분 거래대금 상위 %d | %s | 매 루프 갱신",
        UNIVERSE_VOLUME_LOOKBACK_MINUTES,
        UNIVERSE_TOP_N,
        INTERVAL,
    )
    send_startup_strategy_notice()

    try:
        while True:
            # 1시간봉 종가 확정 후 5초 대기
            sleep_sec = seconds_to_next_hour() + 5
            log.info("다음 캔들까지 %.1f초 대기…", sleep_sec)
            time.sleep(sleep_sec)

            # ── 상태 로드 / 일자 리셋 ──────────────────────────────────────────────
            state = ensure_state_schema(load_state())
            equity = get_total_equity(upbit)
            if state.get("starting_equity", 0) == 0 and equity > 0:
                state["starting_equity"] = equity
            reset_if_new_day(state, equity)
            removed_positions = reconcile_positions_with_balances(upbit, state)
            if removed_positions:
                log.info("stale 포지션 정리 완료: %s", ", ".join(removed_positions))

            # ── 리스크 체크 ────────────────────────────────────────────────────────
            ok, risk_reason = risk_check(state)
            if not ok:
                log.info("거래 중지: %s", risk_reason)
                maybe_send_hourly_status(upbit, state, equity, risk_reason=risk_reason)
                save_state(state)
                continue

            active_universe = refresh_active_universe(state)
            btc_df = fetch_df("KRW-BTC")

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
                p = df.iloc[-3] if len(df) >= 3 else None
                cfg = get_strategy_config(ticker)
                sell_ok, sell_reason, armed, highest, overbought_seen, bb_break_seen = should_sell(
                    entry_price,
                    current_price,
                    row=c,
                    previous=p,
                    entry_time=pos.get("entry_time"),
                    current_time=c.name,
                    partial_taken=bool(pos.get("partial_taken", False)),
                    bb_break_seen=bool(pos.get("bb_break_seen", False)),
                    overbought_seen=bool(pos.get("overbought_seen", False)),
                    config=cfg,
                    take_profit_armed=bool(pos.get("take_profit_armed", False)),
                    highest_price=float(pos.get("highest_price", entry_price) or entry_price),
                )
                pos["take_profit_armed"] = armed
                pos["highest_price"] = highest
                pos["overbought_seen"] = overbought_seen
                pos["bb_break_seen"] = bb_break_seen
                pos["stop_price"] = adaptive_stop_price(entry_price, c, cfg)
                log.info(
                    "[%s] 보유 | 진입=%s 현재=%s | %s",
                    ticker,
                    f"{entry_price:,.0f}",
                    f"{current_price:,.0f}",
                    sell_reason,
                )

                if sell_ok:
                    sym = coin_symbol(ticker)
                    bal = retry_api("매도 잔고 조회", ticker, lambda: float(upbit.get_balance(sym) or 0))
                    if bal is None:
                        continue
                    if bal * current_price < MIN_ORDER_KRW:
                        log.warning("[%s] 잔고 부족으로 매도 불가 — 포지션 제거", ticker)
                        del state["positions"][ticker]
                        continue

                    if sell_reason.startswith("PARTIAL_TAKE_PROFIT_50"):
                        sell_volume = float(bal) * 0.5
                        if sell_volume * current_price < MIN_ORDER_KRW:
                            continue
                        fill = place_market_sell(upbit, ticker=ticker, volume=sell_volume)
                        if fill is None:
                            continue
                        invest_krw = float(pos.get("invest_krw", 0))
                        partial_invest = invest_krw * 0.5
                        pnl_rate = (current_price / entry_price - 1) - ROUND_TRIP_FEE
                        pnl_krw = partial_invest * pnl_rate
                        pos["invest_krw"] = partial_invest
                        pos["partial_taken"] = True
                        pos["take_profit_armed"] = True
                        notify_trade_execution(
                            side="부분매도",
                            ticker=ticker,
                            price=current_price,
                            invest_krw=partial_invest,
                            reason=sell_reason,
                            pnl_rate=pnl_rate,
                            pnl_krw=pnl_krw,
                        )
                        record_trade_result(state, ticker, pnl_krw)
                        continue
                    if sell_reason.startswith("PARTIAL_TAKE_PROFIT_25_BB_REV"):
                        sell_volume = float(bal) * 0.5
                        if sell_volume * current_price < MIN_ORDER_KRW:
                            continue
                        fill = place_market_sell(upbit, ticker=ticker, volume=sell_volume)
                        if fill is None:
                            continue
                        invest_krw = float(pos.get("invest_krw", 0))
                        partial_invest = invest_krw * 0.5
                        pnl_rate = (current_price / entry_price - 1) - ROUND_TRIP_FEE
                        pnl_krw = partial_invest * pnl_rate
                        pos["invest_krw"] = partial_invest
                        notify_trade_execution(
                            side="부분매도",
                            ticker=ticker,
                            price=current_price,
                            invest_krw=partial_invest,
                            reason=sell_reason,
                            pnl_rate=pnl_rate,
                            pnl_krw=pnl_krw,
                        )
                        record_trade_result(state, ticker, pnl_krw)
                        continue

                    if sell_reason.startswith(("STOP_LOSS", "MA5_MA20_DEAD_CROSS", "TIME_EXIT")):
                        fill = place_market_sell(upbit, ticker=ticker, volume=float(bal))
                    else:
                        fill = place_limit_order_and_wait(
                            upbit,
                            ticker=ticker,
                            side="sell",
                            price=current_price,
                            volume=float(bal),
                        )
                    if fill is None:
                        continue

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

            # ── 신규 진입 검토 ─────────────────────────────────────────────────────
            concurrent = len(state["positions"])
            if concurrent >= MAX_CONCURRENT:
                log.info("동시 보유 한도(%d) 도달 — 신규 진입 건너뜀", MAX_CONCURRENT)
            elif not active_universe:
                log.warning("활성 유니버스 없음 — 신규 진입 건너뜀")
            else:
                try:
                    krw_balance = float(upbit.get_balance("KRW") or 0)
                except Exception as exc:
                    log.error("KRW 잔고 조회 실패: %s", exc)
                    save_state(state)
                    continue

                for ticker in active_universe:
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
                        market_df=btc_df,
                    )
                    log.info("[%s] %s", ticker, buy_reason)

                    if not buy_ok:
                        continue

                    current_price = retry_api(
                        "매수 현재가 조회",
                        ticker,
                        lambda: float(pyupbit.get_current_price(ticker) or 0),
                    )
                    if current_price is None or current_price <= 0:
                        notify_order_error(ticker, "매수 현재가 조회", "current price unavailable")
                        continue

                    last_closed = df.iloc[-2]
                    stop_price = adaptive_stop_price(current_price, last_closed, cfg)
                    trade_ratio = PER_TRADE_RATIO_SPIKE if "ENTRY:PRIMARY_SPIKE" in buy_reason else PER_TRADE_RATIO
                    invest_krw = min(krw_balance, equity * trade_ratio)
                    volume = invest_krw / current_price if current_price > 0 else 0.0
                    if invest_krw < MIN_ORDER_KRW:
                        log.warning("[%s] KRW 잔고 부족 (%.0f < %d)", ticker, invest_krw, MIN_ORDER_KRW)
                        continue

                    fill = place_limit_order_and_wait(
                        upbit,
                        ticker=ticker,
                        side="buy",
                        price=current_price,
                        volume=volume,
                    )
                    if fill is not None:
                        state["positions"][ticker] = {
                            "entry_price": current_price,
                            "entry_time":  datetime.now().isoformat(),
                            "invest_krw":  invest_krw,
                            "atr_at_entry": float(last_closed.get("atr14", 0) or 0),
                            "stop_price": stop_price,
                            "highest_price": current_price,
                            "take_profit_armed": False,
                            "partial_taken": False,
                            "overbought_seen": False,
                            "bb_break_seen": False,
                        }
                        krw_balance -= invest_krw
                        log.info(
                            "[%s] 매수 완료 | 투자=%s KRW | %s",
                            ticker, f"{invest_krw:,.0f}", buy_reason,
                        )
                        notify_trade_execution(
                            side="매수",
                            ticker=ticker,
                            price=current_price,
                            invest_krw=invest_krw,
                            reason=buy_reason,
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
    except KeyboardInterrupt:
        log.info("사용자 종료 요청(Ctrl+C) — 봇을 종료합니다.")
        if state is not None:
            save_state(state)
        send_shutdown_notice(state)


if __name__ == "__main__":
    main()
