# btc_inv

업비트 KRW 마켓 대상 1시간봉 자동매매/백테스트 프로젝트입니다.  
실거래와 백테스트가 같은 [strategy.py](/Users/ht_mac_mini/Documents/dev/git_btc_inv/btc_inv-claude-add-claude-documentation-6Gajn/strategy.py)를 사용해서 신호 드리프트를 줄였습니다.

## 현재 전략 요약

- 타임프레임: `minute60`
- 유니버스: `KRW-USDT` 제외, 거래대금/거래량 증가 혼합 후보에서 상위 `10`개 선별
- 진입: `PRIMARY` 중심 눌림목 진입, BTC 시장 필터, VWAP/POC 가드 적용
- 청산: ATR 기반 손절, 부분 익절, 트레일링 익절, VWAP 약화 청산
- 포지션: 현재 설정 기준 동시 보유 최대 `1`개
- 알림: 시작 시, 매시 정각 상태, 종료 시 텔레그램 알림

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

`.env`에 아래 값을 채워야 합니다.

```env
UPBIT_ACCESS=...
UPBIT_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## 실행

실거래 실행:

```bash
.venv/bin/python bot.py
```

백테스트 실행:

```bash
.venv/bin/python backtest.py --days 365 --universe-mode daily
```

테스트:

```bash
.venv/bin/python -m pytest
PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m py_compile bot.py strategy.py universe.py backtest.py
```

## 주요 파일

- [bot.py](/Users/ht_mac_mini/Documents/dev/git_btc_inv/btc_inv-claude-add-claude-documentation-6Gajn/bot.py): 실거래 루프, 주문, 리스크 관리, 텔레그램 알림
- [strategy.py](/Users/ht_mac_mini/Documents/dev/git_btc_inv/btc_inv-claude-add-claude-documentation-6Gajn/strategy.py): 지표 계산, 진입/청산 규칙, 최종 튜닝 프로파일
- [universe.py](/Users/ht_mac_mini/Documents/dev/git_btc_inv/btc_inv-claude-add-claude-documentation-6Gajn/universe.py): 유니버스 후보 생성 및 필터링
- [backtest.py](/Users/ht_mac_mini/Documents/dev/git_btc_inv/btc_inv-claude-add-claude-documentation-6Gajn/backtest.py): 1년치 백테스트, 캐시 재사용, 포트폴리오 시뮬레이션

## 최근 검증 결과

최종 튜닝 프로파일 기준 최근 1년 백테스트 결과:

- 기간: `2025-05-05 ~ 2026-05-05`
- 총수익률: `+26.38%`
- MDD: `-6.75%`
- 거래 수: `39`
- 승률: `100.00%`

## 주의 사항

- 실거래 코드는 실제 주문을 전송합니다.
- 백테스트 캐시 파일(`data_cache_*`, `indicator_cache_*`)은 대용량이므로 보통 커밋하지 않습니다.
- 전략 파라미터는 [strategy.py](/Users/ht_mac_mini/Documents/dev/git_btc_inv/btc_inv-claude-add-claude-documentation-6Gajn/strategy.py)의 `FINAL_TUNED_PROFILE`에서 관리합니다.
