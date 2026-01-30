# 알고리듬 트레이딩 6조 (MACD 변형 + 파라미터 최적화)

이 레포는 **표준 MACD(12,26,9)** 와, 조별 과제에서 구현한 두 가지 변형 전략을 비교/검증합니다.

- **STD MACD**: EMA 기반 MACD 라인과 시그널 라인의 **교차(cross)** 매매
- **Custom MACD 1**: (fast/slow/signal) 유한기간 EMA를 **WEMA(가중 EMA)** 로 계산 후 **교차(cross)** 매매 + **alpha(0~1) 튜닝**
- **Custom MACD 2**: MACD를 미분 근사로 해석하여 **극값(극소/극대) 기반** 매매 신호 생성 + **alpha 튜닝**

---

## 폴더 구조

```
.
├─ scripts/
│  └─ run_baseline.py
├─ src/
│  └─ algtrading/
│     ├─ data.py
│     ├─ indicators.py
│     ├─ metrics.py
│     ├─ optimize.py
│     ├─ plotting.py
│     └─ strategy.py
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

---

## 실행 방법

### 1) 환경 세팅

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

> 인터넷이 필요합니다(데이터는 `yfinance`로 다운로드).

### 2) 실험/재현 실행

```bash
python scripts/run_baseline.py
```

실행 결과:
- 두 종목(ETF 1개 + 주식 1개)의 **최근 10년 일봉 데이터**를 다운로드
- MDD 제약 하에서 (fast, slow, signal, alpha) 최적 탐색
- STD vs Custom1 vs Custom2의 성능 지표(Sharpe, MDD, 제약Sharpe, 거래횟수) 출력
- 각 종목에 대해 **3개 전략의 매수/매도 시점**을 3개 패널로 시각화

---

## 기본 설정(제출본과 동일)

`scripts/run_baseline.py` 기준으로 제출본과 동일하게 아래 설정을 사용합니다.

- 대상 티커
  - `102110.KS` : TIGER ETF KOSPI 200
  - `068270.KS` : 셀트리온
- 데이터 구간: 최근 10년 (일봉)
- 탐색 범위
  - `fast` : 5~12
  - `slow` : 18~32 (step=2)
  - `signal` : 3~14
  - `alpha` : 0.05~0.50
- MDD 제약
  - ETF: `-0.35`
  - 주식: `-0.59`

---

## 전략 설명 (요약)

### STD MACD (12,26,9)
- EMA(12), EMA(26) 차이를 MACD 라인으로 사용
- MACD 라인과 시그널 EMA(9) 교차 시 매수/매도

### Custom MACD 1 (WEMA + Cross)
- 표준 EMA 대신 **유한기간 WEMA**로 EMA를 계산
- (fast, slow, signal, alpha) 를 함께 최적화
- 교차 이벤트(+1/-1)를 이용해 매매

### Custom MACD 2 (극값 기반)
- MACD 라인을 1차 미분(속도), Histogram을 2차 미분(가속도) 근사로 해석
- **역배열(MACD<0)** 에서 Histogram이 음→양 전환: 극소점 매수
- **정배열(MACD>0)** 에서 Histogram이 양→음 전환: 극대점 매도
- 매수/매도 신호는 순차적으로 필터링(매수 후 매도만 가능)

---

## 주의사항

- `yfinance`는 네트워크 상태/요청 제한에 따라 다운로드가 실패할 수 있습니다.
- 최적화는 (fast × slow × signal × alpha_grid) 조합을 평가하므로, 탐색 범위를 크게 잡으면 시간이 오래 걸립니다.
