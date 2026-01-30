from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    cum_max = equity.cummax()
    drawdown = equity / cum_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(daily_ret: pd.Series, ann_factor: int = 252) -> float:
    r = daily_ret.dropna().astype(float)
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return -np.inf
    return float((mu / sd) * np.sqrt(ann_factor))


def sharpe_from_events_with_mdd_constraint(
    close: pd.Series,
    event: pd.Series,
    *,
    fee_rate: float = 0.0,
    mdd_threshold: float = -0.8,
    ann_factor: int = 252,
) -> float:
    df = pd.DataFrame({"close": close.astype(float), "event": event.astype(int)}).dropna(subset=["close"])
    df["ret"] = df["close"].pct_change()

    target = pd.Series(np.nan, index=df.index, dtype=float)
    target[df["event"] == 1] = 1.0
    target[df["event"] == -1] = 0.0
    pos = target.shift(1).ffill().fillna(0.0)

    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * float(fee_rate)

    strat_ret = pos * df["ret"] - cost
    equity = (1.0 + strat_ret.fillna(0.0)).cumprod()

    mdd = max_drawdown(equity)
    if mdd < float(mdd_threshold):
        return -np.inf

    return sharpe_ratio(strat_ret, ann_factor=ann_factor)


def metrics_from_events_with_mdd_constraint(
    close: pd.Series,
    event: pd.Series,
    *,
    fee_rate: float = 0.0,
    mdd_threshold: float = -0.8,
    ann_factor: int = 252,
) -> dict:
    df = pd.DataFrame({"close": close.astype(float), "event": event.astype(int)}).dropna(subset=["close"])
    df["ret"] = df["close"].pct_change()

    target = pd.Series(np.nan, index=df.index, dtype=float)
    target[df["event"] == 1] = 1.0
    target[df["event"] == -1] = 0.0
    pos = target.shift(1).ffill().fillna(0.0)

    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * float(fee_rate)

    daily_ret = pos * df["ret"] - cost
    equity = (1.0 + daily_ret.fillna(0.0)).cumprod()

    mdd = float(max_drawdown(equity))
    sharpe = float(sharpe_ratio(daily_ret, ann_factor=ann_factor))
    sharpe_mdd = sharpe if (mdd >= float(mdd_threshold)) else -np.inf

    dpos = pos.diff().fillna(0.0)
    n_trades = int((dpos == 1.0).sum())

    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "sharpe_mdd": sharpe_mdd,
        "n_trades": n_trades,
    }


def event_to_pos_and_trades(event: pd.Series, index: pd.DatetimeIndex):
    event = event.reindex(index).fillna(0).astype(int)

    target = pd.Series(np.nan, index=index, dtype=float)
    target[event == 1] = 1.0
    target[event == -1] = 0.0

    pos = target.shift(1).ffill().fillna(0.0)

    dpos = pos.diff().fillna(0.0)
    buy_idx = index[dpos == 1.0]
    sell_idx = index[dpos == -1.0]

    return pos, buy_idx, sell_idx
