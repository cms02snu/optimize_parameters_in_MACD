from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import build_macd_standard, build_macd_wema, macd_cross_events, wema
from .metrics import sharpe_from_events_with_mdd_constraint, metrics_from_events_with_mdd_constraint


def extrema_signal(
    close: pd.Series,
    fast_span: int,
    slow_span: int,
    signal_span: int,
    alpha: float,
) -> pd.Series:
    """Extrema-based buy/sell signals from the original submission.

    Returns a series with values:
    - 1: buy
    - -1: sell
    - 0: no signal
    """

    if fast_span is None or slow_span is None or signal_span is None:
        raise ValueError(
            "This strategy only supports finite-span EMA."
        )
    if fast_span >= slow_span:
        raise ValueError("fast_span must be smaller than slow_span.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")

    def ema_finite(series: pd.Series, alpha_: float, N: int) -> pd.Series:
        return wema(series, alpha=alpha_, N=N)

    ema_fast = ema_finite(close, alpha, fast_span)
    ema_slow = ema_finite(close, alpha, slow_span)

    macd_line = ema_fast - ema_slow
    signal_line = ema_finite(macd_line.dropna(), alpha, signal_span)
    signal_line = signal_line.reindex(macd_line.index)
    hist = macd_line - signal_line

    hist_prev = hist.shift(1)

    buy_cond = (macd_line < 0) & (hist_prev < 0) & (hist > 0)
    sell_cond = (macd_line > 0) & (hist_prev > 0) & (hist < 0)

    raw = pd.Series(0, index=close.index, dtype=int)
    raw[buy_cond] = 1
    raw[sell_cond] = -1

    # sequential filtering: buy -> sell -> buy ...
    sig = pd.Series(0, index=close.index, dtype=int)
    pos = 0
    for i in range(len(raw)):
        if raw.iloc[i] == 1 and pos == 0:
            sig.iloc[i] = 1
            pos = 1
        elif raw.iloc[i] == -1 and pos == 1:
            sig.iloc[i] = -1
            pos = 0

    nan_mask = hist_prev.isna()
    sig[nan_mask] = 0

    return sig


# Alias keeping the original Korean function name
극값기반_매매신호 = extrema_signal


def macd_strategy_score_custom1(
    df: pd.DataFrame,
    fast_N: int,
    slow_N: int,
    signal_N: int,
    alpha: float,
    *,
    price_col: str = "Adj Close",
    fee_rate: float = 0.0,
    mdd_threshold: float = -0.8,
) -> float:
    close = df[price_col].astype(float).copy().dropna()
    macd_df = build_macd_wema(close, fast_N=fast_N, slow_N=slow_N, signal_N=signal_N, alpha=alpha)
    event = macd_cross_events(macd_df).reindex(close.index).fillna(0).astype(int)
    return sharpe_from_events_with_mdd_constraint(close, event, fee_rate=fee_rate, mdd_threshold=mdd_threshold)


def macd_strategy_score_custom2(
    df: pd.DataFrame,
    fast_N: int,
    slow_N: int,
    signal_N: int,
    alpha: float,
    *,
    price_col: str = "Adj Close",
    fee_rate: float = 0.0,
    mdd_threshold: float = -0.8,
) -> float:
    close = df[price_col].astype(float).copy().dropna()
    event = extrema_signal(close, fast_N, slow_N, signal_N, alpha)
    return sharpe_from_events_with_mdd_constraint(close, event, fee_rate=fee_rate, mdd_threshold=mdd_threshold)


def standard_macd_metrics(
    df: pd.DataFrame,
    *,
    price_col: str = "Adj Close",
    fee_rate: float = 0.0,
    mdd_threshold: float = -0.8,
) -> dict:
    close = df[price_col].astype(float).dropna()
    macd_df = build_macd_standard(close, 12, 26, 9)
    event = macd_cross_events(macd_df).reindex(close.index).fillna(0).astype(int)
    return metrics_from_events_with_mdd_constraint(close, event, fee_rate=fee_rate, mdd_threshold=mdd_threshold)


def best_wema_macd_metrics(
    df: pd.DataFrame,
    best: dict,
    *,
    price_col: str = "Adj Close",
    fee_rate: float = 0.0,
    mdd_threshold: float = -0.8,
) -> dict:
    close = df[price_col].astype(float).dropna()
    macd_df = build_macd_wema(
        close,
        fast_N=int(best["fast_N"]),
        slow_N=int(best["slow_N"]),
        signal_N=int(best["signal_N"]),
        alpha=float(best["alpha"]),
    )
    event = macd_cross_events(macd_df).reindex(close.index).fillna(0).astype(int)
    return metrics_from_events_with_mdd_constraint(close, event, fee_rate=fee_rate, mdd_threshold=mdd_threshold)


def compare(
    df: pd.DataFrame,
    best_custom1: dict,
    best_custom2: dict,
    *,
    name: str,
    mdd_th: float,
    fee_rate: float = 0.0,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    met_std = standard_macd_metrics(df, fee_rate=fee_rate, mdd_threshold=mdd_th, price_col=price_col)
    met_best_custom1 = best_wema_macd_metrics(df, best_custom1, fee_rate=fee_rate, mdd_threshold=mdd_th, price_col=price_col)

    event = extrema_signal(df[price_col].astype(float), best_custom2["fast_N"], best_custom2["slow_N"], best_custom2["signal_N"], best_custom2["alpha"])
    met_best_custom2 = metrics_from_events_with_mdd_constraint(df[price_col].astype(float), event, fee_rate=fee_rate, mdd_threshold=mdd_th, ann_factor=252)

    return pd.DataFrame(
        [
            {"asset": name, "strategy": "STD MACD (12,26,9)", **met_std},
            {
                "asset": name,
                "strategy": f"Custom MACD 1 ({best_custom1['fast_N']},{best_custom1['slow_N']},{best_custom1['signal_N']}, alpha={best_custom1['alpha']:.4f})",
                **met_best_custom1,
            },
            {
                "asset": name,
                "strategy": f"Custom MACD 2 ({best_custom2['fast_N']},{best_custom2['slow_N']},{best_custom2['signal_N']}, alpha={best_custom2['alpha']:.4f})",
                **met_best_custom2,
            },
        ]
    )
