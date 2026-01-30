from __future__ import annotations

import numpy as np
import pandas as pd


def wema(series: pd.Series, alpha: float, N: int) -> pd.Series:
    """Weighted EMA used in the original submission."""

    x = series.astype(float).to_numpy()
    out = np.full_like(x, np.nan, dtype=float)

    w = alpha ** np.arange(N, dtype=float)  # length N
    w_sum = w.sum()

    y_valid = np.convolve(x, w[::-1], mode="valid") / w_sum
    out[N - 1 :] = y_valid

    return pd.Series(out, index=series.index)


def build_macd_wema(
    close: pd.Series, fast_N: int, slow_N: int, signal_N: int, alpha: float
) -> pd.DataFrame:
    """Custom MACD (WEMA) components."""

    ema_fast = wema(close, alpha=alpha, N=fast_N)
    ema_slow = wema(close, alpha=alpha, N=slow_N)
    macd_line = ema_fast - ema_slow
    ema_signal = wema(macd_line, alpha=alpha, N=signal_N)
    return pd.DataFrame(
        {"ema_fast": ema_fast, "ema_slow": ema_slow, "macd_line": macd_line, "ema_signal": ema_signal},
        index=close.index,
    )


def build_macd_standard(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Standard MACD using pandas ewm."""

    close = close.astype(float)
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    ema_signal = macd_line.ewm(span=int(signal), adjust=False).mean()
    return pd.DataFrame(
        {"ema_fast": ema_fast, "ema_slow": ema_slow, "macd_line": macd_line, "ema_signal": ema_signal},
        index=close.index,
    )


def macd_cross_events(
    macd_df: pd.DataFrame, col_macd: str = "macd_line", col_signal: str = "ema_signal"
) -> pd.Series:
    """Return +1/-1/0 events when MACD histogram changes sign."""

    macd = macd_df[col_macd].astype(float)
    sig = macd_df[col_signal].astype(float)
    hist = macd - sig

    sign = np.sign(hist.to_numpy())
    event = np.zeros_like(sign, dtype=int)

    prev = sign[:-1]
    curr = sign[1:]

    bullish = (prev <= 0) & (curr > 0)
    bearish = (prev >= 0) & (curr < 0)

    event[1:][bullish] = 1
    event[1:][bearish] = -1

    return pd.Series(event, index=macd_df.index, name="event")
