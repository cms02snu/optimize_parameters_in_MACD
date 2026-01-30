from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_ohlcv(
    ticker: str,
    *,
    interval: str = "1d",
    period: str = "max",
    years: int | None = 10,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data via yfinance and optionally keep only last `years`."""

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
        progress=False,
        threads=True,
    )

    df = df.dropna()

    # Drop MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if years is not None:
        end = df.index.max()
        start = end - pd.DateOffset(years=int(years))
        df = df.loc[df.index >= start].copy()

    return df
