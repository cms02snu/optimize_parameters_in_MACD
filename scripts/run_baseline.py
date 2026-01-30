"""Runner script for the Team 6 algorithmic trading project.

This script reproduces the original workflow:
1) Download last 10 years daily data for two tickers
2) Optimize MACD params for Custom1 and Custom2 under MDD constraints
3) Compare STD MACD vs Custom1 vs Custom2
4) Plot buy/sell points for each strategy

"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pandas as pd

from algtrading.data import download_ohlcv
from algtrading.optimize import optimize_macd_params_custom1, optimize_macd_params_custom2
from algtrading.strategy import compare
from algtrading.plotting import plot_macd_signals_3


TICKER1 = "102110.KS"  # TIGER ETF KOSPI 200
TICKER2 = "068270.KS"  # 셀트리온
INTERVAL = "1d"


def main() -> None:
    # 1) Download data
    df1 = download_ohlcv(TICKER1, interval=INTERVAL, years=10, auto_adjust=False)
    df2 = download_ohlcv(TICKER2, interval=INTERVAL, years=10, auto_adjust=False)

    # 2) Optimize parameters (same settings as the original submission)
    RANGE_FAST = range(5, 13, 1)
    RANGE_SLOW = range(18, 34, 2)
    RANGE_SIGNAL = range(3, 15, 1)
    ALPHA_BOUNDS = (0.05, 0.50)
    MDD_THRESHOLD = [-0.35, -0.59]

    best1_custom1, hist1_custom1 = optimize_macd_params_custom1(
        df1,
        fast_grid=RANGE_FAST,
        slow_grid=RANGE_SLOW,
        signal_grid=RANGE_SIGNAL,
        alpha_bounds=ALPHA_BOUNDS,
        fee_rate=0.0,
        price_col="Adj Close",
        top_k_refine=10,
        alpha_coarse_grid=5,
        mdd_threshold=MDD_THRESHOLD[0],
    )

    print("BEST PARAMS (Custom1, ETF):", best1_custom1)
    print(hist1_custom1.head(20))

    best2_custom1, hist2_custom1 = optimize_macd_params_custom1(
        df2,
        fast_grid=RANGE_FAST,
        slow_grid=RANGE_SLOW,
        signal_grid=RANGE_SIGNAL,
        alpha_bounds=ALPHA_BOUNDS,
        fee_rate=0.0,
        price_col="Adj Close",
        top_k_refine=10,
        alpha_coarse_grid=5,
        mdd_threshold=MDD_THRESHOLD[1],
    )

    print("BEST PARAMS (Custom1, STOCK):", best2_custom1)
    print(hist2_custom1.head(20))

    best1_custom2, hist1_custom2 = optimize_macd_params_custom2(
        df1,
        fast_grid=RANGE_FAST,
        slow_grid=RANGE_SLOW,
        signal_grid=RANGE_SIGNAL,
        alpha_bounds=ALPHA_BOUNDS,
        fee_rate=0.0,
        price_col="Adj Close",
        top_k_refine=10,
        alpha_coarse_grid=5,
        mdd_threshold=MDD_THRESHOLD[0],
    )

    print("BEST PARAMS (Custom2, ETF):", best1_custom2)
    print(hist1_custom2.head(20))

    best2_custom2, hist2_custom2 = optimize_macd_params_custom2(
        df2,
        fast_grid=RANGE_FAST,
        slow_grid=RANGE_SLOW,
        signal_grid=RANGE_SIGNAL,
        alpha_bounds=ALPHA_BOUNDS,
        fee_rate=0.0,
        price_col="Adj Close",
        top_k_refine=10,
        alpha_coarse_grid=5,
        mdd_threshold=MDD_THRESHOLD[1],
    )

    print("BEST PARAMS (Custom2, STOCK):", best2_custom2)
    print(hist2_custom2.head(20))

    # 3) Compare strategies
    res1 = compare(df1, best1_custom1, best1_custom2, name=TICKER1, mdd_th=MDD_THRESHOLD[0], fee_rate=0.0)
    res2 = compare(df2, best2_custom1, best2_custom2, name=TICKER2, mdd_th=MDD_THRESHOLD[1], fee_rate=0.0)

    result = pd.concat([res1, res2], ignore_index=True)
    print(result)

    # 4) Visualize buy/sell points (3 strategies each)
    plot_macd_signals_3(
        df1,
        best_custom1=best1_custom1,
        best_custom2=best1_custom2,
        price_col="Adj Close",
        title=f"{TICKER1} (ETF)",
        start="2017-01-01",
    )

    plot_macd_signals_3(
        df2,
        best_custom1=best2_custom1,
        best_custom2=best2_custom2,
        price_col="Adj Close",
        title=f"{TICKER2} (STOCK)",
        start="2017-01-01",
    )


if __name__ == "__main__":
    main()
