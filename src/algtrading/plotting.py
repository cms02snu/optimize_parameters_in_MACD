from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .indicators import build_macd_standard, build_macd_wema, macd_cross_events
from .metrics import event_to_pos_and_trades
from .strategy import extrema_signal


def plot_macd_signals_3(
    df: pd.DataFrame,
    best_custom1: dict,
    best_custom2: dict,
    *,
    price_col: str = "Adj Close",
    title: str = "",
    start: str | None = None,
    end: str | None = None,
) -> None:
    """For one asset, plot (STD / Custom1 / Custom2) signals in 3 panels."""

    close = df[price_col].astype(float).dropna()
    if start is not None:
        close = close.loc[pd.to_datetime(start) :]
    if end is not None:
        close = close.loc[: pd.to_datetime(end)]

    # 1) Standard
    macd_std = build_macd_standard(close, 12, 26, 9)
    event_std = macd_cross_events(macd_std)
    _, buy_std, sell_std = event_to_pos_and_trades(event_std, close.index)

    # 2) Custom1 (WEMA cross)
    macd_c1 = build_macd_wema(
        close,
        fast_N=int(best_custom1["fast_N"]),
        slow_N=int(best_custom1["slow_N"]),
        signal_N=int(best_custom1["signal_N"]),
        alpha=float(best_custom1["alpha"]),
    )
    event_c1 = macd_cross_events(macd_c1)
    _, buy_c1, sell_c1 = event_to_pos_and_trades(event_c1, close.index)

    # 3) Custom2 (Extrema)
    event_c2 = extrema_signal(
        close,
        fast_span=int(best_custom2["fast_N"]),
        slow_span=int(best_custom2["slow_N"]),
        signal_span=int(best_custom2["signal_N"]),
        alpha=float(best_custom2["alpha"]),
    )
    _, buy_c2, sell_c2 = event_to_pos_and_trades(event_c2, close.index)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    ax0 = axes[0]
    ax0.plot(close.index, close.values, linewidth=1.2, label="Price")
    ax0.scatter(buy_std, close.loc[buy_std], marker="^", s=70, label="Buy")
    ax0.scatter(sell_std, close.loc[sell_std], marker="v", s=70, label="Sell")
    ax0.set_title((title + " - STD") if title else "STD MACD (12,26,9)")
    ax0.set_ylabel("Price")
    ax0.legend(loc="best")

    ax1 = axes[1]
    ax1.plot(close.index, close.values, linewidth=1.2, label="Price")
    ax1.scatter(buy_c1, close.loc[buy_c1], marker="o", s=35, label="Buy")
    ax1.scatter(sell_c1, close.loc[sell_c1], marker="x", s=45, label="Sell")
    ax1.set_title(
        (title + " - Custom1")
        if title
        else f"Custom1 (WEMA cross)  N=({best_custom1['fast_N']},{best_custom1['slow_N']},{best_custom1['signal_N']}), alpha={best_custom1['alpha']:.4f}"
    )
    ax1.set_ylabel("Price")
    ax1.legend(loc="best")

    ax2 = axes[2]
    ax2.plot(close.index, close.values, linewidth=1.2, label="Price")
    ax2.scatter(buy_c2, close.loc[buy_c2], marker="s", s=40, label="Buy")
    ax2.scatter(sell_c2, close.loc[sell_c2], marker="D", s=40, label="Sell")
    ax2.set_title(
        (title + " - Custom2")
        if title
        else f"Custom2 (Extrema)   N=({best_custom2['fast_N']},{best_custom2['slow_N']},{best_custom2['signal_N']}), alpha={best_custom2['alpha']:.4f}"
    )
    ax2.set_ylabel("Price")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()
