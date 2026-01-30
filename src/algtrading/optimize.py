from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from .strategy import macd_strategy_score_custom1, macd_strategy_score_custom2


def optimize_macd_params_custom1(
    df: pd.DataFrame,
    *,
    fast_grid=range(4, 31, 1),
    slow_grid=range(10, 121, 2),
    signal_grid=range(3, 31, 1),
    alpha_bounds=(0.01, 0.99),
    fee_rate: float = 0.0,
    price_col: str = "Adj Close",
    top_k_refine: int = 50,
    alpha_coarse_grid: int = 15,
    mdd_threshold: float = -0.8,
):
    a_lo, a_hi = map(float, alpha_bounds)
    alpha_grid = np.linspace(a_lo, a_hi, int(alpha_coarse_grid))

    coarse_rows = []

    for fast_N in tqdm(fast_grid):
        for slow_N in slow_grid:
            if fast_N >= slow_N:
                continue
            for signal_N in signal_grid:
                best_s = -np.inf
                best_a = np.nan

                for a in alpha_grid:
                    s = macd_strategy_score_custom1(
                        df=df,
                        fast_N=int(fast_N),
                        slow_N=int(slow_N),
                        signal_N=int(signal_N),
                        alpha=float(a),
                        price_col=price_col,
                        fee_rate=float(fee_rate),
                        mdd_threshold=float(mdd_threshold),
                    )
                    if s > best_s:
                        best_s = s
                        best_a = float(a)

                coarse_rows.append(
                    {
                        "fast_N": int(fast_N),
                        "slow_N": int(slow_N),
                        "signal_N": int(signal_N),
                        "alpha_coarse": float(best_a),
                        "score_coarse": float(best_s),
                    }
                )

    coarse_df = pd.DataFrame(coarse_rows).sort_values("score_coarse", ascending=False).reset_index(drop=True)
    refine_df = coarse_df.head(int(top_k_refine)).copy()

    refined_rows = []

    for _, row in tqdm(refine_df.iterrows()):
        fast_N = int(row["fast_N"])
        slow_N = int(row["slow_N"])
        signal_N = int(row["signal_N"])

        def obj(a):
            s = macd_strategy_score_custom1(
                df=df,
                fast_N=fast_N,
                slow_N=slow_N,
                signal_N=signal_N,
                alpha=float(a),
                price_col=price_col,
                fee_rate=float(fee_rate),
                mdd_threshold=float(mdd_threshold),
            )
            if not np.isfinite(s):
                return 1e18
            return -s

        res = minimize_scalar(obj, bounds=(a_lo, a_hi), method="bounded")
        a_star = float(res.x)
        if np.isfinite(res.fun) and res.fun < 1e17:
            s_star = -float(res.fun)
        else:
            s_star = -np.inf

        refined_rows.append(
            {
                "fast_N": fast_N,
                "slow_N": slow_N,
                "signal_N": signal_N,
                "alpha_star": a_star,
                "score_star": s_star,
            }
        )

    refined_df = pd.DataFrame(refined_rows).sort_values("score_star", ascending=False).reset_index(drop=True)

    if len(refined_df) == 0 or not np.isfinite(refined_df.loc[0, "score_star"]):
        best_row = coarse_df.iloc[0].to_dict()
        best = {
            "fast_N": int(best_row["fast_N"]),
            "slow_N": int(best_row["slow_N"]),
            "signal_N": int(best_row["signal_N"]),
            "alpha": float(best_row["alpha_coarse"]),
            "score": float(best_row["score_coarse"]),
            "mdd_threshold": float(mdd_threshold),
        }
    else:
        best_row = refined_df.iloc[0].to_dict()
        best = {
            "fast_N": int(best_row["fast_N"]),
            "slow_N": int(best_row["slow_N"]),
            "signal_N": int(best_row["signal_N"]),
            "alpha": float(best_row["alpha_star"]),
            "score": float(best_row["score_star"]),
            "mdd_threshold": float(mdd_threshold),
        }

    history = coarse_df.merge(refined_df, on=["fast_N", "slow_N", "signal_N"], how="left")
    return best, history


def optimize_macd_params_custom2(
    df: pd.DataFrame,
    *,
    fast_grid=range(4, 31, 1),
    slow_grid=range(10, 121, 2),
    signal_grid=range(3, 31, 1),
    alpha_bounds=(0.01, 0.99),
    fee_rate: float = 0.0,
    price_col: str = "Adj Close",
    top_k_refine: int = 50,
    alpha_coarse_grid: int = 15,
    mdd_threshold: float = -0.8,
):
    a_lo, a_hi = map(float, alpha_bounds)
    alpha_grid = np.linspace(a_lo, a_hi, int(alpha_coarse_grid))

    coarse_rows = []

    for fast_N in tqdm(fast_grid):
        for slow_N in slow_grid:
            if fast_N >= slow_N:
                continue
            for signal_N in signal_grid:
                best_s = -np.inf
                best_a = np.nan

                for a in alpha_grid:
                    s = macd_strategy_score_custom2(
                        df=df,
                        fast_N=int(fast_N),
                        slow_N=int(slow_N),
                        signal_N=int(signal_N),
                        alpha=float(a),
                        price_col=price_col,
                        fee_rate=float(fee_rate),
                        mdd_threshold=float(mdd_threshold),
                    )
                    if s > best_s:
                        best_s = s
                        best_a = float(a)

                coarse_rows.append(
                    {
                        "fast_N": int(fast_N),
                        "slow_N": int(slow_N),
                        "signal_N": int(signal_N),
                        "alpha_coarse": float(best_a),
                        "score_coarse": float(best_s),
                    }
                )

    coarse_df = pd.DataFrame(coarse_rows).sort_values("score_coarse", ascending=False).reset_index(drop=True)
    refine_df = coarse_df.head(int(top_k_refine)).copy()

    refined_rows = []

    for _, row in tqdm(refine_df.iterrows()):
        fast_N = int(row["fast_N"])
        slow_N = int(row["slow_N"])
        signal_N = int(row["signal_N"])

        def obj(a):
            s = macd_strategy_score_custom2(
                df=df,
                fast_N=fast_N,
                slow_N=slow_N,
                signal_N=signal_N,
                alpha=float(a),
                price_col=price_col,
                fee_rate=float(fee_rate),
                mdd_threshold=float(mdd_threshold),
            )
            if not np.isfinite(s):
                return 1e18
            return -s

        res = minimize_scalar(obj, bounds=(a_lo, a_hi), method="bounded")
        a_star = float(res.x)
        if np.isfinite(res.fun) and res.fun < 1e17:
            s_star = -float(res.fun)
        else:
            s_star = -np.inf

        refined_rows.append(
            {
                "fast_N": fast_N,
                "slow_N": slow_N,
                "signal_N": signal_N,
                "alpha_star": a_star,
                "score_star": s_star,
            }
        )

    refined_df = pd.DataFrame(refined_rows).sort_values("score_star", ascending=False).reset_index(drop=True)

    if len(refined_df) == 0 or not np.isfinite(refined_df.loc[0, "score_star"]):
        best_row = coarse_df.iloc[0].to_dict()
        best = {
            "fast_N": int(best_row["fast_N"]),
            "slow_N": int(best_row["slow_N"]),
            "signal_N": int(best_row["signal_N"]),
            "alpha": float(best_row["alpha_coarse"]),
            "score": float(best_row["score_coarse"]),
            "mdd_threshold": float(mdd_threshold),
        }
    else:
        best_row = refined_df.iloc[0].to_dict()
        best = {
            "fast_N": int(best_row["fast_N"]),
            "slow_N": int(best_row["slow_N"]),
            "signal_N": int(best_row["signal_N"]),
            "alpha": float(best_row["alpha_star"]),
            "score": float(best_row["score_star"]),
            "mdd_threshold": float(mdd_threshold),
        }

    history = coarse_df.merge(refined_df, on=["fast_N", "slow_N", "signal_N"], how="left")
    return best, history
