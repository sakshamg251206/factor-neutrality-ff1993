"""Industry momentum (Moskowitz & Grinblatt 1999) and its factor-hedged version.

Timing convention (the one place look-ahead bias could creep in):
    signal[t]  uses returns up to and including month t-skip   (known at end of t)
    weights[t] are formed from signal[t] at the end of month t
    pnl[t+1] = sum_i weights[t, i] * r[t+1, i]
so `strategy_returns` multiplies *lagged* weights by current returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(returns: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Cumulative return over months [t-skip-lookback+1, t-skip]; NaN unless all present."""
    log_r = np.log1p(returns)
    return np.expm1(log_r.rolling(lookback, min_periods=lookback).sum().shift(skip))


def long_short_weights(signal: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Equal-weight long the top-n and short the bottom-n names each month.

    Long leg sums to +1, short leg to -1 (a zero-investment portfolio).
    Months with fewer than 2n valid signals get zero weight.
    """
    rank_hi = signal.rank(axis=1, ascending=False, method="first")
    rank_lo = signal.rank(axis=1, ascending=True, method="first")
    enough = signal.notna().sum(axis=1) >= 2 * n
    w = (rank_hi <= n).astype(float) / n - (rank_lo <= n).astype(float) / n
    return w.where(enough, 0.0, axis=0).where(signal.notna(), 0.0)


def strategy_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Gross return: weights formed at t-1 earn returns at t.

    An industry with a valid signal but a missing next-month return contributes
    zero (it cannot happen in the 1963+ sample; checked in tests).
    """
    return (weights.shift(1) * returns.fillna(0.0)).sum(axis=1).rename("gross")


def turnover(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """One-way turnover sum_i |w_t,i - w_{t-1,i}^drift| per rebalance.

    Between rebalances each leg's weights drift with returns, so the trade needed
    at t is measured against drifted, not stale, weights.
    """
    r = returns.fillna(0.0)
    prev = weights.shift(1).fillna(0.0)
    grown = prev * (1 + r)
    long_sum = grown.clip(lower=0).sum(axis=1).replace(0, np.nan)
    short_sum = grown.clip(upper=0).sum(axis=1).abs().replace(0, np.nan)
    drifted = (grown.clip(lower=0).div(long_sum, axis=0)
               + grown.clip(upper=0).div(short_sum, axis=0)).fillna(0.0)
    return (weights - drifted).abs().sum(axis=1).rename("turnover")


def net_returns(gross: pd.Series, turn: pd.Series, cost_bps: float) -> pd.Series:
    """Charge `cost_bps` per unit of one-way turnover. Trading at end of t-1 hits month t."""
    return (gross - turn.shift(1).fillna(0.0) * cost_bps / 1e4).rename(f"net_{cost_bps:g}bps")


def build(returns: pd.DataFrame, lookback: int = 12, skip: int = 1, n: int = 10,
          cost_bps: float = 0.0) -> pd.DataFrame:
    """Gross, turnover and net returns for one parameter choice."""
    w = long_short_weights(momentum_signal(returns, lookback, skip), n)
    gross = strategy_returns(w, returns)
    turn = turnover(w, returns)
    out = pd.DataFrame({"gross": gross, "turnover": turn,
                        "net": net_returns(gross, turn, cost_bps)})
    first = w.abs().sum(axis=1).gt(0).idxmax()
    return out.loc[out.index > first]


def hedge_ex_ante(strategy: pd.Series, factors: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Factor-neutralise using betas estimated only on data before month t.

    beta_{t-1} = OLS slopes of strategy on factors over months t-window..t-1.
    hedged_t   = strategy_t - beta_{t-1}' f_t
    The factors are zero-cost long-short portfolios (Mkt-RF is an excess
    return), so the hedge is a tradable overlay. An in-sample (ex-post) hedge
    would subtract the full-sample beta, which uses future data and makes
    the hedged series look more neutral than any real trader could achieve.
    """
    df = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    betas = {}
    y, X = df["y"].to_numpy(), df[factors.columns].to_numpy()
    for i in range(window, len(df)):
        Xi = np.column_stack([np.ones(window), X[i - window:i]])
        betas[df.index[i]] = np.linalg.lstsq(Xi, y[i - window:i], rcond=None)[0][1:]
    B = pd.DataFrame.from_dict(betas, orient="index", columns=factors.columns)
    hedge = (B * factors.loc[B.index]).sum(axis=1)
    return pd.DataFrame({"hedged": df.loc[B.index, "y"] - hedge, "hedge": hedge}).join(
        B.add_prefix("beta_"))
