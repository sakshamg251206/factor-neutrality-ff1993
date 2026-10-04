"""Performance statistics and multiple-testing tools."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def summary(r: pd.Series) -> pd.Series:
    """Annualised summary of a monthly (excess or zero-cost) return series."""
    r = r.dropna()
    n = len(r)
    mu, sd = r.mean(), r.std()
    return pd.Series({
        "Months": n, "Ann. mean (%)": 1200 * mu, "Ann. vol (%)": 100 * np.sqrt(12) * sd,
        "Sharpe": np.sqrt(12) * mu / sd, "t(mean)": mu / sd * np.sqrt(n),
        "Skew": stats.skew(r), "Ex. kurt": stats.kurtosis(r),
        "Max DD (%)": 100 * drawdown(r).min(), "Hit rate (%)": 100 * (r > 0).mean(),
    })


def deflated_sharpe(r: pd.Series, sr_trials: np.ndarray) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), per-period units.

    SR0 is the Sharpe ratio expected from the best of N skill-less trials:
      SR0 = sqrt(Var[SR_n]) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e)))
    DSR = Phi( (SR - SR0) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) )
    i.e. the probability the observed SR beats that luck benchmark, after
    adjusting for non-normal returns (skew g3, kurtosis g4).
    """
    r = r.dropna()
    T, sr = len(r), r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    N = len(sr_trials)
    var_sr = np.var(sr_trials, ddof=1)
    sr0 = np.sqrt(var_sr) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N)
                             + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"SR_monthly": sr, "SR0_monthly": sr0, "N_trials": N, "DSR": float(stats.norm.cdf(z))}


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm (1979) step-down family-wise error correction."""
    p = pvalues.sort_values()
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p.to_numpy()).clip(max=1.0)
    return pd.Series(adj, index=p.index).reindex(pvalues.index)


if __name__ == "__main__":
    assert np.allclose(holm(pd.Series([0.01, 0.04, 0.03])).to_numpy(), [0.03, 0.06, 0.06])
    dd = drawdown(pd.Series([0.1, -0.5, 0.2]))
    assert np.isclose(dd.min(), -0.5)
    print("performance self-check ok")
