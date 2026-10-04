"""Time-series factor regressions (Black-Jensen-Scholes / Fama-French 1993 style).

    r_t - rf_t = alpha + sum_k beta_k f_{k,t} + e_t

Inference uses Newey-West (HAC) standard errors because strategy returns can be
autocorrelated and heteroskedastic. The joint test that all intercepts are zero
is Gibbons, Ross & Shanken (1989).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.rolling import RollingOLS

# SMB5 is the five-factor file's SMB (an average of size sorts on B/M, OP and
# INV); it differs from the FF3 SMB (corr ~0.98), so each model uses its own.
MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF3+UMD": ["Mkt-RF", "SMB", "HML", "UMD"],
    "FF5": ["Mkt-RF", "SMB5", "HML", "RMW", "CMA"],
    "FF5+UMD": ["Mkt-RF", "SMB5", "HML", "RMW", "CMA", "UMD"],
}


def nw_lags(n_obs: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


@dataclass
class FactorFit:
    params: pd.Series   # 'const' is alpha (monthly, decimal)
    tvalues: pd.Series  # Newey-West t-statistics
    pvalues: pd.Series
    bse: pd.Series
    r2_adj: float
    resid: pd.Series
    nobs: int

    @property
    def alpha(self) -> float:
        return float(self.params["const"])

    @property
    def alpha_t(self) -> float:
        return float(self.tvalues["const"])


def fit(y: pd.Series, X: pd.DataFrame, hac: bool = True) -> FactorFit:
    """OLS of y on X with an intercept; NaN rows dropped jointly."""
    df = pd.concat([y.rename("y"), X], axis=1, sort=True).dropna()
    Xc = sm.add_constant(df[X.columns])
    if hac:
        res = sm.OLS(df["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(df))})
    else:
        res = sm.OLS(df["y"], Xc).fit()
    return FactorFit(res.params, res.tvalues, res.pvalues, res.bse,
                     float(res.rsquared_adj), res.resid, int(res.nobs))


def fit_models(y: pd.Series, factors: pd.DataFrame, models: dict[str, list[str]] = MODELS
               ) -> dict[str, FactorFit]:
    return {name: fit(y, factors[cols]) for name, cols in models.items()
            if set(cols) <= set(factors.columns)}


def grs(excess: pd.DataFrame, factors: pd.DataFrame) -> dict[str, float]:
    """Gibbons-Ross-Shanken F-test that all N intercepts are jointly zero.

    F = (T-N-K)/N * a' S^-1 a / (1 + mu' O^-1 mu) ~ F(N, T-N-K)
    with S the residual covariance and O the factor covariance (both MLE, /T),
    mu the factor means. Assumes iid normal residuals, as in FF93 Table 9c.
    """
    df = pd.concat([excess, factors], axis=1).dropna()
    R, F = df[excess.columns].to_numpy(), df[factors.columns].to_numpy()
    T, N, K = len(df), R.shape[1], F.shape[1]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    alpha, resid = B[0], R - X @ B
    sigma = resid.T @ resid / T
    mu = F.mean(axis=0)
    omega = np.atleast_2d(np.cov(F, rowvar=False, bias=True))
    quad_a = alpha @ np.linalg.solve(sigma, alpha)
    quad_f = mu @ np.linalg.solve(omega, mu)
    f_stat = (T - N - K) / N * quad_a / (1 + quad_f)
    return {"F": float(f_stat), "p": float(stats.f.sf(f_stat, N, T - N - K)),
            "T": T, "N": N, "K": K,
            "mean_abs_alpha": float(np.abs(alpha).mean())}


def rolling(y: pd.Series, X: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling OLS coefficients; row t uses months t-window+1..t only.

    Adds 'alpha_se' (plain OLS standard error of the window's intercept).
    """
    df = pd.concat([y.rename("y"), X], axis=1, sort=True).dropna()
    res = RollingOLS(df["y"], sm.add_constant(df[X.columns]), window=window).fit()
    out = res.params.rename(columns={"const": "alpha"})
    out["alpha_se"] = res.bse["const"]
    return out.dropna()


def variance_decomposition(fit_: FactorFit, factors: pd.DataFrame) -> pd.Series:
    """Share of strategy variance from each factor (Euler allocation) and residual.

    Var(r) = b' S_f b + var(e). Factor k's share = b_k (S_f b)_k / Var(r);
    shares sum to one and a factor can contribute negatively (hedging).
    """
    cols = [c for c in fit_.params.index if c != "const"]
    f = factors.loc[fit_.resid.index, cols]
    b = fit_.params[cols].to_numpy()
    cov = np.atleast_2d(np.cov(f.to_numpy(), rowvar=False))
    contrib = b * (cov @ b)
    resid_var = fit_.resid.var()
    total = contrib.sum() + resid_var
    return pd.Series([*contrib, resid_var], index=[*cols, "Residual"]) / total


def mean_attribution(fit_: FactorFit, factors: pd.DataFrame) -> pd.Series:
    """Average return = alpha + sum_k b_k * mean(f_k) (holds exactly in-sample for OLS)."""
    cols = [c for c in fit_.params.index if c != "const"]
    f = factors.loc[fit_.resid.index, cols].mean()
    return pd.concat([pd.Series({"Alpha": fit_.alpha}), fit_.params[cols] * f])
