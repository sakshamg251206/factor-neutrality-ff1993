"""Market-regime labels.

`bear` and `high_vol` are ex-ante: the label for month t uses only data through
t-1, so conditioning on them is something a trader could have done. NBER
recession dates are announced months after the fact; they are used only for
descriptive attribution, never as a trading signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bear_market(mkt_excess: pd.Series, months: int = 24) -> pd.Series:
    """True if the trailing `months` cumulative market excess return to t-1 is negative
    (Cooper, Gutierrez & Hameed 2004 'down market' state)."""
    cum = np.expm1(np.log1p(mkt_excess).rolling(months).sum())
    return (cum.shift(1) < 0).where(cum.shift(1).notna()).rename("bear")


def high_volatility(mkt_excess: pd.Series, months: int = 12, min_history: int = 60) -> pd.Series:
    """True if trailing realized vol (to t-1) exceeds its own expanding median (to t-1)."""
    vol = mkt_excess.rolling(months).std().shift(1)
    median = vol.expanding(min_history).median()
    return (vol > median).where(median.notna()).rename("high_vol")


def label(mkt_excess: pd.Series, recession: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Bear market (ex-ante)": bear_market(mkt_excess),
        "High volatility (ex-ante)": high_volatility(mkt_excess),
        "NBER recession (ex-post)": recession.reindex(mkt_excess.index).astype(bool),
    })
