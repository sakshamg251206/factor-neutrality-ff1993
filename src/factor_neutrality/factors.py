"""Factor construction.

SMB and HML follow Fama & French (1993, sec. 2.1.2) exactly, using the six
value-weighted size/BE-ME portfolios:

    SMB = 1/3 (S/L + S/M + S/H) - 1/3 (B/L + B/M + B/H)
    HML = 1/2 (S/H + B/H)       - 1/2 (S/L + B/L)

TERM and DEF are *proxies*. FF93 use Ibbotson long-term government and corporate
bond returns, which are not public. We turn FRED yields into returns by pricing
a par bond: buy at par with coupon y_{t-1}, then reprice at y_t one month later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIX = {"SL": "SMALL LoBM", "SM": "ME1 BM2", "SH": "SMALL HiBM",
       "BL": "BIG LoBM", "BM": "ME2 BM2", "BH": "BIG HiBM"}


def smb_hml(six: pd.DataFrame) -> pd.DataFrame:
    p = {k: six[v] for k, v in SIX.items()}
    smb = (p["SL"] + p["SM"] + p["SH"]) / 3 - (p["BL"] + p["BM"] + p["BH"]) / 3
    hml = (p["SH"] + p["BH"]) / 2 - (p["SL"] + p["BL"]) / 2
    return pd.DataFrame({"SMB": smb, "HML": hml})


def par_bond_return(yield_pct: pd.Series, maturity_years: float) -> pd.Series:
    """Monthly holding return of a par bond priced off a yield series (in %).

    At t-1 we buy a bond with semiannual coupon c = y_{t-1} at price 1. At t its
    price is the PV of the remaining cash flows at yield y_t (maturity treated as
    constant, i.e. the bond is rolled). Return = price - 1 + c/12 (accrued coupon).
    Uses y_{t-1} and y_t only, so it is observable at the end of month t.
    """
    y = yield_pct / 100.0
    c, y_new = y.shift(1), y
    n = 2 * maturity_years
    disc = (1 + y_new / 2) ** (-n)
    price = (c / y_new) * (1 - disc) + disc
    return (price - 1 + c / 12).rename(f"R_{yield_pct.name}")


def term_def(gov_yield: pd.Series, corp_yield: pd.Series, rf: pd.Series,
             maturity_years: float = 10.0) -> pd.DataFrame:
    """TERM = R_gov - RF; DEF = R_corp - R_gov (same maturity, so DEF ~ spread risk)."""
    r_gov = par_bond_return(gov_yield, maturity_years)
    r_corp = par_bond_return(corp_yield, maturity_years)
    out = pd.DataFrame({"TERM": r_gov - rf, "DEF": r_corp - r_gov})
    return out.dropna()


if __name__ == "__main__":
    # Self-check: unchanged yield -> return equals one month of coupon.
    flat = pd.Series([5.0, 5.0, 5.0], name="Y")
    assert np.allclose(par_bond_return(flat, 10).iloc[1:], 0.05 / 12)
    # Yield up 1pp on a 10y bond -> loss of roughly duration (~7.8) * 1%.
    up = pd.Series([5.0, 6.0], name="Y")
    assert -0.08 < par_bond_return(up, 10).iloc[1] < -0.06
    print("factors self-check ok")
