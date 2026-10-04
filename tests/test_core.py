"""Tests for the pieces whose failure would silently corrupt results:
look-ahead in signals/weights/hedges, the GRS statistic, factor formulas,
parsing, turnover and attribution identities."""
import numpy as np
import pandas as pd
import pytest

from factor_neutrality import data, factors, performance as pf, regimes, regression as rg, strategy as st

RNG = np.random.default_rng(0)
IDX = pd.date_range("2000-01-31", periods=240, freq="ME")


def _panel(n_assets=30, T=240):
    return pd.DataFrame(RNG.normal(0.01, 0.05, (T, n_assets)), index=IDX[:T],
                        columns=[f"a{i}" for i in range(n_assets)])


# --- look-ahead bias ---------------------------------------------------------

def test_signal_and_strategy_ignore_future_returns():
    r = _panel()
    cut = 150
    r2 = r.copy()
    r2.iloc[cut:] = RNG.normal(0, 0.2, r2.iloc[cut:].shape)  # rewrite the future
    s1, s2 = st.momentum_signal(r, 12, 1), st.momentum_signal(r2, 12, 1)
    pd.testing.assert_frame_equal(s1.iloc[:cut], s2.iloc[:cut])
    w1, w2 = st.long_short_weights(s1, 5), st.long_short_weights(s2, 5)
    pd.testing.assert_frame_equal(w1.iloc[:cut], w2.iloc[:cut])
    g1, g2 = st.strategy_returns(w1, r), st.strategy_returns(w2, r2)
    pd.testing.assert_series_equal(g1.iloc[:cut], g2.iloc[:cut])


def test_strategy_uses_weights_formed_last_month():
    r = pd.DataFrame({"x": [0.0, 0.10, -0.20], "y": [0.0, 0.0, 0.0]}, index=IDX[:3])
    w = pd.DataFrame({"x": [1.0, 0.0, 0.0], "y": [-1.0, 0.0, 0.0]}, index=IDX[:3])
    g = st.strategy_returns(w, r)
    assert g.iloc[1] == pytest.approx(0.10)  # weights from month 0 earn month-1 returns
    assert g.iloc[2] == pytest.approx(0.0)   # month-1 weights are zero


def test_skip_month_excludes_most_recent_return():
    r = pd.DataFrame({"x": np.zeros(14)}, index=IDX[:14])
    r.iloc[-1] = 0.5  # only the latest month is non-zero
    assert st.momentum_signal(r, 12, skip=1).iloc[-1, 0] == pytest.approx(0.0)
    assert st.momentum_signal(r, 12, skip=0).iloc[-1, 0] == pytest.approx(0.5)


def test_ex_ante_hedge_ignores_future():
    y = pd.Series(RNG.normal(0, 0.04, 200), IDX[:200])
    f = pd.DataFrame(RNG.normal(0, 0.04, (200, 2)), IDX[:200], columns=["A", "B"])
    h1 = st.hedge_ex_ante(y, f, 60)
    y2, f2 = y.copy(), f.copy()
    y2.iloc[120:] += 1.0
    f2.iloc[120:] *= 3
    h2 = st.hedge_ex_ante(y2, f2, 60)
    pd.testing.assert_frame_equal(h1.loc[:IDX[119]], h2.loc[:IDX[119]])


def test_ex_ante_hedge_removes_a_stable_beta():
    f = pd.DataFrame(RNG.normal(0, 0.04, (240, 1)), IDX, columns=["A"])
    y = 1.5 * f["A"] + RNG.normal(0, 0.005, 240)
    h = st.hedge_ex_ante(y, f, 60)
    assert abs(rg.fit(h["hedged"], f).params["A"]) < 0.05


def test_regime_labels_are_ex_ante():
    m = pd.Series(RNG.normal(0.005, 0.04, 240), IDX)
    m2 = m.copy()
    m2.iloc[100] = -0.9  # a crash in month 100 cannot change month-100 labels
    for fn in (regimes.bear_market, regimes.high_volatility):
        a, b = fn(m), fn(m2)
        assert a.iloc[:101].equals(b.iloc[:101])


# --- statistics ----------------------------------------------------------------

def test_grs_equals_fama_french_1993_formula():
    """FF93 Table 9c footnote: F = A' S^-1 A (T-K-L+1) / (L (T-K) a11), with
    S the unbiased residual covariance, K = 1 + #factors, a11 = [(X'X)^-1]_11."""
    T, N, Kf = 300, 6, 3
    F = RNG.normal(0.005, 0.04, (T, Kf))
    R = 0.002 + F @ RNG.normal(1, 0.3, (Kf, N)) + RNG.normal(0, 0.02, (T, N))
    ours = rg.grs(pd.DataFrame(R), pd.DataFrame(F, columns=["f1", "f2", "f3"]))["F"]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    E = R - X @ B
    K = Kf + 1
    S = E.T @ E / (T - K)
    a11 = np.linalg.inv(X.T @ X)[0, 0]
    ff = B[0] @ np.linalg.solve(S, B[0]) * (T - K - N + 1) / (N * (T - K) * a11)
    assert ours == pytest.approx(ff, rel=1e-10)


def test_grs_has_correct_size_under_null():
    rejections = 0
    for _ in range(400):
        F = RNG.normal(0.005, 0.04, (120, 1))
        R = F @ np.ones((1, 5)) + RNG.normal(0, 0.02, (120, 5))
        rejections += rg.grs(pd.DataFrame(R), pd.DataFrame(F, columns=["m"]))["p"] < 0.05
    assert 0.02 < rejections / 400 < 0.09


def test_mean_attribution_sums_to_average_return():
    f = pd.DataFrame(RNG.normal(0.005, 0.04, (240, 2)), IDX, columns=["A", "B"])
    y = 0.002 + 0.8 * f["A"] - 0.3 * f["B"] + RNG.normal(0, 0.01, 240)
    ft = rg.fit(y, f)
    assert rg.mean_attribution(ft, f).sum() == pytest.approx(y.mean())
    assert rg.variance_decomposition(ft, f).sum() == pytest.approx(1.0)


def test_deflated_sharpe_falls_with_more_trials():
    r = pd.Series(RNG.normal(0.01, 0.04, 300))
    few = pf.deflated_sharpe(r, RNG.normal(0.1, 0.05, 5))["DSR"]
    many = pf.deflated_sharpe(r, RNG.normal(0.1, 0.05, 500))["DSR"]
    assert many < few


# --- factor formulas, parsing, accounting -------------------------------------

def test_smb_hml_formula():
    six = pd.DataFrame([[0.03, 0.02, 0.01, 0.006, 0.004, 0.002]], columns=list(factors.SIX.values()))
    out = factors.smb_hml(six).iloc[0]
    assert out["SMB"] == pytest.approx((0.03 + 0.02 + 0.01) / 3 - (0.006 + 0.004 + 0.002) / 3)
    assert out["HML"] == pytest.approx((0.01 + 0.002) / 2 - (0.03 + 0.006) / 2)


def test_par_bond_return():
    flat = pd.Series([5.0, 5.0], name="Y")
    assert factors.par_bond_return(flat, 10).iloc[1] == pytest.approx(0.05 / 12)
    up = pd.Series([5.0, 6.0], name="Y")
    assert -0.08 < factors.par_bond_return(up, 10).iloc[1] < -0.06


def test_parse_french_csv_tables_and_missing():
    text = ("Header text\n\n  Value Weighted\n,A,B\n196301,  1.00, -99.99\n196302, 2.00, 3.00\n\n"
            "  Equal Weighted\n,A,B\n196301, 5.00, 6.00\n\nAnnual\n,A,B\n1963, 9.0, 9.0\n")
    vw = data.parse_french_csv(text, 0)
    assert vw.shape == (2, 2) and np.isnan(vw.iloc[0, 1]) and vw.iloc[1, 0] == pytest.approx(0.02)
    assert data.parse_french_csv(text, 1).iloc[0, 1] == pytest.approx(0.06)
    assert vw.index[0] == pd.Timestamp("1963-01-31")
    with pytest.raises(ValueError):
        data.parse_french_csv(text, 2)  # annual block is not a monthly table


def test_turnover_full_flip_is_four():
    r = pd.DataFrame(0.0, index=IDX[:3], columns=["x", "y"])
    w = pd.DataFrame({"x": [1.0, 1.0, -1.0], "y": [-1.0, -1.0, 1.0]}, index=IDX[:3])
    t = st.turnover(w, r)
    assert t.iloc[1] == pytest.approx(0.0) and t.iloc[2] == pytest.approx(4.0)


def test_holm():
    assert np.allclose(pf.holm(pd.Series([0.01, 0.04, 0.03])).to_numpy(), [0.03, 0.06, 0.06])


# --- data integrity (needs the cached raw files; skipped otherwise) -----------

@pytest.mark.skipif(not (data.RAW / "49_Industry_Portfolios_CSV.zip").exists(), reason="raw data not downloaded")
def test_no_held_industry_has_missing_next_month_return():
    ind = data.industries_49()
    w = st.long_short_weights(st.momentum_signal(ind), 10).loc["1963-07":]
    held_missing = (w.shift(1).fillna(0).ne(0) & ind.loc[w.index].isna()).sum().sum()
    assert held_missing == 0
