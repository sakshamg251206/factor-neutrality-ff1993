"""Reproduce every table, figure and in-text number of the paper.

    python scripts/run_all.py

Stages: replication -> strategy & attribution -> out-of-sample & robustness
-> regimes -> international. Outputs go to results/tables, results/figures,
results/macros.tex (numbers quoted in the paper) and results/summary.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from factor_neutrality import data, factors, performance as pf, plots, regimes, regression as rg, strategy as st

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
IS = slice("1963-07", "1991-12")       # Fama & French (1993) sample
OOS = slice("1992-01", None)           # everything after their sample
FULL = slice("1963-07", None)
PERIODS = {"In-sample 1963-07..1991-12": IS, "Out-of-sample 1992-01..": OOS, "Full 1963-07..": FULL}
BASE = dict(lookback=12, skip=1, n=10)  # a-priori choice (Jegadeesh-Titman 12-1 convention)
COST_BPS = 25.0                         # base one-way cost per unit turnover
HEDGE = ["Mkt-RF", "SMB", "HML", "UMD"]
MACROS: dict[str, str] = {}
SUMMARY: dict[str, object] = {}


# --- output helpers ---------------------------------------------------------

def table(df: pd.DataFrame, name: str, fmt: str = "{:.2f}", index: bool = True) -> pd.DataFrame:
    """Write CSV (full precision) and a booktabs LaTeX fragment (formatted)."""
    df.to_csv(TAB / f"{name}.csv", index=index)
    shown = df.map(lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                   else ("" if isinstance(v, (float, np.floating)) else v))
    n_idx = df.index.nlevels if index else 0
    if n_idx > 1:  # flatten so LaTeX needs no \multirow and column counts match
        shown = shown.reset_index()
    tex = shown.to_latex(index=index and n_idx == 1, escape=True,
                         column_format="l" * n_idx + "r" * len(df.columns))
    tex = tex.replace("NaN", "")
    (TAB / f"{name}.tex").write_text(tex)
    return df


def coef_tex(est: pd.DataFrame, tstat: pd.DataFrame, name: str, extra: pd.DataFrame | None = None) -> None:
    """Paper table: estimates with t-statistics in parentheses underneath."""
    cols = list(est.columns) + (list(extra.columns) if extra is not None else [])
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             " & " + " & ".join(c.replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    for r in est.index:
        e = [f"{v:.2f}" if np.isfinite(v) else "" for v in est.loc[r]]
        t = [f"({v:.2f})" if np.isfinite(v) else "" for v in tstat.loc[r]]
        x = [f"{v:.2f}" for v in extra.loc[r]] if extra is not None else []
        lines += [f"{r} & " + " & ".join(e + x) + " \\\\",
                  " & " + " & ".join(t + [""] * len(x)) + " \\\\[2pt]"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


def macro(name: str, value: float | int | str, fmt: str = "{:.2f}") -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MACROS[name] = fmt.format(value) if isinstance(value, (float, np.floating)) else str(value)


def ann_alpha(fit: rg.FactorFit) -> float:
    return 1200 * fit.alpha  # % per year (12 x monthly)


# --- data -------------------------------------------------------------------

def load() -> dict[str, object]:
    ff3, ff5, umd = data.ff3(), data.ff5(), data.umd()
    bonds = factors.term_def(data.fred("GS10"), data.fred("BAA"), ff3["RF"], maturity_years=20)
    f_all = ff3.drop(columns="RF").join(umd)                        # 1926/27 onward
    # FF3 Mkt-RF/SMB/HML for every model; FF5's own SMB (SMB5) for FF5 models.
    f = (ff3.drop(columns="RF").join(ff5[["SMB", "RMW", "CMA"]].rename(columns={"SMB": "SMB5"}), how="inner")
         .join(umd).join(bonds))                                    # 1963-07 onward
    end = min(f.index[-1], ff3.index[-1])
    f = f.loc[:end]
    rec = data.fred("USREC")
    plots.RECESSIONS = rec
    return dict(ff3=ff3, ff5=ff5, f=f, f_all=f_all, rf=ff3["RF"], six=data.six_portfolios(),
                p25=data.portfolios_25(), ind=data.industries_49(), rec=rec, end=end,
                vintage=data.french_vintage("F-F_Research_Data_Factors"))


# --- stage 1: replication of Fama & French (1993) ---------------------------

def grid(values: pd.Series) -> pd.DataFrame:
    """S1B1..S5B5 series -> 5x5 frame (rows size quintile, cols BE/ME quintile)."""
    g = pd.DataFrame(np.asarray(values, dtype=float).reshape(5, 5),
                     index=["Small", "2", "3", "4", "Big"], columns=["Low", "2", "3", "4", "High"])
    return g


def replication(d: dict) -> None:
    ref = pd.read_csv(ROOT / "data" / "ff93_reference.csv")
    scal = ref[ref.portfolio.isna()].set_index("statistic")["value"]
    refgrid = {s: ref[ref.statistic == s].set_index("portfolio")["value"]
               for s in ref.statistic.unique() if (ref.statistic == s).sum() == 25}
    ff3, f, rf = d["ff3"], d["f"], d["rf"]

    # 1a. Reconstruct SMB and HML from the six building-block portfolios.
    rec = factors.smb_hml(d["six"])
    rows = {}
    for c in ["SMB", "HML"]:
        for lab, per in [("1926-2026", slice(None)), ("FF93 sample", IS)]:
            a, b = rec[c].loc[per].dropna(), ff3[c].loc[per]
            a, b = a.align(b, join="inner")
            rows[(c, lab)] = {"Corr": a.corr(b), "RMSE (bp/mo)": 1e4 * np.sqrt(((a - b) ** 2).mean()),
                              "Mean rebuilt (%/mo)": 100 * a.mean(), "Mean published (%/mo)": 100 * b.mean(),
                              "Months": len(a)}
    table(pd.DataFrame(rows).T, "T01_factor_reconstruction", "{:.4f}")
    macro("SmbCorr", rows[("SMB", "1926-2026")]["Corr"], "{:.4f}")
    macro("HmlCorr", rows[("HML", "1926-2026")]["Corr"], "{:.4f}")
    macro("SmbRmse", rows[("SMB", "1926-2026")]["RMSE (bp/mo)"], "{:.2f}")
    macro("HmlRmse", rows[("HML", "1926-2026")]["RMSE (bp/mo)"], "{:.2f}")
    both = pd.concat([pd.DataFrame({"rebuilt": rec[c], "published": ff3[c], "factor": c}) for c in ["SMB", "HML"]]).dropna()
    fig = plots.scatter_45(100 * both.published, 100 * both.rebuilt, None,
                           "Rebuilt vs. published factors, 1926-2026", "Published (%/month)",
                           "Rebuilt from six portfolios (%/month)", both.factor.reset_index(drop=True))
    plots.save(fig, FIG, "F02_factor_reconstruction")

    # 1b. Table 2: factor summary statistics in the FF93 window vs. the paper.
    fis = f.loc[IS]
    n = len(fis)
    rows = {}
    for c in ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]:
        x = fis[c]
        rows[c] = {"Mean (ours)": 100 * x.mean(), "Mean (FF93)": scal.get(f"{c}_mean_pct", np.nan),
                   "SD (ours)": 100 * x.std(), "SD (FF93)": scal.get(f"{c}_sd_pct", np.nan),
                   "t (ours)": x.mean() / x.std() * np.sqrt(n), "t (FF93)": scal.get(f"{c}_t", np.nan)}
    table(pd.DataFrame(rows).T, "T02_factor_stats_vs_ff93")
    macro("NIS", n, "{}")
    for c, nm in [("Mkt-RF", "Mkt"), ("SMB", "Smb"), ("HML", "Hml")]:
        macro(f"{nm}MeanIS", rows[c]["Mean (ours)"])
        macro(f"{nm}TIS", rows[c]["t (ours)"])
    corr = fis[["Mkt-RF", "SMB", "HML"]].corr()
    macro("CorrSmbHml", corr.loc["SMB", "HML"])
    macro("CorrMktSmb", corr.loc["Mkt-RF", "SMB"])
    macro("CorrMktHml", corr.loc["Mkt-RF", "HML"])
    table(pd.DataFrame({"Ours": [corr.loc["SMB", "HML"], corr.loc["Mkt-RF", "SMB"], corr.loc["Mkt-RF", "HML"]],
                        "FF93": [scal["corr_SMB_HML"], scal["corr_Mkt_SMB"], scal["corr_Mkt_HML"]]},
                       index=["corr(SMB,HML)", "corr(Mkt-RF,SMB)", "corr(Mkt-RF,HML)"]), "T03_factor_correlations")

    # 1c. 25 size/BE-ME portfolios: CAPM and FF3 time-series regressions (OLS t, as in FF93).
    ex = d["p25"].sub(rf, axis=0).loc[:d["end"]]
    fits = {m: {p: rg.fit(ex.loc[IS, p], fis[rg.MODELS[m]], hac=False) for p in ex}
            for m in ["CAPM", "FF3"]}
    a_capm = pd.Series({p: 100 * v.alpha for p, v in fits["CAPM"].items()})
    a_ff3 = pd.Series({p: 100 * v.alpha for p, v in fits["FF3"].items()})
    t_ff3 = pd.Series({p: v.alpha_t for p, v in fits["FF3"].items()})
    r2 = {m: pd.Series({p: v.r2_adj for p, v in fits[m].items()}) for m in fits}
    mean_ex = 100 * ex.loc[IS].mean()

    cmp = pd.DataFrame({
        "Mean excess (ours)": mean_ex, "Mean excess (FF93)": refgrid["mean_excess_pct"],
        "CAPM alpha (ours)": a_capm, "CAPM alpha (FF93)": refgrid["capm_alpha_pct"],
        "FF3 alpha (ours)": a_ff3, "FF3 alpha (FF93)": refgrid["ff3_alpha_pct"],
        "FF3 t (ours)": t_ff3, "FF3 t (FF93)": refgrid["ff3_alpha_t"],
        "CAPM R2 (ours)": r2["CAPM"], "CAPM R2 (FF93)": refgrid["capm_r2"],
        "FF3 R2 (ours)": r2["FF3"], "FF3 R2 (FF93)": refgrid["ff3_r2"]})
    table(cmp, "T04_25portfolios_vs_ff93")
    agree = {}
    for lab, a, b in [("Mean excess", "Mean excess (ours)", "Mean excess (FF93)"),
                      ("CAPM alpha", "CAPM alpha (ours)", "CAPM alpha (FF93)"),
                      ("FF3 alpha", "FF3 alpha (ours)", "FF3 alpha (FF93)"),
                      ("FF3 alpha t", "FF3 t (ours)", "FF3 t (FF93)"),
                      ("CAPM R2", "CAPM R2 (ours)", "CAPM R2 (FF93)"),
                      ("FF3 R2", "FF3 R2 (ours)", "FF3 R2 (FF93)")]:
        agree[lab] = {"Corr(ours, FF93)": cmp[a].corr(cmp[b]),
                      "Mean abs diff": (cmp[a] - cmp[b]).abs().mean(),
                      "Max abs diff": (cmp[a] - cmp[b]).abs().max()}
    agree = table(pd.DataFrame(agree).T, "T05_replication_agreement", "{:.3f}")
    macro("CorrCapmAlphaRep", agree.loc["CAPM alpha", "Corr(ours, FF93)"])
    macro("CorrFfAlphaRep", agree.loc["FF3 alpha", "Corr(ours, FF93)"])
    macro("MadFfAlphaRep", agree.loc["FF3 alpha", "Mean abs diff"])
    macro("SgAlpha", a_ff3["S1B1"])
    macro("SgT", t_ff3["S1B1"])
    macro("BgAlpha", a_ff3["S5B1"])
    macro("BgT", t_ff3["S5B1"])
    macro("NFfRNinety", int((r2["FF3"] > 0.9).sum()), "{}")
    macro("NCapmRNinety", int((r2["CAPM"] > 0.9).sum()), "{}")
    macro("MeanAbsCapmAlpha", a_capm.abs().mean())
    macro("MeanAbsFfAlpha", a_ff3.abs().mean())

    plots.save(plots.heatmaps({"CAPM alpha (%/mo)": grid(a_capm), "FF3 alpha (%/mo)": grid(a_ff3),
                               "FF3 alpha t-stat": grid(t_ff3)},
                              "25 size/BE-ME portfolios, July 1963 - Dec 1991: what the market misses and FF3 absorbs",
                              vmax=None), FIG, "F03_alpha_heatmaps_is")
    plots.save(plots.heatmaps({"CAPM adj. R2": grid(r2["CAPM"]), "FF3 adj. R2": grid(r2["FF3"])},
                              "Common variation explained, July 1963 - Dec 1991", diverging=False, vmax=1.0),
               FIG, "F04_r2_heatmaps_is")
    long = pd.concat([pd.DataFrame({"ours": a_capm, "ff93": refgrid["capm_alpha_pct"], "g": "CAPM"}),
                      pd.DataFrame({"ours": a_ff3, "ff93": refgrid["ff3_alpha_pct"], "g": "FF3"})])
    plots.save(plots.scatter_45(long.ff93, long.ours, None, "Replicated vs. published intercepts (25 portfolios)",
                                "FF93 Table 9a intercept (%/month)", "Our intercept (%/month)",
                                long.g.reset_index(drop=True)), FIG, "F05_intercepts_vs_ff93")

    # 1d. GRS tests, replication window and beyond.
    models = {"TERM+DEF": ["TERM", "DEF"], "CAPM": ["Mkt-RF"], "SMB+HML": ["SMB", "HML"],
              "FF3": ["Mkt-RF", "SMB", "HML"], "FF3+TERM+DEF": ["Mkt-RF", "SMB", "HML", "TERM", "DEF"],
              "FF3+UMD": rg.MODELS["FF3+UMD"], "FF5": rg.MODELS["FF5"], "FF5+UMD": rg.MODELS["FF5+UMD"]}
    paper_f = {"TERM+DEF": scal["GRS_F_TERM_DEF"], "CAPM": scal["GRS_F_CAPM"], "SMB+HML": scal["GRS_F_SMB_HML"],
               "FF3": scal["GRS_F_FF3"], "FF3+TERM+DEF": scal["GRS_F_FF3_TERM_DEF"]}
    rows = {}
    for m, cols in models.items():
        r = {"FF93 F (32 assets)": paper_f.get(m, np.nan)}
        for lab, per in PERIODS.items():
            g = rg.grs(ex.loc[per], f.loc[per, cols])
            short = lab.split()[0]
            r[f"F {short}"], r[f"p {short}"], r[f"|a| {short} (%)"] = g["F"], g["p"], 100 * g["mean_abs_alpha"]
        rows[m] = r
    grs = table(pd.DataFrame(rows).T, "T06_grs_tests", "{:.3f}")
    macro("GrsCapmIS", grs.loc["CAPM", "F In-sample"])
    macro("GrsCapmISp", grs.loc["CAPM", "p In-sample"], "{:.3f}")
    macro("GrsFfIS", grs.loc["FF3", "F In-sample"])
    macro("GrsFfISp", grs.loc["FF3", "p In-sample"], "{:.3f}")
    macro("GrsFfOOS", grs.loc["FF3", "F Out-of-sample"])
    macro("GrsFfOOSp", grs.loc["FF3", "p Out-of-sample"], "{:.3f}")
    macro("GrsFfFull", grs.loc["FF3", "F Full"])
    macro("GrsFfFullp", grs.loc["FF3", "p Full"], "{:.4f}")
    macro("GrsFfiveFull", grs.loc["FF5", "F Full"])
    macro("GrsFfiveFullp", grs.loc["FF5", "p Full"], "{:.4f}")
    macro("GrsFfTdIS", grs.loc["FF3+TERM+DEF", "F In-sample"])
    macro("GrsTdIS", grs.loc["TERM+DEF", "F In-sample"])

    # 1e. Actual vs. model-implied mean returns, in- and out-of-sample.
    for tag, per in [("is", IS), ("oos", OOS)]:
        parts = []
        for m in ["CAPM", "FF3"]:
            cols = rg.MODELS[m]
            pred = {p: (rg.fit(ex.loc[per, p], f.loc[per, cols], hac=False).params[cols]
                        * f.loc[per, cols].mean()).sum() for p in ex}
            parts.append(pd.DataFrame({"pred": 1200 * pd.Series(pred), "act": 1200 * ex.loc[per].mean(), "g": m}))
        long = pd.concat(parts)
        title = "FF93 sample 1963-1991" if tag == "is" else f"After FF93: 1992-{d['end'].year}"
        plots.save(plots.scatter_45(long.pred, long.act, None,
                                    f"Average vs. model-implied excess returns, {title}",
                                    "Model-implied (beta x factor premium, %/yr)", "Realised average (%/yr)",
                                    long.g.reset_index(drop=True)), FIG, f"F06_actual_vs_predicted_{tag}")

    # 1f. Bond factors (our proxies) and factor cumulative returns.
    tdv = pd.DataFrame({c: {"Mean (%/mo)": 100 * fis[c].mean(), "SD (%/mo)": 100 * fis[c].std(),
                            "corr with Mkt-RF": fis[c].corr(fis["Mkt-RF"])} for c in ["TERM", "DEF"]}).T
    table(tdv, "T07_bond_factor_proxies")
    plots.save(plots.cumulative(f.loc[FULL, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]],
                                f"Factor returns, July 1963 - {d['end']:%b %Y} (growth of $1, log scale; grey = NBER recessions)"),
               FIG, "F01_factor_cumulative")
    plots.save(plots.cumulative(f.loc[FULL, ["TERM", "DEF"]],
                                "Bond-market factor proxies from FRED yields (growth of $1)", log=False),
               FIG, "F07_bond_factor_cumulative")
    SUMMARY["replication"] = {"grs": grs.round(4).to_dict(), "agreement": agree.round(4).to_dict()}


# --- stage 2: strategy, raw performance and factor attribution --------------

def legs(ind: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    w = st.long_short_weights(st.momentum_signal(ind, BASE["lookback"], BASE["skip"]), BASE["n"])
    r = ind.fillna(0.0)
    return pd.DataFrame({"Long leg": (w.clip(lower=0).shift(1) * r).sum(axis=1) - rf,
                         "Short leg": (-w.clip(upper=0).shift(1) * r).sum(axis=1) - rf})


def reg_table(y: pd.Series, f: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = {}
    for m in models:
        ft = rg.fit(y, f[rg.MODELS[m]])
        r = {"alpha (%/yr)": ann_alpha(ft), "t(alpha)": ft.alpha_t}
        for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]:
            k = "SMB5" if c == "SMB" and "SMB5" in ft.params else c  # FF5 rows: FF5's SMB
            r[c] = ft.params.get(k, np.nan)
            r[f"t({c})"] = ft.tvalues.get(k, np.nan)
        r.update({"adj R2": ft.r2_adj, "N": ft.nobs})
        rows[m] = r
    return pd.DataFrame(rows).T


def strategies(d: dict) -> dict[str, pd.Series]:
    f, rf, ind = d["f"], d["rf"], d["ind"]
    s = st.build(ind, **BASE, cost_bps=COST_BPS)
    s = s.loc[FULL].loc[:d["end"]]
    lg = legs(ind, rf).loc[s.index]
    sv = (d["p25"]["S1B5"] - rf).loc[s.index].rename("Small-value tilt")
    hed = st.hedge_ex_ante(s["gross"], f.loc[s.index, HEDGE]).reindex(s.index)
    ser = pd.DataFrame({"Market (Mkt-RF)": f.loc[s.index, "Mkt-RF"], "IMOM gross": s["gross"],
                        f"IMOM net ({COST_BPS:g} bp)": s["net"], "IMOM long leg": lg["Long leg"],
                        "IMOM short leg": lg["Short leg"], "IMOM ex-ante hedged": hed["hedged"],
                        "Small-value tilt": sv})

    # Raw performance by period.
    blocks = []
    for lab, per in PERIODS.items():
        t = ser.loc[per].apply(pf.summary).T
        t.insert(0, "Period", lab.split()[0])
        blocks.append(t)
    perf = pd.concat(blocks)
    perf.index.name = "Series"
    table(perf.reset_index().set_index(["Period", "Series"]), "T08_performance", "{:.2f}")
    full = ser.apply(pf.summary).T
    macro("ImomMean", full.loc["IMOM gross", "Ann. mean (%)"])
    macro("ImomVol", full.loc["IMOM gross", "Ann. vol (%)"])
    macro("ImomSharpe", full.loc["IMOM gross", "Sharpe"])
    macro("ImomT", full.loc["IMOM gross", "t(mean)"])
    macro("ImomMaxDD", full.loc["IMOM gross", "Max DD (%)"], "{:.0f}")
    macro("ImomSkew", full.loc["IMOM gross", "Skew"])
    macro("ImomNetMean", full.loc[f"IMOM net ({COST_BPS:g} bp)", "Ann. mean (%)"])
    macro("ImomNetSharpe", full.loc[f"IMOM net ({COST_BPS:g} bp)", "Sharpe"])
    macro("MktSharpe", full.loc["Market (Mkt-RF)", "Sharpe"])
    macro("MktMean", full.loc["Market (Mkt-RF)", "Ann. mean (%)"])
    macro("HedMean", full.loc["IMOM ex-ante hedged", "Ann. mean (%)"])
    macro("HedSharpe", full.loc["IMOM ex-ante hedged", "Sharpe"])
    macro("HedVol", full.loc["IMOM ex-ante hedged", "Ann. vol (%)"])
    macro("SvMean", full.loc["Small-value tilt", "Ann. mean (%)"])
    macro("SvSharpe", full.loc["Small-value tilt", "Sharpe"])
    macro("Turnover", 100 * s["turnover"].mean(), "{:.0f}")
    macro("CostBps", COST_BPS, "{:.0f}")
    macro("StartImom", f"{s.index[0]:%B %Y}")
    macro("EndSample", f"{d['end']:%B %Y}")
    macro("NFull", len(s), "{}")
    macro("Vintage", d["vintage"])

    # Factor attribution (Newey-West t-stats), full sample.
    models = ["CAPM", "FF3", "FF3+UMD", "FF5", "FF5+UMD"]
    regs = {}
    for nm, y in [("imom", s["gross"]), ("imom_net", s["net"]), ("hedged", hed["hedged"].dropna()),
                  ("smallvalue", sv)]:
        regs[nm] = table(reg_table(y, f, models), f"T09_regressions_{nm}")
        coefs6 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
        est = regs[nm][["alpha (%/yr)", *coefs6]].rename(columns={"alpha (%/yr)": "alpha (%/yr)"})
        tst = regs[nm][["t(alpha)", *[f"t({c})" for c in coefs6]]]
        tst.columns = est.columns
        coef_tex(est, tst, f"P_regressions_{nm}", regs[nm][["adj R2"]].rename(columns={"adj R2": "adj. $R^2$"}))
    for m, tag in [("CAPM", "Capm"), ("FF3", "Ff"), ("FF3+UMD", "FfUmd"), ("FF5", "Ffive"), ("FF5+UMD", "FfiveUmd")]:
        macro(f"Imom{tag}Alpha", regs["imom"].loc[m, "alpha (%/yr)"])
        macro(f"Imom{tag}T", regs["imom"].loc[m, "t(alpha)"])
        macro(f"Imom{tag}R", regs["imom"].loc[m, "adj R2"])
        macro(f"Net{tag}Alpha", regs["imom_net"].loc[m, "alpha (%/yr)"])
        macro(f"Net{tag}T", regs["imom_net"].loc[m, "t(alpha)"])
        macro(f"Hed{tag}Alpha", regs["hedged"].loc[m, "alpha (%/yr)"])
        macro(f"Hed{tag}T", regs["hedged"].loc[m, "t(alpha)"])
        macro(f"Sv{tag}Alpha", regs["smallvalue"].loc[m, "alpha (%/yr)"])
        macro(f"Sv{tag}T", regs["smallvalue"].loc[m, "t(alpha)"])
    macro("ImomUmdBeta", regs["imom"].loc["FF5+UMD", "UMD"])
    macro("ImomHmlBetaFf", regs["imom"].loc["FF3", "HML"])
    macro("ImomMktBetaCapm", regs["imom"].loc["CAPM", "Mkt-RF"])
    macro("SvSmbBeta", regs["smallvalue"].loc["FF3", "SMB"])
    macro("SvHmlBeta", regs["smallvalue"].loc["FF3", "HML"])
    macro("SvMktBetaCapm", regs["smallvalue"].loc["CAPM", "Mkt-RF"])

    # Figures: cumulative, drawdowns, alpha shrinkage, loadings.
    plots.save(plots.cumulative(ser[["Market (Mkt-RF)", "IMOM gross", f"IMOM net ({COST_BPS:g} bp)",
                                     "IMOM ex-ante hedged", "Small-value tilt"]],
                                "Strategies vs. the market (growth of $1, log scale; grey = NBER recessions)"),
               FIG, "F08_strategy_cumulative")
    plots.save(plots.cumulative(ser[["IMOM long leg", "IMOM short leg", "Market (Mkt-RF)"]],
                                "Industry momentum legs (excess of T-bill) vs. the market"), FIG, "F09_strategy_legs")
    plots.save(plots.drawdowns(ser[["Market (Mkt-RF)", "IMOM gross", "IMOM ex-ante hedged", "Small-value tilt"]]
                               .apply(pf.drawdown), "Drawdowns from running peak"), FIG, "F10_drawdowns")
    est = pd.DataFrame({"IMOM": regs["imom"]["alpha (%/yr)"], "Small-value tilt": regs["smallvalue"]["alpha (%/yr)"],
                        "IMOM hedged": regs["hedged"]["alpha (%/yr)"]})
    se = pd.DataFrame({k: (v["alpha (%/yr)"] / v["t(alpha)"]).abs() for k, v in
                       [("IMOM", regs["imom"]), ("Small-value tilt", regs["smallvalue"]), ("IMOM hedged", regs["hedged"])]})
    plots.save(plots.coef_bars(est, se, "Alpha shrinks as known factors are added (95% Newey-West CI)",
                               "Annualised alpha (%)"), FIG, "F11_alpha_by_model")
    coefs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    b = regs["imom"][coefs].T
    bse = (regs["imom"][coefs] / regs["imom"][[f"t({c})" for c in coefs]].to_numpy()).abs().T
    plots.save(plots.coef_bars(b, bse, "Industry momentum factor loadings by model (95% Newey-West CI)", "Loading"),
               FIG, "F12_imom_loadings")

    # Rolling exposures and alpha (60-month windows, uses only the window's data).
    roll = rg.rolling(s["gross"], f[HEDGE], 60)
    plots.save(plots.rolling_lines(roll[HEDGE], "Industry momentum: rolling 60-month FF3+UMD betas", "Beta"),
               FIG, "F13_rolling_betas")
    ra = pd.DataFrame({"Rolling alpha (FF3+UMD)": 1200 * roll["alpha"]})
    rc = rg.rolling(s["gross"], f[["Mkt-RF"]], 60)
    ra["Rolling alpha (CAPM)"] = 1200 * rc["alpha"]
    band = pd.DataFrame({"Rolling alpha (FF3+UMD)": 1200 * 1.96 * roll["alpha_se"],
                         "Rolling alpha (CAPM)": 1200 * 1.96 * rc["alpha_se"]})
    plots.save(plots.rolling_lines(ra.dropna(), "Rolling 60-month alpha, %/yr (band = +/-1.96 OLS SE)", "% per year", band),
               FIG, "F14_rolling_alpha")
    sv_roll = rg.rolling(sv, f[["Mkt-RF", "SMB", "HML"]], 60)
    plots.save(plots.rolling_lines(sv_roll[["Mkt-RF", "SMB", "HML"]],
                                   "Small-value tilt: rolling 60-month FF3 betas", "Beta"), FIG, "F15_rolling_betas_smallvalue")

    # Residual returns: split IMOM into factor-explained part and alpha + residual.
    ft = rg.fit(s["gross"], f[rg.MODELS["FF5+UMD"]])
    cols = rg.MODELS["FF5+UMD"]
    explained = (f.loc[ft.resid.index, cols] * ft.params[cols]).sum(axis=1)
    parts = pd.DataFrame({"IMOM gross": s["gross"].loc[ft.resid.index],
                          "Factor-explained (FF5+UMD)": explained,
                          "Alpha + residual": ft.alpha + ft.resid})
    plots.save(plots.cumulative(parts, "Industry momentum = factor exposure + alpha + residual (FF5+UMD)"),
               FIG, "F16_residual_returns")

    # Risk decomposition and mean-return attribution.
    shares = pd.DataFrame({m: rg.variance_decomposition(rg.fit(s["gross"], f[rg.MODELS[m]]), f)
                           for m in models}).T
    shares["SMB"] = shares["SMB"].fillna(shares.pop("SMB5"))
    shares = shares.fillna(0.0)
    shares = shares[[c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "Residual"] if c in shares]]
    table(shares * 100, "T10_variance_decomposition_imom")
    plots.save(plots.stacked_shares(shares * 100, "Where industry-momentum risk comes from (share of variance, %)",
                                    "% of return variance"), FIG, "F17_risk_decomposition")
    macro("UmdVarShare", 100 * shares.loc["FF5+UMD", "UMD"], "{:.0f}")
    macro("ResidVarShare", 100 * shares.loc["FF5+UMD", "Residual"], "{:.0f}")
    attr_rows = {}
    for nm, y, m in [("IMOM", s["gross"], "FF5+UMD"), ("Small-value tilt", sv, "FF3")]:
        fit_ = rg.fit(y, f[rg.MODELS[m]])
        a = 1200 * rg.mean_attribution(fit_, f)
        attr_rows[f"{nm} ({m})"] = a
        plots.save(plots.waterfall(a, f"{nm}: average return = factor premia x loadings + alpha ({m})",
                                   "% per year"), FIG, f"F18_attribution_{'imom' if nm == 'IMOM' else 'smallvalue'}")
    table(pd.DataFrame(attr_rows).T, "T11_mean_attribution")
    macro("UmdContribution", attr_rows["IMOM (FF5+UMD)"]["UMD"])

    # Did the ex-ante hedge actually neutralise exposures? Rolling realised betas.
    rb_raw = rg.rolling(s["gross"], f[HEDGE], 36)
    rb_hed = rg.rolling(hed["hedged"].dropna(), f[HEDGE], 36)
    plots.save(plots.rolling_lines(pd.DataFrame({"UMD beta, unhedged": rb_raw["UMD"], "UMD beta, ex-ante hedged": rb_hed["UMD"],
                                                 "Mkt beta, unhedged": rb_raw["Mkt-RF"], "Mkt beta, ex-ante hedged": rb_hed["Mkt-RF"]}),
                                   "Realised 36-month betas before and after the ex-ante hedge", "Beta"),
               FIG, "F19_hedge_effectiveness")
    # Spanning in both directions: a regression of IMOM on UMD cannot tell which
    # one is the more primitive factor, so also ask whether UMD has alpha vs. IMOM.
    rows = {}
    for lab, per in PERIODS.items():
        short = lab.split()[0]
        x = f.loc[per].join(s["gross"].rename("IMOM"), how="inner")
        a = rg.fit(x["IMOM"], x[["Mkt-RF", "SMB", "HML", "UMD"]])
        b = rg.fit(x["UMD"], x[["Mkt-RF", "SMB", "HML", "IMOM"]])
        rows[(short, "IMOM on FF3+UMD")] = {"alpha (%/yr)": ann_alpha(a), "t": a.alpha_t,
                                            "slope on other": a.params["UMD"], "adj R2": a.r2_adj}
        rows[(short, "UMD on FF3+IMOM")] = {"alpha (%/yr)": ann_alpha(b), "t": b.alpha_t,
                                            "slope on other": b.params["IMOM"], "adj R2": b.r2_adj}
    span = pd.DataFrame(rows).T
    span.index.names = ["Period", "Regression"]
    span = table(span, "T20_spanning_both_directions")
    for p, ptag in [("Full", "Full"), ("In-sample", "IS"), ("Out-of-sample", "OOS")]:
        macro(f"UmdOnImomAlpha{ptag}", span.loc[(p, "UMD on FF3+IMOM"), "alpha (%/yr)"])
        macro(f"UmdOnImomT{ptag}", span.loc[(p, "UMD on FF3+IMOM"), "t"])
    macro("UmdOnImomSlope", span.loc[("Full", "UMD on FF3+IMOM"), "slope on other"])
    SUMMARY["strategy"] = {k: v.round(4).to_dict() for k, v in regs.items()}
    return {"gross": s["gross"], "net": s["net"], "turnover": s["turnover"], "hedged": hed["hedged"], "sv": sv}


# --- stage 3: out-of-sample, robustness, costs, multiple testing ------------

def robustness(d: dict, S: dict) -> None:
    f, f_all, ind, rf = d["f"], d["f_all"], d["ind"], d["rf"]
    tests: list[dict] = []  # every alpha test we run on IMOM variants, for multiple-testing

    def record(label: str, fit_: rg.FactorFit) -> None:
        tests.append({"test": label, "alpha (%/yr)": ann_alpha(fit_), "t": fit_.alpha_t, "p": float(fit_.pvalues["const"])})

    # 3a. In-sample vs. out-of-sample attribution.
    rows = {}
    for nm, y in [("IMOM gross", S["gross"]), ("IMOM net", S["net"]), ("IMOM hedged", S["hedged"]),
                  ("Small-value tilt", S["sv"])]:
        for m in ["CAPM", "FF3", "FF3+UMD", "FF5+UMD"]:
            for lab, per in list(PERIODS.items()):
                ft = rg.fit(y.loc[per].dropna(), f.loc[per, rg.MODELS[m]])
                rows[(nm, m, lab.split()[0])] = {"alpha (%/yr)": ann_alpha(ft), "t": ft.alpha_t,
                                                 "adj R2": ft.r2_adj, "N": ft.nobs}
                if nm.startswith("IMOM"):
                    record(f"{nm} | {m} | {lab.split()[0]}", ft)
    isoos = pd.DataFrame(rows).T
    isoos.index.names = ["Strategy", "Model", "Period"]
    table(isoos, "T12_is_vs_oos")
    a = isoos["alpha (%/yr)"].unstack(["Model"]).astype(float)
    t = isoos["t"].unstack(["Model"]).astype(float)
    order = [(s_, p) for s_ in ["IMOM gross", "IMOM net", "IMOM hedged", "Small-value tilt"]
             for p in ["In-sample", "Out-of-sample"]]
    a, t = a.loc[order, ["CAPM", "FF3", "FF3+UMD", "FF5+UMD"]], t.loc[order, ["CAPM", "FF3", "FF3+UMD", "FF5+UMD"]]
    a.index = t.index = [f"{s_}, {'IS' if p == 'In-sample' else 'OOS'}" for s_, p in order]
    coef_tex(a, t, "P_is_vs_oos")
    for m, tag in [("CAPM", "Capm"), ("FF3", "Ff"), ("FF3+UMD", "FfUmd"), ("FF5+UMD", "FfiveUmd")]:
        for p, ptag in [("In-sample", "IS"), ("Out-of-sample", "OOS")]:
            macro(f"Imom{tag}Alpha{ptag}", isoos.loc[("IMOM gross", m, p), "alpha (%/yr)"])
            macro(f"Imom{tag}T{ptag}", isoos.loc[("IMOM gross", m, p), "t"])
            macro(f"Sv{tag}Alpha{ptag}", isoos.loc[("Small-value tilt", m, p), "alpha (%/yr)"])
            macro(f"Sv{tag}T{ptag}", isoos.loc[("Small-value tilt", m, p), "t"])
    for p, ptag in [("In-sample", "IS"), ("Out-of-sample", "OOS")]:
        x = S["gross"].loc[IS if ptag == "IS" else OOS]
        macro(f"ImomMean{ptag}", 1200 * x.mean())
        macro(f"ImomSharpe{ptag}", np.sqrt(12) * x.mean() / x.std())
    for strat, fname in [("IMOM gross", "imom"), ("Small-value tilt", "smallvalue")]:
        sub = isoos.loc[strat]
        est = sub["alpha (%/yr)"].unstack("Period")[["In-sample", "Out-of-sample"]]
        se = (sub["alpha (%/yr)"] / sub["t"]).abs().unstack("Period")[["In-sample", "Out-of-sample"]]
        order = ["CAPM", "FF3", "FF3+UMD", "FF5+UMD"]
        plots.save(plots.coef_bars(est.loc[order], se.loc[order],
                                   f"{strat}: alpha in the FF93 sample vs. after it (95% NW CI)",
                                   "Annualised alpha (%)"), FIG, f"F20_is_vs_oos_{fname}")

    # 3b. Subperiods, including the pre-1963 period FF93 never saw (FF3+UMD only).
    s_long = st.build(ind, **BASE)["gross"]
    subs = {"1927-07..1963-06": slice("1927-07", "1963-06"), "1963-07..1979-12": slice("1963-07", "1979-12"),
            "1980-01..1999-12": slice("1980-01", "1999-12"), "2000-01..": slice("2000-01", None),
            "Excl. Januaries (1963-07..)": None}
    rows = {}
    for lab, per in subs.items():
        if per is None:
            y = S["gross"][S["gross"].index.month != 1]
            fx = f
        else:
            y = s_long.loc[per]
            fx = f_all if per.start < "1963-07" else f
        r = {"Mean (%/yr)": 1200 * y.mean(), "Sharpe": np.sqrt(12) * y.mean() / y.std(), "N": len(y)}
        for m in ["CAPM", "FF3", "FF3+UMD", "FF5+UMD"]:
            if set(rg.MODELS[m]) <= set(fx.columns):
                ft = rg.fit(y, fx[rg.MODELS[m]])
                r[f"{m} alpha"], r[f"{m} t"] = ann_alpha(ft), ft.alpha_t
                record(f"IMOM gross | {m} | {lab}", ft)
        rows[lab] = r
    sub = table(pd.DataFrame(rows).T, "T13_subperiods")
    macro("PreAlphaFfUmd", sub.loc["1927-07..1963-06", "FF3+UMD alpha"])
    macro("PreTFfUmd", sub.loc["1927-07..1963-06", "FF3+UMD t"])
    macro("PreMean", sub.loc["1927-07..1963-06", "Mean (%/yr)"])
    macro("PostTwoKAlphaFfUmd", sub.loc["2000-01..", "FF5+UMD alpha"])
    macro("PostTwoKTFfUmd", sub.loc["2000-01..", "FF5+UMD t"])

    # 3c. Parameter grid: chosen in-sample, judged out-of-sample.
    grid_rows, series = {}, {}
    for L in [3, 6, 9, 12]:
        for k in [0, 1]:
            for n in [5, 10]:
                g = st.build(ind, L, k, n)["gross"]
                gi, go = g.loc[IS].loc["1963-07":], g.loc[OOS].loc[:d["end"]]
                fi = rg.fit(gi, f.loc[IS, rg.MODELS["FF5+UMD"]])
                fo = rg.fit(go, f.loc[OOS, rg.MODELS["FF5+UMD"]])
                record(f"Grid L={L},skip={k},n={n} | FF5+UMD | In-sample", fi)
                record(f"Grid L={L},skip={k},n={n} | FF5+UMD | Out-of-sample", fo)
                grid_rows[(L, k, n)] = {"Sharpe IS": np.sqrt(12) * gi.mean() / gi.std(),
                                        "Sharpe OOS": np.sqrt(12) * go.mean() / go.std(),
                                        "alpha IS": ann_alpha(fi), "t IS": fi.alpha_t,
                                        "alpha OOS": ann_alpha(fo), "t OOS": fo.alpha_t,
                                        "SR_monthly_IS": gi.mean() / gi.std()}
                series[(L, k, n)] = g
    gdf = pd.DataFrame(grid_rows).T
    gdf.index.names = ["lookback", "skip", "n"]
    gdf = gdf.astype(float)
    best = gdf["Sharpe IS"].idxmax()
    table(gdf.drop(columns="SR_monthly_IS"), "T14_parameter_grid")
    rho = stats.spearmanr(gdf["Sharpe IS"], gdf["Sharpe OOS"]).statistic
    macro("GridN", len(gdf), "{}")
    macro("GridBest", f"$L={best[0]}$, skip $={best[1]}$, $n={best[2]}$")
    macro("GridBestSharpeIS", gdf.loc[best, "Sharpe IS"])
    macro("GridBestSharpeOOS", gdf.loc[best, "Sharpe OOS"])
    macro("GridBestAlphaOOS", gdf.loc[best, "alpha OOS"])
    macro("GridBestTOOS", gdf.loc[best, "t OOS"])
    macro("GridMedianSharpeOOS", gdf["Sharpe OOS"].median())
    macro("GridRho", rho)
    base_key = (BASE["lookback"], BASE["skip"], BASE["n"])
    macro("BaseRankIS", int(gdf["Sharpe IS"].rank(ascending=False)[base_key]), "{}")
    dsr_best = pf.deflated_sharpe(series[best].loc[IS].loc["1963-07":], gdf["SR_monthly_IS"].to_numpy())
    dsr_base = pf.deflated_sharpe(series[base_key].loc[IS].loc["1963-07":], gdf["SR_monthly_IS"].to_numpy())
    macro("DsrBest", dsr_best["DSR"], "{:.3f}")
    macro("DsrBase", dsr_base["DSR"], "{:.3f}")
    macro("SrZero", np.sqrt(12) * dsr_best["SR0_monthly"])
    fig, axes = plots.plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, col in zip(axes, ["Sharpe IS", "Sharpe OOS"]):
        t = gdf[col].unstack(["skip", "n"])
        t.columns = [f"skip={a}, n={b}" for a, b in t.columns]
        t.index = [f"L={i}" for i in t.index]
        plots.grid_heatmap(t, col.replace("IS", "in-sample 1963-91").replace("OOS", "out-of-sample 1992-"),
                           vmax=gdf[["Sharpe IS", "Sharpe OOS"]].abs().max().max(), ax=ax)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.suptitle("Parameter grid: annualised Sharpe ratio of industry momentum", x=0.01, y=1.04, ha="left", fontweight="bold")
    plots.save(fig, FIG, "F21_parameter_grid")

    # 3d. Transaction costs sweep and break-even.
    rows = {}
    for c in range(0, 105, 5):
        net = st.net_returns(S["gross"], S["turnover"], c)
        ft = rg.fit(net, f[rg.MODELS["FF5+UMD"]])
        ftc = rg.fit(net, f[rg.MODELS["CAPM"]])
        rows[c] = {"Mean (%/yr)": 1200 * net.mean(), "Sharpe": np.sqrt(12) * net.mean() / net.std(),
                   "CAPM alpha": ann_alpha(ftc), "CAPM t": ftc.alpha_t,
                   "FF5+UMD alpha": ann_alpha(ft), "FF5+UMD t": ft.alpha_t}
    cost = pd.DataFrame(rows).T
    cost.index.name = "cost (bp one-way)"
    table(cost.loc[[0, 10, 25, 50, 75, 100]], "T15_transaction_costs")
    to = S["turnover"].shift(1).fillna(0).mean() * 12 / 100  # alpha lost per bp of cost, %/yr
    macro("BreakEvenMean", 1200 * S["gross"].mean() / to, "{:.0f}")
    macro("BreakEvenAlpha", cost.loc[0, "FF5+UMD alpha"] / to, "{:.0f}")
    macro("CostTwoSig", float(np.interp(1.96, cost["FF5+UMD t"].to_numpy()[::-1], cost.index.to_numpy()[::-1])), "{:.0f}")
    fig, axes = plots.plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    axes[0].plot(cost.index, cost["Sharpe"], color=plots.SERIES[0], marker="o", ms=3)
    axes[0].set_ylabel("Net Sharpe")
    axes[0].set_title("Industry momentum after trading costs")
    axes[1].plot(cost.index, cost["FF5+UMD t"], color=plots.SERIES[1], marker="o", ms=3, label="FF5+UMD alpha t")
    axes[1].plot(cost.index, cost["CAPM t"], color=plots.SERIES[0], marker="o", ms=3, label="CAPM alpha t")
    for y, lab in [(1.96, "t = 1.96"), (3.0, "t = 3.0 (HLZ)")]:
        axes[1].axhline(y, color=plots.INK2, lw=0.8, ls="--")
        axes[1].text(100, y, lab, fontsize=7, color=plots.INK2, ha="right", va="bottom")
    axes[1].set_ylabel("t(alpha)")
    axes[1].set_xlabel("One-way cost per unit turnover (bp)")
    axes[1].legend(loc="upper right")
    plots.save(fig, FIG, "F22_transaction_costs")

    # 3e. Hedge-window robustness.
    rows = {}
    for w in [36, 60, 120]:
        h = st.hedge_ex_ante(S["gross"], f[HEDGE], w)["hedged"]
        ft = rg.fit(h, f[rg.MODELS["FF5+UMD"]])
        record(f"IMOM hedged (window {w}) | FF5+UMD | Full", ft)
        rows[f"{w}-month window"] = {"Mean (%/yr)": 1200 * h.mean(), "Vol (%/yr)": 100 * np.sqrt(12) * h.std(),
                                     "Sharpe": np.sqrt(12) * h.mean() / h.std(),
                                     "FF5+UMD alpha": ann_alpha(ft), "t": ft.alpha_t,
                                     "Residual UMD beta": ft.params["UMD"], "N": len(h)}
    table(pd.DataFrame(rows).T, "T16_hedge_windows")

    # 3f. Multiple testing across everything we ran.
    mt = pd.DataFrame(tests).drop_duplicates("test").set_index("test")
    mt["Holm p"] = pf.holm(mt["p"])
    mt["|t| > 1.96"] = mt["t"].abs() > 1.96
    mt["|t| > 3 (HLZ)"] = mt["t"].abs() > 3.0
    table(mt.sort_values("t", ascending=False), "T17_multiple_testing", "{:.3f}")
    umd_tests = mt[mt.index.str.contains("UMD")]
    macro("NTests", len(mt), "{}")
    macro("NUmdTests", len(umd_tests), "{}")
    macro("NUmdTwo", int((umd_tests["t"] > 1.96).sum()), "{}")
    macro("NUmdThree", int((umd_tests["t"] > 3.0).sum()), "{}")
    macro("NUmdHolm", int((umd_tests["Holm p"] < 0.05).sum()), "{}")
    macro("NCapmFfTests", int((~mt.index.str.contains("UMD")).sum()), "{}")
    macro("NCapmFfHolm", int((mt.loc[~mt.index.str.contains("UMD"), "Holm p"] < 0.05).sum()), "{}")
    fig, ax = plots.plt.subplots(figsize=(7.2, 3.4))
    srt = mt.sort_values("t")
    colors = [plots.SERIES[1] if "UMD" in i else plots.SERIES[0] for i in srt.index]
    ax.scatter(range(len(srt)), srt["t"], c=colors, s=14, edgecolor=plots.SURFACE, lw=0.8)
    for y, lab in [(1.96, "1.96"), (3.0, "3.0 (Harvey-Liu-Zhu)")]:
        ax.axhline(y, color=plots.INK2, lw=0.8, ls="--")
        ax.text(0, y + 0.1, lab, fontsize=7, color=plots.INK2)
    ax.axhline(0, color=plots.INK2, lw=0.6)
    ax.scatter([], [], color=plots.SERIES[0], label="CAPM / FF3 / FF5 benchmarks")
    ax.scatter([], [], color=plots.SERIES[1], label="Benchmarks including UMD")
    ax.legend(loc="upper left")
    ax.set_xticks([])
    ax.set_xlabel(f"All {len(mt)} alpha tests run on industry-momentum variants, sorted")
    ax.set_ylabel("t(alpha)")
    ax.set_title("Multiple testing: only momentum-blind benchmarks clear the bar")
    plots.save(fig, FIG, "F23_multiple_testing")
    SUMMARY["robustness"] = {"grid_best": list(map(int, best)), "dsr_best": dsr_best, "dsr_base": dsr_base,
                             "grid_spearman_is_oos": rho}


# --- stage 4: regimes ---------------------------------------------------------

def regime_analysis(d: dict, S: dict) -> None:
    f = d["f"]
    lab = regimes.label(f["Mkt-RF"], d["rec"]).loc[S["gross"].index]
    rows, betas, ses = {}, {}, {}
    for reg in lab:
        for state in [True, False]:
            m = lab[reg] == state
            y = S["gross"][m.to_numpy()]
            ft = rg.fit(y, f[HEDGE])
            name = f"{reg}: {'yes' if state else 'no'}"
            rows[name] = {"Months": len(y), "Mean (%/yr)": 1200 * y.mean(),
                          "Sharpe": np.sqrt(12) * y.mean() / y.std(),
                          "FF3+UMD alpha": ann_alpha(ft), "t": ft.alpha_t,
                          **{c: ft.params[c] for c in HEDGE}}
            short = {"Bear market (ex-ante)": "Bear", "High volatility (ex-ante)": "HighVol",
                     "NBER recession (ex-post)": "Recession"}[reg] + (" yes" if state else " no")
            betas[short] = ft.params[HEDGE]
            ses[short] = ft.bse[HEDGE]
    rt = table(pd.DataFrame(rows).T, "T18_regimes")
    order = ["Bear yes", "Bear no", "HighVol yes", "HighVol no", "Recession yes", "Recession no"]
    plots.save(plots.coef_bars(pd.DataFrame(betas)[order], pd.DataFrame(ses)[order],
                               "Industry momentum exposures by regime (FF3+UMD, 95% NW CI)", "Loading"),
               FIG, "F24_regime_exposures")
    macro("BearMktBeta", rt.loc["Bear market (ex-ante): yes", "Mkt-RF"])
    macro("BullMktBeta", rt.loc["Bear market (ex-ante): no", "Mkt-RF"])
    macro("BearMean", rt.loc["Bear market (ex-ante): yes", "Mean (%/yr)"])
    macro("BullMean", rt.loc["Bear market (ex-ante): no", "Mean (%/yr)"])
    macro("BearMonths", int(rt.loc["Bear market (ex-ante): yes", "Months"]), "{}")
    macro("BearAlpha", rt.loc["Bear market (ex-ante): yes", "FF3+UMD alpha"])
    macro("BullAlpha", rt.loc["Bear market (ex-ante): no", "FF3+UMD alpha"])
    macro("HiVolMean", rt.loc["High volatility (ex-ante): yes", "Mean (%/yr)"])
    macro("LoVolMean", rt.loc["High volatility (ex-ante): no", "Mean (%/yr)"])
    macro("RecMean", rt.loc["NBER recession (ex-post): yes", "Mean (%/yr)"])
    macro("ExpMean", rt.loc["NBER recession (ex-post): no", "Mean (%/yr)"])
    macro("BearUmdBeta", rt.loc["Bear market (ex-ante): yes", "UMD"])
    macro("BullUmdBeta", rt.loc["Bear market (ex-ante): no", "UMD"])


# --- stage 5: international (does "small-value alpha" = factor exposure abroad?) --

def international(d: dict) -> None:
    rows, est, se = {}, {}, {}
    for region in ["North_America", "Europe", "Japan", "Asia_Pacific_ex_Japan", "Developed"]:
        rf_ = data.regional_factors(region)
        p = data.portfolios_25(region)
        ex = p.sub(rf_["RF"], axis=0)
        sv = ex["S1B5"]
        name = region.replace("_", " ")
        r = {"Months": len(sv), "Mean (%/yr)": 1200 * sv.mean()}
        for m in ["CAPM", "FF3"]:
            ft = rg.fit(sv, rf_[rg.MODELS[m]])
            r[f"{m} alpha"], r[f"{m} t"] = ann_alpha(ft), ft.alpha_t
            est.setdefault(m, {})[name] = ann_alpha(ft)
            se.setdefault(m, {})[name] = abs(ann_alpha(ft) / ft.alpha_t)
            g = rg.grs(ex, rf_[rg.MODELS[m]])
            r[f"GRS {m}"], r[f"GRS {m} p"] = g["F"], g["p"]
            r[f"mean|a| {m} (%/mo)"] = 100 * g["mean_abs_alpha"]
        r["SMB beta"] = rg.fit(sv, rf_[rg.MODELS["FF3"]]).params["SMB"]
        r["HML beta"] = rg.fit(sv, rf_[rg.MODELS["FF3"]]).params["HML"]
        rows[name] = r
    it = table(pd.DataFrame(rows).T, "T19_international_smallvalue")
    plots.save(plots.coef_bars(pd.DataFrame(est), pd.DataFrame(se),
                               "Small-value tilt abroad, 1990-2026: CAPM vs. regional FF3 alpha (95% NW CI)",
                               "Annualised alpha (%)"), FIG, "F25_international")
    macro("IntlStart", "July 1990")
    macro("IntlNCapmSig", int((it["CAPM t"] > 1.96).sum()), "{}")
    macro("IntlNFfSig", int((it["FF3 t"] > 1.96).sum()), "{}")
    for name, tag in [("Developed", "Dev"), ("Japan", "Jpn"), ("Europe", "Eur"), ("North America", "Na"),
                      ("Asia Pacific ex Japan", "Apx")]:
        macro(f"{tag}CapmAlpha", it.loc[name, "CAPM alpha"])
        macro(f"{tag}CapmT", it.loc[name, "CAPM t"])
        macro(f"{tag}FfAlpha", it.loc[name, "FF3 alpha"])
        macro(f"{tag}FfT", it.loc[name, "FF3 t"])
    SUMMARY["international"] = it.round(4).to_dict()


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    print("data vintage:", d["vintage"], "| sample end:", d["end"].date())
    replication(d)
    print("replication done")
    S = strategies(d)
    print("strategy & attribution done")
    robustness(d, S)
    print("robustness done")
    regime_analysis(d, S)
    international(d)
    print("regimes & international done")
    (ROOT / "results" / "macros.tex").write_text(
        "% Auto-generated by scripts/run_all.py -- do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))
    (ROOT / "results" / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=str))
    print(f"{len(MACROS)} macros, {len(list(TAB.glob('*.csv')))} tables, {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
