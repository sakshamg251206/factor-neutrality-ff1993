# Alpha or Factor Exposure?
### A replication of Fama & French (1993) and a factor-neutrality test of industry momentum, 1963–2026

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)
![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23144116.svg)](https://doi.org/10.5281/zenodo.23144116)
**Zenodo DOI:** [10.5281/zenodo.23144116](https://doi.org/10.5281/zenodo.23144116) (v1.0.0) · all versions: [10.5281/zenodo.23144115](https://doi.org/10.5281/zenodo.23144115)

> **Research question.** *Is this strategy actually generating alpha, or am I just being compensated for taking known systematic factor risk?*

📄 **Paper:** [`paper/Factor_Neutrality_FF1993_Garg_2026.pdf`](paper/Factor_Neutrality_FF1993_Garg_2026.pdf) (27 pages)

---

## TL;DR

| | Result |
|---|---|
| **Replication** | Rebuilt SMB/HML correlate **1.0000** with the published factors (RMSE 0.3 bp/month). The 25-portfolio intercepts correlate **0.98** (CAPM) and **0.92** (FF3) with FF93 Table 9a. GRS F = **1.98 / 1.43** (CAPM / FF3) vs. the paper's 1.91 / 1.56. |
| **Raw strategy** | Industry momentum: **8.63 %/yr**, t = 4.46, Sharpe 0.56 (market 0.47), July 1963 – Aug 2026. |
| **"Alpha" under CAPM / FF3** | **9.40 %/yr (t = 4.90)** / **10.78 %/yr (t = 5.75)**. Looks like exceptional skill. |
| **After adding the momentum factor UMD** | R² jumps 0.05 → **0.67**; FF5+UMD alpha **2.76 %/yr (t = 2.46)**. Insignificant in-sample (t = 0.98) and out-of-sample (t = 1.60). |
| **After 25 bp trading costs** | FF5+UMD alpha **0.14 %/yr (t = 0.12)**. Break-even cost for the alpha: ~26 bp. |
| **Multiple testing** | 62 alpha tests with UMD in the benchmark: **0** exceed t = 3, **0** survive Holm. |
| **Control (small-value tilt)** | FF3 explains it in the FF93 sample (α = 0.70 %/yr, t = 0.91), but an alpha re-appears after 1991 (3.60 %/yr, t = 3.01) and in **all 5** international regions. |

**Answer:** industry momentum's apparent alpha is overwhelmingly **compensation for exposure to the momentum factor**, and what remains does not survive costs or multiple-testing corrections. The small-value control adds a caution: "explained by factors" is a statement about a model *in a sample*.

---

## Motivation

A track record that beats the market says nothing yet about skill. If the same returns can be
reproduced by a passive combination of cheap, well-known factor portfolios, the "alpha" is really
a **risk premium**. It carries those factors' crash risk and does not deserve active fees.
Fama & French (1993) gave the standard tool for telling the two apart: regress the strategy's
excess returns on factor returns, and read the **intercept** as alpha.

## The Fama–French (1993) paper

*Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
Journal of Financial Economics, 33(1), 3–56.* [doi:10.1016/0304-405X(93)90023-5](https://doi.org/10.1016/0304-405X(93)90023-5)

* **Five factors:** the market (RM−RF), size (**SMB**), book-to-market (**HML**), and two bond factors (**TERM**, **DEF**).
* **Method:** time-series regressions (Black–Jensen–Scholes) of 25 size/BE-ME stock portfolios and 7 bond portfolios, July 1963 – December 1991.
* **Findings:** SMB and HML capture strong common variation (three-factor R² > 0.9 for 21 of 25 portfolios). Intercepts are close to zero except in the small-growth corner. TERM/DEF dominate bond returns but add little for stocks. The GRS test rejects marginally.

## Methodology

**The regression.** For strategy excess return $r_t$ and factor returns $f_t$:

$$r_t = \alpha + \beta' f_t + \varepsilon_t \quad\Longrightarrow\quad E[r_t] = \alpha + \beta' E[f_t]$$

Average return = alpha + Σ (loading × factor premium). Because the factors are tradable zero-cost
portfolios, $\beta' f_t$ can be replicated passively, so only $\alpha$ is skill. *Factor neutrality*
means $\beta = 0$. The test is whether $\hat\alpha$ survives once the relevant factors are included.

**Joint test (GRS).** $F = \frac{T-N-K}{N}\,\frac{\hat\alpha'\hat\Sigma^{-1}\hat\alpha}{1+\bar f'\hat\Omega^{-1}\bar f} \sim F_{N,T-N-K}$.
A unit test proves our implementation equals the FF93 Table 9c formula.

| Component | Choice |
|---|---|
| Strategy | 49 industries; signal = return over months t−11..t−1 (skip month t); long top 10, short bottom 10, equal-weighted, monthly. Parameters fixed **a priori**. |
| Benchmarks | CAPM, FF3, FF3+UMD, FF5, FF5+UMD |
| Inference | Newey–West HAC standard errors (lags = ⌊4(T/100)^{2/9}⌋); OLS for the replication (as in FF93) |
| Factor-neutral version | **Ex-ante** hedge: betas from the trailing 60 months only, then subtract β̂ₜ₋₁′fₜ |
| Out-of-sample | IS = FF93 window (1963-07 → 1991-12); OOS = 1992-01 → 2026-08; also 1927–1963 |
| Robustness | 16-variant grid chosen IS and judged OOS; deflated Sharpe ratio; cost sweep 0–100 bp; hedge windows; regimes; international |
| Multiple testing | All 90 alpha tests logged; Holm correction; Harvey–Liu–Zhu t > 3 |

**Bias controls.** No look-ahead (unit-tested for signals, P&L, hedge and regime labels). No
survivorship (CRSP includes delistings). No shuffling (strictly chronological splits). See
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the full audit: where each bias could enter and
which test guards it.

## Dataset

Public data only: the **Kenneth R. French Data Library** (files built from the 202608 CRSP database)
and **FRED** (GS10, BAA, USREC). Details, vintages and a "when is each variable known" table are in
[`data/README.md`](data/README.md). Raw files are downloaded on first run and hashed into
`data/raw/MANIFEST.json`. They are not redistributed.

## Factor definitions

| Factor | Definition |
|---|---|
| **Mkt−RF** | Value-weighted market return minus the one-month T-bill |
| **SMB** | ⅓(S/L + S/M + S/H) − ⅓(B/L + B/M + B/H) from 2×3 size × BE/ME sorts (NYSE breakpoints, formed each June) |
| **HML** | ½(S/H + B/H) − ½(S/L + B/L) |
| **RMW, CMA** | Robust-minus-weak profitability; conservative-minus-aggressive investment (FF 2015) |
| **UMD** | High minus low prior return (months −12 to −2), 2×3 size × momentum sorts, monthly (Carhart 1997) |
| **TERM** | Long-term government bond return − T-bill. *Proxy:* 20-year par bond priced off GS10 |
| **DEF** | Corporate − government bond return. *Proxy:* same-maturity par bonds off BAA and GS10 |

## Main findings

1. **The replication works.** Every FF93 headline is reproduced within data-revision noise:
   SMB 0.26 vs 0.27 %/mo, HML 0.38 vs 0.40, corr(SMB, HML) −0.09 vs −0.08, small-growth FF3
   intercept −0.37 (t = −3.43) vs −0.34 (t = −3.16), FF3 R² > 0.9 in 22 vs 21 of 25 portfolios.
2. **Industry momentum has a large CAPM/FF3 "alpha"**: 9.40 and 10.78 %/yr. FF3 makes it look
   *better*, because the strategy is short HML (β = −0.30) and value had a positive premium.
3. **That alpha is mostly momentum-factor exposure.** The UMD loading is 0.87 (t ≈ 30). UMD explains
   67 % of the strategy's variance and contributes 6.2 of its 8.6 %/yr. The alpha falls by about
   three-quarters.
4. **The remainder is fragile.** It is insignificant IS and OOS and before 1963 (t = 0.52). It is
   zero after 25 bp costs, below t = 3 in all 62 UMD-controlled tests, and 0/62 survive Holm.
5. **Neutralising exposure removes the return.** The ex-ante hedged strategy earns 2.64 %/yr
   (Sharpe 0.28) with near-zero realised loadings.
6. **Spanning runs both ways.** UMD has no robust alpha against industry momentum either (t = 1.54).
   They are largely the same risk, and the data cannot say which is more primitive.
7. **The FF3 model is sample-dependent.** GRS: F = 1.43 (p = 0.086) in 1963–91, but F = 3.71
   (p < 0.001) after. The small-value tilt is fully explained in-sample, but has a significant FF3
   alpha after 1991 and in all five international regions.
8. **Regimes.** Returns are far lower after bear markets (4.75 vs 10.01 %/yr) and in high-volatility
   states (6.16 vs 11.21 %/yr). The UMD loading is stable, so the regime dependence is inherited
   from momentum itself.

*A backtest is a hypothesis, not proof of a profitable strategy. Correlation in these regressions is
not causation.*

## Visual results

All figures: [`results/figures/`](results/figures) (PNG + vector PDF). All tables: [`results/tables/`](results/tables) (CSV + LaTeX).

### Replication of Fama & French (1993)
| | |
|---|---|
| ![](results/figures/F03_alpha_heatmaps_is.png) | ![](results/figures/F04_r2_heatmaps_is.png) |
| CAPM leaves size/value intercepts that FF3 absorbs (except small-growth) | SMB and HML capture common variation |
| ![](results/figures/F05_intercepts_vs_ff93.png) | ![](results/figures/F02_factor_reconstruction.png) |
| Our intercepts vs. FF93 Table 9a | Rebuilt vs. published SMB/HML |

![](results/figures/F01_factor_cumulative.png)

### Strategy vs. benchmark, drawdowns
![](results/figures/F08_strategy_cumulative.png)
| | |
|---|---|
| ![](results/figures/F09_strategy_legs.png) | ![](results/figures/F10_drawdowns.png) |

### Factor attribution: is it alpha?
![](results/figures/F11_alpha_by_model.png)
| | |
|---|---|
| ![](results/figures/F12_imom_loadings.png) | ![](results/figures/F17_risk_decomposition.png) |
| Factor exposures by model | Risk decomposition (share of variance) |
| ![](results/figures/F18_attribution_imom.png) | ![](results/figures/F18_attribution_smallvalue.png) |
| Factor contribution to mean return: industry momentum | ... and the small-value control |

![](results/figures/F16_residual_returns.png)

### Rolling betas, rolling alpha, hedge effectiveness
| | |
|---|---|
| ![](results/figures/F13_rolling_betas.png) | ![](results/figures/F14_rolling_alpha.png) |
| ![](results/figures/F19_hedge_effectiveness.png) | ![](results/figures/F15_rolling_betas_smallvalue.png) |

### In-sample vs. out-of-sample
| | |
|---|---|
| ![](results/figures/F20_is_vs_oos_imom.png) | ![](results/figures/F20_is_vs_oos_smallvalue.png) |
| ![](results/figures/F06_actual_vs_predicted_is.png) | ![](results/figures/F06_actual_vs_predicted_oos.png) |

### Robustness: parameters, costs, regimes, multiple testing, international
| | |
|---|---|
| ![](results/figures/F21_parameter_grid.png) | ![](results/figures/F22_transaction_costs.png) |
| ![](results/figures/F24_regime_exposures.png) | ![](results/figures/F23_multiple_testing.png) |
| ![](results/figures/F25_international.png) | ![](results/figures/F07_bond_factor_cumulative.png) |

### Key tables

**Industry momentum: the neutralisation ladder** (July 1963 – Aug 2026, Newey–West t in parentheses)

| Model | α (%/yr) | Mkt-RF | SMB | HML | RMW | CMA | UMD | adj. R² |
|---|---|---|---|---|---|---|---|---|
| CAPM | **9.40** (4.90) | −0.11 | | | | | | 0.01 |
| FF3 | **10.78** (5.75) | −0.13 | −0.07 | −0.30 | | | | 0.05 |
| FF3+UMD | **2.31** (2.09) | 0.03 | −0.05 | −0.01 | | | 0.86 | 0.67 |
| FF5 | **10.42** (4.92) | −0.11 | −0.08 | −0.39 | −0.04 | 0.24 | | 0.05 |
| FF5+UMD | **2.76** (2.46) | 0.03 | −0.08 | 0.01 | −0.13 | −0.01 | 0.87 | 0.67 |

**GRS tests on the 25 size/BE-ME portfolios**

| Model | FF93 F (32 assets) | F 1963–91 (p) | F 1992–2026 (p) |
|---|---|---|---|
| CAPM | 1.91 | 1.98 (0.004) | 3.84 (<0.001) |
| FF3 | 1.56 | 1.43 (0.086) | 3.71 (<0.001) |
| FF5 | — | 1.11 (0.33) | 3.34 (<0.001) |

The other 21 tables (performance by period, subperiods, grid, costs, regimes, international,
spanning, multiple-testing ledger, replication cell-by-cell) are in [`results/tables/`](results/tables).

## Reproduction

Requires Python 3.14 (tested 3.14.4), internet access for the first run, and a LaTeX distribution
for the PDF.

```bash
git clone https://github.com/sakshamg251206/factor-neutrality-ff1993.git && cd factor-neutrality-ff1993
make setup          # venv + pinned dependencies (requirements.txt; full lock in requirements-lock.txt)
make all            # tests -> data download -> all tables/figures -> notebooks -> paper PDF
```

Or step by step: `make test`, `make results` (≈15 s after download), `make notebooks`, `make paper`.
Every number in the paper is written by `scripts/run_all.py` to `results/macros.tex` and
`\input` by LaTeX, so no number is typed by hand. Results are deterministic (no randomness outside
the unit tests, which use a fixed seed). A download made after the French Data Library's next
monthly update will differ slightly; compare `data/raw/MANIFEST.json` hashes.

## Repository structure

```
├── src/factor_neutrality/   # library code
│   ├── data.py              #   download, cache, parse French & FRED data (+ SHA-256 manifest)
│   ├── factors.py           #   SMB/HML construction; TERM/DEF bond proxies
│   ├── strategy.py          #   industry momentum, turnover, costs, ex-ante hedge
│   ├── regression.py        #   factor regressions (Newey–West), GRS, rolling, attribution
│   ├── performance.py       #   performance stats, drawdowns, deflated Sharpe, Holm
│   ├── regimes.py           #   ex-ante bear / high-vol labels; NBER (ex-post)
│   └── plots.py             #   publication figure style
├── scripts/run_all.py       # end-to-end pipeline: every table, figure, macro
├── tests/test_core.py       # look-ahead, GRS, formulas, parsing, accounting tests
├── notebooks/               # 01 replication · 02 attribution · 03 OOS & robustness (executed)
├── data/                    # README (documentation), ff93_reference.csv (paper's numbers), raw/ (cache)
├── results/                 # tables/ (CSV+TeX), figures/ (PNG+PDF), macros.tex, summary.json
├── paper/                   # main.tex, references.bib, compiled PDF
├── docs/RESEARCH_PLAN.md    # workflow, hypotheses, bias & leakage audit
├── CITATION.cff · .zenodo.json · LICENSE · Makefile · pyproject.toml · requirements*.txt
```

## Paper

[`paper/Factor_Neutrality_FF1993_Garg_2026.pdf`](paper/Factor_Neutrality_FF1993_Garg_2026.pdf)
Sections: Abstract · Introduction · Literature · Hypotheses · Data · Factor construction ·
Methodology · Replication · Results · Factor attribution · Robustness · Out-of-sample analysis ·
Economic interpretation · Limitations · Conclusion · Future research · References.

## Limitations (short)

Portfolio-level backtest (within-industry turnover, shorting costs and borrow ignored); TERM/DEF
are yield-based proxies, and FF93's seven bond portfolios are unavailable (GRS on 25 vs 32 assets);
French data are revised over time; GRS assumes normal iid errors. See paper §13.

## Citation

```bibtex
@software{garg2026factorneutrality,
  author  = {Garg, Saksham},
  title   = {Alpha or Factor Exposure? A Replication of Fama and French (1993) and a
             Factor-Neutrality Test of Industry Momentum, 1963--2026},
  year    = {2026},
  version = {1.0.0},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.23144116},
  url     = {https://doi.org/10.5281/zenodo.23144116}
}
```
Please also cite Fama & French (1993). Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## Zenodo

Archived on Zenodo: **v1.0.0**, DOI [10.5281/zenodo.23144116](https://doi.org/10.5281/zenodo.23144116)
(record: https://zenodo.org/records/23144116). The concept DOI [10.5281/zenodo.23144115](https://doi.org/10.5281/zenodo.23144115)
always resolves to the latest version. Each GitHub release creates a new Zenodo version.

## Acknowledgements & licence

Data: Kenneth R. French Data Library; Federal Reserve Bank of St. Louis (FRED). Code and drafting
were assisted by an AI coding assistant (Anthropic's Claude); all results come from the published
pipeline. Code: MIT licence.
