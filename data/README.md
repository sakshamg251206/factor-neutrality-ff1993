# Data documentation

All data are public and downloaded on demand by `src/factor_neutrality/data.py` into `data/raw/`.
Raw files are **not** redistributed in this repository (they belong to their providers); instead
`data/raw/MANIFEST.json` records each file's URL, SHA-256 hash and download time, so you can tell
whether your download matches the one behind the published results.

## Vintage

The Kenneth R. French Data Library rebuilds its files monthly and revises history when CRSP and
Compustat are corrected. The results in this repository use files **built from the 202608 CRSP
database** (downloaded October 2026). A later download will give slightly different numbers; the
paper's conclusions are about magnitudes that are far larger than these revisions.

## Series

| File / series | Source | Content | Used for | Sample |
|---|---|---|---|---|
| `F-F_Research_Data_Factors` | French | Mkt-RF, SMB, HML, RF (monthly, %) | FF3 factors, risk-free rate | 1926-07 → |
| `F-F_Research_Data_5_Factors_2x3` | French | Mkt-RF, SMB, HML, RMW, CMA, RF | FF5 factors (its own SMB kept as `SMB5`) | 1963-07 → |
| `F-F_Momentum_Factor` | French | Mom (= UMD) | Carhart momentum factor | 1927-01 → |
| `6_Portfolios_2x3` | French | 6 value-weighted size × BE/ME portfolios | Rebuilding SMB and HML | 1926-07 → |
| `25_Portfolios_5x5` | French | 25 value-weighted size × BE/ME portfolios | FF93 test assets; small-value tilt (S1B5) | 1926-07 → |
| `49_Industry_Portfolios` | French | 49 value-weighted industry portfolios | Strategy universe | 1926-07 → (Hlth from 1969, Softw from 1965) |
| `{Region}_3_Factors` | French | Regional Mkt-RF, SMB, HML, RF (USD) | International tests | 1990-07 → |
| `{Region}_25_Portfolios_ME_BE-ME` | French | Regional 25 size × BE/ME portfolios (USD) | International small-value tilt, GRS | 1990-07 → |
| `GS10` | FRED | 10-year Treasury constant-maturity yield, monthly average | TERM proxy | 1953-04 → |
| `BAA` | FRED | Moody's seasoned Baa corporate yield, monthly average | DEF proxy | 1919-01 → |
| `USREC` | FRED | NBER recession indicator | Descriptive regime labels | 1854 → |

Regions: `Developed`, `North_America`, `Europe`, `Japan`, `Asia_Pacific_ex_Japan`.

## Processing

* Percent returns are converted to decimals; French's missing-value codes (-99.99, -999) become NaN.
* Every observation is indexed by its month-end date.
* Only the first table in each French CSV (value-weighted monthly returns) is used.
* **TERM and DEF are approximations.** FF93 used Ibbotson long-term government and corporate bond
  returns, which are proprietary. We convert yields into returns by pricing a 20-year par bond:
  bought at par at t-1 with coupon y(t-1), repriced at y(t). TERM = R(GS10) - RF; DEF = R(BAA) - R(GS10).
  Monthly-average yields smooth these returns, and Baa is lower-grade than Ibbotson's composite.

## When each variable is known (look-ahead audit)

| Variable | Known at | Notes |
|---|---|---|
| Portfolio and factor returns for month t | end of month t | Used only for month-t P&L or for signals formed at the end of t |
| BE/ME sorts behind SMB/HML | June of year t | Book equity from fiscal years ending in t-1 (at least six months old); done by the library |
| Industry-momentum signal at t | end of month t | Uses returns for months t-11..t-1 only (skip month t) |
| Ex-ante hedge betas for month t | end of month t-1 | Trailing 60-month OLS ending at t-1 |
| Bear-market / high-volatility labels for t | end of month t-1 | Trailing market returns and an expanding median ending at t-1 |
| NBER recession dates | months to years later | **Ex-post**; used only for descriptive attribution |

## Survivorship

CRSP includes delisted firms and delisting returns, so the French portfolios do not suffer from
survivorship bias. The 49-industry classification is fixed over time. Compustat backfilling affects
early book equity; FF93 address it with a two-year listing requirement that the library retains.
