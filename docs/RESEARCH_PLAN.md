# Research plan and bias audit

## Question
Is a systematic strategy generating alpha, or is it being paid for exposure to known factors?

## Workflow (as executed)
1. **Read** Fama & French (1993), sections 1–7 and Tables 1–9c. Transcribe the replication targets
   into `data/ff93_reference.csv`. OCR-ambiguous cells were checked against each table's own
   t-statistics using t = mean / sd · √342.
2. **Replicate** on the paper's sample (1963-07 → 1991-12): rebuild SMB/HML, Table 2 factor
   statistics, Tables 4/6 R², Table 9a intercepts, Table 9c GRS statistics.
3. **Strategy.** Industry momentum (49 industries; 12-1 formation; top/bottom 10; monthly). The
   parameters were fixed a priori from the literature before any results were seen.
4. **Neutralise.** CAPM → FF3 → FF3+UMD → FF5 → FF5+UMD; ex-ante beta hedge; spanning both ways.
5. **Validate.** IS/OOS split at the end of the FF93 sample; pre-1963 sample; 16-variant grid chosen
   IS and judged OOS; deflated Sharpe ratio; cost sweep; hedge-window sensitivity; ex-ante regimes;
   Holm and Harvey–Liu–Zhu multiple-testing ledger; international control test.
6. **Write up**, publish the code, then archive it on Zenodo, in that order.

## Pre-stated hypotheses
H1 replication · H2 factor compensation (control) · H3 raw strategy premium ·
H4 factor neutrality (main) · H5 out-of-sample decay · H6 regime dependence. See paper §3.

## Bias & leakage audit: where it was checked

| Risk | Where it could enter | Safeguard | Verified by |
|---|---|---|---|
| Look-ahead in signal | `strategy.momentum_signal` | rolling sum shifted by `skip`; NaN until full window | `test_signal_and_strategy_ignore_future_returns`, `test_skip_month_excludes_most_recent_return` |
| Look-ahead in P&L | `strategy.strategy_returns` | weights lagged one month before multiplying returns | `test_strategy_uses_weights_formed_last_month` |
| Look-ahead in hedge | `strategy.hedge_ex_ante` | betas from months t-60..t-1 only | `test_ex_ante_hedge_ignores_future` |
| Look-ahead in regimes | `regimes.py` | labels use data to t-1; NBER flagged ex-post | `test_regime_labels_are_ex_ante` |
| Look-ahead in accounting data | factor construction | library forms portfolios in June using fiscal-year t-1 book equity | FF93 §2.1.2 |
| Parameter snooping | grid | base spec fixed a priori; grid chosen IS, judged OOS; DSR | `T14`, `F21` |
| Multiple testing | all alpha tests | every test logged; Holm; t > 3 | `T17`, `F23` |
| Survivorship | universe | CRSP incl. delistings; fixed industry set | `data/README.md` |
| Missing data | industries starting late | excluded until 12 months of history; no held industry has a missing return | `test_no_held_industry_has_missing_next_month_return` |
| Data vintage | French revisions | SHA-256 manifest; vintage reported in paper | `data/raw/MANIFEST.json` |
