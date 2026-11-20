# Findings: walk-forward baselines on NQ futures (Phase 1)

## Question

Does a small, pre-declared signal on NQ futures beat buy-and-hold after costs, out of sample? Phase 1 evaluates only the baselines: buy-and-hold, and a momentum rule (long when the trailing 24-bar return is above zero, short otherwise), at horizons of 1 and 2 hourly bars.

## Data

One-hour NQ=F bars from Yahoo Finance, an unofficial source, downloaded on 2026-11-19 as a fixed snapshot. 13,735 of 13,736 rows are used; the last bar was still forming. Yahoo's contract-roll handling is unverified, so the 16:00 New York bar on each third Friday of March, June, September and December is treated as missing (9 bars). For 2026-06-19, a holiday, the previous trading day's bar is used: an assumption, not a verified CME fact, declared on 2026-11-20 before any result. Gap-spanning samples are excluded, leaving 13,111 valid samples at H=1 and 12,497 at H=2. Of 493 zero-volume bars, 471 are the 18:00 New York bar, so the volume z-score was dropped from Phase 1 and is left to Phase 2.

## Method

Walk-forward, expanding window, 18 folds (test blocks of 500 bars, 235 in the last), with purge and embargo. A signal uses data up to the close of bar t, enters at the open of t+1 and exits at the close of t+H. Sharpe uses daily USD P&L for one contract, annualised with sqrt(252). Costs are assumptions: 7.50 USD per side in the main run (commission 0.00, slippage 1.0 tick, half-spread 0.5 tick) and 10.00 USD in a conservative run. Intervals are 95 percent block bootstraps (2000 paths, 5-day blocks); the permutation test uses 200 circular shifts.

## Results

No evidence of an edge was found. The out-of-sample window, 2025-05-15 to 2026-11-19, has 8,735 bars, 392 trading days, 1.6 years: short, so every interval is several Sharpe units wide.

<!-- RESULTS_TABLE_START -->
Main run, commission 0.00 USD per side, 7.50 USD per side in total (ASSUMPTIONS). Out-of-sample, net and gross, 95 percent block-bootstrap intervals.

| Strategy | H | Gross Sharpe [95% CI] | Net Sharpe [95% CI] | Sides | Sides/day | Net P&L/yr USD | Max DD USD | Hit rate | Break-even cost x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum | 1 | -0.60 [-1.89, 0.75] | -0.67 [-1.96, 0.69] | 1401 | 3.57 | -69,356 | 142,285 | 0.49 | 0.00 |
| momentum | 2 | -0.35 [-1.59, 1.09] | -0.40 [-1.64, 1.04] | 993 | 2.53 | -37,911 | 141,620 | 0.49 | 0.00 |
| buy-and-hold, tradable samples (like-for-like) | 1 | 0.21 [-1.30, 1.68] | 0.21 [-1.30, 1.68] | 1 | 0.00 | 22,566 | 132,105 | 0.54 | 361.03 |
| buy-and-hold, tradable samples (like-for-like) | 2 | 0.16 [-1.37, 1.61] | 0.16 [-1.37, 1.61] | 1 | 0.00 | 15,967 | 131,570 | 0.53 | 255.74 |
| buy-and-hold, passive (reference) | n/a | 0.67 [-0.81, 2.09] | 0.67 [-0.81, 2.08] | 1 | 0.00 | 71,317 | 97,940 | 0.54 | 1138.82 |
<!-- RESULTS_TABLE_END -->

Momentum's net Sharpe is -0.67 [-1.96, 0.69] at H=1 and -0.40 [-1.64, 1.04] at H=2. Like-for-like buy-and-hold has 0.21 [-1.30, 1.68]. Every interval includes zero; the H=1 difference of -0.88 [-3.02, 1.38] is not distinguishable from zero. Permutation p-values are 0.766 (H=1) and 0.682 (H=2); the smallest possible is 0.005. Costs are not the cause: gross Sharpe is -0.60 at H=1.

Passive buy-and-hold (net Sharpe 0.67 [-0.81, 2.08]) earns 111,035 USD gross against 35,200 USD for the tradable line. Of the 75,835 USD gap, 47,555 USD is movement inside bars no signal can trade, mainly the 389 first bars after a gap (47,030 USD, of which 34,785 USD from 66 weekend reopens); 28,280 USD is movement between bars. The tradable line is the like-for-like comparison; the passive line is a labelled reference. Figures are in results/figures.

## Limitations

- Slippage and half-spread are assumptions; there are no quotes or fills.
- Yahoo data, roll handling and the expiry rule are unverified.
- The first test block starts on 2025-05-15, so the April 2025 tariff period (6 of the 10 largest single-bar moves) is in training only.
- The result concerns this rule, with L = 24, on this window and data, and nothing broader.

## Failures and negative results

Momentum's gross Sharpe is negative. Only L = 24 was evaluated, nothing was tuned, and all 42 configurations are logged in results/trials.csv.

## What I did not do in Phase 1

No machine learning, volatility regimes or sensitivity grid, deflated Sharpe, ADF, ACF, PACF, GARCH, risk simulator, or ES run.

## What I would do next

Handle the 18:00 volume, add models with the same permutation test, deflate Sharpe by trial count, and use longer, verified history.
