# Interview notes

Five plain sentences per item, written by Claude Code. Rewrite each one in your own words.

## Step 0: data loader, validation and data report

1. The data module reads the Yahoo CSV, converts every timestamp to America/New_York, and refuses to continue if timestamps are naive, unsorted or duplicated, if high is below low, if a price is not positive, or if a gap is longer than the longest normal weekend or holiday break.
2. It always drops the newest bar, because that bar was still forming when the file was downloaded and its close is not final.
3. It treats the return of the 16:00 New York bar on the third Friday of March, June, September and December as missing in every year, a rule written down before any performance result so the cleaning cannot be tuned to flatter the strategy.
4. It classifies every gap as daily break, weekend, holiday or other, and reports zero-volume bars, off-the-hour bars and the largest single-bar moves, so unusual data is visible instead of hidden.
5. It writes a data report with the sha256 of the file, so every later result is tied to exactly one data snapshot.

## Item 1: walk-forward splitter with purge and embargo

1. The label y(t) is the log of close(t+H) over open(t+1), so it is exactly what a trade signalled at the close of bar t would earn, and the signal bar's own return is never part of it.
2. A sample is dropped when a gap longer than one bar lies between bar t and bar t+H, or when an expiry bar with a missing return lies inside its label window, because that label would contain a move that could not be traded.
3. The splitter trains on an expanding window and tests on the next fixed block of 500 bars, then steps forward one block, so the out-of-sample periods join into one continuous stretch and each test block is used once.
4. Purge removes every training sample whose label window reaches into the test block, and the embargo leaves a gap of at least H bars before the test block, which stops overlapping labels from leaking test information into training.
5. A lookback feature that reaches back across the train and test boundary is not a leak, because it reads only the past, which is information a live trader would have had at that moment.

Deviations from SPEC, declared on 2026-11-20 before any performance result: (a) the volume z-score feature is dropped from Phase 1 because 471 of 493 zero-volume bars are the 18:00 New York bar, and volume handling is decided in Phase 2 with a documented rule; (b) the last test block may be shorter than 500 bars so the final bars of the file are used.

## Item 2: cost model and net-of-cost metrics

1. Every position change pays commission plus slippage plus half the bid-ask spread per contract, and a flip from long to short trades two contracts so it pays two sides, while a position that does not change pays nothing.
2. The half-spread and slippage are fixed tick counts from the config because bar data has no quotes, so they are assumptions that I test at 0x, 1x, 2x and 3x rather than facts.
3. The backtest enters at the open of bar t+1 and exits at the close of bar t+H, which is the same quantity as the label, so a signal can never earn the return of the bar it was computed on, and a planted perfect-foresight signal earns almost nothing once it is lagged.
4. Sharpe is computed by summing P&L to daily USD for one contract per CME trading day and annualising with the square root of 252, because scaling an hourly Sharpe directly would inflate it.
5. The break-even cost is the multiple of my assumed costs at which mean net P&L reaches zero, and it can be solved exactly because net P&L is a straight line in the cost multiplier.

## Item 3: baselines and comparison table

1. Buy-and-hold and one momentum rule (long when the trailing 24-bar return is above zero, short otherwise) are the only two yardsticks, so any later model must beat them after costs on the same bars, the same valid samples and the same cost model.
2. I report two buy-and-hold lines: a tradable one that only earns the bars a signal could trade, which is the fair comparison, and a passive one that also earns weekend and daily-break moves, which is a reference only.
3. At each quarterly expiry the buy-and-hold line earns nothing on the roll bar and pays a roll cost, and the momentum rule pays the same roll cost for the position it holds, because the glued Yahoo series has a fake jump there.
4. Every Sharpe comes with a 95 percent circular block-bootstrap interval on daily P&L, resampling five-day blocks so that short-range dependence between days survives, because a plain shuffle would make the interval too narrow.
5. Every strategy, horizon, commission run and cost multiplier that is evaluated is written to results/trials.csv by the run itself, which is rebuilt each time, so the trial count for the deflated Sharpe comes from a file and not from memory.

## Item 4: permutation test and automated leakage tests

1. The permutation test asks how often a net Sharpe as high as the real one would appear if the signal carried no information about the future, and the p-value is (1 + the number of permuted Sharpes at least as large) divided by (N + 1), so with 200 permutations it can never be below 1/201.
2. For the momentum rule there is nothing to train, so I shift the signal circularly against the returns, which keeps the signal's own autocorrelation and trade count but breaks its link to the market; for a trained model I would instead shuffle the training labels inside each fold and retrain, so the test only ever destroys the relationship the model is meant to learn.
3. The truncation test cuts the data at many points and checks that every feature value before the cut is identical to the full-data value, which catches centred windows, negative shifts, back-fill and full-sample normalisation, and it caught a real calendar-lookahead bug in my own expiry-bar rule.
4. The canary is a feature built from the next bar's return, and it is needed because a leakage test that passes everything is worthless: the truncation test must fail on it and a naive backtest of it must show an absurd Sharpe.
5. The label-boundary and fit-on-train-only tests check, fold by fold, that no training label window reaches into the test block and that the model never saw a bar at or after the start of its test block.

## Item 5: findings write-up, summary and figures

1. The pipeline writes results/summary.md and inserts the results table into docs/FINDINGS.md by script, and a test fails if the prose quotes any number that is not in the summary, so I cannot retype a result wrongly.
2. The first line of Results says that no evidence of an edge was found, because every Sharpe interval includes zero and the permutation p-values are 0.766 and 0.682, and a null result stated plainly is the honest outcome.
3. The passive buy-and-hold earns much more than the tradable one because it also collects the first hour after each weekend and daily break and the moves between bars, which no signal computed before the gap could trade, so the tradable line is the fair comparison.
4. The write-up says the window is only 1.6 years, that the costs and the June 2026 expiry fallback are assumptions, that the Yahoo roll handling is unverified, and that April 2025 sits in training only, so a reader knows what the result can and cannot support.
5. The figures use real data only, label the costs as assumptions in their titles, and show the equity curves, the drawdowns and the permutation histogram with the real Sharpe marked against the shifted-signal null.

## Item 6: make run, pytest and CI

1. One command, `make run`, reads the config and the local data file, uses the single seed from the config, and writes every table, figure, summary and `run_info.json`, so every number in the write-up traces to a file produced by that command.
2. `run_info.json` records the git commit, the pinned and installed package versions, the sha256 of the data file and the config, which ties a result to one data snapshot and one code version.
3. `make reproduce-check` runs the pipeline twice and compares the sha256 of every result file, which shows that the same seed and the same data give identical outputs on this machine.
4. `make smoke` runs the same code on synthetic random-walk bars and writes only files whose names start with SYNTHETIC, so the pipeline can be tested in CI without the private data and a synthetic number can never be mistaken for a result.
5. CI runs ruff, pytest and the synthetic smoke run on Python 3.13, but it cannot run on the real data because the data is private, so it proves the code runs and the tests pass and does not prove that the real results reproduce.
