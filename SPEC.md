# Signal Research Lab - Full Project Specification

Nov 19, 2026 · @yotam

## 1. Goal and success criteria

The goal is to show, with honest evidence, that you can build and test a trading signal on NQ futures without fooling yourself. The deliverable is a reproducible research pipeline for the Brevan Howard Tel Aviv quant internship application. It is not a profitable strategy. A correct method that finds no edge is a good outcome.

**The project succeeds when:**

- One command (`make run`) reproduces every number in the write-up from a clean checkout, with fixed seeds.
- No lookahead exists, and an automated test proves it instead of a claim.
- Every model is compared to buy-and-hold and one simple rule, after costs, on identical splits.
- A permutation test shows the signal disappears when labels are shuffled.
- Weak and negative results are reported as weak and negative.
- You can explain every module in an interview without notes.

**Non-goals:**

- No live trading, no broker connection, no order execution.
- No claim of a profitable strategy unless the decision rule in section 8 is met.
- No synthetic data in reported results. Synthetic data is allowed only inside unit tests and the CI smoke run, and is labelled as such.
- No NLP or LLM signal (for example FOMC statements). It stays out of scope unless every other item is finished, and then only as a clearly marked stretch.

## 2. Research question, target and scope

**The question:** does any small, pre-declared set of features predict the direction of NQ over the next H bars, out of sample, after realistic costs, better than buy-and-hold and one simple rule?

**Timing convention (everything depends on this, and it is tested):**

- A signal is computed from data up to and including the close of bar t.
- The trade is entered at the open of bar t+1. A signal can never earn the return of the bar it was computed on.
- The position is held for H bars and exited at the close of bar t+H.
- The label is y(t) = log(close(t+H) / open(t+1)). It is the same quantity the strategy earns, so label and P&L cannot disagree.
- Classification target: sign of y(t). Regression target (ridge): y(t) itself.
- When H is larger than 1, labels of neighbouring samples overlap. That is why splits need purging (section 5, item 1).

**Features (declared in advance, small, scale-free):**

- Lagged log returns over 1, 5 and 20 bars.
- Rolling realised volatility over 20 bars.
- Distance of price from its moving average, in units of rolling volatility.
- Volume z-score over a rolling window.
- Bar range divided by rolling average range.
- Intraday data only: time of day. Day of week for all data.
- Every feature uses a trailing window ending at t. No feature selection on the full sample.

**Baselines (declared in advance):**

- Buy-and-hold, one contract.
- One simple rule: momentum, long when the trailing L-bar return is above zero and short otherwise. L is fixed in the config before any result is seen. Mean reversion is not run as a second rule, because every extra rule is an extra trial that must be counted in the deflated Sharpe.

**Scope:**

- Instrument: NQ continuous front-month, the main study. ES: the same pipeline and the same config, run once as a robustness check with no re-tuning.
- Data decided: 1-hour bars of NQ=F from Yahoo Finance (yfinance), unadjusted, downloaded on 2026-11-19. It has 13,736 bars from 2024-06-30 to 2026-11-19, timestamps in America/New\_York, and no missing values in the six columns. The series starts on a Sunday evening, so it covers nearly the full Globex session and not only regular hours. Because bars are 1 hour, H = 1 means one hour. The source is unofficial and its handling of contract rolls is not verified, see the roll check in section 3.
- Sample size is a real constraint. Ten years of daily bars is about 2,500 observations, which is thin for gradient boosting and an MLP. Regular-hours 5-minute bars give 78 bars per day, about 19,500 per year, but then costs matter much more. The write-up must state which one was used and what that means for power.

## 3. Data

The project needs two different kinds of data, and neither is in the repo. Market data drives the signal research. Your own trade history drives only the risk simulator (item 9). Claude Code must ask for the path of each file and must never generate substitute data for reported results.

**Market data (NQ main, ES robustness):**

- Columns: timestamp, open, high, low, close, volume. Contract id if the vendor provides it.
- Timestamps are timezone-aware. CME session logic uses America/Chicago, and daylight saving transitions must be tested.
- The loader validates and fails loudly on: unsorted or duplicate timestamps, high below low, non-positive prices, naive timestamps, and gaps larger than the expected session breaks. It writes a data report (row count, date range, gaps, sha256 of the file) into the results folder.
- **Contract rolls:** a continuous series built by gluing contracts together contains fake returns at roll dates. Either use a vendor series that is already adjusted and record which adjustment, or compute returns inside each contract and drop the return that spans a roll. Features are built from returns, not price levels, because back-adjusted price levels can drift and even go negative.
- Roll dates for the equity index futures are quarterly (March, June, September, December). The loader must report the largest single-bar moves so any unflagged roll gap is visible.

**Data quality findings from the first download (2026-11-19):**

- The last bar was still forming when the data was downloaded. The loader always drops the most recent bar.
- Gaps between bars are mostly normal: the daily break, weekends and US holidays. Four long gaps end at 00:00, so some Sunday-evening bars are missing, for example on 2025-07-14 and 2025-09-22. The data report lists them and does not fix them.
- The 16:00 New York bar on three quarterly expiry Fridays has a move far above the typical daily maximum hourly move of 0.57 percent: 2024-09-20 (1.01 percent), 2024-12-20 (2.86 percent) and 2025-03-21 (2.64 percent). The five later expiries do not show it. It is probably a roll artifact of the Yahoo series. This is not verified, and the cause is unknown.
- **Cleaning rule, fixed before any performance result is seen:** the return of the 16:00 New York bar on the third Friday of March, June, September and December is treated as missing in every year, whatever its size. Any sample whose feature window or label window contains a missing return is excluded.
- **Gap-aware labels:** a sample is excluded when any gap longer than one bar interval (daily break, weekend, holiday) lies between bar t and bar t+H, because the label would include a move that could not be traded from the signal.
- The write-up states these findings and the number of samples each rule removed.

**Contract facts used by the cost model:**

- NQ: tick size 0.25 index points, tick value 5 USD, multiplier 20 USD per point.
- ES: tick size 0.25 index points, tick value 12.50 USD, multiplier 50 USD per point.
- Commission per side per contract and slippage in ticks are not facts. They are assumptions that live in the config, with a comment naming the source. See open decision D4.

**Your trade history (risk simulator only):**

- One row per closed trade: entry time, exit time, side, quantity, instrument, gross P&L, fees.
- Stays on your machine. Never committed.

**Privacy and licensing:**

- The `data/` folder is in `.gitignore` from the first commit. No raw prices and no trade history are ever committed.
- Exchange data is usually licensed and cannot be republished. A public repo shows only aggregate results and code.
- Tests use tiny hand-made or generated fixtures, labelled as synthetic.

## 4. Repo structure, tooling and reproducibility

The code is split into small modules, one per concept, so each can be explained in one sentence.

```
signal-research-lab/
  README.md
  Makefile
  requirements.txt            pinned versions
  config/default.yaml         every parameter and every assumption
  data/                       gitignored, local only
  src/signal_lab/
    data.py                   load and validate
    features.py               trailing-window features
    labels.py                 forward return label
    splits.py                 walk-forward with purge and embargo
    costs.py                  commission, slippage, half-spread
    backtest.py               signal to positions to net P&L
    baselines.py              buy-and-hold, momentum rule
    metrics.py                Sharpe, drawdown, turnover, bootstrap CI
    stats_tests.py            permutation test, deflated Sharpe, ADF
    models.py                 ridge, logistic, gradient boosting, MLP
    regimes.py                volatility regimes, sensitivity grid
    timeseries.py             ACF, PACF, GARCH
    risk_sim.py               bootstrap, drawdown, ruin, Kelly
    report.py                 tables and figures
  scripts/run_all.py          the single entry point
  tests/                      see section 7
  results/                    generated, small tables and figures
  docs/FINDINGS.md            the one-page write-up
  .github/workflows/ci.yml
```

**Tooling:**

- Python 3.11 or newer. pandas, numpy, scipy, scikit-learn, statsmodels, arch (for GARCH), matplotlib, pyyaml, pytest, ruff.
- Gradient boosting uses scikit-learn's own implementation, so there is no extra dependency to explain.

**Config:**

- One YAML file holds paths, bar frequency, H, L, window lengths, cost assumptions, number of permutations, number of bootstrap paths, and the seed.
- A parameter that is not in the config is not allowed to exist in the code as a magic number.

**Reproducibility:**

- One seed in the config, passed explicitly into every random operation through numpy's `default_rng`. No global random state.
- Every run writes `results/run_info.json` with the git commit hash, package versions, the sha256 of each data file, and the config used. Results are tied to a data version.
- Tiny floating point differences across machines are possible for gradient boosting and the MLP. Tests that compare numbers use tolerances, and the write-up says so.
- Results are committed as small tables and figures. Raw data is never committed.

## 5. Phase 1 - core (items 1 to 6)

Phase 1 must be finished, tested and committed before any Phase 2 work starts. Each item is one commit. Each item lists what it does, what it outputs, and the test that must pass.

### Item 1. Walk-forward framework

- **What:** an expanding train window with a minimum length, followed by a fixed-length test block. The window then steps forward by one test block. Rolling train windows are a config option. The out-of-sample signals of all test blocks are concatenated into one series, with the fold id attached.
- **Purge:** drop from the training set every sample whose label window (bars t+1 to t+H) reaches into the test block. This is what prevents the label overlap leak.
- **Embargo:** keep a gap of at least H bars between the end of train and the start of test. In a forward-only walk-forward, features that look back across the boundary are not a leak, because they use only the past. Be ready to explain exactly why.
- **Fit on train only:** scalers, imputers and models are fitted on the training rows of each fold and nothing else. Hyperparameter choice happens only inside the training window, using an inner chronological split.
- **Test block is touched once.** No parameter is changed after looking at test results.
- **Output:** `splits.py` yields (train\_index, test\_index) pairs. `results/folds.csv` lists fold id, train range, test range, number of samples.
- **Tests:** for every fold the latest label end in train is before the first test timestamp. Test blocks do not overlap and together form one contiguous out-of-sample period. Checked on small deterministic arrays.
- **Commit:** `phase1 item 1 - walk-forward splitter with purge and embargo`

### Item 2. Realistic costs

- **What:** every change of position pays commission, slippage and half the bid-ask spread. Cost in USD per trade = contracts traded x (commission per side + (slippage ticks + half-spread ticks) x tick value). Costs are charged on position changes only.
- **Bid-ask with bar data:** bar data has no quotes, so the half-spread is a fixed number of ticks from the config and is labelled as an assumption in the write-up.
- **Sharpe:** strategy P&L is aggregated to daily P&L first, and then annualised with the square root of 252. Intraday bar Sharpe is never scaled directly. Risk-free rate is zero and the write-up says so.
- **Report for every strategy:** gross Sharpe, net Sharpe, trades per day, cost drag (gross minus net), the break-even cost level at which net Sharpe reaches zero, and net Sharpe at 0x, 1x, 2x and 3x the assumed costs.
- **Tests:** zero trades means zero cost. A hand-computed three-trade example equals the function output. Net is never above gross when costs are non-negative. A signal computed at bar t cannot earn the return of bar t (execution lag test, with a planted perfect-foresight signal).
- **Commit:** `phase1 item 2 - cost model and net-of-cost metrics`

### Item 3. Baselines

- **Buy-and-hold:** long one contract over the whole out-of-sample window, with one entry cost and a roll cost at each roll date. The roll cost is a config assumption.
- **Simple rule:** momentum with the lookback L from the config. One rule only.
- **Comparison:** one table on the same out-of-sample window, the same cost model and the same splits: net Sharpe, annualised return, volatility, max drawdown, turnover, hit rate.
- **Uncertainty:** every Sharpe gets a 95 percent confidence interval from a block bootstrap, so a difference of 0.1 is not presented as a finding.
- **Rule:** every later model is compared to these two rows and nothing else.
- **Tests:** buy-and-hold on a toy price series equals the known price change. Baseline signals pass the leakage test of item 4.
- **Commit:** `phase1 item 3 - baselines and comparison table`

### Item 4. Permutation test and automated leakage test

**Permutation test.** It asks: if the features carried no information about the future, how often would I see a net Sharpe this high?

- For a trained model: shuffle the training labels inside each fold, retrain, produce the out-of-sample signal, and record the net Sharpe. Repeat N times (the config sets N, for example 200, and the write-up states it). The p-value is (1 + number of permuted Sharpes at least as large as the real one) / (N + 1).
- For a rule with nothing to train (the momentum baseline): apply random circular shifts of the signal against the returns. This keeps the signal's own autocorrelation and breaks its link to returns.
- The permuted Sharpe distribution should be centred near zero or slightly negative because of costs. A distribution centred elsewhere signals a bug.
- The p-value is not corrected for the number of things you tried. The deflated Sharpe in item 10 handles that.

**Automated leakage tests (pytest):**

- **Truncation test:** cut the data at random points t. Features computed on the truncated data must equal the features of the full data for every timestamp up to t. This catches centred rolling windows, negative shifts, back-fill and full-sample normalisation.
- **Label boundary test:** for each fold, no training label window overlaps the test block.
- **Fit-on-train-only test:** a wrapper around fit records the latest timestamp it saw. It must be before the test start.
- **Canary test (a test of the test):** deliberately add a feature equal to the future return. The leakage tests must fail on it, and the naive backtest must show an absurdly high Sharpe. If the leakage test passes on the canary, the leakage test is broken.
- **Ordering test:** data is sorted and timestamps are unique.
- **Permutation test sanity:** on synthetic data with a planted signal the test detects it. On pure noise it does not reject at the 1 percent level across several fixed seeds.
- **Commit:** `phase1 item 4 - permutation test and leakage tests`

### Item 5. One-page findings write-up

- `docs/FINDINGS.md`, about 500 to 600 words, with these headings: Question, Data, Method, Results, Limitations, Failures and negative results, What I would do next.
- The results table is generated by the pipeline into `results/summary.md` and inserted by script, so numbers are not retyped by hand. Numbers quoted in the prose are checked by hand against that file before the commit.
- Negative results are required content, not an appendix. If the signal is not distinguishable from the permutation null, the document says that in the first line of Results.
- **Commit:** `phase1 item 5 - findings write-up`

### Item 6. One command, tests and CI

- **Makefile targets:** `setup` (virtual environment and pinned install), `test`, `run` (data validation, every Phase 1 experiment, report), `smoke` (the whole pipeline on a tiny synthetic fixture), `reproduce-check` (run twice and compare result hashes), `clean`.
- **`make run`** is a thin wrapper over `scripts/run_all.py`. It reads the config, uses the fixed seed, and writes tables, figures, `summary.md` and `run_info.json`.
- **CI (GitHub Actions)** runs on every push and pull request: install, ruff, pytest, and `make smoke`.
- **Honest limit of CI:** the real data is private, so CI cannot run `make run` on it. CI proves that the code runs and the tests pass. It does not prove the real results reproduce. That is proven locally by `make reproduce-check`, and the README says so.
- **Commit:** `phase1 item 6 - make run, pytest and CI`

**Gate 1 (all required before Phase 2):** fresh clone, `make setup`, `make test` green, `make run` finishes on the real data, `make reproduce-check` shows identical hashes, the canary test fails as designed, `docs/FINDINGS.md` exists, CI is green.

## 6. Phase 2 - final stage (items 7 to 11)

Phase 2 starts only after Gate 1. Every Phase 2 item reuses the Phase 1 splitter, cost model, baselines, permutation test and leakage tests. Nothing in Phase 2 gets its own private evaluation code.

### Item 7. ML models inside the walk-forward loop

- **Models:** ridge regression (forward return target), logistic regression (direction target), gradient boosting, and a small MLP (one or two small hidden layers).
- **Signal from a model:** sign of the predicted return, or predicted probability above 0.5 plus a dead zone from the config, inside which the position is flat. The dead zone cuts turnover and cost.
- **Hyperparameters:** a small fixed grid, chosen by an inner chronological split inside each training window. The MLP early-stopping validation set is the last 20 percent of the training window in time order, never a random split. The number of configurations tried is logged, because the deflated Sharpe needs it.
- **Comparison:** one table with ridge, logistic, gradient boosting, MLP, momentum rule and buy-and-hold on identical splits and identical costs. Each model also gets the permutation test.
- **Interpretation:** ridge coefficients and permutation importance per fold, with a note on how stable they are across folds.
- **Expectation to state honestly in the write-up:** on a small financial sample, flexible models usually overfit. A null result here is a normal and acceptable finding.
- **Tests:** all models use the same splitter object. Results are identical across two runs with the same seed. The fit-on-train-only test is extended to every model.
- **Commit:** `phase2 item 7 - ML models in walk-forward loop`

### Item 8. Stability analysis

- **Volatility regimes:** classify each bar as high or low volatility by comparing trailing realised volatility with its own past distribution (an expanding median computed up to t). A full-sample median is a lookahead and is forbidden. Report per regime: net Sharpe with confidence interval, hit rate, number of bars.
- **Time stability:** net Sharpe by fold and by calendar year, and how many folds are positive.
- **Parameter sensitivity:** a grid over lookback L, horizon H, dead zone and cost multiplier. Show net Sharpe as a heatmap. Report the share of grid cells with positive net Sharpe. A single bright cell surrounded by losses is a sign of overfitting, not of an edge.
- **Rule:** the best cell is never chosen and promoted. The whole grid counts as trials in the trial log.
- **Tests:** the regime label at time t is unchanged when later data is removed. Grid results are deterministic.
- **Commit:** `phase2 item 8 - regime and sensitivity analysis`

### Item 9. Risk and position-sizing simulator

- **Inputs:** (a) the per-trade or per-day net P&L of the signal from the walk-forward run, and (b) your own closed-trade history.
- **Method:** a block bootstrap of the P&L sequence generates many equity paths (the config sets the number, for example 10,000) over a fixed horizon. A block bootstrap keeps some of the serial dependence that a plain shuffle destroys.
- **Outputs:** distribution of maximum drawdown (median and 95th percentile), distribution of terminal equity, and probability of ruin. The ruin threshold (for example a loss of a stated percentage of starting capital, or an account drawdown limit) is a config parameter. See open decision D6.
- **Fractional Kelly:** estimate the growth-optimal fraction from per-trade mean and variance (approximately mean divided by variance for small edges), then simulate sizing at 0.25, 0.5 and 1.0 of that fraction. Show median growth, drawdown and ruin probability for each.
- **Estimation error is part of the result:** bootstrap the Kelly estimate itself and show its confidence interval. With a short trade history the interval is wide, which is the reason full Kelly is dangerous.
- **Stated limits:** the bootstrap assumes the future resembles the sample. A short or lucky history gives a flattering simulation. Your own history is one realisation, selected by the fact that you kept trading.
- **Tests:** a coin-flip case with a known edge matches the closed-form gambler's ruin probability within tolerance. Same seed gives the same paths.
- **Commit:** `phase2 item 9 - risk simulator with bootstrap and fractional Kelly`

### Item 10. Deflated Sharpe, ADF, ACF and PACF, GARCH

- **Trial log:** from Phase 1 on, every strategy, model and parameter configuration that is evaluated is appended automatically to `results/trials.csv`. The number of trials is counted from this file, not from memory.
- **Deflated Sharpe ratio:** the Bailey and Lopez de Prado adjustment of the observed Sharpe for the number of trials, the variance of trial Sharpes, sample length, skewness and kurtosis. Output: the probability that the true Sharpe is above zero after deflation. Implement from the published formula and cite the paper in the code.
- **ADF test (statsmodels):** on price level, log returns and the main features. Report statistic, p-value and lag choice. Expectation: price level fails to reject a unit root, returns reject it. Note that ADF on a back-adjusted series has its own caveats.
- **ACF and PACF:** of returns and of squared returns, with confidence bands, plus Ljung-Box tests on both. Expectation: little autocorrelation in returns, clear autocorrelation in squared returns (volatility clustering).
- **GARCH(1,1)** with Student-t innovations (arch package) on returns. Report parameters, persistence (alpha plus beta), and Ljung-Box on squared standardised residuals as a fit check.
- **Use of GARCH:** one-step-ahead volatility forecasts inside the walk-forward loop, fitted on train and forecast on test. Compare with simple rolling realised volatility as a forecast of next-period realised variance. Report which one forecasts better, and say if the answer is that neither does.
- **Tests:** deflated Sharpe falls as trials rise, rises with sample length, and equals the probabilistic Sharpe ratio when trials equal 1. GARCH parameters are roughly recovered on a simulated GARCH process. ADF rejects on white noise and does not on a simulated random walk.
- **Commit:** `phase2 item 10 - deflated Sharpe, ADF, ACF and PACF, GARCH`

### Item 11. Update write-up and README

- Add Phase 2 results to `docs/FINDINGS.md` and keep the one-page core. Extra tables go into a short appendix.
- A section called What did not work is mandatory.
- README sections: what this is, headline results with their caveats, how to run, repo structure, data note (data not included), limitations, and a status table with one row per item showing done, partial or not done.
- **Commit:** `phase2 item 11 - findings and README update`

**Gate 2:** same checks as Gate 1, plus all Phase 2 tests green and `make run` regenerating every Phase 2 table and figure.

## 7. Testing strategy and CI

The tests exist to catch the specific ways quant research lies to its author. Each test file below maps to one such failure.

| Test file | What it proves |
| --- | --- |
| test\_data\_validation | Bad data (unsorted, duplicate, naive timestamps, high below low) is rejected. |
| test\_features\_no\_lookahead | Truncating the data never changes earlier feature values. |
| test\_splits | Purge and embargo hold for every fold. Test blocks are contiguous and do not overlap. |
| test\_execution\_lag | A signal at bar t cannot earn the return of bar t. A planted perfect-foresight signal earns nothing when lagged correctly. |
| test\_costs | Zero trades cost zero. Hand-computed examples match. Net does not exceed gross. |
| test\_leakage\_canary | A feature built from the future return is caught by the leakage tests. |
| test\_fit\_on\_train\_only | No model or scaler sees test rows during fit. |
| test\_permutation | A planted signal is detected. Noise is not. |
| test\_regimes\_no\_lookahead | Regime labels do not change when later data is removed. |
| test\_risk\_sim | Bootstrap results match a closed-form case. Same seed gives the same paths. |
| test\_stats | Deflated Sharpe, ADF and GARCH behave correctly on simulated data. |
| test\_reproducibility | Two runs with the same seed give identical result hashes. |

**Rules:**

- Test fixtures are small and synthetic, created inside the tests, and never used for any reported number.
- Chasing coverage percentage is not a goal. The goal is that each failure mode above has a test.
- Run `pytest` before every commit. A commit with a failing test is not allowed.

**CI (GitHub Actions):** triggers on push and pull request. Steps: check out, set up Python, install pinned requirements, run ruff, run pytest, run `make smoke`. It cannot run on the real data. The README states this limit in plain words.

## 8. Reporting standards and honesty rules

A result is called evidence of an edge only if all five conditions below hold. Otherwise the write-up says no evidence of an edge was found. These thresholds are choices, they are written down before the real run, and they are not changed after seeing results.

1. The 95 percent bootstrap confidence interval of net Sharpe excludes zero.
2. The permutation test p-value is below 0.05.
3. The deflated Sharpe probability is above 0.95, using the honest trial count from `results/trials.csv`.
4. The result holds in both volatility regimes and in most folds or years.
5. Net Sharpe stays positive when costs are doubled.

**Every results table shows:** out-of-sample period, number of bars, number of trades, gross and net Sharpe with confidence interval, max drawdown, turnover.

**Rules:**

- Net and gross are always shown together.
- Every evaluated configuration is in the trial log. Nothing evaluated is omitted from the write-up.
- The test period is never used to choose anything.
- Weak results are named plainly, for example: the model does not beat momentum after costs.
- Any number in the write-up can be traced to a file in `results/` produced by `make run`.
- Figures: net equity curves against baselines, drawdowns, the permutation histogram with the real Sharpe marked, regime bars with confidence intervals, and the sensitivity heatmap. Axes and units are labelled.
- Where something could not be verified, the text says so. Nothing is estimated and presented as measured.

## 9. Build order, commits, gates and hour estimates

The full build is roughly 38 to 43 hours of work, before the time you need to read and understand each diff. These are my estimates, not measured. They were made without seeing any of your data, and they assume the data is ready on day one. If the data is not ready, the first step grows.

| Step | Work | Hours (estimate) | Priority |
| --- | --- | --- | --- |
| 0 | Repo skeleton, config, `.gitignore`, data loader and validation, data report | 3 | Must |
| 1 | Walk-forward splitter with purge and embargo | 3 to 4 | Must |
| 2 | Cost model, net metrics, execution lag test | 3 | Must |
| 3 | Baselines, comparison table, bootstrap confidence intervals | 2 | Must |
| 4 | Permutation test, leakage tests, canary | 3 to 4 | Must |
| 5 | One-page write-up | 2 | Must |
| 6 | Makefile, run script, pytest, CI, reproduce check | 2 | Must |
| Gate 1 | Clean-clone verification | 1 | Must |
| 7 | ML models in the loop | 5 to 6 | Should |
| 10 | Deflated Sharpe, ADF, ACF and PACF, GARCH | 4 to 5 | Should |
| 8 | Regimes and sensitivity grid | 3 to 4 | Could |
| 9 | Risk simulator, Kelly | 5 | Could |
| 11 | Write-up and README update | 2 | Must if any Phase 2 item ships |

Phase 1 is about 19 to 21 hours including Gate 1. Phase 2 is about 19 to 22 hours. Add 30 to 50 percent for reading and understanding every diff, because you must be able to explain the code. That reading time is the real cost of using Claude Code on a project you will be questioned on.

**If time runs short, cut in this order:** the Kelly part of item 9, then the sensitivity grid of item 8, then the ES robustness run. Keep items 7 and 10. Never cut the leakage tests, the costs or the permutation test, because those are what make the project credible.

**Working rules for Claude Code:**

- Start in plan mode. The first output is the proposed structure and the list of data files needed. No code before approval.
- Work on a branch per phase (`phase1`, `phase2`). Merge to `main` through a pull request only after the gate passes.
- One commit per item, message as listed in sections 5 and 6, with `pytest` green first.
- After each item, Claude Code writes five plain sentences explaining what the module does and why, in `docs/INTERVIEW_NOTES.md`, and you rewrite them in your own words.
- After each item, report the real numbers it produced, including bad ones.
- If a data file, a path or an assumption is missing, stop and ask. Do not substitute.
- No em dashes anywhere. Code stays simple: short functions, clear names, comments that say why.

## 10. Pitfalls that make quant results fake

These are the mistakes interviewers look for. Each one has a place in the design above that prevents it.

- **Lookahead through rolling windows.** Centred windows, negative shifts, back-fill, and normalising with the full-sample mean all leak the future. Prevented by trailing-only features and the truncation test.
- **Same-bar execution.** Using the close of bar t to decide and to trade at that same close earns a return that was not available. Prevented by the next-open convention and the execution lag test.
- **Overlapping labels.** With H above 1, neighbouring labels share future bars, so a random or unpurged split leaks. Prevented by purge and embargo.
- **Full-sample regime or threshold.** Defining high volatility with the median of the whole history uses the future. Prevented by expanding statistics.
- **Contract roll gaps.** A glued series shows a fake return at each roll. Prevented by roll-aware returns and the data report.
- **Timezone and daylight saving errors.** Misaligned sessions create fake lead-lag. Prevented by timezone-aware timestamps and tests across a daylight saving change.
- **Costs too light.** Slippage is worse at the open, on news and in fast markets. Prevented by the break-even cost and the 0x to 3x cost table.
- **Sharpe annualisation errors.** Scaling a 5-minute Sharpe by the square root of bars per year inflates it. Prevented by aggregating to daily P&L first.
- **Choosing after looking.** Tuning on the test result, or re-running until it works. Prevented by touching the test block once, and by the trial log.
- **Multiple testing.** Many tries guarantee one lucky result. Prevented by the deflated Sharpe with an honest trial count.
- **Small sample with flexible models.** Gradient boosting and an MLP fit noise. Prevented by baselines, the permutation test and honest reporting of null results.
- **Reading too much into p-values and Sharpe differences.** Prevented by confidence intervals on every Sharpe.
- **Silent drift.** A package update changes results. Prevented by pinned requirements and `run_info.json`.

## 11. Interview preparation

For each module, you should be able to answer these questions in your own words, in about three sentences each. Write your answers in `docs/INTERVIEW_NOTES.md`. Do not paste Claude Code's version. If you cannot answer one without reading, that module is not yet yours.

| Module | Questions you must be able to answer |
| --- | --- |
| Walk-forward and purge | Why not a random train-test split? What is purging, what is an embargo, and why do overlapping labels need them? Why is a lookback feature that crosses the train-test boundary not a leak? |
| Timing convention | Walk me through one trade from the signal at the close of bar t to its exit. Where could lookahead sneak in? |
| Costs | Why does net Sharpe fall below gross? What is break-even cost? Why is the half-spread an assumption with bar data? |
| Baselines | Why compare to buy-and-hold and a simple rule? What does it mean if the ML model loses to momentum? |
| Permutation test | What null hypothesis does it test? Why shuffle labels only in training? Why use circular shifts for a rule with nothing to train? |
| Leakage tests | How does the truncation test work? Why is a canary test needed? |
| ML models | Why do flexible models overfit financial data? How did you choose hyperparameters without touching the test block? |
| Regimes and sensitivity | Why can a full-sample median not define a regime? What does a lone good cell in the heatmap tell you? |
| Risk simulator | Why a block bootstrap and not a plain shuffle? What does the bootstrap assume? Why fractional Kelly? How noisy is the Kelly estimate? |
| Deflated Sharpe | What problem does it solve? What goes into it? Why does it need the trial count? |
| ADF, ACF, GARCH | Why test returns and not prices? What does autocorrelation in squared returns mean? Why Student-t innovations? |
| The whole project | What would you do with more time or better data? What is the main weakness of this study? |

## 12. CV rules and open decisions

**CV rules:**

- The CV lists only what was built and run, with the real numbers. Nothing planned.
- A status table in the README marks each of the 11 items as done, partial or not done. The CV summary is generated from that table.
- The CV never says the strategy is profitable unless all five conditions in section 8 are met. A sentence such as "built a leakage-tested walk-forward framework and found no edge after costs" is stronger in front of a quant interviewer than an unsupported profit claim.
- Template, to be filled only after the results exist: "Built a walk-forward research framework on NQ futures (purged and embargoed splits, automated leakage tests, permutation tests) and evaluated \[list of models actually run\] against buy-and-hold and a momentum baseline after costs. Result: \[real net Sharpe with interval, and the plain conclusion\]."

**Open decisions (the build cannot start before D1, D4 and D5 are answered):**

| ID | Decision | Why it matters | Suggested default |
| --- | --- | --- | --- |
| D1 | NQ market data: source, bar frequency, date range, regular hours or full session, how the continuous contract is adjusted | Without price data there is nothing to run. Frequency sets sample size and cost sensitivity. | Decided: Yahoo NQ=F hourly bars, see section 2. The sample is only about 2.4 years, and the write-up must say so. |
| D2 | ES data for the robustness run | Needed only for the robustness check. | Same source and format as NQ. Skip if unavailable. |
| D3 | Path and columns of your closed-trade history | Item 9 cannot run without it. | A file with entry time, exit time, side, quantity, instrument, gross P&L, fees. |
| D4 | Commission per side per contract at your broker, and slippage in ticks | The cost model is only as honest as these numbers. | Your real commission. One tick of slippage per side as a starting assumption, tested at 0x to 3x. |
| D5 | Horizon H and momentum lookback L | Both must be fixed before any result is seen. | H = 1 bar. H = 2 bars (two hours) as a secondary horizon. L = 24 bars, about one trading day. Proposed, waiting for your approval. Fixed now, not tuned. |
| D6 | Ruin threshold for the simulator | Probability of ruin has no meaning without a threshold. | Your own account loss limit or evaluation drawdown rule, as a percentage of starting capital. |
| D7 | Repo public or private | Public means only code and aggregate results. | Public, with `data/` gitignored. |
| D8 | Time budget | Decides how much of Phase 2 is realistic. | Finish Phase 1 and Gate 1 first. Then items 7 and 10. |
