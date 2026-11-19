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
