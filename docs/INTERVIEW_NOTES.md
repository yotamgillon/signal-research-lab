# Interview notes

Five plain sentences per item, written by Claude Code. Rewrite each one in your own words.

## Step 0: data loader, validation and data report

1. The data module reads the Yahoo CSV, converts every timestamp to America/New_York, and refuses to continue if timestamps are naive, unsorted or duplicated, if high is below low, if a price is not positive, or if a gap is longer than the longest normal weekend or holiday break.
2. It always drops the newest bar, because that bar was still forming when the file was downloaded and its close is not final.
3. It treats the return of the 16:00 New York bar on the third Friday of March, June, September and December as missing in every year, a rule written down before any performance result so the cleaning cannot be tuned to flatter the strategy.
4. It classifies every gap as daily break, weekend, holiday or other, and reports zero-volume bars, off-the-hour bars and the largest single-bar moves, so unusual data is visible instead of hidden.
5. It writes a data report with the sha256 of the file, so every later result is tied to exactly one data snapshot.
