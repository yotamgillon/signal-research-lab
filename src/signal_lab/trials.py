"""Trial log: every strategy configuration that is evaluated is recorded here.

The deflated Sharpe (Phase 2) counts trials from results/trials.csv, not from memory. The file
is rebuilt from scratch by each run, so rerunning does not inflate the count. Anything that is
evaluated must go through this log, including any other momentum lookback L.
"""
import pandas as pd


class TrialLog:
    def __init__(self):
        self.rows = []

    def add(self, **row):
        self.rows.append(row)

    def write(self, path):
        table = pd.DataFrame(self.rows, dtype=object).fillna("n/a")  # object keeps ints as ints
        table.to_csv(path, index_label="trial_id")

    def __len__(self):
        return len(self.rows)
