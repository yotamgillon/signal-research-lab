"""Automated leakage checks. Each returns a list of violations, empty when everything is clean.

The pytest files assert the list is empty for clean inputs and non-empty for deliberately
leaky ones (the canary), so a broken check cannot pass silently.
"""
import numpy as np


def truncation_violations(build, raw_bars, cut_points):
    """Cut the data at each point. Features of the cut data must equal the full-data features.

    build: raw bars -> feature frame (must include every step from the raw bars on). A feature
    that uses a centred window, a negative shift, back-fill or full-sample statistics changes
    when later data is removed, and shows up here.
    """
    full = build(raw_bars)
    violations = []
    for cut in cut_points:
        part = build(raw_bars.iloc[: cut + 1])
        same = np.isclose(full.iloc[: cut + 1].to_numpy(float), part.to_numpy(float),
                          equal_nan=True)
        bad_columns = [c for c, ok in zip(full.columns, same.all(axis=0), strict=True) if not ok]
        if bad_columns:
            violations.append({"cut_bar": int(cut), "columns": bad_columns,
                               "rows_differing": int((~same.all(axis=1)).sum())})
    return violations


def label_boundary_violations(label_end, folds):
    """Folds where a training label window reaches into the test block."""
    label_end = np.asarray(label_end, dtype=float)
    return [{"fold": i, "latest_train_label_end": float(np.nanmax(label_end[train])),
             "test_start": int(test[0])}
            for i, (train, test) in enumerate(folds)
            if np.nanmax(label_end[train]) >= test[0]]


def fit_boundary_violations(recorded_latest_positions, test_starts):
    """Fits that saw a bar at or after the start of their test block."""
    return [{"fold": i, "latest_position_seen": int(seen), "test_start": int(start)}
            for i, (seen, start) in enumerate(zip(recorded_latest_positions, test_starts,
                                                  strict=True))
            if seen >= start]


class RecordingFit:
    """Wraps a fit_predict function and records the latest bar position seen in training.

    X_train must carry the bar position in column `position_column`, so the wrapper can tell
    what the fit saw.
    """

    def __init__(self, fit_predict, position_column=0):
        self.fit_predict = fit_predict
        self.position_column = position_column
        self.latest_seen = []

    def __call__(self, x_train, y_train, x_test):
        self.latest_seen.append(x_train[:, self.position_column].max())
        return self.fit_predict(x_train, y_train, x_test)


def ordering_violations(index):
    """Timestamps must be sorted and unique."""
    problems = []
    if not index.is_monotonic_increasing:
        problems.append("timestamps are not sorted")
    if index.has_duplicates:
        problems.append("duplicate timestamps")
    return problems


def canary_feature(bars):
    """A feature that deliberately uses the future: the log return of the NEXT bar.

    It cannot exist at the close of bar t. Every leakage check must fail on it.
    """
    return np.log(bars["close"].shift(-1) / bars["open"].shift(-1))
