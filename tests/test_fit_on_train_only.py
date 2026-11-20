"""No model or scaler sees test rows during fit. Toy least-squares model, synthetic data."""
import numpy as np

from signal_lab.leakage import RecordingFit, fit_boundary_violations
from signal_lab.splits import block_bounds, walk_forward_splits
from signal_lab.stats_tests import walk_forward_signal

N = 800
SPLITS = {"scheme": "expanding", "min_train_bars": 300, "test_bars": 100, "embargo_bars": 2,
          "rolling_train_bars": 300, "keep_partial_last_block": True}


def toy_data(seed=0):
    rng = np.random.default_rng(seed)
    features = np.column_stack([np.arange(N), rng.normal(size=(N, 2))])  # column 0 = bar position
    labels = 0.4 * features[:, 1] + rng.normal(size=N)
    label_end = np.arange(N) + 1.0
    label_end[-1] = np.nan
    return features, labels, label_end


def least_squares(x_train, y_train, x_test):
    """Standardise with TRAIN statistics only, then fit with an intercept. Column 0 is ignored."""
    mean, std = x_train[:, 1:].mean(axis=0), x_train[:, 1:].std(axis=0)
    design = np.column_stack([np.ones(len(x_train)), (x_train[:, 1:] - mean) / std])
    beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
    test_design = np.column_stack([np.ones(len(x_test)), (x_test[:, 1:] - mean) / std])
    return test_design @ beta


def test_every_fit_ends_before_its_test_block():
    features, labels, label_end = toy_data()
    recorder = RecordingFit(least_squares)
    walk_forward_signal(recorder, features, labels, label_end, SPLITS)
    starts = [start for start, _ in block_bounds(N, SPLITS)]
    assert len(recorder.latest_seen) == len(starts)
    assert fit_boundary_violations(recorder.latest_seen, starts) == []


def test_the_check_catches_a_fit_that_peeks_at_the_test_block():
    features, labels, label_end = toy_data()
    recorder = RecordingFit(least_squares)
    for (_, test) in walk_forward_splits(label_end, SPLITS):
        peek = np.arange(0, test[-1] + 1)  # leaky: trains through the end of the test block
        recorder(features[peek], labels[peek], features[test])
    starts = [start for start, _ in block_bounds(N, SPLITS)]
    assert len(fit_boundary_violations(recorder.latest_seen, starts)) == len(starts)


def test_test_labels_and_later_data_never_change_a_folds_signal():
    features, labels, label_end = toy_data()
    base = walk_forward_signal(least_squares, features, labels, label_end, SPLITS)
    start, stop = block_bounds(N, SPLITS)[2]
    changed_labels = labels.copy()
    changed_labels[start:stop] = 1e6          # the test block's own labels
    changed_features = features.copy()
    changed_features[stop + 5 :, 1:] = 1e6    # everything after the block
    again = walk_forward_signal(least_squares, changed_features, changed_labels, label_end, SPLITS)
    assert np.array_equal(base[start:stop], again[start:stop], equal_nan=True)
