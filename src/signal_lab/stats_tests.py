"""Permutation tests: how often would a net Sharpe this high appear if the signal carried no
information about the future?

p-value = (1 + number of permuted Sharpes at least as large as the real one) / (N + 1).
The smallest possible p-value is 1 / (N + 1). It is NOT corrected for the number of things
tried: the deflated Sharpe (Phase 2) does that using results/trials.csv.
"""
import numpy as np

from signal_lab.backtest import add_roll_sides, net_pnl_usd, simulate
from signal_lab.metrics import daily_pnl, sharpe
from signal_lab.splits import walk_forward_splits


def summarize(real, permuted, label):
    permuted = np.asarray(permuted, dtype=float)
    count = int((permuted >= real).sum())  # NaN compares as False
    n = len(permuted)
    return {
        "method": label,
        "real_net_sharpe": float(real),
        "n_permutations": n,
        "p_value": (1 + count) / (n + 1),
        "minimum_possible_p_value": 1 / (n + 1),
        "permuted_at_least_as_large": count,
        "permuted_mean": float(np.nanmean(permuted)),
        "permuted_std": float(np.nanstd(permuted)),
        "permuted_5th_percentile": float(np.nanpercentile(permuted, 5)),
        "permuted_95th_percentile": float(np.nanpercentile(permuted, 95)),
    }


def circular_shift_test(statistic, signal, n_permutations, min_shift, seed):
    """For a rule with nothing to train: shift the signal circularly against the returns.

    A shift keeps the signal's own autocorrelation and its trade count but breaks its link to
    the returns. statistic(signal) -> net Sharpe. Returns the summary and the permuted values.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    shifts = np.random.default_rng(seed).integers(min_shift, n - min_shift + 1, size=n_permutations)
    permuted = np.array([statistic(np.roll(signal, k)) for k in shifts])
    return summarize(statistic(signal), permuted, "circular shift of the signal"), permuted


def walk_forward_signal(fit_predict, features, labels, label_end, splits_cfg, rng=None):
    """Out-of-sample signal (sign of the prediction) from the walk-forward loop.

    Each fold fits on its training rows only. With an rng, the training labels of the fold are
    shuffled before fitting, which is the permutation null. Rows with NaN features or labels
    are never trained on and get no signal.
    """
    signal = np.full(len(features), np.nan)
    usable = np.isfinite(features).all(axis=1) & np.isfinite(labels)
    for train, test in walk_forward_splits(label_end, splits_cfg):
        train = train[usable[train]]
        train_labels = labels[train] if rng is None else rng.permutation(labels[train])
        prediction = fit_predict(features[train], train_labels, features[test])
        signal[test] = np.where(usable[test], np.sign(prediction), np.nan)
    return signal


def label_shuffle_test(statistic, fit_predict, features, labels, label_end, splits_cfg,
                       n_permutations, seed):
    """For a trained model: shuffle training labels inside each fold, retrain, re-evaluate."""
    real_signal = walk_forward_signal(fit_predict, features, labels, label_end, splits_cfg)
    rng = np.random.default_rng(seed)
    permuted = np.array([
        statistic(walk_forward_signal(fit_predict, features, labels, label_end, splits_cfg, rng))
        for _ in range(n_permutations)
    ])
    return summarize(statistic(real_signal), permuted, "training-label shuffle"), permuted


def net_sharpe_statistic(part, valid, horizon, per_side_usd, cfg):
    """A function signal -> net Sharpe on the bars `part`, with the full cost model and rolls."""
    def statistic(signal):
        sim = simulate(part, signal, valid, horizon, cfg["instrument"]["multiplier_usd"])
        add_roll_sides(sim, part["expiry_bar"].to_numpy(), cfg["costs"]["roll_round_trip_sides"])
        daily = daily_pnl(net_pnl_usd(sim, per_side_usd), part.index, cfg["data"])
        return sharpe(daily, cfg["metrics"])
    return statistic
