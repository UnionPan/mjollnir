"""The exact 1-D two-means split behind the Avellaneda-Stoikov regime calibration."""

import numpy as np

from mjollnir.calibration.physical.as_calibration import two_means_1d


def _within_ss(x, labels):
    return sum(((x[labels == k] - x[labels == k].mean()) ** 2).sum() for k in (0, 1) if (labels == k).any())


def test_two_means_finds_the_split_no_other_cut_beats():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.15, 0.02, 200), rng.normal(0.45, 0.05, 80)])
    rng.shuffle(x)
    labels = two_means_1d(x)
    assert x[labels == 0].max() < x[labels == 1].min()  # contiguous, low cluster labelled 0
    best = min(_within_ss(x, (x > c).astype(int)) for c in np.sort(x)[:-1])
    assert _within_ss(x, labels) <= best + 1e-12
    assert 190 <= (labels == 0).sum() <= 210


def test_two_means_degenerate_inputs():
    assert two_means_1d(np.array([])).size == 0
    assert two_means_1d(np.array([1.0])).tolist() == [0]
    assert sorted(two_means_1d(np.array([2.0, 1.0])).tolist()) == [0, 1]
