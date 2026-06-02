import numpy as np
from mlx_tm import (ThermometerBinarizer, SoftmaxBinarizer, GaussianizeBinarizer,
                    GMMBinarizer, RotationBinarizer)

X = np.random.RandomState(0).randn(120, 8)


def test_thermometer_width_binary_and_monotone():
    B = ThermometerBinarizer(n_bins=5).fit_transform(X)
    assert B.shape == (120, 8 * 5)
    assert set(np.unique(B)).issubset({0, 1})
    # thermometer is ordinal: for each feature, bits are non-increasing across thresholds
    Bf = B.reshape(120, 8, 5)
    assert np.all(Bf[:, :, 1:] <= Bf[:, :, :-1])


def test_softmax_width_and_binary():
    B = SoftmaxBinarizer(n_bins=7).fit_transform(X)
    assert B.shape == (120, 8 * 7)
    assert set(np.unique(B)).issubset({0, 1})
    assert B.reshape(120, 8, 7).sum(axis=2).min() >= 1     # at least the nearest bin lights up


def test_gaussianize_width_and_binary():
    B = GaussianizeBinarizer(n_bins=4).fit_transform(X)
    assert B.shape == (120, 8 * 4)
    assert set(np.unique(B)).issubset({0, 1})


def test_gmm_one_hot_per_feature():
    B = GMMBinarizer(n_components=3, seed=0).fit_transform(X)
    assert B.shape == (120, 8 * 3)
    assert np.array_equal(B.reshape(120, 8, 3).sum(axis=2), np.ones((120, 8)))   # exactly one bit


def test_rotation_modes_binary_and_deterministic():
    for mode in ("random", "hadamard", "itq"):
        rb = RotationBinarizer(n_bits=16, mode=mode, seed=0).fit(X)
        B1, B2 = rb.transform(X), rb.transform(X)
        assert set(np.unique(B1)).issubset({0, 1})
        assert np.array_equal(B1, B2)                       # deterministic
        assert B1.shape[0] == 120 and 1 <= B1.shape[1] <= 16
