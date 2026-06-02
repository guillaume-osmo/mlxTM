import numpy as np
from mlx_tm import (ThermometerBinarizer, SoftmaxBinarizer, GaussianizeBinarizer,
                    GMMBinarizer, RotationBinarizer, count_thermometer)

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


def test_count_thermometer():
    counts = np.array([[0, 1, 2, 3], [2, 0, 5, 1]])           # (2 mols, 4 substructures)
    B = count_thermometer(counts, max_count=3)
    assert B.shape == (2, 4 * 3)
    assert set(np.unique(B)).issubset({0, 1})
    # the count>=1 block reproduces a binary presence fingerprint
    assert np.array_equal(B[:, :4], (counts >= 1).astype(np.int8))
    # monotone across thresholds: (count>=1) >= (count>=2) >= (count>=3)
    Bf = B.reshape(2, 3, 4)                                   # (n, max_count, F)
    assert np.all(Bf[:, 1:, :] <= Bf[:, :-1, :])
    # multiplicity: a count of 3 lights all three bits; a count of 1 lights only the first
    assert [B[0, 3], B[0, 4 + 3], B[0, 8 + 3]] == [1, 1, 1]   # feature 3, count 3
    assert [B[0, 1], B[0, 4 + 1], B[0, 8 + 1]] == [1, 0, 0]   # feature 1, count 1


def test_rotation_modes_binary_and_deterministic():
    for mode in ("random", "hadamard", "itq"):
        rb = RotationBinarizer(n_bits=16, mode=mode, seed=0).fit(X)
        B1, B2 = rb.transform(X), rb.transform(X)
        assert set(np.unique(B1)).issubset({0, 1})
        assert np.array_equal(B1, B2)                       # deterministic
        assert B1.shape[0] == 120 and 1 <= B1.shape[1] <= 16
