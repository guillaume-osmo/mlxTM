import numpy as np
from mlx_tm import rpcholesky_select


def test_returns_valid_unique_indices():
    X = np.random.RandomState(0).randn(80, 200)
    sel = rpcholesky_select(X, k=32, seed=0)
    assert len(sel) <= 32
    assert len(set(sel.tolist())) == len(sel)            # no duplicate pivots
    assert sel.min() >= 0 and sel.max() < 200


def test_deterministic_with_seed():
    X = np.random.RandomState(1).randn(60, 150)
    assert np.array_equal(rpcholesky_select(X, k=20, seed=3),
                          rpcholesky_select(X, k=20, seed=3))


def test_k_capped_to_n_features():
    X = np.random.RandomState(2).randn(40, 10)
    assert len(rpcholesky_select(X, k=50, seed=0)) <= 10


def test_label_free():
    # Target-independent: selection must work without any labels.
    X = np.random.RandomState(3).randn(50, 100)
    sel = rpcholesky_select(X, k=16)
    assert len(sel) <= 16
