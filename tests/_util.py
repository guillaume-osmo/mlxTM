"""Shared test helpers: a self-contained Noisy-XOR generator (no data files)."""
import numpy as np


def make_xor(n, seed=0, noise=0.0, n_feat=12):
    """y = x0 XOR x1 over n_feat random bits; `noise` fraction of labels flipped."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, n_feat)).astype(np.int32)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int32)
    if noise:
        flip = rng.rand(n) < noise
        y[flip] = 1 - y[flip]
    return X, y


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())
