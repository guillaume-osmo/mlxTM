import numpy as np
from mlx_tm import DenseTsetlinMachine, NumpyTM
from _util import make_xor, acc


def test_predict_shapes():
    X, y = make_xor(200, seed=1)
    tm = DenseTsetlinMachine(n_clauses=10, T=10, s=3.9, seed=0).fit(X, y, epochs=2)
    assert tm.predict(X).shape == (200,)
    assert tm.decision_function(X).shape == (200, 2)


def test_deterministic():
    X, y = make_xor(400, seed=2, noise=0.2)
    p1 = DenseTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=7).fit(X, y, epochs=5).predict(X)
    p2 = DenseTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=7).fit(X, y, epochs=5).predict(X)
    assert np.array_equal(p1, p2)


def test_dense_learns_noisy_xor():
    Xtr, Ytr = make_xor(1500, seed=0, noise=0.3)
    Xte, Yte = make_xor(1500, seed=99, noise=0.0)
    tm = DenseTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=42).fit(Xtr, Ytr, epochs=30)
    assert acc(tm.predict(Xte), Yte) > 0.9


def test_numpy_oracle_learns_noisy_xor():
    Xtr, Ytr = make_xor(2000, seed=0, noise=0.3)
    Xte, Yte = make_xor(2000, seed=99, noise=0.0)
    tm = NumpyTM(n_clauses=20, T=15, s=3.9, boost_true_positive_feedback=False, seed=42)
    tm.fit(Xtr, Ytr, epochs=40)
    assert acc(tm.predict(Xte), Yte) > 0.9
