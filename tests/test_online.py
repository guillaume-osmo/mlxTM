import numpy as np
from mlx_tm import OnlineTsetlinMachine, TMClassifierMLX
from _util import make_xor, acc


def test_online_learns_noisy_xor():
    # CUDA-style online backend: per-example updates looped inside the Metal kernel.
    Xtr, Ytr = make_xor(1500, seed=0, noise=0.25)
    Xte, Yte = make_xor(1500, seed=99, noise=0.0)
    tm = OnlineTsetlinMachine(n_clauses=40, T=80, s=3.9, max_weight=8,
                              weighted_clauses=True, seed=42).fit(Xtr, Ytr, epochs=40)
    assert acc(tm.predict(Xte), Yte) > 0.85


def test_online_weight_shape_and_decision():
    X, y = make_xor(200, seed=1)
    tm = OnlineTsetlinMachine(n_clauses=40, T=80, s=3.9, max_weight=8,
                              weighted_clauses=True, seed=0).fit(X, y, epochs=3)
    assert tuple(tm.W.shape) == (tm.C,)              # per-clause integer weight
    cs = tm.decision_function(X)
    assert cs.shape == (200, 2)                      # (n_examples, n_classes)


def test_online_backend_via_classifier():
    Xtr, Ytr = make_xor(1500, seed=2, noise=0.2)
    Xte, Yte = make_xor(1500, seed=7, noise=0.0)
    clf = TMClassifierMLX(number_of_clauses=40, T=80, s=3.9, backend="online",
                          weighted_clauses=True, max_weight=8, seed=1).fit(Xtr, Ytr, epochs=40)
    assert acc(clf.predict(Xte), Yte) > 0.85


def test_online_unweighted_runs():
    X, y = make_xor(300, seed=3)
    tm = OnlineTsetlinMachine(n_clauses=40, T=80, s=3.9, weighted_clauses=False, seed=0).fit(X, y, epochs=5)
    assert tm.decision_function(X).shape == (300, 2)
