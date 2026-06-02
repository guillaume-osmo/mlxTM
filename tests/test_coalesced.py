import numpy as np
from mlx_tm import CoalescedTsetlinMachine
from _util import make_xor, acc


def test_weight_bank_shape():
    X, y = make_xor(200, seed=1)
    tm = CoalescedTsetlinMachine(n_clauses=40, T=100, s=3.9, max_weight=8, seed=0).fit(X, y, epochs=2)
    assert tuple(tm.W.shape) == (2, tm.C)        # per-class signed integer weights


def test_coalesced_learns_noisy_xor():
    # Coalesced needs T scaled to the (growing) weight magnitude: small max_weight, large T.
    Xtr, Ytr = make_xor(1500, seed=0, noise=0.25)
    Xte, Yte = make_xor(1500, seed=99, noise=0.0)
    tm = CoalescedTsetlinMachine(n_clauses=40, T=400, s=3.9, max_weight=8,
                                 weighted_clauses=True, seed=42).fit(Xtr, Ytr, epochs=40)
    assert acc(tm.predict(Xte), Yte) > 0.85
