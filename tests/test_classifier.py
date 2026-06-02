import numpy as np
import pytest
from mlx_tm import TMClassifierMLX
from _util import make_xor, acc

BACKENDS = ["dense", "bitpacked", "coalesced", "bitplane"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_dispatch_and_shapes(backend):
    X, y = make_xor(300, seed=1, noise=0.2)
    kw = dict(number_of_clauses=20, T=15, s=3.9, backend=backend, seed=0)
    if backend == "coalesced":
        kw.update(T=200, weighted_clauses=True, max_weight=8)
    clf = TMClassifierMLX(**kw)
    clf.fit(X, y)                                  # one epoch
    yhat, cs = clf.predict(X, return_class_sums=True)
    assert yhat.shape == (300,)
    assert cs.shape == (300, 2)
    assert set(np.unique(yhat)).issubset({0, 1})


def test_incremental_epochs_run():
    X, y = make_xor(400, seed=2, noise=0.2)
    clf = TMClassifierMLX(number_of_clauses=20, T=15, s=3.9, backend="dense", seed=0)
    for ep in range(5):
        clf.fit(X, y, incremental=(ep > 0))
    assert acc(clf.predict(X), y) > 0.6           # learning, not asserting convergence here
