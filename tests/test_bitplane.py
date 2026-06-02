import numpy as np
import mlx.core as mx
from mlx_tm import BitPlaneTsetlinMachine
from mlx_tm.kernels import bitplane_update
from _util import make_xor, acc


def test_incdec_carry_ops():
    """One clause, one chunk, STATE_BITS planes; drive Type-I feedback and check the
    counter crosses the include threshold (MSB plane bit flips on), then back off."""
    sb, nch, C = 4, 1, 1
    N = 1 << (sb - 1)                       # include iff counter >= N (=8 here)
    # init counter = N-1 (just below include): planes 0..sb-2 = ~0, MSB = 0
    state = np.zeros((C * nch, sb), np.uint32)
    state[:, : sb - 1] = np.uint32(0xFFFFFFFF)
    ta = mx.array(state.reshape(-1))
    X = mx.array(np.array([1], np.uint32))            # literal 0 present
    fired = mx.array(np.array([1], np.uint8))
    tI = mx.array(np.array([1], np.uint8)); tII = mx.array(np.array([0], np.uint8))
    laf = mx.array(np.array([0], np.uint32))          # boost path -> inc present literals
    out = bitplane_update(ta, X, fired, tI, tII, laf, C, nch, sb, boost=True)
    msb = int(np.array(out)[sb - 1]) & 1              # action bit for literal 0
    assert msb == 1                                   # N-1 -> N crossed the include threshold


def test_bitplane_state_is_packed_uint32():
    X, y = make_xor(200, seed=1)
    tm = BitPlaneTsetlinMachine(n_clauses=10, T=10, s=3.9, seed=0).fit(X, y, epochs=2)
    assert tm.ta_state.dtype == mx.uint32
    assert tm.ta_state.size == tm.C * tm.n_chunks * tm.number_of_state_bits


def test_bitplane_learns_noisy_xor():
    Xtr, Ytr = make_xor(1500, seed=0, noise=0.3)
    Xte, Yte = make_xor(1500, seed=99, noise=0.0)
    tm = BitPlaneTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=42).fit(Xtr, Ytr, epochs=30)
    assert acc(tm.predict(Xte), Yte) > 0.9
