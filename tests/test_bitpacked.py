import numpy as np
from mlx_tm import DenseTsetlinMachine, BitPackedTsetlinMachine, pack_bits_uint32
from _util import make_xor, acc


def _unpack(P, nbits):
    n, nch = P.shape
    out = np.zeros((n, nch * 32), dtype=np.uint8)
    for p in range(32):
        out[:, p::32] = (P >> np.uint32(p)) & np.uint32(1)
    return out[:, :nbits]


def test_pack_roundtrip_exact():
    rng = np.random.RandomState(0)
    for nbits in (24, 32, 100, 2048, 4096):
        bits = (rng.rand(37, nbits) < 0.5).astype(np.uint8)
        assert np.array_equal(_unpack(pack_bits_uint32(bits), nbits), bits)


def test_bitpacked_matches_dense_exactly():
    # Same seed + config -> identical training -> the bitwise kernel must reproduce
    # the dense matmul class sums bit-for-bit.
    Xtr, Ytr = make_xor(800, seed=0, noise=0.3)
    Xte, _ = make_xor(300, seed=5, noise=0.0)
    cfg = dict(n_clauses=20, T=15, s=3.9, seed=42)
    cs_d = DenseTsetlinMachine(**cfg).fit(Xtr, Ytr, epochs=10).decision_function(Xte)
    cs_p = BitPackedTsetlinMachine(**cfg).fit(Xtr, Ytr, epochs=10).decision_function(Xte)
    assert np.abs(cs_d - cs_p).max() == 0.0


def test_bitpacked_learns_noisy_xor():
    Xtr, Ytr = make_xor(1500, seed=0, noise=0.3)
    Xte, Yte = make_xor(1500, seed=99, noise=0.0)
    tm = BitPackedTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=42).fit(Xtr, Ytr, epochs=30)
    assert acc(tm.predict(Xte), Yte) > 0.9
