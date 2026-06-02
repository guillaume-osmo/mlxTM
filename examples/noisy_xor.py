"""Noisy-XOR: the canonical Tsetlin Machine sanity check (self-contained, no data files)."""
import numpy as np
from mlx_tm import DenseTsetlinMachine


def make_xor(n, seed, noise=0.0, n_feat=12):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, n_feat)).astype(np.int32)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int32)
    if noise:
        flip = rng.rand(n) < noise
        y[flip] = 1 - y[flip]
    return X, y


if __name__ == "__main__":
    Xtr, Ytr = make_xor(5000, seed=0, noise=0.4)     # 40% label noise on train
    Xte, Yte = make_xor(5000, seed=1, noise=0.0)     # clean test
    tm = DenseTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=42)
    for ep in range(30):
        tm.fit(Xtr, Ytr, epochs=1, incremental=(ep > 0))
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1:2d}  test acc {(tm.predict(Xte) == Yte).mean():.4f}")
