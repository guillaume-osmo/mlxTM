"""NumPy reference / oracle: classic partitioned multiclass Tsetlin Machine.

Faithful translation of cair/TsetlinMachine MultiClassTsetlinMachine.pyx
(arXiv:1804.01508), vectorised over clauses but otherwise identical semantics.
Used to (a) lock the feedback logic and (b) act as the CPU baseline (since tmu
does not compile on this Mac). The MLX dense backend mirrors this exactly.
"""
import numpy as np


class NumpyTM:
    def __init__(self, n_clauses, T, s, n_classes=2, n_state_bits=8,
                 boost_true_positive_feedback=True, seed=0):
        # Round clause count up to a whole multiple of n_classes (partitioning).
        cpc = max(1, n_clauses // n_classes)
        self.C = cpc * n_classes
        self.T = float(T)
        self.s = float(s)
        self.n_classes = n_classes
        self.N = 1 << (n_state_bits - 1)
        self.max_state = (1 << n_state_bits) - 1
        self.boost = boost_true_positive_feedback
        self.rng = np.random.RandomState(seed)
        self.class_of = np.arange(self.C) // cpc
        within = np.arange(self.C) % cpc
        self.sign = np.where(within % 2 == 0, 1, -1).astype(np.int32)
        onehot = (self.class_of[None, :] == np.arange(n_classes)[:, None]).astype(np.int32)
        self.S = onehot * self.sign[None, :]            # (n_classes, C) voting matrix
        self.ta = None

    def _init(self, f):
        self.f = f
        self.L = 2 * f
        self.ta = self.rng.randint(self.N - 1, self.N + 1, size=(self.C, self.L)).astype(np.int32)

    @staticmethod
    def _lits(X):
        return np.concatenate([X, 1 - X], axis=1)

    def _eval(self, lits, predict):
        inc = (self.ta >= self.N).astype(np.int32)       # (C,L)
        miss = inc @ (1 - lits).T                        # (C,B)
        fired = (miss == 0).astype(np.int32)
        if predict:
            cnt = inc.sum(1, keepdims=True)
            fired = fired * (cnt > 0)
        return fired

    def decision_function(self, X):
        return (self.S @ self._eval(self._lits(X.astype(np.int32)), True)).T

    def predict(self, X):
        return self.decision_function(X).argmax(1)

    def _type_I(self, mask, lit, fired):
        invs, ps = 1.0 / self.s, (self.s - 1.0) / self.s
        firing = (mask & (fired > 0))[:, None]
        nonf = (mask & (fired == 0))[:, None]
        r = self.rng.rand(self.C, self.L)
        inc_pos = (lit == 1)[None, :] & (True if self.boost else (r < ps))
        dec_f = (lit == 0)[None, :] & (r < invs)
        delta_f = inc_pos.astype(np.int32) - dec_f.astype(np.int32)
        delta_n = -((r < invs).astype(np.int32))
        self.ta = np.clip(self.ta + firing * delta_f + nonf * delta_n, 0, self.max_state)

    def _type_II(self, mask, lit, fired):
        inc = self.ta >= self.N
        firing = (mask & (fired > 0))[:, None]
        inc2 = ((lit == 0)[None, :]) & (~inc)
        self.ta = np.clip(self.ta + (firing & inc2).astype(np.int32), 0, self.max_state)

    def _update(self, lit, target):
        inc = (self.ta >= self.N).astype(np.int32)
        fired = ((inc @ (1 - lit)) == 0)                 # (C,) training eval (empty clause fires)
        cs = np.clip(self.S @ fired.astype(np.int32), -self.T, self.T)
        neg = self.rng.randint(self.n_classes)
        while neg == target:
            neg = self.rng.randint(self.n_classes)
        up_t = (self.T - cs[target]) / (2 * self.T)
        up_n = (self.T + cs[neg]) / (2 * self.T)
        sel_t = (self.class_of == target) & (self.rng.rand(self.C) <= up_t)
        sel_n = (self.class_of == neg) & (self.rng.rand(self.C) <= up_n)
        spos = self.sign >= 0
        tI = (sel_t & spos) | (sel_n & ~spos)            # target+ and neg- -> Type I
        tII = (sel_t & ~spos) | (sel_n & spos)           # target- and neg+ -> Type II
        self._type_I(tI, lit, fired)
        self._type_II(tII, lit, fired)

    def fit(self, X, Y, epochs=1, incremental=False, shuffle=True):
        X = np.asarray(X).astype(np.int32)
        Y = np.asarray(Y).astype(np.int32)
        if self.ta is None or not incremental:
            self._init(X.shape[1])
        lits = self._lits(X)
        n = X.shape[0]
        for _ in range(epochs):
            idx = np.arange(n)
            if shuffle:
                self.rng.shuffle(idx)
            for i in idx:
                self._update(lits[i], int(Y[i]))
        return self


if __name__ == "__main__":
    # Self-contained Noisy-XOR sanity check (no external data files).
    rng = np.random.RandomState(0)

    def xor(n, noise=0.0):
        X = rng.randint(0, 2, (n, 12)).astype(np.int32)
        y = (X[:, 0] ^ X[:, 1]).astype(np.int32)
        if noise:
            flip = rng.rand(n) < noise
            y[flip] = 1 - y[flip]
        return X, y

    Xtr, Ytr = xor(2000, 0.4)
    Xte, Yte = xor(2000, 0.0)
    tm = NumpyTM(n_clauses=20, T=15, s=3.9, boost_true_positive_feedback=False, seed=42)
    tm.fit(Xtr, Ytr, epochs=40)
    print("Noisy-XOR test acc:", (tm.predict(Xte) == Yte).mean())
