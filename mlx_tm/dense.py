"""Option 1 — Dense MLX Tsetlin Machine (classic multiclass, arXiv:1804.01508).

Verified against the NumPy oracle (`numpy_ref.py`), which is a line-for-line port
of cair/TsetlinMachine's MultiClassTsetlinMachine.pyx and reaches 100% on Noisy XOR.

Representation (everything is an MLX tensor, so it runs on the Apple GPU):

    ta_state : (C, 2f) int16   automaton counter per (clause, literal), in [0, 2N-1]
                                "include literal" iff state >= N  (the action bit)
    sign     : (C,)     +/-1    clause vote sign; clauses are partitioned across classes
    S        : (n_classes, C)   voting matrix = sign where clause in class else 0

Inference is a matmul + step:

    include    = (ta_state >= N)                  # (C, 2f)
    miss       = include @ (1 - literals).T       # (C, B)  included-but-absent count
    clause_out = (miss == 0)                      # (C, B)
    class_sums = S @ clause_out                   # (n_classes, B)  <-- the matmul
    pred       = argmax(class_sums)

Training is the per-example Type I / Type II automaton feedback, vectorised across
the whole (clauses x literals) state tensor.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


class DenseTsetlinMachine:
    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float,
        n_classes: int = 2,
        number_of_state_bits: int = 8,
        boost_true_positive_feedback: bool = False,
        seed: int = 0,
    ):
        self.n_clauses_req = int(n_clauses)
        self.T = float(T)
        self.s = float(s)
        self.n_classes = int(n_classes)
        self.number_of_state_bits = int(number_of_state_bits)
        self.N = 1 << (number_of_state_bits - 1)         # action threshold: include iff state >= N
        self.max_state = (1 << number_of_state_bits) - 1
        self.boost = bool(boost_true_positive_feedback)
        self.seed = int(seed)
        self._rng = np.random.RandomState(seed)
        self.ta = None
        self.f = None
        self.L = None

    # ------------------------------------------------------------------ setup
    def _build_partition(self):
        cpc = max(1, self.n_clauses_req // self.n_classes)
        self.C = cpc * self.n_classes                    # whole multiple of n_classes
        class_of = np.arange(self.C) // cpc
        within = np.arange(self.C) % cpc
        sign = np.where(within % 2 == 0, 1, -1).astype(np.int32)
        onehot = (class_of[None, :] == np.arange(self.n_classes)[:, None]).astype(np.int32)
        self.class_of = mx.array(class_of)
        self.sign = mx.array(sign)
        self.S = mx.array(onehot * sign[None, :], dtype=mx.float32)   # (n_classes, C)

    def _init_params(self, n_features: int):
        self.f = int(n_features)
        self.L = 2 * self.f
        self._build_partition()
        mx.random.seed(self.seed)
        init = mx.random.randint(self.N - 1, self.N + 1, shape=(self.C, self.L))
        self.ta = init.astype(mx.int16)
        mx.eval(self.ta)

    @staticmethod
    def _literals(X: mx.array) -> mx.array:
        return mx.concatenate([X, 1.0 - X], axis=1)      # [x, ~x] -> (B, 2f)

    # -------------------------------------------------------------- inference
    def _eval(self, lits: mx.array, predict: bool) -> mx.array:
        include = (self.ta >= self.N).astype(mx.float32)             # (C, L)
        miss = include @ (1.0 - lits).T                             # (C, B)
        fired = (miss == 0).astype(mx.float32)
        if predict:
            cnt = include.sum(axis=1, keepdims=True)
            fired = fired * (cnt > 0).astype(mx.float32)            # empty clause abstains
        return fired

    def decision_function(self, X) -> np.ndarray:
        lits = self._literals(mx.array(np.asarray(X), dtype=mx.float32))
        class_sums = (self.S @ self._eval(lits, predict=True)).T     # (B, n_classes)
        mx.eval(class_sums)
        return np.array(class_sums)

    def predict(self, X) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1).astype(np.int32)

    # --------------------------------------------------------------- feedback
    def _type_I(self, mask: mx.array, lit_row: mx.array, fired: mx.array):
        """Combats false negatives. Firing: reinforce present literals, weaken absent.
        Non-firing: forget (decrement everything with prob 1/s)."""
        inv_s, ps = 1.0 / self.s, (self.s - 1.0) / self.s
        firing = (mask & fired)[:, None]
        nonfiring = (mask & (~fired))[:, None]
        r = mx.random.uniform(shape=self.ta.shape)
        is1 = lit_row == 1.0
        is0 = lit_row == 0.0
        inc_pos = is1 if self.boost else (is1 & (r < ps))
        dec_fire = is0 & (r < inv_s)
        delta_fire = inc_pos.astype(mx.int16) - dec_fire.astype(mx.int16)
        delta_non = -((r < inv_s).astype(mx.int16))
        delta = firing.astype(mx.int16) * delta_fire + nonfiring.astype(mx.int16) * delta_non
        self.ta = mx.clip(self.ta + delta, 0, self.max_state).astype(mx.int16)

    def _type_II(self, mask: mx.array, lit_row: mx.array, fired: mx.array):
        """Combats false positives: include absent literals that are currently excluded."""
        include = self.ta >= self.N
        firing = (mask & fired)[:, None]
        inc2 = (lit_row == 0.0) & (~include)
        self.ta = mx.clip(self.ta + (firing & inc2).astype(mx.int16), 0, self.max_state).astype(mx.int16)

    def _update_example(self, lit: mx.array, target: int):
        lit_row = lit[None, :]
        include = (self.ta >= self.N).astype(mx.float32)
        fired = (include @ (1.0 - lit)) == 0                         # (C,) bool, computed once
        cs = mx.clip(self.S @ fired.astype(mx.float32), -self.T, self.T)   # (n_classes,)
        neg = self._rng.randint(self.n_classes)
        while neg == target:
            neg = self._rng.randint(self.n_classes)
        up_t = (self.T - cs[target]) / (2.0 * self.T)
        up_n = (self.T + cs[neg]) / (2.0 * self.T)
        r1 = mx.random.uniform(shape=(self.C,))
        r2 = mx.random.uniform(shape=(self.C,))
        sel_t = (self.class_of == target) & (r1 <= up_t)
        sel_n = (self.class_of == neg) & (r2 <= up_n)
        spos = self.sign >= 0
        type_I_mask = (sel_t & spos) | (sel_n & (~spos))             # target+ / neg-
        type_II_mask = (sel_t & (~spos)) | (sel_n & spos)            # target- / neg+
        self._type_I(type_I_mask, lit_row, fired)
        self._type_II(type_II_mask, lit_row, fired)

    def fit(self, X, Y, epochs: int = 1, incremental: bool = False,
            shuffle: bool = True, eval_every: int = 64):
        Xm = mx.array(np.asarray(X), dtype=mx.float32)
        Yn = np.asarray(Y).astype(np.int32)
        if self.ta is None or not incremental:
            self.n_classes = max(self.n_classes, int(Yn.max()) + 1)
            self._init_params(Xm.shape[1])
        lits = self._literals(Xm)
        n = Xm.shape[0]
        for _ in range(epochs):
            idx = np.arange(n)
            if shuffle:
                self._rng.shuffle(idx)
            for step, i in enumerate(idx):
                self._update_example(lits[int(i)], int(Yn[i]))
                if (step + 1) % eval_every == 0:
                    mx.eval(self.ta)
            mx.eval(self.ta)
        return self
