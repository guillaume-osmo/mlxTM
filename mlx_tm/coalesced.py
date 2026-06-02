"""Option 2 — Coalesced weighted MLX Tsetlin Machine (arXiv:2108.07594).

Extends the verified dense TM (`dense.py`). Where the classic dense machine
*partitions* clauses across classes (each clause votes for exactly one class with
a fixed +/- sign), the coalesced machine keeps ONE shared clause pool and gives
every class its own signed integer weight per clause:

    ta_state : (C, 2f) int16    automaton counter per (clause, literal), in [0, 2N-1]
                                 "include literal" iff state >= N  (the action bit)
    W        : (n_classes, C) int32   per-class signed clause weights (sign = vote
                                      direction, magnitude = vote strength)

Clause evaluation is identical to dense (`_eval`). Inference is a weighted matmul:

    fired      = (include @ (1 - literals).T == 0)   # (C, B)
    class_sums = (W.float @ fired).T                 # (B, n_classes)   <-- the matmul
    pred       = argmax(class_sums)

Training is the coalesced per-example update (target / not-target passes), mirroring
tmu's CoalescedTsetlinMachine.update: for each pass a single shared `fired` vector is
computed once and the inherited Type I / Type II automaton feedback is steered by the
*sign* of that class's weights, and (for weighted_clauses) the weight magnitudes are
nudged toward/away on firing clauses. We REUSE `dense._type_I` / `_type_II` verbatim —
they already act on an arbitrary clause mask, so partitioning never mattered to them.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .dense import DenseTsetlinMachine
from .bitpacked import packed_eval as _packed_eval


class CoalescedTsetlinMachine(DenseTsetlinMachine):
    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float,
        n_classes: int = 2,
        number_of_state_bits: int = 8,
        boost_true_positive_feedback: bool = False,
        max_weight: int = 255,
        weighted_clauses: bool = True,
        packed_eval: bool = False,
        seed: int = 0,
    ):
        super().__init__(
            n_clauses=n_clauses,
            T=T,
            s=s,
            n_classes=n_classes,
            number_of_state_bits=number_of_state_bits,
            boost_true_positive_feedback=boost_true_positive_feedback,
            seed=seed,
        )
        self.max_weight = int(max_weight)
        self.weighted_clauses = bool(weighted_clauses)
        # When True, inference clause evaluation uses the bit-packed Metal kernel
        # (`bitpacked.packed_eval`) instead of the inherited dense matmul. Training is
        # unaffected — `_update_example` computes `fired` inline, not via `_eval`.
        self.packed_eval = bool(packed_eval)
        self.W = None

    # -------------------------------------------------------------- inference
    def _eval(self, lits: mx.array, predict: bool) -> mx.array:
        if self.packed_eval:
            return _packed_eval(self.ta, self.N, self.C, lits, predict)
        return super()._eval(lits, predict)

    # ------------------------------------------------------------------ setup
    def _init_params(self, n_features: int):
        # One shared clause pool: C = requested clause count (no per-class partition).
        self.f = int(n_features)
        self.L = 2 * self.f
        self.C = self.n_clauses_req
        mx.random.seed(self.seed)
        init = mx.random.randint(self.N - 1, self.N + 1, shape=(self.C, self.L))
        self.ta = init.astype(mx.int16)
        # Per-class signed weights, init random +/-1.
        self.W = (mx.random.randint(0, 2, shape=(self.n_classes, self.C)) * 2 - 1).astype(mx.int32)
        mx.eval(self.ta, self.W)

    # -------------------------------------------------------------- inference
    def decision_function(self, X) -> np.ndarray:
        lits = self._literals(mx.array(np.asarray(X), dtype=mx.float32))
        fired = self._eval(lits, predict=True)                       # (C, B)
        class_sums = (self.W.astype(mx.float32) @ fired).T           # (B, n_classes)
        mx.eval(class_sums)
        return np.array(class_sums)

    # --------------------------------------------------------------- feedback
    def _set_weight_row(self, cls: int, new_row: mx.array):
        """Functionally overwrite one class row of W (avoids in-place scatter)."""
        rows = mx.arange(self.n_classes)
        self.W = mx.where((rows == cls)[:, None], new_row[None, :], self.W)

    def _update_example(self, lit: mx.array, target: int):
        lit_row = lit[None, :]
        include = (self.ta >= self.N).astype(mx.float32)
        fired = (include @ (1.0 - lit)) == 0                          # (C,) bool, computed ONCE
        fired_f = fired.astype(mx.float32)

        # ---- TARGET pass -------------------------------------------------
        w_t = self.W[target]
        cs_t = mx.clip((w_t.astype(mx.float32) * fired_f).sum(), -self.T, self.T)
        up_t = (self.T - cs_t) / (2.0 * self.T)
        sel = mx.random.uniform(shape=(self.C,)) <= up_t
        w_pos = w_t >= 0
        self._type_I(sel & w_pos, lit_row, fired)                     # positive-weight clauses
        self._type_II(sel & (~w_pos), lit_row, fired)                 # negative-weight clauses
        if self.weighted_clauses:
            wsel = mx.random.uniform(shape=(self.C,)) <= up_t
            inc = (wsel & w_pos & fired).astype(mx.int32)
            self._set_weight_row(target, mx.minimum(w_t + inc, self.max_weight))

        # ---- NOT-TARGET pass --------------------------------------------
        nt = int(self._rng.randint(self.n_classes))
        while nt == target:
            nt = int(self._rng.randint(self.n_classes))
        w_n = self.W[nt]
        cs_n = mx.clip((w_n.astype(mx.float32) * fired_f).sum(), -self.T, self.T)  # SAME fired
        up_n = (self.T + cs_n) / (2.0 * self.T)
        sel2 = mx.random.uniform(shape=(self.C,)) <= up_n
        w_pos_n = w_n >= 0
        self._type_I(sel2 & (~w_pos_n), lit_row, fired)               # negative-weight clauses
        self._type_II(sel2 & w_pos_n, lit_row, fired)                 # positive-weight clauses
        if self.weighted_clauses:
            wsel2 = mx.random.uniform(shape=(self.C,)) <= up_n
            dec = (wsel2 & (~w_pos_n) & fired).astype(mx.int32)
            self._set_weight_row(nt, mx.maximum(w_n - dec, -self.max_weight))

    def fit(self, X, Y, epochs: int = 1, incremental: bool = False,
            shuffle: bool = True, eval_every: int = 64, verbose: int = 0):
        """verbose: 0=silent, 1=epoch-level, 2=epoch+example every eval_every steps."""
        import time as _time
        Xm = mx.array(np.asarray(X), dtype=mx.float32)
        Yn = np.asarray(Y).astype(np.int32)
        if self.ta is None or not incremental:
            self.n_classes = max(self.n_classes, int(Yn.max()) + 1)
            self._init_params(Xm.shape[1])
        lits = self._literals(Xm)
        n = Xm.shape[0]
        t0 = _time.time()
        for ep in range(epochs):
            idx = np.arange(n)
            if shuffle:
                self._rng.shuffle(idx)
            for step, i in enumerate(idx):
                self._update_example(lits[int(i)], int(Yn[i]))
                if (step + 1) % eval_every == 0:
                    mx.eval(self.ta, self.W)    # flush graph; prevents unbounded graph growth
                    if verbose >= 2:
                        print(f"  ep {ep+1}/{epochs}  ex {step+1}/{n}  "
                              f"({_time.time()-t0:.0f}s)", flush=True)
            mx.eval(self.ta, self.W)            # flush at end of every epoch
            if verbose >= 1:
                print(f"  epoch {ep+1}/{epochs}  ({_time.time()-t0:.0f}s)", flush=True)
            # Hard flush every 5 epochs to prevent graph accumulation across epochs
            if (ep + 1) % 5 == 0:
                import gc; gc.collect()
        return self
