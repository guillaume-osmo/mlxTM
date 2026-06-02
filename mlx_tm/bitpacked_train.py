"""Fully bit-packed (bit-plane) TRAINING for a Tsetlin Machine on the Apple GPU.

Ports cair/PyTsetlinMachineCUDA's `inc`/`dec`/`update_clause` (kernels.py
lines 43-172) to a custom Metal kernel (`kernels.bitplane_update`). Unlike the
`BitPackedTsetlinMachine` in `bitpacked.py` — which packs bits only for the
clause-eval kernel but keeps int8 states and the dense feedback — here the
automaton states themselves live as STATE_BITS bit-planes packed into uint32
words and the Type I / Type II feedback runs entirely as bit-parallel ripple
counters in Metal.

Representation (mirrors the CUDA `prepare`):

    ta_state : (C*NCHUNKS*STATE_BITS,) uint32
               plane b of clause c / chunk ch at index (c*NCHUNKS+ch)*STATE_BITS+b
               MSB plane (b = STATE_BITS-1) IS the include/action mask
    init     : planes 0..STATE_BITS-2 = 0xFFFFFFFF, MSB plane = 0
               -> counter 2^(STATE_BITS-1) - 1 = N-1, just below the include
                  threshold (action = 0), exactly like the int8 dense init [N-1, N].

All randomness and the Type I / Type II clause selection are precomputed on the
HOST in MLX (no RNG in Metal): per example we compute `fired`, the per-clause
selection masks `tI` / `tII` (identical to dense.py / numpy_ref.py), and the
per-literal 1/s `la_feedback` mask, then the kernel applies the carry ops and
returns the full updated state.

Padding literals (when L is not a multiple of 32) are made inert by forcing
their X bit to 1 (an always-present literal never blocks firing and is never
added by Type II) and zeroing their la_feedback bit (their counters never move).
The MSB/action plane therefore never gets a spurious include in a padding slot.

Public API matches DenseTsetlinMachine: fit / decision_function / predict.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .bitpacked import pack_bits_uint32
from .kernels import clause_eval_packed, bitplane_update, bitplane_update_batch


class BitPlaneTsetlinMachine:
    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float,
        n_classes: int = 2,
        number_of_state_bits: int = 8,
        boost_true_positive_feedback: bool = False,
        batch_size: int = 0,           # 0 -> per-example online; >1 -> batched kernel
        seed: int = 0,
    ):
        self.n_clauses_req = int(n_clauses)
        self.T = float(T)
        self.s = float(s)
        self.n_classes = int(n_classes)
        self.number_of_state_bits = int(number_of_state_bits)
        self.N = 1 << (number_of_state_bits - 1)          # include iff counter >= N
        self.boost = bool(boost_true_positive_feedback)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._rng = np.random.RandomState(seed)
        self.ta_state = None                              # (C*NCHUNKS*STATE_BITS,) uint32
        self.f = None
        self.L = None

    # ------------------------------------------------------------------ setup
    @property
    def n_chunks(self) -> int:
        return (self.L + 31) // 32

    def _build_partition(self):
        cpc = max(1, self.n_clauses_req // self.n_classes)
        self.C = cpc * self.n_classes                     # whole multiple of n_classes
        class_of = np.arange(self.C) // cpc
        within = np.arange(self.C) % cpc
        sign = np.where(within % 2 == 0, 1, -1).astype(np.int32)
        onehot = (class_of[None, :] == np.arange(self.n_classes)[:, None]).astype(np.int32)
        self.class_of = mx.array(class_of)
        self.sign = mx.array(sign)
        self.S = mx.array(onehot * sign[None, :], dtype=mx.float32)   # (n_classes, C)

    def _valid_mask(self) -> np.ndarray:
        """(NCHUNKS,) uint32 with 1s only in the L real literal positions."""
        nch = self.n_chunks
        bits = np.zeros((1, nch * 32), dtype=np.uint32)
        bits[:, : self.L] = 1
        return pack_bits_uint32(bits).reshape(-1)         # (NCHUNKS,)

    def _init_params(self, n_features: int):
        self.f = int(n_features)
        self.L = 2 * self.f
        self._build_partition()
        mx.random.seed(self.seed)
        nch = self.n_chunks
        sb = self.number_of_state_bits
        # planes 0..sb-2 = all ones, MSB plane = 0  (counter N-1, action=0)
        state = np.zeros((self.C * nch, sb), dtype=np.uint32)
        if sb > 1:
            state[:, : sb - 1] = np.uint32(0xFFFFFFFF)
        # zero out padding-literal bits in every plane so padding counters are 0
        # and (being absent in X) never become included.
        valid = self._valid_mask()                        # (NCHUNKS,) uint32
        valid_tile = np.tile(valid, self.C)               # (C*NCHUNKS,)
        state = state * valid_tile[:, None]               # padding planes -> 0
        self.ta_state = mx.array(state.reshape(-1))       # (C*NCHUNKS*STATE_BITS,)
        self._valid = mx.array(valid)
        mx.eval(self.ta_state)

    @staticmethod
    def _literals(X: np.ndarray) -> np.ndarray:
        return np.concatenate([X, 1 - X], axis=1)         # [x, ~x] -> (B, 2f)

    def _action_words(self) -> mx.array:
        """Extract the MSB (action/include) plane of every (clause, chunk): (C*NCHUNKS,) uint32."""
        sb = self.number_of_state_bits
        planes = self.ta_state.reshape(self.C * self.n_chunks, sb)
        return planes[:, sb - 1]                          # (C*NCHUNKS,)

    def _pack_examples(self, lits: np.ndarray) -> np.ndarray:
        """Pack literal rows into uint32 chunks and force padding literals present (=1)."""
        Xp = pack_bits_uint32(lits.astype(np.uint32))     # (B, NCHUNKS)
        # padding bits (beyond L) -> 1 so they never block firing / get added.
        pad = (~np.array(self._valid)) & np.uint32(0xFFFFFFFF)
        return Xp | pad[None, :]

    # -------------------------------------------------------------- inference
    def _eval(self, lits: np.ndarray, predict: bool) -> mx.array:
        Xp = mx.array(self._pack_examples(lits).reshape(-1))
        B = lits.shape[0]
        fired = clause_eval_packed(self._action_words(), Xp, self.C, B, self.n_chunks)  # (C, B)
        if predict:
            # empty clause (no real literal included) abstains, like dense.
            action = self._action_words().reshape(self.C, self.n_chunks)
            cnt = mx.sum(_popcount32(action & self._valid[None, :]), axis=1, keepdims=True)
            fired = fired * (cnt > 0).astype(mx.float32)
        return fired

    def decision_function(self, X) -> np.ndarray:
        lits = self._literals(np.asarray(X).astype(np.int32))
        class_sums = (self.S @ self._eval(lits, predict=True)).T          # (B, n_classes)
        mx.eval(class_sums)
        return np.array(class_sums)

    def predict(self, X) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1).astype(np.int32)

    # --------------------------------------------------------------- feedback
    def _update_example(self, lit_packed: mx.array, fired_c: mx.array, target: int):
        """lit_packed (NCHUNKS,) uint32 for this example; fired_c (C,) float32."""
        cs = mx.clip(self.S @ fired_c, -self.T, self.T)                   # (n_classes,)
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
        tI = (sel_t & spos) | (sel_n & (~spos))                          # target+ / neg-
        tII = (sel_t & (~spos)) | (sel_n & spos)                         # target- / neg+

        # per-literal 1/s feedback mask, iid per (clause, literal); pad bits -> 0.
        r = mx.random.uniform(shape=(self.C, self.n_chunks * 32)) < (1.0 / self.s)
        laf_np = pack_bits_uint32(np.array(r).astype(np.uint32))         # (C, NCHUNKS)
        laf_np = laf_np & np.array(self._valid)[None, :]                 # zero padding bits
        la_feedback = mx.array(laf_np.reshape(-1))                       # (C*NCHUNKS,)

        self.ta_state = bitplane_update(
            self.ta_state,
            lit_packed,
            fired_c.astype(mx.uint8),
            tI.astype(mx.uint8),
            tII.astype(mx.uint8),
            la_feedback,
            self.C, self.n_chunks, self.number_of_state_bits, self.boost,
        )

    def _rng_other(self, t: int) -> int:
        nt = self._rng.randint(self.n_classes)
        while nt == t:
            nt = self._rng.randint(self.n_classes)
        return nt

    def _fit_batched(self, Xp_mx, Yn, epochs, shuffle, eval_every):
        """Mini-batch training: one Metal launch per batch (B examples), state
        evolving in registers per clause. Selection/`fired` are computed from the
        batch-start state; `la_feedback` is generated inside the kernel."""
        B = self.batch_size
        n = Yn.shape[0]
        inv_s = 1.0 / self.s
        spos = (self.sign >= 0)[:, None]                                  # (C,1)
        for _ in range(epochs):
            idx = np.arange(n)
            if shuffle:
                self._rng.shuffle(idx)
            for start in range(0, n, B):
                bidx = idx[start:start + B]
                pad = B - len(bidx)
                if pad:
                    bidx = np.concatenate([bidx, np.full(pad, bidx[0], dtype=bidx.dtype)])
                Xb = Xp_mx[mx.array(bidx)]                                # (B, NCHUNKS)
                tgt = Yn[bidx]                                           # (B,)
                # clause outputs at batch start (training eval: empty clause fires)
                fired = clause_eval_packed(self._action_words(), Xb.reshape(-1),
                                           self.C, B, self.n_chunks)      # (C, B)
                cs = np.array(mx.clip(self.S @ fired, -self.T, self.T))   # (n_classes, B)
                ar = np.arange(B)
                neg = np.array([self._rng_other(int(t)) for t in tgt])
                up_t = ((self.T - cs[tgt, ar]) / (2.0 * self.T)).astype(np.float32)
                up_n = ((self.T + cs[neg, ar]) / (2.0 * self.T)).astype(np.float32)
                r1 = mx.random.uniform(shape=(self.C, B))
                r2 = mx.random.uniform(shape=(self.C, B))
                sel_t = (self.class_of[:, None] == mx.array(tgt)[None, :]) & (r1 <= mx.array(up_t)[None, :])
                sel_n = (self.class_of[:, None] == mx.array(neg)[None, :]) & (r2 <= mx.array(up_n)[None, :])
                tI = (sel_t & spos) | (sel_n & (~spos))                  # (C, B)
                tII = (sel_t & (~spos)) | (sel_n & spos)
                if pad:                                                  # zero pad columns -> no-op
                    keep = mx.array(np.concatenate([np.ones(B - pad), np.zeros(pad)]).astype(np.float32))
                    tI = (tI.astype(mx.float32) * keep[None, :]) > 0
                    tII = (tII.astype(mx.float32) * keep[None, :]) > 0
                seed = int(self._rng.randint(1, 2 ** 31 - 1))
                self.ta_state = bitplane_update_batch(
                    self.ta_state, Xb.reshape(-1),
                    fired.astype(mx.uint8).reshape(-1),
                    tI.astype(mx.uint8).reshape(-1),
                    tII.astype(mx.uint8).reshape(-1),
                    self._valid, seed, inv_s,
                    self.C, B, self.n_chunks, self.number_of_state_bits, self.boost)
                mx.eval(self.ta_state)
        return self

    def fit(self, X, Y, epochs: int = 1, incremental: bool = False,
            shuffle: bool = True, eval_every: int = 64):
        Xn = np.asarray(X).astype(np.int32)
        Yn = np.asarray(Y).astype(np.int32)
        if self.ta_state is None or not incremental:
            self.n_classes = max(self.n_classes, int(Yn.max()) + 1)
            self._init_params(Xn.shape[1])
        lits = self._literals(Xn)                                        # (n, L)
        Xp_all = self._pack_examples(lits)                               # (n, NCHUNKS) uint32
        Xp_mx = mx.array(Xp_all)
        if self.batch_size and self.batch_size > 1:
            return self._fit_batched(Xp_mx, Yn, epochs, shuffle, eval_every)
        n = Xn.shape[0]
        for _ in range(epochs):
            idx = np.arange(n)
            if shuffle:
                self._rng.shuffle(idx)
            for step, i in enumerate(idx):
                i = int(i)
                # training-time clause output (empty clause fires) via packed eval.
                fired_c = clause_eval_packed(
                    self._action_words(), Xp_mx[i], self.C, 1, self.n_chunks
                ).reshape(self.C)                                        # (C,) float32
                self._update_example(Xp_mx[i], fired_c, int(Yn[i]))
                if (step + 1) % eval_every == 0:
                    mx.eval(self.ta_state)
            mx.eval(self.ta_state)
        return self


def _popcount32(x: mx.array) -> mx.array:
    """Bit population count of a uint32 MLX array (returned as float32)."""
    x = x.astype(mx.uint32)
    x = x - ((x >> 1) & mx.array(0x55555555, dtype=mx.uint32))
    x = (x & mx.array(0x33333333, dtype=mx.uint32)) + ((x >> 2) & mx.array(0x33333333, dtype=mx.uint32))
    x = (x + (x >> 4)) & mx.array(0x0F0F0F0F, dtype=mx.uint32)
    x = x + (x >> 8)
    x = x + (x >> 16)
    return (x & mx.array(0x3F, dtype=mx.uint32)).astype(mx.float32)
