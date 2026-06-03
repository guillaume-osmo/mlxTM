"""Option 4 — CUDA-style ONLINE-batched training on the Apple GPU.

The mini-batch backends (`bitplane`, weighted or not) are 17x faster than the
per-example `coalesced` path but UNDER-TRAIN: they apply only ~n/B weight+selection
updates per epoch instead of n, which collapses accuracy on rare-positive targets
(CYP2D6: 0.42 vs 0.70 online). PyTsetlinMachineCUDA solves the same problem the
right way: it keeps the per-EXAMPLE (online) update but runs it on the GPU, looping
many examples *inside* one kernel launch so there is neither a host round-trip per
example (the coalesced bottleneck) nor mini-batch staleness.

This is the MLX/Metal port of that design, with one simplification that makes it
exact and race-free instead of CUDA's relaxed global-atomic class_sum:

    ONE THREADGROUP PER CLASS. PyTsetlinMachineCUDA is dense-partitioned — each
    class owns its own clause pool (kernels.py:193) with a per-clause sign and an
    optional learned weight (MAX_WEIGHT). We give class k its own threadgroup; its
    threads stride over class k's clauses. Per example we (1) recompute every
    clause output from the LIVE state, (2) threadgroup-reduce the signed-weighted
    votes to class_sum, (3) apply Type I/II feedback + the weight nudge. Because a
    class owns its clauses, concurrent class-threadgroups never write shared state,
    so class_sum needs no atomics and the result is the exact online update.

    The CUDA margin rule (target sum +T for the true class, -T otherwise) reduces,
    for binary problems, to dense.py's exact target/negative update probabilities.

State evolves inside the kernel across a chunk of examples (we copy ta_state ->
out / weights once, then read+write `out`), so one launch trains a whole chunk;
the chunk size only bounds kernel duration, it is NOT a mini-batch.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .kernels import _get_kernel_h, _BITPLANE_HEADER, clause_eval_packed
from .bitpacked import pack_bits_uint32


# Counter-based RNG + per-literal 1/s mask, generated on-device (no (C,L,n) host tensor).
_ONLINE_RNG = """
    inline uint hash32(uint x) {
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return x;
    }
    inline float urand(uint s) { return (float)(hash32(s) >> 8) * (1.0f / 16777216.0f); }
    inline uint bernoulli_mask(uint seed_base, float inv_s) {
        uint mask = 0u;
        for (int b = 0; b < 32; ++b)
            if (urand(seed_base * 32u + (uint)b) < inv_s) mask |= (1u << b);
        return mask;
    }
"""
_ONLINE_HEADER = _BITPLANE_HEADER + _ONLINE_RNG

# Grid = TG_SIZE * n_classes threads (one threadgroup per class). Two outputs:
# `out` (updated ta_state) and `wout` (updated weights). Reads/writes evolve `out`
# in place across the example chunk, so the chunk trains fully online.
_ONLINE_TRAIN_SRC = """
    uint gid = thread_position_in_grid.x;
    uint k   = gid / TG_SIZE;                 // class index (= threadgroup)
    uint tid = gid % TG_SIZE;                 // thread within the class group
    if (k >= NCLASSES) return;

    threadgroup int tg_part[TG_SIZE];
    threadgroup int tg_cs;

    uint base_clause = k * CPC;
    float inv_s = inv_s_buf[0];
    uint  gseed = seed[0];
    int   Tf    = THRESHOLD;
    uint  nexc  = n_ex_chunk[0];

    // ---- copy this class's clauses (ta_state -> out, win -> wout) ----
    for (uint j = tid; j < CPC; j += TG_SIZE) {
        uint c = base_clause + j;
        for (uint ch = 0; ch < NCHUNKS; ++ch) {
            uint b0 = (c * NCHUNKS + ch) * STATE_BITS;
            for (int b = 0; b < STATE_BITS; ++b) out[b0 + b] = ta_state[b0 + b];
        }
        wout[c] = win[c];
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint ei = 0; ei < nexc; ++ei) {
        uint e   = order[ei];
        int  lab = labels[e];

        // ---- PASS 1: class_sum over this class's clauses (signed, weighted) ----
        int part = 0;
        for (uint j = tid; j < CPC; j += TG_SIZE) {
            uint c = base_clause + j;
            bool fired = true;
            for (uint ch = 0; ch < NCHUNKS; ++ch) {
                uint action = out[(c * NCHUNKS + ch) * STATE_BITS + (STATE_BITS - 1)];
                if ((action & X[e * NCHUNKS + ch]) != action) { fired = false; break; }
            }
            if (fired) { int sgn = (j & 1u) ? -1 : 1; part += sgn * (int)wout[c]; }
        }
        tg_part[tid] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = TG_SIZE >> 1; off > 0; off >>= 1) {     // tree reduction
            if (tid < off) tg_part[tid] += tg_part[tid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            int cs = tg_part[0];
            if (cs > Tf) cs = Tf; else if (cs < -Tf) cs = -Tf;
            tg_cs = cs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int cs   = tg_cs;
        int yT   = (lab == (int)k) ? Tf : -Tf;          // margin target
        int tgt  = (cs > yT) ? -1 : 1;
        float prob = (float)abs(yT - cs) / (2.0f * (float)Tf);

        // ---- PASS 2: per-clause Type I / Type II feedback (+ weight nudge) ----
        for (uint j = tid; j < CPC; j += TG_SIZE) {
            uint c = base_clause + j;
            uint rseed = (gseed * N_EX_TOTAL + e) * TOTAL_C + c;
            if (urand(rseed) <= prob) {
                int sgn = (j & 1u) ? -1 : 1;
                bool fired = true;
                for (uint ch = 0; ch < NCHUNKS; ++ch) {
                    uint action = out[(c * NCHUNKS + ch) * STATE_BITS + (STATE_BITS - 1)];
                    if ((action & X[e * NCHUNKS + ch]) != action) { fired = false; break; }
                }
                if (tgt * sgn > 0) {                              // Type I
                    if (WEIGHTED && fired && (int)wout[c] < MAXW) wout[c] = wout[c] + 1;
                    for (uint ch = 0; ch < NCHUNKS; ++ch) {
                        uint b0 = (c * NCHUNKS + ch) * STATE_BITS;
                        uint planes[STATE_BITS];
                        for (int b = 0; b < STATE_BITS; ++b) planes[b] = out[b0 + b];
                        uint Xw  = X[e * NCHUNKS + ch];
                        uint laf = bernoulli_mask(rseed * NCHUNKS + ch + 1u, inv_s) & valid[ch];
                        if (fired) {
                            uint inc_mask = BOOST ? Xw : (Xw & (~laf));
                            bp_inc(planes, inc_mask, STATE_BITS);
                            bp_dec(planes, (~Xw) & laf, STATE_BITS);
                        } else {
                            bp_dec(planes, laf, STATE_BITS);
                        }
                        for (int b = 0; b < STATE_BITS; ++b) out[b0 + b] = planes[b];
                    }
                } else if (tgt * sgn < 0 && fired) {             // Type II
                    if (WEIGHTED && (int)wout[c] > 1) wout[c] = wout[c] - 1;
                    for (uint ch = 0; ch < NCHUNKS; ++ch) {
                        uint b0 = (c * NCHUNKS + ch) * STATE_BITS;
                        uint planes[STATE_BITS];
                        for (int b = 0; b < STATE_BITS; ++b) planes[b] = out[b0 + b];
                        uint action = planes[STATE_BITS - 1];
                        bp_inc(planes, (~X[e * NCHUNKS + ch]) & (~action) & valid[ch], STATE_BITS);
                        for (int b = 0; b < STATE_BITS; ++b) out[b0 + b] = planes[b];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
"""


def _online_train_chunk(ta_state, weights, Xp, labels, order, n_ex_chunk, seed, inv_s,
                        valid, cpc, n_classes, n_chunks, state_bits, T, max_weight,
                        weighted, boost, n_ex_total, tg_size):
    total_c = cpc * n_classes
    kernel = _get_kernel_h(
        "tm_online_train",
        ["ta_state", "win", "labels", "order", "n_ex_chunk", "seed",
         "inv_s_buf", "valid", "X"],
        ["out", "wout"],
        _ONLINE_TRAIN_SRC, _ONLINE_HEADER,
    )
    out, wout = kernel(
        inputs=[ta_state, weights, labels, order,
                mx.array([n_ex_chunk], dtype=mx.uint32), mx.array([seed], dtype=mx.uint32),
                mx.array([inv_s], dtype=mx.float32), valid, Xp],
        template=[("CPC", cpc), ("NCLASSES", n_classes), ("NCHUNKS", n_chunks),
                  ("STATE_BITS", state_bits), ("THRESHOLD", int(T)), ("MAXW", int(max_weight)),
                  ("WEIGHTED", 1 if weighted else 0), ("BOOST", 1 if boost else 0),
                  ("TOTAL_C", total_c), ("N_EX_TOTAL", int(n_ex_total)), ("TG_SIZE", tg_size)],
        grid=(tg_size * n_classes, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(ta_state.size,), (weights.size,)],
        output_dtypes=[mx.uint32, mx.int32],
    )
    return out, wout


class OnlineTsetlinMachine:
    """Dense-partitioned, optionally weighted TM trained fully online on the GPU.

    `n_clauses` is the TOTAL clause budget (to match the coalesced/tmu convention);
    it is split evenly into `n_clauses // n_classes` clauses per class, half positive
    half negative sign. API matches the other backends: fit / decision_function / predict.
    """

    def __init__(self, n_clauses, T, s, n_classes=2, number_of_state_bits=8,
                 boost_true_positive_feedback=False, weighted_clauses=False,
                 max_weight=255, seed=0, chunk=256, tg_size=256):
        self.n_clauses_req = int(n_clauses)
        self.T = float(T)
        self.s = float(s)
        self.n_classes = int(n_classes)
        self.number_of_state_bits = int(number_of_state_bits)
        self.boost = bool(boost_true_positive_feedback)
        self.weighted = bool(weighted_clauses)
        self.max_weight = int(max_weight)
        self.seed = int(seed)
        self.chunk = int(chunk)
        self.tg_size = int(tg_size)
        self._rng = np.random.RandomState(seed)
        self.ta_state = None
        self.W = None
        self.f = None
        self.L = None

    @property
    def n_chunks(self) -> int:
        return (self.L + 31) // 32

    def _valid_mask(self) -> np.ndarray:
        bits = np.zeros((1, self.n_chunks * 32), dtype=np.uint32)
        bits[:, : self.L] = 1
        return pack_bits_uint32(bits).reshape(-1)

    def _init_params(self, n_features: int):
        self.f = int(n_features)
        self.L = 2 * self.f
        self.cpc = max(1, self.n_clauses_req // self.n_classes)
        self.C = self.cpc * self.n_classes
        self.N = 1 << (self.number_of_state_bits - 1)
        sb, nch = self.number_of_state_bits, self.n_chunks
        # init counter N-1 (action 0): planes 0..sb-2 all ones, MSB plane 0; pad bits 0.
        state = np.zeros((self.C * nch, sb), dtype=np.uint32)
        if sb > 1:
            state[:, : sb - 1] = np.uint32(0xFFFFFFFF)
        valid = self._valid_mask()
        state = state * np.tile(valid, self.C)[:, None]
        self.ta_state = mx.array(state.reshape(-1))
        self.W = mx.ones((self.C,), dtype=mx.int32)
        self._valid = mx.array(valid)
        # per-class signed-weight gather matrix for inference (built lazily in eval)
        within = np.arange(self.C) % self.cpc
        self._sign = np.where(within % 2 == 0, 1, -1).astype(np.int32)
        self._class_of = (np.arange(self.C) // self.cpc).astype(np.int32)
        mx.eval(self.ta_state, self.W)

    def _pack(self, Xint: np.ndarray) -> np.ndarray:
        lits = np.concatenate([Xint, 1 - Xint], axis=1).astype(np.uint32)   # (n, 2f)
        Xp = pack_bits_uint32(lits)                                          # (n, NCHUNKS)
        pad = (~np.array(self._valid)) & np.uint32(0xFFFFFFFF)               # padding -> present
        return (Xp | pad[None, :]).reshape(-1)                               # (n*NCHUNKS,)

    def fit(self, X, Y, epochs: int = 1, incremental: bool = False, shuffle: bool = True):
        Xint = np.asarray(X).astype(np.int32)
        Yn = np.asarray(Y).astype(np.int32)
        if self.ta_state is None or not incremental:
            self.n_classes = max(self.n_classes, int(Yn.max()) + 1)
            self._init_params(Xint.shape[1])
        n = Xint.shape[0]
        Xp = mx.array(self._pack(Xint))
        labels = mx.array(Yn)
        inv_s = 1.0 / self.s
        ch = self.chunk
        for _ in range(epochs):
            idx = np.arange(n)
            if shuffle:
                self._rng.shuffle(idx)
            for start in range(0, n, ch):
                oc = idx[start:start + ch].astype(np.uint32)
                ne = len(oc)
                if ne < ch:                                  # pad to fixed chunk length
                    oc = np.concatenate([oc, np.zeros(ch - ne, dtype=np.uint32)])
                self.ta_state, self.W = _online_train_chunk(
                    self.ta_state, self.W, Xp, labels, mx.array(oc), ne,
                    int(self._rng.randint(1, 2 ** 31 - 1)), inv_s, self._valid,
                    self.cpc, self.n_classes, self.n_chunks, self.number_of_state_bits,
                    self.T, self.max_weight, self.weighted, self.boost, n, self.tg_size)
            mx.eval(self.ta_state, self.W)
        return self

    def _action_words(self) -> mx.array:
        sb = self.number_of_state_bits
        return self.ta_state.reshape(self.C * self.n_chunks, sb)[:, sb - 1]   # (C*NCHUNKS,)

    def decision_function(self, X) -> np.ndarray:
        Xint = np.asarray(X).astype(np.int32)
        B = Xint.shape[0]
        Xp = mx.array(self._pack(Xint))
        action = self._action_words()
        fired = clause_eval_packed(action, Xp, self.C, B, self.n_chunks)      # (C, B)
        aw = action.reshape(self.C, self.n_chunks)
        nonempty = ((aw & self._valid[None, :]).sum(axis=1) > 0).astype(mx.float32)  # abstain mask
        fired = fired * nonempty[:, None]
        # signed-weight class matrix Sw (n_classes, C)
        sw = np.zeros((self.n_classes, self.C), dtype=np.float32)
        W = np.array(self.W)
        sw[self._class_of, np.arange(self.C)] = self._sign * W
        class_sums = (mx.array(sw) @ fired).T                                 # (B, n_classes)
        mx.eval(class_sums)
        return np.array(class_sums)

    def predict(self, X) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1).astype(np.int32)
