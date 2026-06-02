"""Custom Metal kernels for bit-packed Tsetlin clause evaluation.

Ports the cair/PyTsetlinMachineCUDA clause-evaluation idiom to Apple Metal:
literals and the clause include/action mask are packed into uint32 chunks, and a
clause fires for an example iff every included literal is present, i.e.

    for all chunks ch:  (action[ch] & X[ch]) == action[ch]

which is pure bitwise AND + compare — no multiply, 32 literals per word. Uses the
mlx-addons `mx.fast.metal_kernel` + cached-compile idiom.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_CACHE: dict = {}

# Each thread owns one (clause, example) pair and ANDs across all uint32 chunks.
_CLAUSE_EVAL_SRC = """
    uint gid = thread_position_in_grid.x;
    if (gid >= TOTAL) return;
    uint c = gid / B;
    uint e = gid % B;
    uint fired = 1u;
    for (uint ch = 0; ch < NCHUNKS; ++ch) {
        uint a = action[c * NCHUNKS + ch];
        uint x = Xp[e * NCHUNKS + ch];
        if ((a & x) != a) { fired = 0u; break; }
    }
    out[gid] = (float) fired;
"""


def _get_kernel(name, input_names, output_names, source):
    if name not in _KERNEL_CACHE:
        _KERNEL_CACHE[name] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
        )
    return _KERNEL_CACHE[name]


def clause_eval_packed(action: mx.array, Xp: mx.array, n_clauses: int,
                       batch: int, n_chunks: int) -> mx.array:
    """action (C*NCHUNKS,) uint32, Xp (B*NCHUNKS,) uint32 -> clause_out (C, B) float32."""
    total = n_clauses * batch
    kernel = _get_kernel("tm_clause_eval", ["action", "Xp"], ["out"], _CLAUSE_EVAL_SRC)
    (out,) = kernel(
        inputs=[action, Xp],
        template=[("TOTAL", total), ("B", batch), ("NCHUNKS", n_chunks)],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )
    return out.reshape(n_clauses, batch)


# ---------------------------------------------------------------------------
# Bit-plane TRAINING kernel — ported from cair/PyTsetlinMachineCUDA inc/dec/
# update_clause (kernels.py lines 43-172). STATE_BITS planes per literal are
# packed into uint32 words; plane b of clause c / chunk ch lives at
#     ta_state[(c*NCHUNKS + ch)*STATE_BITS + b]
# and the MSB plane (b = STATE_BITS-1) is the include/action mask. inc/dec are
# the ripple-carry counter ops applied bit-parallel to all 32 automata in a word.
# ---------------------------------------------------------------------------

# Helper functions live in `header=` (outside the auto-generated kernel body).
# `planes` is a thread-local array of STATE_BITS uint32 words for one (clause,
# chunk); `active` flags which of the 32 automata to step.
_BITPLANE_HEADER = """
    // Increment the 32 counters flagged in `active` (ripple carry, saturating).
    // `sb` = number of bit-planes (template STATE_BITS is only substituted in
    // the kernel body, so it is passed in explicitly here).
    inline void bp_inc(thread uint *planes, uint active, int sb) {
        uint carry = active;
        for (int b = 0; b < sb; ++b) {
            if (carry == 0u) break;
            uint carry_next = planes[b] & carry;   // overflow bits -> next plane
            planes[b] = planes[b] ^ carry;          // increment via XOR
            carry = carry_next;
        }
        if (carry > 0u) {                           // overflow: saturate to all-ones
            for (int b = 0; b < sb; ++b) planes[b] |= carry;
        }
    }

    // Decrement the 32 counters flagged in `active` (ripple borrow, saturating).
    inline void bp_dec(thread uint *planes, uint active, int sb) {
        uint carry = active;
        for (int b = 0; b < sb; ++b) {
            if (carry == 0u) break;
            uint carry_next = (~planes[b]) & carry;  // borrow bits -> next plane
            planes[b] = planes[b] ^ carry;           // decrement via XOR
            carry = carry_next;
        }
        if (carry > 0u) {                            // underflow: saturate to zero
            for (int b = 0; b < sb; ++b) planes[b] &= ~carry;
        }
    }
"""

# One thread per clause. Reads the full packed state and writes the full updated
# state (MLX kernel outputs are fresh arrays — no in-place mutation). Mirrors the
# host-selected Type I / Type II feedback (tI / tII per clause) from dense.py.
_BITPLANE_UPDATE_SRC = """
    uint c = thread_position_in_grid.x;
    if (c >= CLAUSES) return;

    uchar do_I  = tI[c];
    uchar do_II = tII[c];
    uchar did_fire = fired[c];

    for (uint ch = 0; ch < NCHUNKS; ++ch) {
        uint base = (c * NCHUNKS + ch) * STATE_BITS;

        // Load this (clause, chunk)'s STATE_BITS planes into registers.
        uint planes[STATE_BITS];
        for (int b = 0; b < STATE_BITS; ++b) planes[b] = ta_state[base + b];

        if (do_I) {
            uint Xw  = X[ch];
            uint laf = la_feedback[c * NCHUNKS + ch];
            if (did_fire) {
                // reinforce present literals; with boost, ignore the 1/s mask
                uint inc_mask = BOOST ? Xw : (Xw & (~laf));
                bp_inc(planes, inc_mask, STATE_BITS);
                bp_dec(planes, (~Xw) & laf, STATE_BITS);   // weaken absent literals (prob 1/s)
            } else {
                bp_dec(planes, laf, STATE_BITS);           // forget everything (prob 1/s)
            }
        } else if (do_II) {
            if (did_fire) {
                uint action_word = planes[STATE_BITS - 1];   // MSB plane = include mask
                // include absent literals that are currently excluded
                bp_inc(planes, (~X[ch]) & (~action_word), STATE_BITS);
            }
        }
        // else: no feedback selected -> planes unchanged (still written out).

        for (int b = 0; b < STATE_BITS; ++b) out[base + b] = planes[b];
    }
"""

# Bit-plane kernels need a `header=` for inc/dec, so they are compiled/cached
# here (the existing _get_kernel above is intentionally left untouched).
def _get_kernel_h(name, input_names, output_names, source, header):
    key = (name, "h")
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
        )
    return _KERNEL_CACHE[key]


def bitplane_update(ta_state: mx.array, X: mx.array, fired: mx.array,
                    tI: mx.array, tII: mx.array, la_feedback: mx.array,
                    n_clauses: int, n_chunks: int, state_bits: int,
                    boost: bool) -> mx.array:
    """One online example's bit-plane Type I/II update.

    ta_state    (C*NCHUNKS*STATE_BITS,) uint32 — packed planes (MSB = action).
    X           (NCHUNKS,)              uint32 — this example's packed literals.
    fired       (C,)                    uint8  — clause output (MSB action plane).
    tI, tII     (C,)                    uint8  — host-selected feedback masks.
    la_feedback (C*NCHUNKS,)            uint32 — per-literal 1/s mask.
    Returns the full updated ta_state (C*NCHUNKS*STATE_BITS,) uint32.
    """
    kernel = _get_kernel_h(
        "tm_bitplane_update",
        ["ta_state", "X", "fired", "tI", "tII", "la_feedback"],
        ["out"],
        _BITPLANE_UPDATE_SRC,
        _BITPLANE_HEADER,
    )
    (out,) = kernel(
        inputs=[ta_state, X, fired, tI, tII, la_feedback],
        template=[("CLAUSES", n_clauses), ("NCHUNKS", n_chunks),
                  ("STATE_BITS", state_bits), ("BOOST", 1 if boost else 0)],
        grid=(n_clauses, 1, 1),
        threadgroup=(min(256, n_clauses), 1, 1),
        output_shapes=[(ta_state.size,)],
        output_dtypes=[mx.uint32],
    )
    return out


# ---------------------------------------------------------------------------
# BATCHED bit-plane update — the throughput port of the CUDA design.
#
# One thread per clause processes a whole batch of B examples: it loads each
# (clause, chunk)'s STATE_BITS planes ONCE into registers, applies all B examples'
# Type I/II updates in place (state evolves across the batch, as in CUDA's in-place
# global mutation), and writes the planes out ONCE. This amortises the full-state
# copy (the per-example kernel's bottleneck) and the launch overhead across B.
#
# The 1/s `la_feedback` mask is generated INSIDE the kernel (xorshift32, seeded per
# (clause, example, chunk, batch-seed)). Materialising it on the host would be a
# (C, B, 2f) tensor — hundreds of MB at QSAR scale — so an in-kernel counter RNG
# (like CUDA's curand) is the only scalable choice. Clause outputs (`fired`) and the
# Type I/II selection (`tI`/`tII`) are still precomputed host-side from the
# batch-start state: this is the standard mini-batch TM (the cair CUDA NoisyXOR demo
# itself uses batch_size=100). Pad slots carry tI=tII=0 (no-op), so the last partial
# batch needs no separate kernel.
# ---------------------------------------------------------------------------

_BITPLANE_RNG = """
    inline uint xorshift32(thread uint &st) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st;
    }
    // 32-bit mask, each bit set iid ~ Bernoulli(inv_s).
    inline uint bernoulli_mask(uint seed_base, float inv_s) {
        uint st = seed_base * 2654435761u + 1442695041u;
        if (st == 0u) st = 0x9E3779B9u;
        uint mask = 0u;
        for (int b = 0; b < 32; ++b) {
            uint r = xorshift32(st);
            float u = (float)(r >> 8) * (1.0f / 16777216.0f);   // [0,1)
            if (u < inv_s) mask |= (1u << b);
        }
        return mask;
    }
"""
_BITPLANE_BATCH_HEADER = _BITPLANE_HEADER + _BITPLANE_RNG

_BITPLANE_UPDATE_BATCH_SRC = """
    uint c = thread_position_in_grid.x;
    if (c >= CLAUSES) return;
    uint bseed = seed[0];
    float inv_s = inv_s_buf[0];

    for (uint ch = 0; ch < NCHUNKS; ++ch) {
        uint base = (c * NCHUNKS + ch) * STATE_BITS;
        uint planes[STATE_BITS];
        for (int b = 0; b < STATE_BITS; ++b) planes[b] = ta_state[base + b];
        uint valid = valid_mask[ch];

        for (uint e = 0; e < BATCH; ++e) {
            uint idx = c * BATCH + e;
            uchar do_I  = tI[idx];
            uchar do_II = tII[idx];
            uchar did_fire = fired[idx];
            uint Xw = Xb[e * NCHUNKS + ch];
            if (do_I) {
                uint sb_seed = ((bseed * CLAUSES + c) * BATCH + e) * NCHUNKS + ch;
                uint laf = bernoulli_mask(sb_seed, inv_s) & valid;
                if (did_fire) {
                    uint inc_mask = BOOST ? Xw : (Xw & (~laf));
                    bp_inc(planes, inc_mask, STATE_BITS);
                    bp_dec(planes, (~Xw) & laf, STATE_BITS);
                } else {
                    bp_dec(planes, laf, STATE_BITS);
                }
            } else if (do_II) {
                if (did_fire) {
                    uint action_word = planes[STATE_BITS - 1];
                    bp_inc(planes, (~Xw) & (~action_word) & valid, STATE_BITS);
                }
            }
        }
        for (int b = 0; b < STATE_BITS; ++b) out[base + b] = planes[b];
    }
"""


def bitplane_update_batch(ta_state: mx.array, Xb: mx.array, fired: mx.array,
                          tI: mx.array, tII: mx.array, valid_mask: mx.array,
                          seed: int, inv_s: float, n_clauses: int, batch: int,
                          n_chunks: int, state_bits: int, boost: bool) -> mx.array:
    """One mini-batch bit-plane Type I/II update (one thread per clause, B examples).

    ta_state   (C*NCHUNKS*STATE_BITS,) uint32  packed planes (MSB = action).
    Xb         (B*NCHUNKS,)            uint32  packed literals for the B examples.
    fired      (C*B,)                  uint8   clause outputs at batch start.
    tI, tII    (C*B,)                  uint8   host-selected feedback masks (pad cols 0).
    valid_mask (NCHUNKS,)              uint32  real-literal mask (zeros padding bits).
    seed       int                             per-batch RNG seed (vary each batch).
    inv_s      float                           1/s feedback probability.
    Returns the full updated ta_state.
    """
    kernel = _get_kernel_h(
        "tm_bitplane_update_batch",
        ["ta_state", "Xb", "fired", "tI", "tII", "valid_mask", "seed", "inv_s_buf"],
        ["out"],
        _BITPLANE_UPDATE_BATCH_SRC,
        _BITPLANE_BATCH_HEADER,
    )
    (out,) = kernel(
        inputs=[ta_state, Xb, fired, tI, tII, valid_mask,
                mx.array([seed], dtype=mx.uint32), mx.array([inv_s], dtype=mx.float32)],
        template=[("CLAUSES", n_clauses), ("BATCH", batch), ("NCHUNKS", n_chunks),
                  ("STATE_BITS", state_bits), ("BOOST", 1 if boost else 0)],
        grid=(n_clauses, 1, 1),
        threadgroup=(min(256, n_clauses), 1, 1),
        output_shapes=[(ta_state.size,)],
        output_dtypes=[mx.uint32],
    )
    return out
