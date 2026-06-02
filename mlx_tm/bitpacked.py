"""Option 3 — bit-packed uint32 Tsetlin Machine with a Metal clause-eval kernel.

Literals and the clause include/action mask are packed into uint32 chunks (32
literals/word) and clause evaluation runs as bitwise AND + compare in a custom
Metal kernel (`kernels.clause_eval_packed`) — the cair/PyTsetlinMachineCUDA design
ported to Apple Metal. This is the high-throughput inference path: ~32x less memory
for the masks and no multiplies.

It subclasses the *verified* DenseTsetlinMachine and overrides only the evaluation
path, so training uses the exact same (validated) Type I / Type II feedback. The
automaton states live as int8 (shared with the dense feedback); the include mask is
packed on the fly for the kernel. A fully bit-plane-packed update kernel (the CUDA
`inc`/`dec` carry ops) is the documented next step.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .dense import DenseTsetlinMachine
from .kernels import clause_eval_packed


def pack_bits_uint32(bits: np.ndarray) -> np.ndarray:
    """(n, nbits) {0,1} -> (n, nchunks) uint32, with literal 32*ch+p in bit p of chunk ch."""
    n, nbits = bits.shape
    nchunks = (nbits + 31) // 32
    padded = np.zeros((n, nchunks * 32), dtype=np.uint32)
    padded[:, :nbits] = bits.astype(np.uint32)
    weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    return (padded.reshape(n, nchunks, 32) * weights).sum(axis=2).astype(np.uint32)


_PACK_WEIGHTS = (np.uint32(1) << np.arange(32, dtype=np.uint32))       # bit p -> 1<<p


def pack_bits_uint32_mx(bits: mx.array) -> mx.array:
    """On-GPU bit-packing: (..., nbits) in {0,1} -> (..., nchunks) uint32.

    MLX-native twin of pack_bits_uint32 (identical layout: literal 32*ch+p occupies
    bit p of chunk ch). Stays on the Apple GPU — avoids the host numpy round-trip that
    otherwise dominates inference and dwarfs the clause-eval kernel.
    """
    lead = tuple(bits.shape[:-1])
    nbits = bits.shape[-1]
    nchunks = (nbits + 31) // 32
    pad = nchunks * 32 - nbits
    if pad:
        bits = mx.concatenate([bits, mx.zeros(lead + (pad,), dtype=bits.dtype)], axis=-1)
    bits = bits.reshape(lead + (nchunks, 32)).astype(mx.uint32)
    weights = mx.array(_PACK_WEIGHTS)                                  # (32,) uint32
    return (bits * weights).sum(axis=-1).astype(mx.uint32)            # distinct bits -> no overflow


def packed_eval(ta: mx.array, N: int, n_clauses: int, lits: mx.array,
                predict: bool) -> mx.array:
    """Bit-packed Metal clause evaluation, reusable across TM variants.

    ta/lits stay on the GPU: the include/action mask and the batch literals are packed
    into uint32 chunks with pack_bits_uint32_mx (no numpy round-trip), then clauses are
    evaluated with the bitwise `clause_eval_packed` Metal kernel. Returns fired (C, B).
    """
    n_chunks = (ta.shape[1] + 31) // 32
    include = ta >= N                                                  # (C, L) bool, on GPU
    action_p = pack_bits_uint32_mx(include).reshape(-1)                # (C*nchunks,)
    Xp = pack_bits_uint32_mx(lits).reshape(-1)                         # (B*nchunks,)
    B = lits.shape[0]
    fired = clause_eval_packed(action_p, Xp, n_clauses, B, n_chunks)   # (C, B)
    if predict:
        cnt = include.astype(mx.int32).sum(axis=1, keepdims=True)
        fired = fired * (cnt > 0).astype(mx.float32)
    return fired


class BitPackedTsetlinMachine(DenseTsetlinMachine):
    @property
    def n_chunks(self) -> int:
        return (self.L + 31) // 32

    def _eval(self, lits: mx.array, predict: bool) -> mx.array:
        # Pack the clause include/action mask and the batch literals into uint32 chunks,
        # then evaluate clauses with the bitwise Metal kernel.
        return packed_eval(self.ta, self.N, self.C, lits, predict)
