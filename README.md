# mlx-tm

**Tsetlin Machines on the Apple GPU.** A from-scratch [MLX](https://github.com/ml-explore/mlx)
implementation of the Tsetlin Machine ([Granmo 2018](https://arxiv.org/abs/1804.01508)) — an
interpretable, bit-native rule learner — with a custom **Metal bit-plane training kernel** and
continuous→bit feature utilities.

The reference Tsetlin stacks ([cair/tmu](https://github.com/cair/tmu),
[PyTsetlinMachineCUDA](https://github.com/cair/PyTsetlinMachineCUDA)) are CUDA-only and do not run
on Apple Silicon. `mlx-tm` is native Metal/MLX.

## Install

```bash
git clone https://github.com/guillaume-osmo/mlxTM && cd mlxTM
pip install -e .                   # core: mlx + numpy
pip install -e ".[examples]"       # + rdkit, scikit-learn for the QSAR example
```

## Quickstart

```python
import numpy as np
from mlx_tm import DenseTsetlinMachine

rng = np.random.RandomState(0)
X = rng.randint(0, 2, (2000, 12)).astype(np.int32)
y = (X[:, 0] ^ X[:, 1]).astype(np.int32)            # XOR of two bits

tm = DenseTsetlinMachine(n_clauses=20, T=15, s=3.9, seed=42).fit(X, y, epochs=30)
print((tm.predict(X) == y).mean())                  # ~1.0
```

## Backends

| class | description |
|-------|-------------|
| `DenseTsetlinMachine` | states as MLX tensors; clause evaluation is a matmul + step, Type I/II feedback vectorised over (clauses × literals). |
| `BitPackedTsetlinMachine` | uint32 bit-packed literals + a custom `mx.fast.metal_kernel` doing `(action & X) == action` over chunks (the PyTsetlinMachineCUDA clause-eval, on Metal). Bit-exact with `dense`. |
| `CoalescedTsetlinMachine` | one shared clause pool + per-class signed integer weights. |
| `BitPlaneTsetlinMachine` | fully bit-packed **training** — the CUDA `inc`/`dec` ripple-carry counter ops ported to a Metal kernel (`batch_size=` for the batched path). |
| `TMClassifierMLX` | scikit-learn / `tmu`-style wrapper: `fit(incremental=)`, `predict(return_class_sums=)`, `backend="dense"\|"bitpacked"\|"coalesced"\|"bitplane"`. |

All backends share the classic multiclass semantics and are checked against a NumPy oracle
(`NumpyTM`) and each other (the bit-packed kernel reproduces the dense class sums bit-for-bit).

## Feature utilities (continuous → bits)

Tsetlin Machines consume Boolean literals. `mlx_tm` includes converters and a label-free
feature selector:

```python
from mlx_tm import ThermometerBinarizer, RotationBinarizer, rpcholesky_select

bits = ThermometerBinarizer(n_bins=4).fit_transform(X_continuous)   # ordinal threshold bits
sel  = rpcholesky_select(X_continuous, k=1024)                      # target-independent selection
```

- `ThermometerBinarizer`, `SoftmaxBinarizer`, `GaussianizeBinarizer`, `GMMBinarizer` — per-feature codes.
- `RotationBinarizer` (`random` / `hadamard` / `itq`) — rotation-space binarization (ITQ / QuaRot / QuIP# lineage).
- `rpcholesky_select` — randomly-pivoted Cholesky on the feature Gram; uses **no labels**.

## Tests

```bash
pip install -e ".[test]"
pytest
```

Unit tests cover: pack/unpack round-trips, the bit-plane `inc`/`dec` carry ops, dense↔bit-packed
exactness, binarizer properties (width, binary, thermometer monotonicity, GMM one-hot), RPCholesky
validity/determinism, the classifier API across all backends, and Noisy-XOR learning for every
trainer. (Requires an Apple-Silicon GPU for the Metal-backed tests.)

## Examples

```bash
python examples/noisy_xor.py                 # canonical sanity check
python examples/qsar_ecfp.py data.csv        # ECFP -> Coalesced TM (needs the [examples] extra)
```

## License

MIT © 2026 Guillaume Godin.
