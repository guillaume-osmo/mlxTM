"""mlx_tm — Tsetlin Machines on the Apple GPU (MLX).

A from-scratch MLX implementation of the Tsetlin Machine (arXiv:1804.01508), with a
custom Metal bit-plane training kernel and continuous->bit feature utilities.

Backends
--------
DenseTsetlinMachine      states as MLX tensors; clause eval is a matmul + step.
BitPackedTsetlinMachine  uint32 bit-packed literals + a Metal clause-eval kernel.
CoalescedTsetlinMachine  one shared clause pool + per-class signed integer weights.
BitPlaneTsetlinMachine   fully bit-packed training (CUDA inc/dec carry ops on Metal).
TMClassifierMLX          tmu / PyTsetlinMachineCUDA-compatible wrapper.

Utilities
---------
ThermometerBinarizer, SoftmaxBinarizer, GaussianizeBinarizer, GMMBinarizer,
RotationBinarizer        continuous -> bit converters.
rpcholesky_select        target-independent (label-free) feature selection.
NumpyTM                  CPU reference / correctness oracle.
"""
from .dense import DenseTsetlinMachine
from .bitpacked import BitPackedTsetlinMachine, pack_bits_uint32, packed_eval
from .coalesced import CoalescedTsetlinMachine
from .bitpacked_train import BitPlaneTsetlinMachine
from .classifier import TMClassifierMLX
from .numpy_ref import NumpyTM
from .binarize import (
    ThermometerBinarizer,
    SoftmaxBinarizer,
    GaussianizeBinarizer,
    GMMBinarizer,
    RotationBinarizer,
)
from .feature_select import rpcholesky_select

__version__ = "0.1.0"

__all__ = [
    "DenseTsetlinMachine",
    "BitPackedTsetlinMachine",
    "CoalescedTsetlinMachine",
    "BitPlaneTsetlinMachine",
    "TMClassifierMLX",
    "NumpyTM",
    "ThermometerBinarizer",
    "SoftmaxBinarizer",
    "GaussianizeBinarizer",
    "GMMBinarizer",
    "RotationBinarizer",
    "rpcholesky_select",
    "pack_bits_uint32",
    "packed_eval",
]
