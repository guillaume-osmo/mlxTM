"""Option 2 — drop-in, tmu / PyTsetlinMachineCUDA-compatible API.

`TMClassifierMLX` mirrors the surface that PaulC61/TM-QSAR-Benchmark drives:

    clf = TMClassifierMLX(number_of_clauses=1600, T=8000, s=5.0, platform="MLX")
    for epoch in range(N):
        clf.fit(X_train, Y_train, incremental=True)     # one epoch per call
    yhat, class_sums = clf.predict(X_test, return_class_sums=True)
    prob = expit(class_sums / clf.T)[:, 1]

so the benchmark scripts run unchanged except for the import + platform. Accepts
(and ignores) the extra tmu coalesced kwargs so existing param dicts still work.
The compute runs on the Apple GPU via the `dense` (default) or `bitpacked` backend.
"""

from __future__ import annotations

import numpy as np

from .dense import DenseTsetlinMachine


class TMClassifierMLX:
    def __init__(
        self,
        number_of_clauses: int,
        T: float,
        s: float,
        platform: str = "MLX",
        weighted_clauses: bool = False,
        number_of_state_bits_ta: int = 8,
        boost_true_positive_feedback: int = 0,
        backend: str = "dense",
        max_weight: int = 255,        # coalesced backend only
        batch_size: int = 0,          # bitplane backend only (0 = per-example online)
        seed=None,
        **kwargs,                     # swallow unused tmu coalesced kwargs
    ):
        self.number_of_clauses = int(number_of_clauses)
        self.T = float(T)
        self.s = float(s)
        self.platform = platform
        self.weighted_clauses = weighted_clauses
        self.number_of_state_bits_ta = int(number_of_state_bits_ta)
        self.boost = bool(boost_true_positive_feedback)
        self.backend = backend
        self.max_weight = int(max_weight)
        self.batch_size = int(batch_size)
        self.seed = 0 if seed is None else int(seed)
        self._tm = None

    def _make(self, n_classes: int):
        common = dict(
            n_clauses=self.number_of_clauses, T=self.T, s=self.s,
            n_classes=n_classes, number_of_state_bits=self.number_of_state_bits_ta,
            boost_true_positive_feedback=self.boost, seed=self.seed,
        )
        if self.backend == "bitpacked":
            from .bitpacked import BitPackedTsetlinMachine
            return BitPackedTsetlinMachine(**common)
        if self.backend == "bitplane":
            from .bitpacked_train import BitPlaneTsetlinMachine
            return BitPlaneTsetlinMachine(batch_size=self.batch_size,
                                          weighted_clauses=self.weighted_clauses,
                                          max_weight=self.max_weight, **common)
        if self.backend == "online":
            from .online import OnlineTsetlinMachine
            return OnlineTsetlinMachine(weighted_clauses=self.weighted_clauses,
                                        max_weight=self.max_weight,
                                        chunk=self.batch_size or 256, **common)
        if self.backend == "coalesced":
            from .coalesced import CoalescedTsetlinMachine
            return CoalescedTsetlinMachine(weighted_clauses=self.weighted_clauses,
                                           max_weight=self.max_weight, **common)
        return DenseTsetlinMachine(**common)

    # --- scikit-learn / tmu surface ---------------------------------------
    def fit(self, X, Y, incremental: bool = False, epochs: int = 1, shuffle: bool = True, **kwargs):
        Y = np.asarray(Y).astype(np.int32)
        first = self._tm is None or not incremental
        if first:
            self._tm = self._make(int(Y.max()) + 1)
        self._tm.fit(X, Y, epochs=epochs, incremental=not first, shuffle=shuffle)
        return self

    def predict(self, X, return_class_sums: bool = False, **kwargs):
        class_sums = self._tm.decision_function(X)        # (B, n_classes)
        yhat = class_sums.argmax(axis=1).astype(np.int32)
        if return_class_sums:
            return yhat, class_sums
        return yhat

    def decision_function(self, X):
        return self._tm.decision_function(X)
