"""Float -> bit converters for Tsetlin Machine input.

Two families:
  * Per-feature ordinal  (ThermometerBinarizer, SoftmaxBinarizer): each feature is
    binarized independently. Standard TM practice (quantile binning). b bits/feature.
  * Rotation-space        (RotationBinarizer): rotate the (standardised) feature block,
    then take the sign — 1 bit per rotated dimension. Imports the KV-cache / hashing
    idea (Gong & Lazebnik ITQ 2011; QuIP/QuIP# incoherence processing; QuaRot/SpinQuant
    Hadamard rotation) into TM feature construction: rotation decorrelates features and
    suppresses outliers, so the same information survives in far fewer bits. All variants
    are TARGET-INDEPENDENT (use no labels), like RPCholesky selection.
"""
from __future__ import annotations

import numpy as np


class ThermometerBinarizer:
    def __init__(self, n_bins: int = 8):
        self.n_bins = int(n_bins)
        self.thresholds_ = None                      # (F, n_bins)

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X):
        X = self._clean(X)
        qs = np.linspace(0.0, 1.0, self.n_bins + 2)[1:-1]   # interior quantiles
        self.thresholds_ = np.quantile(X, qs, axis=0).T      # (F, n_bins)
        return self

    def transform(self, X) -> np.ndarray:
        X = self._clean(X)
        bits = (X[:, :, None] > self.thresholds_[None, :, :]).astype(np.int8)  # (N, F, n_bins)
        return bits.reshape(X.shape[0], -1)                  # (N, F * n_bins)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class SoftmaxBinarizer:
    """Soft quantile binning: each feature -> n_bins bits via a softmax over (squared)
    distances to n_bins quantile centers, binarized at membership >= 1/n_bins.

    Unlike the thermometer's cumulative/ordinal code, this is a LOCALIZED code: only the
    1-3 bins nearest the value light up (temperature controls the width). It mirrors
    molFTP's own softmax_temperature aggregation and tends to need fewer active bits.
    """
    def __init__(self, n_bins: int = 7, temperature: float = 1.0):
        self.n_bins = int(n_bins)
        self.temperature = float(temperature)
        self.centers_ = None      # (F, n_bins)
        self.scale_ = None        # (F,)

    @staticmethod
    def _clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X):
        X = self._clean(X)
        qs = np.linspace(0.0, 1.0, self.n_bins + 2)[1:-1]
        self.centers_ = np.quantile(X, qs, axis=0).T          # (F, n_bins)
        sd = X.std(0); sd[sd < 1e-9] = 1.0
        self.scale_ = sd
        return self

    def transform(self, X) -> np.ndarray:
        X = self._clean(X)
        d = -((X[:, :, None] - self.centers_[None]) ** 2) / (
            self.temperature * (self.scale_[None, :, None] ** 2) + 1e-12)
        d = d - d.max(axis=2, keepdims=True)
        e = np.exp(d)
        soft = e / e.sum(axis=2, keepdims=True)               # softmax over bins (N,F,n_bins)
        bits = (soft >= (1.0 / self.n_bins)).astype(np.int8)
        return bits.reshape(X.shape[0], -1)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class GaussianizeBinarizer:
    """Gaussianize each feature (rank -> normal via a quantile transform) THEN thermometer.
    Spreads skewed / heavy-tailed descriptors so equal-width-in-normal-space bins carry more
    even information. (Domain pre-normalisations like dividing size-extensive descriptors by
    molecular weight are a chemistry-specific special case of the same 'normalise first' idea.)"""
    def __init__(self, n_bins: int = 4, n_quantiles: int = 256):
        from sklearn.preprocessing import QuantileTransformer
        self.n_bins = int(n_bins)
        self._nq = int(n_quantiles)
        self.qt = None
        self.therm = ThermometerBinarizer(n_bins)

    @staticmethod
    def _clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X):
        from sklearn.preprocessing import QuantileTransformer
        X = self._clean(X)
        self.qt = QuantileTransformer(output_distribution="normal",
                                      n_quantiles=min(self._nq, X.shape[0]), subsample=10**9)
        self.therm.fit(self.qt.fit_transform(X))
        return self

    def transform(self, X) -> np.ndarray:
        return self.therm.transform(self.qt.transform(self._clean(X)))

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class GMMBinarizer:
    """Per-feature Gaussian-mixture binning: fit a k-component 1-D GMM, hard-assign each
    value to a component (ordered by mean), one-hot -> k bits/feature. A 'mixture-of-Gaussians'
    alternative to fixed quantile bins that adapts bin edges to multimodal descriptor distributions."""
    def __init__(self, n_components: int = 4, seed: int = 0):
        self.k = int(n_components)
        self.seed = int(seed)
        self.models_ = None      # list of (gmm, mean_order)

    @staticmethod
    def _clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X):
        from sklearn.mixture import GaussianMixture
        X = self._clean(X)
        self.models_ = []
        for j in range(X.shape[1]):
            g = GaussianMixture(self.k, random_state=self.seed, covariance_type="diag",
                                max_iter=50, reg_covar=1e-4).fit(X[:, [j]])
            order = np.argsort(g.means_.ravel())          # stable bit order by component mean
            inv = np.argsort(order)
            self.models_.append((g, inv))
        return self

    def transform(self, X) -> np.ndarray:
        X = self._clean(X)
        out = np.zeros((X.shape[0], X.shape[1] * self.k), dtype=np.int8)
        for j, (g, inv) in enumerate(self.models_):
            comp = inv[g.predict(X[:, [j]])]               # ordered component id
            out[np.arange(X.shape[0]), j * self.k + comp] = 1
        return out

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class RotationBinarizer:
    """Rotate standardised features, then sign -> 1 bit per rotated dimension.

    modes (all label-free / target-independent):
      'random'   - random Gaussian projection then sign (SimHash / random hyperplanes).
      'hadamard' - randomised Hadamard transform (sign-flip diag * Hadamard) then sign,
                   the QuaRot / QuIP# incoherence rotation (O(n log n), outlier-killing).
      'itq'      - PCA to n_bits dims, then a rotation iteratively learned to minimise
                   quantisation error to the binary hypercube (Gong & Lazebnik ITQ 2011).
    """
    def __init__(self, n_bits: int = 256, mode: str = "itq", seed: int = 0, itq_iters: int = 50):
        self.n_bits = int(n_bits)
        self.mode = mode
        self.seed = int(seed)
        self.itq_iters = int(itq_iters)
        self.mu_ = self.sd_ = None
        self._params = {}

    def _std(self, X, fit):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.mu_ = X.mean(0)
            sd = X.std(0); sd[sd < 1e-9] = 1.0
            self.sd_ = sd
        return (X - self.mu_) / self.sd_

    def fit(self, X):
        Xc = self._std(X, fit=True)
        n, F = Xc.shape
        rng = np.random.RandomState(self.seed)
        if self.mode == "random":
            self._params["P"] = rng.randn(F, self.n_bits) / np.sqrt(F)
        elif self.mode == "hadamard":
            from scipy.linalg import hadamard
            m = 1 << int(np.ceil(np.log2(max(2, F))))
            self._params["m"] = m
            self._params["D"] = rng.choice([-1.0, 1.0], size=m)
            self._params["H"] = hadamard(m).astype(np.float64) / np.sqrt(m)
        elif self.mode == "itq":
            c = min(self.n_bits, F)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._params["pca"] = Vt[:c].T                     # (F, c)
            V = Xc @ self._params["pca"]                       # (n, c)
            R = np.linalg.qr(rng.randn(c, c))[0]               # random orthogonal init
            for _ in range(self.itq_iters):
                B = np.sign(V @ R); B[B == 0] = 1
                Uu, _, Vv = np.linalg.svd(B.T @ V)
                R = Vv.T @ Uu.T                                # orthogonal Procrustes
            self._params["R"] = R
        else:
            raise ValueError(f"unknown mode {self.mode}")
        return self

    def rotate(self, X) -> np.ndarray:
        """Return the CONTINUOUS rotated coordinates (before sign). Feed these to a
        multi-bit binarizer for the 'rotate then 2-4 bits' (QuaRot-style) recipe."""
        Xc = self._std(X, fit=False)
        if self.mode == "random":
            return Xc @ self._params["P"]
        if self.mode == "hadamard":
            m, D, H = self._params["m"], self._params["D"], self._params["H"]
            Xp = np.zeros((Xc.shape[0], m)); Xp[:, :Xc.shape[1]] = Xc
            return ((Xp * D) @ H)[:, : self.n_bits]
        return (Xc @ self._params["pca"]) @ self._params["R"]   # itq

    def transform(self, X) -> np.ndarray:
        return (self.rotate(X) > 0).astype(np.int8)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
