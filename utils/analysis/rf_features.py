from __future__ import annotations

from typing import Dict

import numpy as np

from utils.model_utils.conversion_helpers import iq_channels_to_complex_matrix


def _safe_float(value) -> float:
    value = np.asarray(value)
    if value.size == 0:
        return 0.0
    out = float(np.real(value).reshape(-1)[0])
    if not np.isfinite(out):
        return 0.0
    return out


def _spectral_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=np.float64)
    p = p / (np.sum(p) + 1e-12)
    return _safe_float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12))


def _spectral_flatness(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=np.float64) + 1e-12
    return _safe_float(np.exp(np.mean(np.log(p))) / np.mean(p))


def complex_mixture_features(mixture: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Extract compact frame-level features from a complex RF mixture.

    Parameters
    ----------
    mixture:
        Complex array of shape ``(n_rx, T)`` or ``(T,)``.
    prefix:
        Optional prefix for feature names.
    """
    X = np.asarray(mixture)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError(f"Expected complex mixture shape (n_rx, T), got {X.shape}")

    X = X.astype(np.complex128, copy=False)
    n_rx, n_time = X.shape
    amp = np.abs(X)
    power = amp**2

    feats: Dict[str, float] = {
        f"{prefix}n_rx": float(n_rx),
        f"{prefix}n_time": float(n_time),
        f"{prefix}power_mean": _safe_float(power.mean()),
        f"{prefix}power_std": _safe_float(power.mean(axis=1).std()),
        f"{prefix}power_imbalance": _safe_float(power.mean(axis=1).max() / (power.mean(axis=1).min() + 1e-12)),
        f"{prefix}amp_mean_all": _safe_float(amp.mean()),
        f"{prefix}amp_std_all": _safe_float(amp.std()),
        f"{prefix}amp_max_all": _safe_float(amp.max()),
        f"{prefix}papr_mean": _safe_float(np.mean(power.max(axis=1) / (power.mean(axis=1) + 1e-12))),
        f"{prefix}papr_max": _safe_float(np.max(power.max(axis=1) / (power.mean(axis=1) + 1e-12))),
    }

    for rr in range(n_rx):
        feats[f"{prefix}ch{rr + 1}_power_mean"] = _safe_float(power[rr].mean())
        feats[f"{prefix}ch{rr + 1}_amp_mean"] = _safe_float(amp[rr].mean())
        feats[f"{prefix}ch{rr + 1}_amp_std"] = _safe_float(amp[rr].std())
        feats[f"{prefix}ch{rr + 1}_papr"] = _safe_float(power[rr].max() / (power[rr].mean() + 1e-12))

    centered = X - X.mean(axis=1, keepdims=True)
    cov = (centered @ centered.conj().T) / max(1, n_time - 1)
    eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
    for ii, eig in enumerate(eigvals, start=1):
        feats[f"{prefix}cov_eig_{ii}"] = _safe_float(eig)
    feats[f"{prefix}cov_condition"] = _safe_float((eigvals[0] + 1e-12) / (eigvals[-1] + 1e-12))

    if n_rx > 1:
        offdiag = cov[~np.eye(n_rx, dtype=bool)]
        offdiag_abs = np.abs(offdiag)
        diag = np.sqrt(np.outer(np.real(np.diag(cov)), np.real(np.diag(cov)))) + 1e-12
        coherence = np.abs(cov / diag)
        coherence_offdiag = coherence[~np.eye(n_rx, dtype=bool)]
        feats[f"{prefix}offdiag_cov_abs_mean"] = _safe_float(offdiag_abs.mean())
        feats[f"{prefix}offdiag_cov_abs_std"] = _safe_float(offdiag_abs.std())
        feats[f"{prefix}offdiag_cov_abs_max"] = _safe_float(offdiag_abs.max())
        feats[f"{prefix}coherence_mean"] = _safe_float(coherence_offdiag.mean())
        feats[f"{prefix}coherence_std"] = _safe_float(coherence_offdiag.std())
        feats[f"{prefix}coherence_max"] = _safe_float(coherence_offdiag.max())
    else:
        feats[f"{prefix}offdiag_cov_abs_mean"] = 0.0
        feats[f"{prefix}offdiag_cov_abs_std"] = 0.0
        feats[f"{prefix}offdiag_cov_abs_max"] = 0.0
        feats[f"{prefix}coherence_mean"] = 0.0
        feats[f"{prefix}coherence_std"] = 0.0
        feats[f"{prefix}coherence_max"] = 0.0

    spec = np.fft.fftshift(np.fft.fft(X, axis=1), axes=1)
    spec_power = np.abs(spec) ** 2
    freqs = np.linspace(-0.5, 0.5, n_time, endpoint=False)
    entropy_vals = []
    flatness_vals = []
    centroid_vals = []
    bandwidth_vals = []
    for rr in range(n_rx):
        p = spec_power[rr]
        p_sum = p.sum() + 1e-12
        centroid = np.sum(freqs * p) / p_sum
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * p) / p_sum)
        entropy_vals.append(_spectral_entropy(p))
        flatness_vals.append(_spectral_flatness(p))
        centroid_vals.append(centroid)
        bandwidth_vals.append(bandwidth)

    feats[f"{prefix}spec_entropy_mean"] = _safe_float(np.mean(entropy_vals))
    feats[f"{prefix}spec_entropy_std"] = _safe_float(np.std(entropy_vals))
    feats[f"{prefix}spec_flatness_mean"] = _safe_float(np.mean(flatness_vals))
    feats[f"{prefix}spec_flatness_std"] = _safe_float(np.std(flatness_vals))
    feats[f"{prefix}spec_centroid_mean"] = _safe_float(np.mean(centroid_vals))
    feats[f"{prefix}spec_centroid_std"] = _safe_float(np.std(centroid_vals))
    feats[f"{prefix}spec_bandwidth_mean"] = _safe_float(np.mean(bandwidth_vals))
    feats[f"{prefix}spec_bandwidth_std"] = _safe_float(np.std(bandwidth_vals))

    return feats


def stacked_iq_features(x_iq: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Feature wrapper for canonical stacked-IQ arrays of shape ``(2*n_rx, T)``."""
    return complex_mixture_features(iq_channels_to_complex_matrix(np.asarray(x_iq)), prefix=prefix)
