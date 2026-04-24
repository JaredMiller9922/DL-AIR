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


def _spectral_rolloff(freqs: np.ndarray, power: np.ndarray, pct: float = 0.85) -> float:
    power = np.asarray(power, dtype=np.float64)
    total = np.sum(power) + 1e-12
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, pct * total, side="left"))
    idx = int(np.clip(idx, 0, len(freqs) - 1))
    return _safe_float(np.abs(freqs[idx]))


def _moment_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean()
    var = np.mean(centered**2) + 1e-12
    return _safe_float(np.mean(centered**4) / (var**2))


def _moment_skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean()
    std = np.sqrt(np.mean(centered**2) + 1e-12)
    return _safe_float(np.mean((centered / std) ** 3))


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
        f"{prefix}n_samples_per_channel": float(n_time),
        f"{prefix}power_mean": _safe_float(power.mean()),
        f"{prefix}power_std": _safe_float(power.mean(axis=1).std()),
        f"{prefix}power_imbalance": _safe_float(power.mean(axis=1).max() / (power.mean(axis=1).min() + 1e-12)),
        f"{prefix}power_range": _safe_float(power.mean(axis=1).max() - power.mean(axis=1).min()),
        f"{prefix}amp_mean_all": _safe_float(amp.mean()),
        f"{prefix}amp_std_all": _safe_float(amp.std()),
        f"{prefix}amp_max_all": _safe_float(amp.max()),
        f"{prefix}amp_cv_all": _safe_float(amp.std() / (amp.mean() + 1e-12)),
        f"{prefix}papr_mean": _safe_float(np.mean(power.max(axis=1) / (power.mean(axis=1) + 1e-12))),
        f"{prefix}papr_max": _safe_float(np.max(power.max(axis=1) / (power.mean(axis=1) + 1e-12))),
        f"{prefix}radius_kurtosis_mean": _safe_float(np.mean([_moment_kurtosis(amp[rr]) for rr in range(n_rx)])),
        f"{prefix}radius_skew_mean": _safe_float(np.mean([_moment_skewness(amp[rr]) for rr in range(n_rx)])),
    }

    for rr in range(n_rx):
        phase = np.angle(X[rr])
        phase_resultant = np.abs(np.mean(np.exp(1j * phase)))
        feats[f"{prefix}ch{rr + 1}_power_mean"] = _safe_float(power[rr].mean())
        feats[f"{prefix}ch{rr + 1}_power"] = _safe_float(power[rr].mean())
        feats[f"{prefix}ch{rr + 1}_amp_mean"] = _safe_float(amp[rr].mean())
        feats[f"{prefix}ch{rr + 1}_amp_std"] = _safe_float(amp[rr].std())
        feats[f"{prefix}ch{rr + 1}_amp_var"] = _safe_float(amp[rr].var())
        feats[f"{prefix}ch{rr + 1}_amp_max"] = _safe_float(amp[rr].max())
        feats[f"{prefix}ch{rr + 1}_papr"] = _safe_float(power[rr].max() / (power[rr].mean() + 1e-12))
        feats[f"{prefix}ch{rr + 1}_real_mean"] = _safe_float(np.real(X[rr]).mean())
        feats[f"{prefix}ch{rr + 1}_imag_mean"] = _safe_float(np.imag(X[rr]).mean())
        feats[f"{prefix}ch{rr + 1}_real_std"] = _safe_float(np.real(X[rr]).std())
        feats[f"{prefix}ch{rr + 1}_imag_std"] = _safe_float(np.imag(X[rr]).std())
        feats[f"{prefix}ch{rr + 1}_phase_mean"] = _safe_float(np.angle(np.mean(np.exp(1j * phase))))
        feats[f"{prefix}ch{rr + 1}_phase_std"] = _safe_float(np.std(phase))
        feats[f"{prefix}ch{rr + 1}_phase_resultant"] = _safe_float(phase_resultant)
        feats[f"{prefix}ch{rr + 1}_iq_kurtosis"] = _safe_float(0.5 * (_moment_kurtosis(np.real(X[rr])) + _moment_kurtosis(np.imag(X[rr]))))
        feats[f"{prefix}ch{rr + 1}_iq_skewness"] = _safe_float(0.5 * (_moment_skewness(np.real(X[rr])) + _moment_skewness(np.imag(X[rr]))))

    centered = X - X.mean(axis=1, keepdims=True)
    cov = (centered @ centered.conj().T) / max(1, n_time - 1)
    eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
    for ii, eig in enumerate(eigvals, start=1):
        feats[f"{prefix}cov_eig_{ii}"] = _safe_float(eig)
    feats[f"{prefix}cov_trace"] = _safe_float(np.real(np.trace(cov)))
    feats[f"{prefix}cov_condition"] = _safe_float((eigvals[0] + 1e-12) / (eigvals[-1] + 1e-12))
    feats[f"{prefix}cov_spectral_entropy"] = _safe_float(_spectral_entropy(np.maximum(eigvals, 0.0)))
    feats[f"{prefix}cov_participation_ratio"] = _safe_float((np.sum(eigvals) ** 2) / (np.sum(eigvals**2) + 1e-12))
    feats[f"{prefix}cov_eig_ratio_12"] = _safe_float((eigvals[0] + 1e-12) / (eigvals[1] + 1e-12)) if len(eigvals) > 1 else 0.0
    feats[f"{prefix}cov_eig_ratio_13"] = _safe_float((eigvals[0] + 1e-12) / (eigvals[2] + 1e-12)) if len(eigvals) > 2 else 0.0
    feats[f"{prefix}cov_eig_ratio_14"] = _safe_float((eigvals[0] + 1e-12) / (eigvals[3] + 1e-12)) if len(eigvals) > 3 else 0.0

    if n_rx > 1:
        offdiag = cov[~np.eye(n_rx, dtype=bool)]
        offdiag_abs = np.abs(offdiag)
        offdiag_phase = np.angle(offdiag)
        diag = np.sqrt(np.outer(np.real(np.diag(cov)), np.real(np.diag(cov)))) + 1e-12
        coherence = np.abs(cov / diag)
        coherence_offdiag = coherence[~np.eye(n_rx, dtype=bool)]
        feats[f"{prefix}offdiag_cov_abs_mean"] = _safe_float(offdiag_abs.mean())
        feats[f"{prefix}offdiag_cov_abs_std"] = _safe_float(offdiag_abs.std())
        feats[f"{prefix}offdiag_cov_abs_max"] = _safe_float(offdiag_abs.max())
        feats[f"{prefix}offdiag_cov_phase_std"] = _safe_float(offdiag_phase.std())
        feats[f"{prefix}offdiag_cov_phase_resultant"] = _safe_float(np.abs(np.mean(np.exp(1j * offdiag_phase))))
        feats[f"{prefix}coherence_mean"] = _safe_float(coherence_offdiag.mean())
        feats[f"{prefix}coherence_std"] = _safe_float(coherence_offdiag.std())
        feats[f"{prefix}coherence_max"] = _safe_float(coherence_offdiag.max())
        feats[f"{prefix}coherence_median"] = _safe_float(np.median(coherence_offdiag))

        phase_diff_means = []
        phase_diff_stds = []
        phase_diff_resultants = []
        for ii in range(n_rx):
            for jj in range(ii + 1, n_rx):
                delta = np.angle(X[ii] * np.conj(X[jj]))
                phase_diff_means.append(np.angle(np.mean(np.exp(1j * delta))))
                phase_diff_stds.append(np.std(delta))
                phase_diff_resultants.append(np.abs(np.mean(np.exp(1j * delta))))
        feats[f"{prefix}phase_diff_mean_abs"] = _safe_float(np.mean(np.abs(phase_diff_means)))
        feats[f"{prefix}phase_diff_std_mean"] = _safe_float(np.mean(phase_diff_stds))
        feats[f"{prefix}phase_diff_resultant_mean"] = _safe_float(np.mean(phase_diff_resultants))
    else:
        feats[f"{prefix}offdiag_cov_abs_mean"] = 0.0
        feats[f"{prefix}offdiag_cov_abs_std"] = 0.0
        feats[f"{prefix}offdiag_cov_abs_max"] = 0.0
        feats[f"{prefix}offdiag_cov_phase_std"] = 0.0
        feats[f"{prefix}offdiag_cov_phase_resultant"] = 0.0
        feats[f"{prefix}coherence_mean"] = 0.0
        feats[f"{prefix}coherence_std"] = 0.0
        feats[f"{prefix}coherence_max"] = 0.0
        feats[f"{prefix}coherence_median"] = 0.0
        feats[f"{prefix}phase_diff_mean_abs"] = 0.0
        feats[f"{prefix}phase_diff_std_mean"] = 0.0
        feats[f"{prefix}phase_diff_resultant_mean"] = 0.0

    spec = np.fft.fftshift(np.fft.fft(X, axis=1), axes=1)
    spec_power = np.abs(spec) ** 2
    freqs = np.linspace(-0.5, 0.5, n_time, endpoint=False)
    entropy_vals = []
    flatness_vals = []
    centroid_vals = []
    bandwidth_vals = []
    rolloff_vals = []
    for rr in range(n_rx):
        p = spec_power[rr]
        p_sum = p.sum() + 1e-12
        centroid = np.sum(freqs * p) / p_sum
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * p) / p_sum)
        entropy = _spectral_entropy(p)
        flatness = _spectral_flatness(p)
        rolloff = _spectral_rolloff(freqs, p, pct=0.85)
        entropy_vals.append(entropy)
        flatness_vals.append(flatness)
        centroid_vals.append(centroid)
        bandwidth_vals.append(bandwidth)
        rolloff_vals.append(rolloff)
        feats[f"{prefix}ch{rr + 1}_spec_entropy"] = _safe_float(entropy)
        feats[f"{prefix}ch{rr + 1}_spec_flatness"] = _safe_float(flatness)
        feats[f"{prefix}ch{rr + 1}_spec_centroid"] = _safe_float(centroid)
        feats[f"{prefix}ch{rr + 1}_spec_bandwidth"] = _safe_float(bandwidth)
        feats[f"{prefix}ch{rr + 1}_spec_rolloff_85"] = _safe_float(rolloff)

    feats[f"{prefix}spec_entropy_mean"] = _safe_float(np.mean(entropy_vals))
    feats[f"{prefix}spec_entropy_std"] = _safe_float(np.std(entropy_vals))
    feats[f"{prefix}spec_flatness_mean"] = _safe_float(np.mean(flatness_vals))
    feats[f"{prefix}spec_flatness_std"] = _safe_float(np.std(flatness_vals))
    feats[f"{prefix}spec_centroid_mean"] = _safe_float(np.mean(centroid_vals))
    feats[f"{prefix}spec_centroid_std"] = _safe_float(np.std(centroid_vals))
    feats[f"{prefix}spec_bandwidth_mean"] = _safe_float(np.mean(bandwidth_vals))
    feats[f"{prefix}spec_bandwidth_std"] = _safe_float(np.std(bandwidth_vals))
    feats[f"{prefix}spec_rolloff_85_mean"] = _safe_float(np.mean(rolloff_vals))
    feats[f"{prefix}spec_rolloff_85_std"] = _safe_float(np.std(rolloff_vals))

    return feats


def stacked_iq_features(x_iq: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Feature wrapper for canonical stacked-IQ arrays of shape ``(2*n_rx, T)``."""
    return complex_mixture_features(iq_channels_to_complex_matrix(np.asarray(x_iq)), prefix=prefix)
