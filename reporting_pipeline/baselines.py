import numpy as np
import torch
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings


def _dominant_bin(signal: np.ndarray) -> int:
    spec = np.abs(np.fft.rfft(signal))
    return int(np.argmax(spec))


def _normalize_complex(z: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(z) ** 2) + 1e-12
    return z / np.sqrt(power)


def _qpsk_phase_correct(z: np.ndarray) -> np.ndarray:
    if len(z) == 0:
        return z
    phase_est = np.angle(np.mean(z ** 4) + 1e-12) / 4.0
    return z * np.exp(-1j * phase_est)


class FastICABaseline:
    """Simple per-frame FastICA baseline with QPSK phase correction."""

    def __init__(self, low_band_hint=14, high_band_hint=36, random_state=0):
        self.low_band_hint = low_band_hint
        self.high_band_hint = high_band_hint
        self.random_state = random_state

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_outputs = []
        x_np = x.detach().cpu().numpy()
        for sample in x_np:
            batch_outputs.append(self._separate_sample(sample))
        return torch.from_numpy(np.stack(batch_outputs, axis=0)).to(x.device)

    def _separate_sample(self, sample_iq: np.ndarray) -> np.ndarray:
        X = sample_iq.T
        n_features = X.shape[1]
        n_components = min(4, n_features)
        ica = FastICA(
            n_components=n_components,
            algorithm="parallel",
            whiten="unit-variance",
            fun="logcosh",
            random_state=self.random_state,
            max_iter=1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            S = ica.fit_transform(X).T.astype(np.float32)

        if S.shape[0] == 2:
            z_low = _normalize_complex(S[0]).astype(np.complex64)
            z_high = _normalize_complex(S[1]).astype(np.complex64)
            out = np.stack([z_low.real, z_low.imag, z_high.real, z_high.imag], axis=0).astype(np.float32)
            return out

        dom_bins = [_dominant_bin(comp) for comp in S]
        low_indices = sorted(range(4), key=lambda i: abs(dom_bins[i] - self.low_band_hint))[:2]
        remaining = [i for i in range(4) if i not in low_indices]
        high_indices = sorted(remaining, key=lambda i: abs(dom_bins[i] - self.high_band_hint))[:2]

        z_low = _qpsk_phase_correct(_normalize_complex(S[low_indices[0]] + 1j * S[low_indices[1]])).astype(np.complex64)
        z_high = _qpsk_phase_correct(_normalize_complex(S[high_indices[0]] + 1j * S[high_indices[1]])).astype(np.complex64)

        out = np.stack([z_low.real, z_low.imag, z_high.real, z_high.imag], axis=0).astype(np.float32)
        return out
