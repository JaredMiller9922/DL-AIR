import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class QPSKConfig:
    n_symbols: int
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    include_preamble: bool = False
    preamble_symbols: int = 50
    normalize_power: bool = True


@dataclass
class InterfererConfig:
    mode: str = "bandlimited_noise"
    normalize_power: bool = True
    fir_len: int = 33


@dataclass
class MixtureConfig:
    n_rx: int = 4
    alpha: float = 1.0
    snr_db: Optional[float] = None
    random_phase: bool = True
    normalize_mixture: bool = False


class RFMixtureGenerator:
    """
    General RF mixture generator.

    Current supported source types:
      - QPSK
      - structured noise-like interferer placeholder

    Output:
      {
        "mixture": complex ndarray of shape (n_rx, n_samples),
        "source_a": complex ndarray of shape (n_samples,),
        "source_b": complex ndarray of shape (n_samples,),
        "H": complex ndarray of shape (n_rx, 2),
        "meta": dict
      }
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # --------------------------
    # Public API
    # --------------------------
    def generate_example(
        self,
        qpsk_cfg: QPSKConfig,
        int_cfg: InterfererConfig,
        mix_cfg: MixtureConfig,
    ) -> Dict[str, Any]:
        """
        Method: generate_example
        inputs: qpsk_cfg - config parameters for qpsk generation,
                int_cfg - config parameters for interference signal generation,
                mix_cfg - config parameters for the mixed signals,
        """
        s_soi, soi_meta = self.generate_qpsk(qpsk_cfg)
        s_int, int_meta = self.generate_interferer(len(s_soi), int_cfg)

        H = self._sample_mixing_matrix(
            n_rx=mix_cfg.n_rx,
            random_phase=mix_cfg.random_phase,
        )

        sources = np.vstack([s_soi, mix_cfg.alpha * s_int])  # shape (2, T)
        mixture = H @ sources  # shape (n_rx, T)

        if mix_cfg.snr_db is not None:
            mixture = self._add_receiver_noise(mixture, mix_cfg.snr_db)

        if mix_cfg.normalize_mixture:
            mixture = self._normalize_complex_power(mixture)

        return {
            "mixture": mixture.astype(np.complex64),
            "source_a": s_soi.astype(np.complex64),
            "source_b": s_int.astype(np.complex64),
            "H": H.astype(np.complex64),
            "meta": {
                "qpsk": soi_meta,
                "interferer": int_meta,
                "alpha": mix_cfg.alpha,
                "n_rx": mix_cfg.n_rx,
                "snr_db": mix_cfg.snr_db,
            },
        }

    # --------------------------
    # Source generation
    # --------------------------
    def generate_qpsk(self, cfg: QPSKConfig):
        total_symbols = cfg.n_symbols
        if cfg.include_preamble:
            total_symbols += cfg.preamble_symbols

        bits = self.rng.integers(0, 2, size=(2 * total_symbols,), endpoint=False)

        symbols = self._bits_to_qpsk(bits)

        shaped = self._pulse_shape_rrc(
            symbols,
            sps=cfg.samples_per_symbol,
            rolloff=cfg.rolloff,
            span_symbols=cfg.rrc_span_symbols,
        )

        if cfg.normalize_power:
            shaped = self._normalize_complex_power(shaped)

        meta = {
            "bits": bits,
            "symbols": symbols,
            "samples_per_symbol": cfg.samples_per_symbol,
            "rolloff": cfg.rolloff,
            "rrc_span_symbols": cfg.rrc_span_symbols,
            "include_preamble": cfg.include_preamble,
            "preamble_symbols": cfg.preamble_symbols,
        }
        return shaped, meta

    def generate_interferer(self, n_samples: int, cfg: InterfererConfig):
        if cfg.mode == "bandlimited_noise":
            x = (
                self.rng.standard_normal(n_samples)
                + 1j * self.rng.standard_normal(n_samples)
            ) / np.sqrt(2.0)

            # crude bandlimiting to make it more structured than white noise
            taps = (
                self.rng.standard_normal(cfg.fir_len)
                + 1j * self.rng.standard_normal(cfg.fir_len)
            ) / np.sqrt(2.0 * cfg.fir_len)
            x = np.convolve(x, taps, mode="same")

        elif cfg.mode == "bursty_bandlimited_noise":
            x = (
                self.rng.standard_normal(n_samples)
                + 1j * self.rng.standard_normal(n_samples)
            ) / np.sqrt(2.0)

            taps = (
                self.rng.standard_normal(cfg.fir_len)
                + 1j * self.rng.standard_normal(cfg.fir_len)
            ) / np.sqrt(2.0 * cfg.fir_len)
            x = np.convolve(x, taps, mode="same")

            envelope = np.zeros(n_samples, dtype=np.float64)
            idx = 0
            while idx < n_samples:
                on = self.rng.integers(20, 120)
                off = self.rng.integers(10, 80)
                envelope[idx:idx + on] = 1.0
                idx += on + off
            x *= envelope[:n_samples]

        else:
            raise ValueError(f"Unsupported interferer mode: {cfg.mode}")

        if cfg.normalize_power:
            x = self._normalize_complex_power(x)

        meta = {
            "mode": cfg.mode,
            "fir_len": cfg.fir_len,
        }
        return x, meta

    # --------------------------
    # Mixing / channel model
    # --------------------------
    def _sample_mixing_matrix(self, n_rx: int, random_phase: bool = True) -> np.ndarray:
        """
        Returns H of shape (n_rx, 2).
        Simple flat-fading narrowband complex mixing matrix.
        """
        H = (
            self.rng.standard_normal((n_rx, 2))
            + 1j * self.rng.standard_normal((n_rx, 2))
        ) / np.sqrt(2.0)

        if random_phase:
            phase = np.exp(1j * self.rng.uniform(0, 2 * np.pi, size=(n_rx, 2)))
            H = np.abs(H) * phase

        # Normalize each column so source power scaling is controlled mostly by alpha
        for k in range(H.shape[1]):
            col_norm = np.linalg.norm(H[:, k]) + 1e-12
            H[:, k] /= col_norm

        return H

    def _add_receiver_noise(self, mixture: np.ndarray, snr_db: float) -> np.ndarray:
        sig_power = np.mean(np.abs(mixture) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = (
            self.rng.standard_normal(mixture.shape)
            + 1j * self.rng.standard_normal(mixture.shape)
        ) / np.sqrt(2.0)
        noise *= np.sqrt(noise_power)
        return mixture + noise

    # --------------------------
    # DSP helpers
    # --------------------------
    def _bits_to_qpsk(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 2 != 0:
            raise ValueError("QPSK needs an even number of bits")

        bit_pairs = bits.reshape(-1, 2)

        # Gray-like mapping
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j,
        }

        syms = np.array([mapping[tuple(b)] for b in bit_pairs], dtype=np.complex128)
        syms /= np.sqrt(2.0)
        return syms

    def _pulse_shape_rrc(
        self,
        symbols: np.ndarray,
        sps: int,
        rolloff: float,
        span_symbols: int,
    ) -> np.ndarray:
        taps = self._rrc_taps(sps=sps, beta=rolloff, span_symbols=span_symbols)

        up = np.zeros(len(symbols) * sps, dtype=np.complex128)
        up[::sps] = symbols

        shaped = np.convolve(up, taps, mode="same")
        return shaped

    def _rrc_taps(self, sps: int, beta: float, span_symbols: int) -> np.ndarray:
        N = span_symbols * sps
        t = np.arange(-N, N + 1, dtype=np.float64) / sps

        taps = np.zeros_like(t)
        for i, ti in enumerate(t):
            if np.isclose(ti, 0.0):
                taps[i] = 1.0 - beta + (4 * beta / np.pi)
            elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
                taps[i] = (
                    beta
                    / np.sqrt(2)
                    * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                    )
                )
            else:
                num = (
                    np.sin(np.pi * ti * (1 - beta))
                    + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                )
                den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
                taps[i] = num / (den + 1e-12)

        taps /= np.sqrt(np.sum(taps ** 2) + 1e-12)
        return taps

    def _normalize_complex_power(self, x: np.ndarray) -> np.ndarray:
        p = np.mean(np.abs(x) ** 2) + 1e-12
        return x / np.sqrt(p)