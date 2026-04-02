import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from utils.model_utils.symbol_utils import rrc_taps


@dataclass
class QPSKConfig:
    n_symbols: int
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    normalize_power: bool = True


@dataclass
class NoiseConfig:
    enabled: bool = False


@dataclass
class MixtureConfig:
    alpha: float = 1.0
    snr_db: Optional[float] = None

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
    def generate_mixture(
        self,
        qpsk_cfg_soi: QPSKConfig,
        qpsk_cfg_int: QPSKConfig,
        noise_cfg: NoiseConfig,
        mix_cfg: MixtureConfig,
    ) -> Dict[str, Any]:
        """
        Method: generate_example
        inputs: qpsk_cfg - config parameters for qpsk generation,
                noise_cf - config parameters for interference signal generation,
                mix_cfg - config parameters for the mixed signals,
        """
        s_soi, s_soi_symbols, soi_meta = self.generate_qpsk(qpsk_cfg_soi)
        s_int, s_int_symbols, int_meta = self.generate_qpsk(qpsk_cfg_int)


        # Mix signals s_soi + α * s_int + noise
        signal = s_soi + mix_cfg.alpha * s_int
        if noise_cfg.enabled:
            noise = self.generate_noise(signal, mix_cfg.snr_db)
            mixture = signal + noise
        else:
            noise = np.zeros_like(signal)
            mixture = signal + noise

        return {
            "mixture": mixture.astype(np.complex64),
            "source_a": s_soi.astype(np.complex64),
            "symbols_a": s_soi_symbols.astype(np.complex64),
            "source_b": s_int.astype(np.complex64),
            "symbols_b": s_int_symbols.astype(np.complex64),
            "noise": noise.astype(np.complex64),
            "meta": {
                "qpsk_soi": soi_meta,
                "qpsk_int": int_meta,
                "alpha": mix_cfg.alpha,
                "snr_db": mix_cfg.snr_db,
            },
        }

    # --------------------------
    # Source generation
    # --------------------------
    def generate_qpsk(self, cfg: QPSKConfig):
        total_symbols = cfg.n_symbols

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
        }
        return shaped, symbols, meta
    
    def generate_noise(self, signal: np.ndarray, snr_db: Optional[float]) -> np.ndarray:
        if snr_db is None:
            return np.zeros_like(signal)

        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        noise = (
            self.rng.standard_normal(signal.shape)
            + 1j * self.rng.standard_normal(signal.shape)
        ) / np.sqrt(2.0)

        noise *= np.sqrt(noise_power)

        return noise

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

    # Apply the pulse shape defined by rrc_taps
    def _pulse_shape_rrc(
        self,
        symbols: np.ndarray,
        sps: int,
        rolloff: float,
        span_symbols: int,
    ) -> np.ndarray:
        taps = rrc_taps(sps=sps, beta=rolloff, span_symbols=span_symbols)

        up = np.zeros(len(symbols) * sps, dtype=np.complex128)
        up[::sps] = symbols

        shaped = np.convolve(up, taps, mode="same")
        return shaped

    def _normalize_complex_power(self, x: np.ndarray) -> np.ndarray:
        p = np.mean(np.abs(x) ** 2) + 1e-12
        return x / np.sqrt(p)