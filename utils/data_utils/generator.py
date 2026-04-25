from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from config import ExperimentConfig
from utils.model_utils.symbol_utils import rrc_taps


@dataclass
class QPSKConfig:
    n_symbols: int = ExperimentConfig.num_symbols
    samples_per_symbol: int = ExperimentConfig.samples_per_symbol
    rolloff: float = ExperimentConfig.rolloff
    rrc_span_symbols: int = ExperimentConfig.rrc_span_symbols
    normalize_power: bool = ExperimentConfig.normalize_power
    num_channels: int = ExperimentConfig.n_rx


@dataclass
class NoiseConfig:
    enabled: bool = ExperimentConfig.noise_enabled
    snr_db: Optional[float] = ExperimentConfig.snr_db
    sigma2: Optional[float] = ExperimentConfig.sigma2


@dataclass
class MixtureConfig:
    alpha: float = ExperimentConfig.noise_alpha
    snr_db: Optional[float] = ExperimentConfig.snr_db
    n_rx: int = ExperimentConfig.n_rx
    random_phase: bool = ExperimentConfig.random_phase
    phase_shift_deg: int = ExperimentConfig.phase_shift_deg
    interference_phase_shift: int = ExperimentConfig.interference_phase_shift
    mixing_mode: str = "phase_only"
    random_complex_mixing: bool = False
    timing_offset: int = 0
    carrier_offset: float = 0.0
    phase_mismatch_deg: float = 0.0
    amplitude_imbalance_db: float = 0.0


@dataclass
class SourceConfig:
    source_type: str = "QPSK"
    n_symbols: int = ExperimentConfig.num_symbols
    samples_per_symbol: int = ExperimentConfig.samples_per_symbol
    rolloff: float = ExperimentConfig.rolloff
    rrc_span_symbols: int = ExperimentConfig.rrc_span_symbols
    normalize_power: bool = ExperimentConfig.normalize_power
    num_channels: int = ExperimentConfig.n_rx
    num_tones: int = 8
    tone_bandwidth: float = 0.35
    colored_alpha: float = 0.92
    chirp_start: float = -0.25
    chirp_stop: float = 0.25
    burst_probability: float = 0.35
    burst_duty_cycle: float = 0.35
    recorded_path: Optional[str] = None
    amplitude: float = 1.0


@dataclass
class SourceFamilyConfig:
    source_a_types: Tuple[str, ...] = ("QPSK",)
    source_b_types: Tuple[str, ...] = ("QPSK", "BPSK", "8PSK", "16QAM", "MULTITONE", "COLORED_NOISE")
    n_symbols_range: Tuple[int, int] = (ExperimentConfig.num_symbols, ExperimentConfig.num_symbols)
    samples_per_symbol_choices: Tuple[int, ...] = (ExperimentConfig.samples_per_symbol,)
    rolloff_range: Tuple[float, float] = (ExperimentConfig.rolloff, ExperimentConfig.rolloff)
    rrc_span_symbols: int = ExperimentConfig.rrc_span_symbols
    alpha_range: Tuple[float, float] = (ExperimentConfig.noise_alpha, ExperimentConfig.noise_alpha)
    snr_db_range: Tuple[float, float] = (ExperimentConfig.snr_db, ExperimentConfig.snr_db)
    timing_offset_range: Tuple[int, int] = (0, 0)
    carrier_offset_range: Tuple[float, float] = (0.0, 0.0)
    phase_mismatch_range_deg: Tuple[float, float] = (0.0, 0.0)
    amplitude_imbalance_db_range: Tuple[float, float] = (0.0, 0.0)
    mixing_modes: Tuple[str, ...] = ("phase_only",)
    n_rx: int = ExperimentConfig.n_rx
    normalize_power: bool = ExperimentConfig.normalize_power
    mit_aligned: bool = False


SUPPORTED_SOURCE_TYPES = {
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "ASK",
    "PAM",
    "OFDM",
    "MULTITONE",
    "COLORED_NOISE",
    "CHIRP",
    "BURSTY",
    "RECORDED",
}


def canonical_source_type(source_type: str) -> str:
    key = str(source_type).replace("-", "").replace("_", "").replace(" ", "").upper()
    aliases = {
        "8PSK": "8PSK",
        "PSK8": "8PSK",
        "16QAM": "16QAM",
        "QAM16": "16QAM",
        "MULTITONE": "MULTITONE",
        "MULTI": "MULTITONE",
        "TONE": "MULTITONE",
        "COLOREDNOISE": "COLORED_NOISE",
        "COLOURNOISE": "COLORED_NOISE",
        "NOISE": "COLORED_NOISE",
        "SWEEP": "CHIRP",
        "CHIRP": "CHIRP",
        "BURST": "BURSTY",
        "BURSTY": "BURSTY",
        "RECORDED": "RECORDED",
        "RECORDEDINTERFERER": "RECORDED",
    }
    return aliases.get(key, key)


def source_config_from_qpsk(cfg: QPSKConfig, source_type: str = "QPSK") -> SourceConfig:
    return SourceConfig(
        source_type=source_type,
        n_symbols=cfg.n_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.num_channels,
    )


def mit_aligned_family_config() -> SourceFamilyConfig:
    return SourceFamilyConfig(
        source_a_types=("QPSK",),
        source_b_types=("QPSK", "BPSK", "8PSK", "MULTITONE", "COLORED_NOISE", "BURSTY"),
        n_symbols_range=(4, 4),
        samples_per_symbol_choices=(16,),
        rolloff_range=(0.25, 0.35),
        rrc_span_symbols=8,
        alpha_range=(0.2, 2.5),
        snr_db_range=(15.0, 35.0),
        timing_offset_range=(-2, 2),
        carrier_offset_range=(-0.01, 0.01),
        phase_mismatch_range_deg=(-10.0, 10.0),
        amplitude_imbalance_db_range=(-2.0, 2.0),
        mixing_modes=("phase_only",),
        n_rx=4,
        mit_aligned=True,
    )


class RFMixtureGenerator:
    """
    General RF mixture generator.

    The original QPSK-only entry point is preserved. New source configs add
    digital modulations, structured interferers, broader channel controls, and
    family sampling for benchmark sweeps.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_mixture(
        self,
        qpsk_cfg_soi: QPSKConfig,
        qpsk_cfg_int: QPSKConfig,
        noise_cfg: NoiseConfig,
        mix_cfg: MixtureConfig,
        soi_message: Optional[str] = None,
        source_a_cfg: Optional[SourceConfig] = None,
        source_b_cfg: Optional[SourceConfig] = None,
        family_cfg: Optional[SourceFamilyConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate a two-source RF mixture.

        Backward-compatible use:
            generate_mixture(qpsk_cfg_soi, qpsk_cfg_int, noise_cfg, mix_cfg)

        Extended use:
            pass SourceConfig objects or SourceFamilyConfig for broader
            modulation/interferer/channel sampling.
        """
        if family_cfg is not None:
            return self.generate_family_mixture(noise_cfg, mix_cfg, family_cfg, soi_message=soi_message)

        source_a_cfg = source_a_cfg or source_config_from_qpsk(qpsk_cfg_soi, "QPSK")
        source_b_cfg = source_b_cfg or source_config_from_qpsk(qpsk_cfg_int, "QPSK")

        s_soi, s_soi_symbols, soi_meta = self.generate_source(source_a_cfg, message=soi_message)
        s_int, s_int_symbols, int_meta = self.generate_source(source_b_cfg)

        s_int = self._apply_timing_offset(s_int, mix_cfg.timing_offset)
        s_int = self._apply_carrier_phase_mismatch(
            s_int,
            carrier_offset=mix_cfg.carrier_offset,
            phase_mismatch_deg=mix_cfg.phase_mismatch_deg,
        )

        if len(s_soi) != len(s_int):
            target_len = max(len(s_soi), len(s_int))
            s_soi = self._resample_complex_to_len(s_soi, target_len)
            s_int = self._resample_complex_to_len(s_int, target_len)

        theta = np.deg2rad(mix_cfg.interference_phase_shift)
        phase_shift = np.exp(1j * theta)

        if mix_cfg.n_rx == 1:
            signal = s_soi + phase_shift * mix_cfg.alpha * s_int
            H = np.array([[1.0, phase_shift * mix_cfg.alpha]], dtype=np.complex128)
        else:
            H = self._select_mixing_matrix(mix_cfg)
            sources = np.vstack([s_soi, phase_shift * mix_cfg.alpha * s_int])
            signal = H @ sources
            signal = self._apply_rx_amplitude_imbalance(signal, mix_cfg.amplitude_imbalance_db)

        if noise_cfg.enabled:
            noise = self.generate_noise(signal, noise_cfg)
            mixture = signal + noise
        else:
            noise = np.zeros_like(signal)
            mixture = signal

        return {
            "mixture": np.asarray(mixture, dtype=np.complex64),
            "source_a": np.asarray(s_soi, dtype=np.complex64),
            "symbols_a": np.asarray(s_soi_symbols, dtype=np.complex64),
            "source_b": np.asarray(s_int, dtype=np.complex64),
            "symbols_b": np.asarray(s_int_symbols, dtype=np.complex64),
            "noise": np.asarray(noise, dtype=np.complex64),
            "H": np.asarray(H, dtype=np.complex64),
            "meta": {
                "qpsk_soi": soi_meta,
                "qpsk_int": int_meta,
                "source_a": soi_meta,
                "source_b": int_meta,
                "alpha": mix_cfg.alpha,
                "snr_db": noise_cfg.snr_db,
                "mixing_mode": mix_cfg.mixing_mode,
                "timing_offset": mix_cfg.timing_offset,
                "carrier_offset": mix_cfg.carrier_offset,
                "phase_mismatch_deg": mix_cfg.phase_mismatch_deg,
                "amplitude_imbalance_db": mix_cfg.amplitude_imbalance_db,
            },
        }

    def generate_family_mixture(
        self,
        noise_cfg: NoiseConfig,
        mix_cfg: MixtureConfig,
        family_cfg: SourceFamilyConfig,
        soi_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_a_cfg = self._sample_source_config(family_cfg, family_cfg.source_a_types)
        source_b_cfg = self._sample_source_config(family_cfg, family_cfg.source_b_types)

        sampled_noise_cfg = replace(
            noise_cfg,
            snr_db=self._sample_uniform(*family_cfg.snr_db_range),
        )
        sampled_mix_cfg = replace(
            mix_cfg,
            alpha=self._sample_uniform(*family_cfg.alpha_range),
            snr_db=sampled_noise_cfg.snr_db,
            n_rx=family_cfg.n_rx,
            mixing_mode=str(self.rng.choice(family_cfg.mixing_modes)),
            timing_offset=int(self.rng.integers(family_cfg.timing_offset_range[0], family_cfg.timing_offset_range[1] + 1)),
            carrier_offset=self._sample_uniform(*family_cfg.carrier_offset_range),
            phase_mismatch_deg=self._sample_uniform(*family_cfg.phase_mismatch_range_deg),
            amplitude_imbalance_db=self._sample_uniform(*family_cfg.amplitude_imbalance_db_range),
        )

        return self.generate_mixture(
            qpsk_cfg_soi=QPSKConfig(),
            qpsk_cfg_int=QPSKConfig(),
            noise_cfg=sampled_noise_cfg,
            mix_cfg=sampled_mix_cfg,
            soi_message=soi_message,
            source_a_cfg=source_a_cfg,
            source_b_cfg=source_b_cfg,
            family_cfg=None,
        )

    def generate_source(self, cfg: SourceConfig, message: Optional[str] = None):
        source_type = canonical_source_type(cfg.source_type)
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type={cfg.source_type}. Supported: {sorted(SUPPORTED_SOURCE_TYPES)}")

        if source_type == "QPSK":
            shaped, symbols, meta = self.generate_qpsk(
                QPSKConfig(
                    n_symbols=cfg.n_symbols,
                    samples_per_symbol=cfg.samples_per_symbol,
                    rolloff=cfg.rolloff,
                    rrc_span_symbols=cfg.rrc_span_symbols,
                    normalize_power=cfg.normalize_power,
                    num_channels=cfg.num_channels,
                ),
                message=message,
            )
        elif source_type in {"BPSK", "8PSK", "16QAM", "ASK", "PAM"}:
            symbols = self._random_constellation_symbols(source_type, cfg.n_symbols)
            shaped = self._pulse_shape_rrc(symbols, cfg.samples_per_symbol, cfg.rolloff, cfg.rrc_span_symbols)
            meta = self._source_meta(cfg, source_type, symbols=symbols)
        elif source_type in {"OFDM", "MULTITONE"}:
            shaped, symbols, meta = self._generate_multitone(cfg, source_type)
        elif source_type == "COLORED_NOISE":
            shaped, symbols, meta = self._generate_colored_noise(cfg)
        elif source_type == "CHIRP":
            shaped, symbols, meta = self._generate_chirp(cfg)
        elif source_type == "BURSTY":
            shaped, symbols, meta = self._generate_bursty(cfg)
        elif source_type == "RECORDED":
            shaped, symbols, meta = self._generate_recorded_placeholder(cfg)
        else:
            raise AssertionError(f"Unhandled source_type={source_type}")

        shaped = cfg.amplitude * np.asarray(shaped, dtype=np.complex128)
        if cfg.normalize_power:
            shaped = self._normalize_complex_power(shaped)

        meta.update(
            {
                "source_type": source_type,
                "n_symbols": cfg.n_symbols,
                "samples_per_symbol": cfg.samples_per_symbol,
                "rolloff": cfg.rolloff,
                "rrc_span_symbols": cfg.rrc_span_symbols,
                "normalize_power": cfg.normalize_power,
            }
        )
        return shaped, np.asarray(symbols, dtype=np.complex128), meta

    def generate_qpsk(self, cfg: QPSKConfig, message: str = None):
        if message is None:
            total_symbols = cfg.n_symbols
            bits = self.rng.integers(0, 2, size=(2 * total_symbols,), endpoint=False)
        else:
            cleaned = message.strip()
            if cleaned and set(cleaned).issubset({"0", "1"}):
                bits = np.array([int(bit) for bit in cleaned], dtype=int)
            else:
                bits = self.symbols_to_bits(message)
            if len(bits) % 2 != 0:
                bits = np.append(bits, 0)
            total_symbols = len(bits) // 2

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
            "source_type": "QPSK",
        }
        return shaped, symbols, meta

    def generate_noise(self, signal: np.ndarray, noise_cfg: NoiseConfig) -> np.ndarray:
        if not noise_cfg.enabled:
            return np.zeros_like(signal)

        noise = (
            self.rng.standard_normal(signal.shape)
            + 1j * self.rng.standard_normal(signal.shape)
        ) / np.sqrt(2.0)

        if noise_cfg.sigma2 is not None:
            noise *= np.sqrt(noise_cfg.sigma2)
        else:
            signal_power = np.mean(np.abs(signal) ** 2)
            snr_db = 25.0 if noise_cfg.snr_db is None else noise_cfg.snr_db
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise *= np.sqrt(noise_power)

        return noise

    def _random_constellation_symbols(self, source_type: str, n_symbols: int) -> np.ndarray:
        if source_type == "BPSK":
            idx = self.rng.integers(0, 2, size=n_symbols)
            return (2 * idx - 1).astype(np.float64).astype(np.complex128)
        if source_type == "8PSK":
            idx = self.rng.integers(0, 8, size=n_symbols)
            return np.exp(1j * 2.0 * np.pi * idx / 8.0)
        if source_type == "16QAM":
            idx = self.rng.integers(0, 16, size=n_symbols)
            levels = np.array([-3, -1, 1, 3], dtype=np.float64)
            syms = levels[idx % 4] + 1j * levels[idx // 4]
            return syms / np.sqrt(np.mean(np.abs(syms) ** 2) + 1e-12)
        if source_type in {"ASK", "PAM"}:
            idx = self.rng.integers(0, 4, size=n_symbols)
            levels = np.array([-3, -1, 1, 3], dtype=np.float64)
            syms = levels[idx].astype(np.complex128)
            return syms / np.sqrt(np.mean(np.abs(syms) ** 2) + 1e-12)
        raise ValueError(f"No constellation generator for {source_type}")

    def _generate_multitone(self, cfg: SourceConfig, source_type: str):
        n = cfg.n_symbols * cfg.samples_per_symbol
        t = np.arange(n, dtype=np.float64) / max(n, 1)
        n_tones = max(1, int(cfg.num_tones))
        max_freq = max(0.02, min(0.49, cfg.tone_bandwidth))
        freqs = self.rng.uniform(-max_freq, max_freq, size=n_tones)
        phases = self.rng.uniform(0.0, 2.0 * np.pi, size=n_tones)
        weights = self.rng.normal(size=n_tones) + 1j * self.rng.normal(size=n_tones)
        tones = np.zeros(n, dtype=np.complex128)
        for freq, phase, weight in zip(freqs, phases, weights):
            tones += weight * np.exp(1j * (2.0 * np.pi * freq * np.arange(n) + phase))
        tones /= np.sqrt(n_tones)
        symbols = tones[:: cfg.samples_per_symbol][: cfg.n_symbols]
        return tones, symbols, self._source_meta(cfg, source_type, freqs=freqs)

    def _generate_colored_noise(self, cfg: SourceConfig):
        n = cfg.n_symbols * cfg.samples_per_symbol
        white = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2.0)
        y = np.zeros(n, dtype=np.complex128)
        a = float(np.clip(cfg.colored_alpha, -0.98, 0.98))
        for ii in range(1, n):
            y[ii] = a * y[ii - 1] + white[ii]
        symbols = y[:: cfg.samples_per_symbol][: cfg.n_symbols]
        return y, symbols, self._source_meta(cfg, "COLORED_NOISE", colored_alpha=a)

    def _generate_chirp(self, cfg: SourceConfig):
        n = cfg.n_symbols * cfg.samples_per_symbol
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        f0 = cfg.chirp_start
        f1 = cfg.chirp_stop
        phase = 2.0 * np.pi * (f0 * np.arange(n) + 0.5 * (f1 - f0) * n * t**2)
        chirp = np.exp(1j * phase)
        symbols = chirp[:: cfg.samples_per_symbol][: cfg.n_symbols]
        return chirp, symbols, self._source_meta(cfg, "CHIRP", chirp_start=f0, chirp_stop=f1)

    def _generate_bursty(self, cfg: SourceConfig):
        base_cfg = replace(cfg, source_type="QPSK")
        base, symbols, meta = self.generate_source(base_cfg)
        n = len(base)
        block = max(1, cfg.samples_per_symbol)
        envelope_blocks = self.rng.random(int(np.ceil(n / block))) < cfg.burst_probability
        if not envelope_blocks.any():
            envelope_blocks[self.rng.integers(0, len(envelope_blocks))] = True
        envelope = np.repeat(envelope_blocks.astype(float), block)[:n]
        if cfg.burst_duty_cycle > 0:
            smooth_len = max(1, int(block * cfg.burst_duty_cycle))
            kernel = np.ones(smooth_len) / smooth_len
            envelope = np.convolve(envelope, kernel, mode="same")
        meta.update({"source_type": "BURSTY", "burst_probability": cfg.burst_probability})
        return base * envelope, symbols, meta

    def _generate_recorded_placeholder(self, cfg: SourceConfig):
        if cfg.recorded_path:
            path = Path(cfg.recorded_path)
            if path.exists() and path.suffix.lower() == ".npy":
                wave = np.load(path).astype(np.complex128)
                symbols = wave[:: cfg.samples_per_symbol][: cfg.n_symbols]
                return wave, symbols, self._source_meta(cfg, "RECORDED", recorded_path=str(path))
            if path.exists() and path.suffix.lower() == ".npz":
                data = np.load(path)
                key = "waveform" if "waveform" in data else data.files[0]
                wave = data[key].astype(np.complex128)
                symbols = wave[:: cfg.samples_per_symbol][: cfg.n_symbols]
                return wave, symbols, self._source_meta(cfg, "RECORDED", recorded_path=str(path))
        wave, symbols, meta = self._generate_colored_noise(cfg)
        meta.update({"source_type": "RECORDED", "recorded_path": cfg.recorded_path, "placeholder": True})
        return wave, symbols, meta

    def _source_meta(self, cfg: SourceConfig, source_type: str, **extra) -> Dict[str, Any]:
        meta = {
            "source_type": source_type,
            "n_symbols": cfg.n_symbols,
            "samples_per_symbol": cfg.samples_per_symbol,
        }
        meta.update(extra)
        return meta

    def _bits_to_qpsk(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 2 != 0:
            raise ValueError("QPSK needs an even number of bits")

        bit_pairs = bits.reshape(-1, 2)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j,
        }
        syms = np.array([mapping[tuple(b)] for b in bit_pairs], dtype=np.complex128)
        syms /= np.sqrt(2.0)
        return syms

    def _pulse_shape_rrc(self, symbols: np.ndarray, sps: int, rolloff: float, span_symbols: int) -> np.ndarray:
        taps = rrc_taps(sps=sps, beta=rolloff, span_symbols=span_symbols)
        up = np.zeros(len(symbols) * sps, dtype=np.complex128)
        up[::sps] = symbols
        return np.convolve(up, taps, mode="same")

    def _normalize_complex_power(self, x: np.ndarray) -> np.ndarray:
        p = np.mean(np.abs(x) ** 2) + 1e-12
        return x / np.sqrt(p)

    def _select_mixing_matrix(self, mix_cfg: MixtureConfig) -> np.ndarray:
        mode = str(mix_cfg.mixing_mode).lower()
        if mix_cfg.random_complex_mixing or mode in {"random_complex", "full_random", "complex"}:
            return self._sample_mixing_matrix(mix_cfg.n_rx, random_phase=True)
        if mode in {"real", "amplitude_only"}:
            return self._sample_mixing_matrix(mix_cfg.n_rx, random_phase=False)
        if mode in {"identity", "sum"}:
            return np.ones((mix_cfg.n_rx, 2), dtype=np.complex128) / np.sqrt(max(1, mix_cfg.n_rx))
        return self._sample_phase_change_matrix(mix_cfg.n_rx, mix_cfg.phase_shift_deg)

    def _sample_mixing_matrix(self, n_rx: int, random_phase: bool = True) -> np.ndarray:
        mags = self.rng.uniform(0.8, 1.2, size=(n_rx, 2))
        if random_phase:
            phases = self.rng.uniform(0.0, 2.0 * np.pi, size=(n_rx, 2))
            H = mags * np.exp(1j * phases)
        else:
            H = mags.astype(np.complex128)
        H /= np.linalg.norm(H, axis=0, keepdims=True) + 1e-12
        return H

    def _sample_phase_change_matrix(self, n_rx: int, phase_change_deg: float = 5.0) -> np.ndarray:
        phase_step = np.deg2rad(phase_change_deg)
        antenna_phases = np.arange(n_rx) * phase_step
        phases = np.zeros((n_rx, 2), dtype=np.float64)
        phases[:, 0] = antenna_phases
        phases[:, 1] = -antenna_phases
        return np.exp(1j * phases).astype(np.complex128)

    def _apply_timing_offset(self, x: np.ndarray, offset: int) -> np.ndarray:
        offset = int(offset)
        if offset == 0:
            return x
        y = np.zeros_like(x)
        if offset > 0:
            y[offset:] = x[:-offset]
        else:
            y[:offset] = x[-offset:]
        return y

    def _apply_carrier_phase_mismatch(
        self,
        x: np.ndarray,
        carrier_offset: float = 0.0,
        phase_mismatch_deg: float = 0.0,
    ) -> np.ndarray:
        if carrier_offset == 0.0 and phase_mismatch_deg == 0.0:
            return x
        n = np.arange(len(x), dtype=np.float64)
        phase0 = np.deg2rad(phase_mismatch_deg)
        return x * np.exp(1j * (2.0 * np.pi * carrier_offset * n + phase0))

    def _apply_rx_amplitude_imbalance(self, signal: np.ndarray, imbalance_db: float) -> np.ndarray:
        if imbalance_db == 0.0 or signal.ndim == 1:
            return signal
        offsets_db = self.rng.uniform(-abs(imbalance_db), abs(imbalance_db), size=(signal.shape[0], 1))
        gains = 10 ** (offsets_db / 20.0)
        return signal * gains

    def _resample_complex_to_len(self, x: np.ndarray, target_len: int) -> np.ndarray:
        if len(x) == target_len:
            return x
        old_idx = np.linspace(0.0, 1.0, len(x))
        new_idx = np.linspace(0.0, 1.0, target_len)
        x_real = np.interp(new_idx, old_idx, np.real(x))
        x_imag = np.interp(new_idx, old_idx, np.imag(x))
        return x_real + 1j * x_imag

    def _sample_source_config(self, family_cfg: SourceFamilyConfig, source_types: Sequence[str]) -> SourceConfig:
        n_min, n_max = family_cfg.n_symbols_range
        n_symbols = int(self.rng.integers(n_min, n_max + 1))
        sps = int(self.rng.choice(family_cfg.samples_per_symbol_choices))
        rolloff = self._sample_uniform(*family_cfg.rolloff_range)
        return SourceConfig(
            source_type=str(self.rng.choice(source_types)),
            n_symbols=n_symbols,
            samples_per_symbol=sps,
            rolloff=rolloff,
            rrc_span_symbols=family_cfg.rrc_span_symbols,
            normalize_power=family_cfg.normalize_power,
            num_channels=family_cfg.n_rx,
        )

    def _sample_uniform(self, lo: Optional[float], hi: Optional[float]) -> Optional[float]:
        if lo is None or hi is None:
            return None
        if lo == hi:
            return float(lo)
        return float(self.rng.uniform(lo, hi))

    def symbols_to_bits(self, symbols):
        bits = []
        for s in symbols:
            ascii_val = ord(s)
            b = format(ascii_val, "08b")
            bits.extend([int(bit) for bit in b])
        return np.array(bits)
